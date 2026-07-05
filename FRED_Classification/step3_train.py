#!/usr/bin/env python3
"""
Step 3: Train and evaluate the FRED pulse classifier.

Run:
    python step3_train.py

Outputs:
    grb_data/classifier_results.png   – confusion matrix + feature importance
    grb_data/fred_classifier.pkl      – saved model (use for prediction)
    grb_data/predictions.csv          – FRED probability for all labeled bursts

Usage after training (predict on new bursts):
    from step3_train import predict_new
    df = predict_new('grb_data/manifest_new.json')
    print(df[['trigger_name', 'fred_prob']].head())
"""

import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics         import (confusion_matrix, classification_report,
                                     ConfusionMatrixDisplay, roc_curve, auc)

OUTDIR = '/workspace/data/grb_data'

# Features the model uses — order matters (matches feature extraction)
FEATURE_COLS = [
    'asymmetry',     # core FRED indicator (fast rise → low value)
    'n_peaks',       # FRED → 1
    'skewness',      # FRED → positive (long tail right)
    'kurtosis',
    'exp_tau',       # exponential decay timescale
    'rise_time',
    'decay_time',
    't90_local',
    't90_catalog',
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_dataset(features_csv: str = f'{OUTDIR}/features.csv'):
    df = pd.read_csv(features_csv)
    df = df.dropna(subset=FEATURE_COLS + ['label'])

    X      = df[FEATURE_COLS].values.astype(float)
    y      = df['label'].values.astype(int)
    names  = df['trigger_name'].values

    print(f"Dataset: {len(df)} bursts")
    print(f"  FRED     : {y.sum()}")
    print(f"  non-FRED : {len(y) - y.sum()}")

    if y.sum() < 3 or (len(y) - y.sum()) < 3:
        raise ValueError(
            "Need at least 3 examples of each class. "
            "Label more bursts in step2_features_label.py"
        )
    return X, y, names, df


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
def build_models() -> dict:
    """Return a dict of named sklearn pipelines to compare."""
    return {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    RandomForestClassifier(
                n_estimators  = 300,
                max_depth     = 6,
                class_weight  = 'balanced',
                random_state  = 42,
            )),
        ]),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    LogisticRegression(
                class_weight = 'balanced',
                C            = 1.0,
                max_iter     = 500,
            )),
        ]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING + EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(X: np.ndarray, y: np.ndarray, names: np.ndarray,
                       df: pd.DataFrame) -> Pipeline:
    """
    Cross-validate all models, pick the best, do full evaluation,
    save plots and the trained model.
    """
    n_splits = min(5, int(y.sum()), int(len(y) - y.sum()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = build_models()

    # ── Cross-validation ──────────────────────────────────────────────────────
    print(f"\n── {n_splits}-fold stratified cross-validation ─────────────────")
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring='f1', error_score='raise')
        results[name] = scores
        print(f"  {name:25s}  F1 = {scores.mean():.3f} ± {scores.std():.3f}")

    best_name  = max(results, key=lambda k: results[k].mean())
    best_model = models[best_name]
    print(f"\n→ Best model: {best_name}")

    # ── Full dataset fit ──────────────────────────────────────────────────────
    best_model.fit(X, y)
    y_pred  = best_model.predict(X)
    y_proba = best_model.predict_proba(X)[:, 1]

    print("\n── Classification report (resubstitution on full training set) ──")
    print(classification_report(y, y_pred,
                                 target_names=['non-FRED', 'FRED'],
                                 digits=3))

    # ── Plots ─────────────────────────────────────────────────────────────────
    _make_plots(best_name, best_model, X, y, y_pred, y_proba)

    # ── Save predictions ──────────────────────────────────────────────────────
    out_df = df.copy()
    out_df['fred_prob'] = y_proba
    out_df['fred_pred'] = y_pred
    out_df = out_df.sort_values('fred_prob', ascending=False)
    pred_path = f'{OUTDIR}/predictions.csv'
    out_df.to_csv(pred_path, index=False)
    print(f"\nSaved predictions → {pred_path}")

    print("\n── Top 10 most FRED-like bursts ──────────────────────────────────")
    cols = ['trigger_name', 'fred_prob', 'label', 'asymmetry', 'n_peaks']
    print(out_df[cols].head(10).to_string(index=False))

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = f'{OUTDIR}/fred_classifier.pkl'
    joblib.dump({'model': best_model, 'feature_cols': FEATURE_COLS,
                 'model_name': best_name}, model_path)
    print(f"\nSaved model → {model_path}")

    return best_model


def _make_plots(model_name: str, model: Pipeline,
                X: np.ndarray, y: np.ndarray,
                y_pred: np.ndarray, y_proba: np.ndarray):
    """Confusion matrix + feature importance + ROC curve."""
    n_plots = 3
    fig, axes = plt.subplots(1, n_plots, figsize=(15, 5))
    fig.suptitle(f'FRED Classifier — {model_name}', fontsize=13,
                 fontweight='bold')

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm   = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['non-FRED', 'FRED'])
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Confusion Matrix')

    # ── Feature importance (RF) or coefficients (LR) ─────────────────────────
    clf = model.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        label = 'Feature importance'
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_[0])
        label = '|Coefficient|'
    else:
        importances = np.ones(len(FEATURE_COLS))
        label = 'Uniform'

    idx = np.argsort(importances)
    colors = ['#EF7B45' if importances[i] == importances.max() else '#5EB1BF'
              for i in idx]
    axes[1].barh([FEATURE_COLS[i] for i in idx], importances[idx],
                 color=colors, edgecolor='white', linewidth=0.5)
    axes[1].set_xlabel(label)
    axes[1].set_title('What the model learned')
    axes[1].invert_yaxis()
    # Annotate highest importance feature
    top_feat = FEATURE_COLS[importances.argmax()]
    axes[1].set_xlabel(f'{label}\n(orange = most important: {top_feat})')

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc     = auc(fpr, tpr)
    axes[2].plot(fpr, tpr, color='#EF7B45', lw=2,
                 label=f'ROC (AUC = {roc_auc:.3f})')
    axes[2].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('ROC Curve')
    axes[2].legend(loc='lower right')
    axes[2].set_xlim([0, 1])
    axes[2].set_ylim([0, 1.02])

    plt.tight_layout()
    plot_path = f'{OUTDIR}/classifier_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation plot → {plot_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION ON NEW BURSTS
# ═══════════════════════════════════════════════════════════════════════════════
def predict_new(new_manifest_json: str,
                model_pkl: str = f'{OUTDIR}/fred_classifier.pkl') -> pd.DataFrame:
    """
    Apply a saved model to unlabeled bursts from a new manifest JSON.
    Returns a DataFrame sorted by fred_prob descending.

    Example:
        df = predict_new('grb_data/manifest_new.json')
        fred_candidates = df[df['fred_prob'] > 0.7]
    """
    from step2_features_label import compute_features, load_ctime

    saved  = joblib.load(model_pkl)
    model  = saved['model']
    f_cols = saved['feature_cols']

    with open(new_manifest_json) as f:
        manifest = json.load(f)

    records = []
    for entry in manifest:
        try:
            t, rate = load_ctime(entry['fits'])
            mask    = (t > -60) & (t < 200)
            feats   = compute_features(t[mask], rate[mask],
                                       t90_catalog=entry.get('t90'))
        except Exception:
            continue
        if feats is None:
            continue
        feats['trigger_name'] = entry['trigger_name']
        feats['t90_catalog']  = entry.get('t90', feats.get('t90_local', 0))
        records.append(feats)

    df   = pd.DataFrame(records)
    X    = df[f_cols].fillna(0).values
    proba = model.predict_proba(X)[:, 1]
    df['fred_prob'] = proba
    df['fred_pred'] = model.predict(X)
    return df.sort_values('fred_prob', ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    features_csv = f'{OUTDIR}/features.csv'
    if not os.path.exists(features_csv):
        print("ERROR: features.csv not found. Run step2_features_label.py first.")
        return

    X, y, names, df = load_dataset(features_csv)
    model = train_and_evaluate(X, y, names, df)

    print("\n── Done ─────────────────────────────────────────────────────────")
    print("To apply this model to more bursts:")
    print("  from step3_train import predict_new")
    print("  df = predict_new('grb_data/manifest.json')")
    print("  print(df[df['fred_prob'] > 0.7][['trigger_name','fred_prob']])")


if __name__ == '__main__':
    main()
