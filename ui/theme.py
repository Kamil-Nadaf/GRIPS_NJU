"""Light CSS polish on top of Streamlit native themes (config.toml).

Widget / sidebar colors come from ``.streamlit/config.toml``
(``[theme.light]``, ``[theme.dark]``, ``[theme.*.sidebar]``).
Users switch Light ↔ Dark via the Streamlit menu (☰ → Settings).
"""

from __future__ import annotations


def inject_theme(theme_name: str = 'light', compact: bool = False) -> str:
    """Return supplemental CSS (fonts, cards). Does not recolor widgets."""
    dark = theme_name == 'dark'
    pad = '0.55rem' if compact else '0.85rem'
    gap = '0.6rem' if compact else '1rem'
    hero = '#e8e4f0' if dark else '#3f3e72'
    muted = '#b0a8bc' if dark else '#8a8494'
    card_bg = '#1f1c2a' if dark else '#ffffff'
    card_border = '#3a3650' if dark else '#e0dce8'
    card_text = '#f3eef5' if dark else '#3d3a45'
    shadow = (
        '0 10px 28px rgba(0, 0, 0, 0.35)' if dark
        else '0 10px 28px rgba(63, 62, 114, 0.08)'
    )
    blush = '#f3b9c155'
    sky = '#acc8e455'
    seafoam = '#addbd444'
    page_tint = (
        f'radial-gradient(1100px 480px at 6% -8%, {blush}, transparent 60%),'
        f'radial-gradient(900px 400px at 94% 0%, {sky}, transparent 55%),'
        f'radial-gradient(700px 360px at 70% 100%, {seafoam}, transparent 50%)'
    )
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
  font-family: 'IBM Plex Sans', sans-serif;
}}

.stApp {{
  background-image: {page_tint};
  background-attachment: fixed;
}}

[data-testid="stHeader"] {{
  background: transparent;
}}

.gbm-hero {{
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 650;
  font-size: clamp(1.8rem, 3vw, 2.6rem);
  line-height: 1.15;
  color: {hero};
  margin: 0 0 0.25rem 0;
  letter-spacing: -0.02em;
}}

.gbm-sub {{
  color: {muted};
  font-size: 0.95rem;
  margin-bottom: {gap};
}}

.gbm-card {{
  background: {card_bg};
  border: 1px solid {card_border};
  border-radius: 20px;
  padding: {pad} 1.1rem;
  margin-bottom: {gap};
  box-shadow: {shadow};
}}

.gbm-card h3, .gbm-card h4 {{
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 550;
  margin: 0 0 0.4rem 0;
  color: {card_text};
}}

.gbm-kicker {{
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.72rem;
  color: {muted};
  margin-bottom: 0.35rem;
}}

.block-container {{
  padding-top: 1.2rem;
  max-width: 1200px;
}}

div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: 0.35rem;
}}

div[data-testid="stTabs"] button[data-baseweb="tab"] {{
  border-radius: 999px !important;
  padding: 0.35rem 0.9rem !important;
}}
"""
