#!/usr/bin/env python3
"""
LIGO Gravitational-Wave Signal Search Pipeline

Searches for a compact-binary coalescence chirp in LIGO Livingston (L1)
strain data, measures its properties, and produces diagnostic plots.

Run with:
    python gw_search.py

Outputs:
    figures/            — diagnostic PNG plots
    stdout              — measured parameters (merger time, SNR, chirp mass, etc.)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS = 4096.0                 # Sampling rate (Hz)
NYQUIST = FS / 2.0          # Nyquist frequency (Hz)
DATA_FILE = 'LIGODATA_L.txt'
FIG_DIR = 'figures'

# Band-pass range for compact-binary chirp (Hz)
F_LOW = 20.0
F_HIGH = 400.0

# Physical constants for chirp-mass estimation
G = 6.67430e-11             # m^3 kg^-1 s^-2
C = 2.99792458e8            # m/s
MSUN = 1.98847e30           # kg

# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------
def load_data(filepath=DATA_FILE):
    """Load strain time series from text file."""
    h = np.loadtxt(filepath)
    n = len(h)
    t = np.arange(n) / FS
    print(f"Loaded {n:,} samples  ({n/FS:.2f} s at {FS:.0f} Hz)")
    return t, h

# ---------------------------------------------------------------------------
# 2. Raw Inspection
# ---------------------------------------------------------------------------
def compute_psd(h, seglen=None):
    """Welch PSD estimate."""
    if seglen is None:
        seglen = int(4 * FS)          # 4-second segments
    freqs, psd = signal.welch(h, FS, nperseg=seglen, window='hann',
                               noverlap=seglen//2, average='median')
    return freqs, psd

def plot_raw_timeseries(t, h, savepath=None):
    """Plot raw strain vs time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, h, color='steelblue', lw=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Strain')
    ax.set_title('Raw LIGO Strain Time Series (L1)')
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    plt.close(fig)

def plot_psd(freqs, psd, savepath=None):
    """Log-log PSD plot with band-pass region highlighted."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(freqs, psd, color='steelblue', lw=1.0, label='Median-Welch PSD')
    ax.axvline(F_LOW, color='red', ls='--', lw=1, label=f'{F_LOW:.0f}–{F_HIGH:.0f} Hz band')
    ax.axvline(F_HIGH, color='red', ls='--', lw=1)
    ax.axvspan(F_LOW, F_HIGH, color='red', alpha=0.08)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'PSD (strain$^2$ / Hz)')
    ax.set_title('Power Spectral Density')
    ax.set_xlim([1, NYQUIST])
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    plt.close(fig)

def plot_spectrogram(t, h, title='Spectrogram', savepath=None,
                     nperseg=None, vmin=None, vmax=None):
    """Plot a time-frequency spectrogram."""
    if nperseg is None:
        nperseg = int(0.5 * FS)       # 0.5 s window → ~2 Hz resolution
    f, time_bins, Sxx = signal.spectrogram(
        h, FS, window='hann', nperseg=nperseg,
        noverlap=nperseg//2, scaling='spectrum'
    )
    # Keep only up to ~600 Hz for visual clarity
    fmax_plot = 600.0
    idx_f = f <= fmax_plot

    fig, ax = plt.subplots(figsize=(12, 5))
    # Use dB scale: 10*log10(Sxx)
    Sxx_dB = 10 * np.log10(Sxx[idx_f, :] + 1e-30)
    im = ax.pcolormesh(time_bins, f[idx_f], Sxx_dB,
                       shading='gouraud', cmap='viridis',
                       vmin=vmin, vmax=vmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    ax.set_ylim([0, fmax_plot])
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Power (dB)')
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    plt.close(fig)
    return f, time_bins, Sxx

# ---------------------------------------------------------------------------
# 3. Data Cleaning
# ---------------------------------------------------------------------------
def bandpass(h, f_low=F_LOW, f_high=F_HIGH, order=4):
    """Zero-phase Butterworth band-pass filter."""
    sos = signal.butter(order, [f_low, f_high], btype='band',
                        fs=FS, output='sos')
    return signal.sosfiltfilt(sos, h)

def whiten(h, freqs=None, psd=None, df=None):
    """
    Whiten the strain time series.

    Parameters
    ----------
    h : array
        Input strain.
    freqs, psd : array, optional
        Pre-computed PSD frequencies and values. If None, computed internally.
    df : float
        Frequency resolution of the PSD. Needed only if freqs/psd provided.

    Returns
    -------
    h_white : array
        Whitened time series (normalized so that noise variance ≈ 1).
    """
    n = len(h)
    # FFT
    hf = np.fft.rfft(h)
    freq_rfft = np.fft.rfftfreq(n, d=1.0/FS)

    if freqs is None or psd is None:
        freqs, psd = compute_psd(h)
        df = freqs[1] - freqs[0]

    # Interpolate PSD onto FFT grid
    # Extrapolate with nearest to avoid zeros at edges
    psd_interp = np.interp(freq_rfft, freqs, psd,
                           left=psd[0], right=psd[-1])

    # Avoid division by zero / very small numbers
    psd_interp = np.maximum(psd_interp, np.max(psd_interp) * 1e-10)

    # Whiten: divide by sqrt(PSD), normalize by sqrt(df) for discrete sum
    # After this, noise in each frequency bin has unit variance.
    hf_white = hf / np.sqrt(psd_interp) * np.sqrt(2 * df)

    # Zero out DC and Nyquist to avoid artifacts
    hf_white[0] = 0.0
    if n % 2 == 0:
        hf_white[-1] = 0.0

    h_white = np.fft.irfft(hf_white, n=n)
    return h_white

# ---------------------------------------------------------------------------
# 4. Signal Identification
# ---------------------------------------------------------------------------
def find_merger_time(t, h_clean):
    """
    Estimate merger time from the peak of |h_clean|.
    Returns the index and time of the peak.
    """
    idx_peak = np.argmax(np.abs(h_clean))
    t_peak = t[idx_peak]
    return idx_peak, t_peak

def extract_frequency_track(h_seg, fs=FS):
    """
    Extract instantaneous frequency via the analytic signal (Hilbert transform).
    Returns frequency array and amplitude envelope.
    """
    analytic = signal.hilbert(h_seg)
    amplitude = np.abs(analytic)
    # Instantaneous phase -> unwrap -> derivative -> frequency
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
    # inst_freq is one sample shorter; pad by repeating last value
    inst_freq = np.append(inst_freq, inst_freq[-1])
    return inst_freq, amplitude

# ---------------------------------------------------------------------------
# 5. Parameter Measurement
# ---------------------------------------------------------------------------
def estimate_snr(h_white, idx_peak, window_s=0.5):
    """
    Estimate SNR from the whitened strain.
    Method: compare peak amplitude to the standard deviation of a
    quiet stretch well away from the signal.
    """
    n = len(h_white)
    win = int(window_s * FS)
    # Exclude a ±2 s region around the peak for noise estimation
    exclude = int(2.0 * FS)
    i0 = max(0, idx_peak - exclude)
    i1 = min(n, idx_peak + exclude)

    # Use the first 8 s and last 8 s as quiet reference
    ref_len = int(8.0 * FS)
    noise_samples = np.concatenate([h_white[:ref_len], h_white[-ref_len:]])
    sigma = np.std(noise_samples)

    peak_amp = np.abs(h_white[idx_peak])
    snr = peak_amp / sigma
    return snr, sigma, peak_amp

def estimate_chirp_mass(f_track, t_track):
    """
    Estimate chirp mass from the frequency evolution during inspiral.

    Uses the Newtonian inspiral relation:
        df/dt = (96/5) * π^(8/3) * (G M_c / c^3)^(5/3) * f^(11/3)

    Parameters
    ----------
    f_track : array
        Instantaneous GW frequency (Hz).
    t_track : array
        Time stamps (s), same length as f_track.

    Returns
    -------
    M_c_msun : float
        Chirp mass in solar masses (median of valid estimates).
    """
    # Smooth the frequency track slightly before differentiating
    f_smooth = median_filter(f_track, size=5)

    # Numerical derivative df/dt
    df_dt = np.gradient(f_smooth, t_track)

    # Only use points where frequency is increasing (inspiral)
    # and within a reasonable range
    valid = (df_dt > 0) & (f_smooth > 30) & (f_smooth < 300)

    if np.sum(valid) < 5:
        return np.nan

    f_v = f_smooth[valid]
    dfdt_v = df_dt[valid]

    # Rearrange Newtonian formula:
    #   M_c = (c^3/G) * [ (5/96) * π^(-8/3) * f^(-11/3) * df/dt ]^(3/5)
    coeff = (5.0 / 96.0) * np.pi**(-8.0/3.0) * f_v**(-11.0/3.0) * dfdt_v
    M_c_kg = (C**3 / G) * coeff**(3.0/5.0)
    M_c_msun = M_c_kg / MSUN

    # Return median; robust against outliers
    return float(np.median(M_c_msun))

# ---------------------------------------------------------------------------
# 6. Main Pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- Load ---
    print("=" * 60)
    print("LIGO GW Signal Search Pipeline")
    print("=" * 60)
    t, h = load_data(DATA_FILE)

    # --- Raw inspection plots ---
    print("\n[1/6] Generating raw inspection plots...")
    plot_raw_timeseries(t, h, savepath=os.path.join(FIG_DIR, '01_raw_timeseries.png'))

    freqs, psd = compute_psd(h)
    plot_psd(freqs, psd, savepath=os.path.join(FIG_DIR, '02_psd.png'))

    plot_spectrogram(t, h, title='Spectrogram (Raw Strain)',
                     savepath=os.path.join(FIG_DIR, '03_spectrogram_raw.png'))

    # --- Clean ---
    print("[2/6] Band-pass filtering (%.0f–%.0f Hz)..." % (F_LOW, F_HIGH))
    h_bp = bandpass(h)

    print("[3/6] Whitening strain...")
    # Use the PSD from the full data; signal is short so median-Welch is noise-dominated
    h_white = whiten(h, freqs=freqs, psd=psd, df=freqs[1]-freqs[0])
    # Also band-pass the whitened strain so we only look in the chirp band
    h_white_bp = bandpass(h_white)

    # --- Whitened inspection plots ---
    print("[4/6] Generating whitened inspection plots...")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, h_white_bp, color='darkgreen', lw=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Whitened Strain (σ)')
    ax.set_title('Whitened & Band-passed Strain')
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '04_whitened_timeseries.png'), dpi=150)
    plt.close(fig)

    plot_spectrogram(t, h_white_bp, title='Spectrogram (Whitened & Band-passed)',
                     savepath=os.path.join(FIG_DIR, '05_spectrogram_whitened.png'))

    # --- Identify signal ---
    print("[5/6] Identifying signal candidate...")
    idx_peak, t_peak = find_merger_time(t, h_white_bp)

    # Zoom window around the signal: ±1.5 s
    zoom_pad = 1.5
    z0 = max(0, int((t_peak - zoom_pad) * FS))
    z1 = min(len(t), int((t_peak + zoom_pad) * FS))
    t_zoom = t[z0:z1]
    h_zoom = h_white_bp[z0:z1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1]})
    axes[0].plot(t_zoom, h_zoom, color='darkgreen', lw=0.8)
    axes[0].axvline(t_peak, color='red', ls='--', lw=1.5, label=f'Merger t = {t_peak:.3f} s')
    axes[0].set_ylabel('Whitened Strain (σ)')
    axes[0].set_title('Zoomed Signal Candidate')
    axes[0].legend()

    # Spectrogram zoom
    nperseg_zoom = int(0.1 * FS)   # 0.1 s → 10 Hz resolution, good for zoom
    f_z, t_z, Sxx_z = signal.spectrogram(
        h_white_bp, FS, window='hann', nperseg=nperseg_zoom,
        noverlap=nperseg_zoom//2, scaling='spectrum'
    )
    # Select time range
    tmask = (t_z >= t_peak - zoom_pad) & (t_z <= t_peak + zoom_pad)
    fmask = f_z <= 500
    Sxx_z_dB = 10 * np.log10(Sxx_z[np.ix_(fmask, tmask)] + 1e-30)
    im = axes[1].pcolormesh(t_z[tmask], f_z[fmask], Sxx_z_dB,
                            shading='gouraud', cmap='viridis')
    axes[1].axvline(t_peak, color='red', ls='--', lw=1.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_ylim([0, 500])
    fig.colorbar(im, ax=axes[1], pad=0.01, label='Power (dB)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '06_signal_zoom.png'), dpi=150)
    plt.close(fig)

    # --- Measure parameters ---
    print("[6/6] Measuring signal properties...")

    # SNR
    snr, sigma_noise, peak_amp = estimate_snr(h_white_bp, idx_peak)

    # Duration & frequency track from band-passed (not whitened) strain for physics
    # Extract a longer segment around the peak for the Hilbert analysis
    seg_pad = 2.0
    s0 = max(0, int((t_peak - seg_pad) * FS))
    s1 = min(len(t), int((t_peak + seg_pad) * FS))
    h_seg = h_bp[s0:s1]
    t_seg = t[s0:s1]

    inst_freq, inst_amp = extract_frequency_track(h_seg)

    # Define inspiral region: where amplitude is significant and frequency is rising
    # Use the amplitude envelope to find where the signal "turns on"
    amp_norm = inst_amp / np.max(inst_amp)
    # Find where amplitude first exceeds ~10% of peak (rough threshold)
    above_thresh = np.where(amp_norm > 0.1)[0]
    if len(above_thresh) > 0:
        idx_inspiral_start = above_thresh[0]
        t_inspiral_start = t_seg[idx_inspiral_start]
    else:
        idx_inspiral_start = 0
        t_inspiral_start = t_seg[0]

    duration = t_peak - t_inspiral_start

    # Frequency range during inspiral
    f_inspiral = inst_freq[idx_inspiral_start:np.argmax(inst_amp)]
    if len(f_inspiral) > 0:
        f_min = np.percentile(f_inspiral, 5)   # 5th percentile to avoid noise floor
        f_max = np.percentile(f_inspiral, 95)  # 95th percentile before merger
    else:
        f_min = np.nan
        f_max = np.nan

    # Chirp mass
    # Use the inspiral portion where frequency is monotonically increasing
    # and above a reasonable threshold
    chirp_mass = estimate_chirp_mass(inst_freq, t_seg)

    # --- Report ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Merger time          : {t_peak:.4f} s")
    print(f"Duration in band     : {duration:.4f} s")
    print(f"Frequency range      : {f_min:.1f} – {f_max:.1f} Hz")
    print(f"Peak whitened amp    : {peak_amp:.2f} σ")
    print(f"Noise σ (reference)  : {sigma_noise:.3f}")
    print(f"SNR estimate         : {snr:.1f}")
    if not np.isnan(chirp_mass):
        print(f"Chirp mass (M_c)     : {chirp_mass:.2f} M☉")
    else:
        print("Chirp mass (M_c)     : could not be estimated")
    print("=" * 60)
    print(f"\nFigures saved to ./{FIG_DIR}/")


if __name__ == '__main__':
    main()
