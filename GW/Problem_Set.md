Problem 2 — Search for a gravitational-wave signal in LIGO data
The file LIGODATA_L.txt is a strain time series h(t) from a LIGO detector (Livingston, L1), sampled at 4096
Hz. Each value is the dimensionless strain — you can think of it as the instantaneous amplitude of the passing
wave. Somewhere in this stretch there may be a real astrophysical signal (most likely the chirp of a compact-
binary coalescence) buried in detector noise. Find it and measure it.
Suggested steps:
(a) Inspect the data. Look at the raw time series, the power spectral density (PSD), and a time–frequency
spectrogram. The Nyquist frequency here is 2048 Hz.
(b) Clean the data. Whiten and/or band-pass the strain (a compact-binary chirp typically sweeps from
tens of Hz up to a few hundred Hz) to suppress the loud low-frequency and narrow-line noise so any real
signal stands out.
(c) Identify the candidate. If it is a chirp, describe qualitatively how its frequency and amplitude evolve as
the binary inspirals and merges.
(d) Measure its properties. Report the merger time, the duration in band, the frequency range, and an
SNR estimate. If you can, estimate the chirp mass from how fast the frequency sweeps upward.
Deliverables: a spectrogram or filtered time series showing the signal; the merger time and frequency
evolution; an SNR estimate; and (bonus) a chirp-mass value.

Solution :
Load the 4096 Hz strain, estimate the noise PSD (e.g. Welch), whiten the data (divide each Fourier component
by the noise amplitude), then band-pass ~30–400 Hz. A Q-transform / spectrogram of the whitened data
should reveal a rising “chirp” track — frequency and amplitude both increasing toward merger, followed by a
rapid cutoff (ringdown). The chirp mass is defined from the component masses as
M_c = (m₁ m₂)^(3/5) / (m₁ + m₂)^(1/5)

It sets how fast the frequency sweeps; at leading (Newtonian) order
df/dt = (96/5) * π^(8/3) * (GM_c/c³)^(5/3) * f^(11/3)

so measuring f and df/dt from the spectrogram yields ℳc. A stellar-mass binary black hole gives ℳc of order
tens of solar masses and a signal lasting a fraction of a second in band. SNR is properly obtained by matched
filtering against templates, but can be estimated from the whitened peak amplitude relative to the noise.

