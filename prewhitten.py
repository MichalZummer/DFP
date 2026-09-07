"""
TEST11 - frequency extraction on the multi-sector PDCSAP light curves.

This is the TEST09 prewhitening code (the ECHO-like one from cell 5), kept as
close to the original as possible. Same criteria:
    SNR_LIMIT = 4.0, NOISE_WINDOW = 1.0, ALPHA = 0.8 (Antoci amplitude test),
    simultaneous L-BFGS refinement of all frequencies, Rayleigh rejection.

Changes forced by the longer baselines:
  - MAX_FREQS lowered from 100 to 50. With ~200k points a full simultaneous
    fit at 100 frequencies costs minutes per step; 50 is far more than the
    ~11 envelope peaks the single-sector data gave.
  - oversample_factor 5 -> 3. The baseline is up to 2600 d, so the Rayleigh
    resolution is ~0.0004 d^-1 and a 0-150 d^-1 periodogram already has
    >1e6 bins. The peak is refined by L-BFGS afterwards anyway.

Resumable: a star already written is skipped.
"""

import os
import glob
import time as clock

import numpy as np
import lightkurve as lk
from scipy.optimize import minimize


def prewhitten(
    INPUT_DIR="lightcurves",
    OUTDIR="periodograms",
    SNR_LIMIT=4.0,
    NOISE_WINDOW=1.0,
    ALPHA=0.8,
    MAX_FREQS=50,
    MAX_FREQUENCY=150,
    OVERSAMPLE=3,
    TIME_BUDGET=900,
):

    os.makedirs(OUTDIR, exist_ok=True)

    if os.path.isfile(INPUT_DIR):
        files = [INPUT_DIR]
    else:
        files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))

    print("INPUT:", INPUT_DIR)
    print("FILES:", files)
    print(f"{len(files)} light curves")

    for nfile, file in enumerate(files, 1):

        name = os.path.basename(file).replace(".txt", "")
        out = os.path.join(OUTDIR, f"{name}_frequencies.txt")

        if os.path.exists(out):
            continue

        t_start = clock.time()

        try:
            data = np.loadtxt(file, skiprows=1)

            time = data[:, 0]
            flux = data[:, 1]

            time = time - np.mean(time)

            baseline = time.max() - time.min()
            cadence = np.median(np.diff(np.sort(time)))

            rayleigh = 1 / baseline

            fmin = rayleigh
            fmax = min(MAX_FREQUENCY, 1 / (2 * cadence))

            lc = lk.LightCurve(time=time, flux=flux)

            pg = lc.to_periodogram(
                normalization="amplitude",
                minimum_frequency=fmin,
                maximum_frequency=fmax,
                oversample_factor=OVERSAMPLE
            )

            original_frequency = pg.frequency.value
            original_amplitude = pg.power.value * 1e6

            df = original_frequency[1] - original_frequency[0]

            # ---------------- simultaneous linear fit ----------------
            def fit_all(freqs):
                X = np.ones((len(time), 1 + 2 * len(freqs)))
                for i, f in enumerate(freqs):
                    phase = 2 * np.pi * f * time
                    X[:, 1 + 2 * i] = np.sin(phase)
                    X[:, 2 + 2 * i] = np.cos(phase)
                coef = np.linalg.lstsq(X, flux, rcond=None)[0]
                return X @ coef, coef

            def error_gradient(freqs):
                model, coef = fit_all(freqs)
                residual = flux - model
                error = np.sum(residual ** 2)
                gradient = np.zeros(len(freqs))
                for i, f in enumerate(freqs):
                    a, b = coef[1 + 2 * i], coef[2 + 2 * i]
                    phase = 2 * np.pi * f * time
                    dmodel = 2 * np.pi * time * (
                        a * np.cos(phase) - b * np.sin(phase)
                    )
                    gradient[i] = -2 * np.sum(residual * dmodel)
                return error, gradient

            # ---------------- prewhitening ----------------
            frequencies_found = []
            blocked = []
            amplitudes = []
            noises = []
            snrs = []

            residual = flux - np.mean(flux)

            for n in range(MAX_FREQS):

                if clock.time() - t_start > TIME_BUDGET:
                    print("   time budget reached", flush=True)
                    break

                pg = lk.LightCurve(time=time, flux=residual).to_periodogram(
                    normalization="amplitude",
                    minimum_frequency=fmin,
                    maximum_frequency=fmax,
                    oversample_factor=OVERSAMPLE
                )

                frequency = pg.frequency.value
                amplitude = pg.power.value * 1e6

                allowed = np.ones(len(frequency), dtype=bool)
                for f in frequencies_found:
                    allowed &= (np.abs(frequency - f) >= rayleigh)
                for f in blocked:
                    allowed &= (np.abs(frequency - f) >= rayleigh)

                if not np.any(allowed):
                    break

                amplitude_search = amplitude.copy()
                amplitude_search[~allowed] = -np.inf

                freq_guess = frequency[np.argmax(amplitude_search)]

                old_freqs = frequencies_found.copy()
                frequencies_found.append(freq_guess)

                start = np.array(frequencies_found)
                bounds = [
                    (max(fmin, f - 2 * df), min(fmax, f + 2 * df))
                    for f in start
                ]

                result = minimize(
                    error_gradient,
                    start,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    options={"maxiter": 10, "ftol": 1e-10}
                )

                frequencies_found = list(result.x)
                new_freq = frequencies_found[-1]

                # Rayleigh test
                if len(frequencies_found) > 1:
                    distance = np.abs(
                        np.array(frequencies_found[:-1]) - new_freq
                    )
                    if np.min(distance) < rayleigh:
                        blocked.append(new_freq)
                        frequencies_found = old_freqs
                        if frequencies_found:
                            model, coef = fit_all(frequencies_found)
                            residual = flux - model
                        else:
                            residual = flux - np.mean(flux)
                        continue

                model, coef = fit_all(frequencies_found)
                residual = flux - model

                a, b = coef[-2], coef[-1]
                amp = np.sqrt(a * a + b * b) * 1e6

                low = max(fmin, new_freq - NOISE_WINDOW / 2)
                high = min(fmax, new_freq + NOISE_WINDOW / 2)

                noise_pg = lk.LightCurve(
                    time=time,
                    flux=residual
                ).to_periodogram(
                    normalization="amplitude",
                    minimum_frequency=low,
                    maximum_frequency=high,
                    oversample_factor=OVERSAMPLE
                )

                noise = np.mean(noise_pg.power.value) * 1e6
                snr = amp / noise

                amp_original = np.interp(
                    new_freq,
                    original_frequency,
                    original_amplitude
                )
                ratio = amp / amp_original

                if snr < SNR_LIMIT:
                    frequencies_found = old_freqs
                    break

                if not (ALPHA < ratio < 1 / ALPHA):
                    blocked.append(new_freq)
                    frequencies_found = old_freqs
                    if frequencies_found:
                        model, coef = fit_all(frequencies_found)
                        residual = flux - model
                    else:
                        residual = flux - np.mean(flux)
                    continue

                amplitudes.append(amp)
                noises.append(noise)
                snrs.append(snr)

            # ---------------- final fit and output ----------------
            if frequencies_found:
                model, coef = fit_all(frequencies_found)
                amps = [
                    np.sqrt(
                        coef[1 + 2 * i] ** 2 +
                        coef[2 + 2 * i] ** 2
                    ) * 1e6
                    for i in range(len(frequencies_found))
                ]

                k = min(len(frequencies_found), len(noises))

                order = np.argsort(frequencies_found[:k])

                with open(out, "w") as f:
                    f.write(
                        "frequency_d-1\tamplitude_ppm\tnoise_ppm\tSNR\n"
                    )
                    for i in order:
                        f.write(
                            f"{frequencies_found[i]:.8f}\t"
                            f"{amps[i]:.3f}\t"
                            f"{noises[i]:.3f}\t"
                            f"{snrs[i]:.2f}\n"
                        )

                print(
                    f"{nfile:>4} {name}: {k} freqs, "
                    f"baseline {baseline:.0f} d, "
                    f"{clock.time() - t_start:.0f} s",
                    flush=True
                )

            else:
                open(out, "w").write(
                    "frequency_d-1\tamplitude_ppm\tnoise_ppm\tSNR\n"
                )
                print(
                    f"{nfile:>4} {name}: none",
                    flush=True
                )

        except Exception as e:
            print(
                f"{nfile:>4} {name}: FAILED {str(e)[:60]}",
                flush=True
            )

    print("\nFinished")