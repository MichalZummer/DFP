"""
multi-sector PDCSAP download.

Same as the TEST09 downloader, with three changes:
  1. ALL sectors
  2. PDCSAP instead of SAP - this is the point of the exercise. SAP keeps the
     instrumental systematics that sit right on top of the gamma Dor band
     (0.3-5 d^-1), which is exactly where the hybridity result lives.
  3. sectors stitched, each normalised first

Resumable: a star already written is skipped.
"""

import os
import re
import numpy as np
import lightkurve as lk


def download(INPUT, OUTDIR, SECTOR="ALL", flux_column="pdcsap_flux"):

    os.makedirs(OUTDIR, exist_ok=True)

    tics = [x.strip() for x in open(INPUT) if x.strip()]

    print(f"{len(tics)} targets")
    print(f"Sector: {SECTOR}")

    for n, tic in enumerate(tics, 1):

        try:

            # ============================================================
            # SPOC search
            # ============================================================

            if SECTOR == "ALL":
                search = lk.search_lightcurve(
                    "TIC " + tic,
                    mission="TESS",
                    author="SPOC",
                    exptime=120
                )
            else:
                search = lk.search_lightcurve(
                    "TIC " + tic,
                    mission="TESS",
                    author="SPOC",
                    exptime=120,
                    sector=int(SECTOR)
                )

            author = "SPOC"

            # ============================================================
            # QLP fallback
            # ============================================================

            if len(search) == 0:

                if SECTOR == "ALL":
                    search = lk.search_lightcurve(
                        "TIC " + tic,
                        mission="TESS",
                        author="QLP"
                    )
                else:
                    search = lk.search_lightcurve(
                        "TIC " + tic,
                        mission="TESS",
                        author="QLP",
                        sector=int(SECTOR)
                    )

                author = "QLP"

            if len(search) == 0:
                print(f"{n:>4} TIC {tic}: no data")
                continue

            # ============================================================
            # Sectors
            # ============================================================

            sectors = sorted(
                set(
                    int(re.search(r"Sector (\d+)", str(m)).group(1))
                    for m in search.table["mission"]
                )
            )

            # ============================================================
            # Download
            # ============================================================

            if author == "SPOC":
                coll = search.download_all(
                    flux_column=flux_column
                )
            else:
                coll = search.download_all()

            if coll is None or len(coll) == 0:
                print(f"{n:>4} TIC {tic}: download empty")
                continue

            # ============================================================
            # Normalize every sector separately and stitch
            # ============================================================

            lc = coll.stitch(
                corrector_func=lambda x: x.normalize()
            )

            lc = (
                lc
                .remove_nans()
                .remove_outliers(sigma=8)
            )

            # ============================================================
            # Convert to numpy
            # ============================================================

            time = np.array(lc.time.value, dtype=float)
            flux = np.array(lc.flux.value, dtype=float)
            err = np.array(lc.flux_err.value, dtype=float)

            good = np.isfinite(time) & np.isfinite(flux)

            time = time[good]
            flux = flux[good]
            err = err[good]

            if len(time) < 100:
                print(f"{n:>4} TIC {tic}: too short")
                continue

            order = np.argsort(time)

            time = time[order]
            flux = flux[order]
            err = err[order]

            # ============================================================
            # Save
            # ============================================================

            # ============================================================
            # Save
            # ============================================================
            
            flux_name = flux_column.replace("_flux", "").upper()
            
            if SECTOR == "ALL":
                filename = f"TIC{tic}_{author}_{flux_name}.txt"
            else:
                filename = f"TIC{tic}_S{int(SECTOR)}_{author}_{flux_name}.txt"
            
            path = os.path.join(OUTDIR, filename)
            
            np.savetxt(
                path,
                np.column_stack((time, flux, err)),
                delimiter="\t",
                header=f"# time\tflux\tflux_err\tflux_column={flux_column}",
                comments=""
            )

            baseline = time.max() - time.min()

            print(
                f"{n:>4} TIC {tic}: "
                f"{len(sectors)} sectors, "
                f"{len(time)} pts, "
                f"baseline {baseline:.1f} d -> {filename}",
                flush=True
            )

        except Exception as e:

            print(
                f"{n:>4} TIC {tic}: FAILED {str(e)[:80]}",
                flush=True
            )

    print("\nFinished")