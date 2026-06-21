"""
Beam cross-section comparison: original polycap C library vs PolyCap.hpp.
Reads the CSVs produced by Test2 (beam_lib_E*.csv, beam_polycap_E*.csv).

Layout per energy:  2 rows x 5 columns
    Row 0 - polycap (C library, reference)   (forestgreen)
    Row 1 - PolyCap.hpp                       (firebrick)
  Columns - exit window | f/2 | focal | 3f/2 | 2f

Run from the project root after building and executing Test2:
    python3 src/tests/plot_beam_comparison.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── geometry constants (must match Test-2.cpp) ────────────────────────────────
PC_FOCAL_OUT = 0.49          # cm
PC_LENGTH    = 4.03          # cm  (for labelling only)
ENERGIES     = [4.0, 6.0, 8.0, 10.0, 12.0]

PLANE_LABELS = [
    "Exit window\nd = 0",
    f"d = f/2\n({PC_FOCAL_OUT*0.5*10:.2f} mm)",
    f"d = focal\n({PC_FOCAL_OUT*10:.1f} mm)",
    f"d = 3f/2\n({PC_FOCAL_OUT*1.5*10:.2f} mm)",
    f"d = 2f\n({PC_FOCAL_OUT*2*10:.1f} mm)",
]
N_PLANES = len(PLANE_LABELS)

ROWS = [
    ("lib",     "polycap (C lib)", "forestgreen"),
    ("polycap", "PolyCap.hpp",     "firebrick"),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_beam(fname):
    """Return (plane_arr, x_mm_arr, y_mm_arr, w_arr) or None if file missing/empty."""
    if not os.path.exists(fname):
        return None
    data = np.loadtxt(fname, delimiter=",", skiprows=1)
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 0].astype(int), data[:, 1] * 10.0, data[:, 2] * 10.0, data[:, 3]


def beam_rms_mm(x, y):
    return np.sqrt(np.mean(x**2 + y**2)) if len(x) else 0.0


def scatter_plane(ax, x_mm, y_mm, color, n_max=15_000):
    n = len(x_mm)
    if n == 0:
        ax.text(0.5, 0.5, "no photons", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        return
    if n > n_max:
        idx = np.random.default_rng(0).choice(n, n_max, replace=False)
        x_mm, y_mm = x_mm[idx], y_mm[idx]
    ax.scatter(x_mm, y_mm, s=0.8, alpha=0.35, color=color, linewidths=0, rasterized=True)
    ax.text(0.02, 0.97, f"RMS={beam_rms_mm(x_mm, y_mm):.3f} mm\nn={n:,}",
            transform=ax.transAxes, fontsize=7, va="top",
            color="black", bbox=dict(fc="white", ec="none", alpha=0.6))


def sym_lim(vals_list, margin=0.1):
    flat = np.concatenate([v for v in vals_list if v is not None and len(v)])
    if len(flat) == 0:
        return 1.0
    p = np.percentile(np.abs(flat), 99)
    return p * (1 + margin) if p > 0 else 1.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(0)
    saved = []
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(base_dir, "test-data", "out")

    for energy in ENERGIES:
        datasets = {key: load_beam(os.path.join(out_dir, f"beam_{key}_E{energy:.1f}.csv"))
                    for key, _, _ in ROWS}
        if all(v is None for v in datasets.values()):
            print(f"  E={energy:.1f} keV - no CSV files found, skipping")
            continue

        fig, axes = plt.subplots(len(ROWS), N_PLANES, figsize=(4 * N_PLANES, 4 * len(ROWS)),
                                 squeeze=False)
        fig.suptitle(
            f"Beam cross-section  --  E = {energy:.1f} keV\n"
            f"pc-236  (L={PC_LENGTH} cm, f_out={PC_FOCAL_OUT} cm)  --  "
            f"polycap (C lib) vs PolyCap.hpp",
            fontsize=13, y=1.01,
        )

        # Per-column symmetric axis limits (same across both rows → fair comparison)
        col_lims = []
        for col in range(N_PLANES):
            xs, ys = [], []
            for key, _, _ in ROWS:
                data = datasets[key]
                if data is None:
                    continue
                plane_arr, x_mm, y_mm, _ = data
                mask = plane_arr == col
                xs.append(x_mm[mask]); ys.append(y_mm[mask])
            col_lims.append(sym_lim(xs + ys))

        for row_idx, (key, label, color) in enumerate(ROWS):
            data = datasets[key]
            for col in range(N_PLANES):
                ax = axes[row_idx, col]
                if data is not None:
                    plane_arr, x_mm, y_mm, _ = data
                    mask = plane_arr == col
                    scatter_plane(ax, x_mm[mask], y_mm[mask], color)
                else:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="gray")
                lim = col_lims[col]
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                ax.set_aspect("equal")
                ax.xaxis.set_major_locator(ticker.MaxNLocator(4, symmetric=True))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4, symmetric=True))
                ax.tick_params(labelsize=7)
                if row_idx == 0:
                    ax.set_title(PLANE_LABELS[col], fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{label}\nY [mm]", fontsize=9)
                if row_idx == len(ROWS) - 1:
                    ax.set_xlabel("X [mm]", fontsize=9)

        plt.tight_layout()
        out = os.path.join(out_dir, f"beam_comparison_E{energy:.1f}keV.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(out)
        print(f"  Saved {out}")

    if saved:
        print(f"\nGenerated {len(saved)} figure(s).")
    else:
        print("\nNo beam CSV files found.  Run ./build/src/Test2 first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
