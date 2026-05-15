"""
Beam cross-section comparison: voxTrace vs polycap
Reads CSV files produced by Test-2 (beam_vt_E*.csv, beam_pc_E*.csv).

Layout per energy:  2 rows × 5 columns
  Row 0 – voxTrace  (blue)
  Row 1 – polycap   (red)
  Columns – exit window | f/2 | focal | 3f/2 | 2f

Run from the project root after building and executing Test2:
    python3 plot_beam_comparison.py
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

PLANE_DISTS = [
    0.0,
    PC_FOCAL_OUT * 0.5,
    PC_FOCAL_OUT,
    PC_FOCAL_OUT * 1.5,
    PC_FOCAL_OUT * 2.0,
]

PLANE_LABELS = [
    f"Exit window\nd = 0",
    f"d = f/2\n({PC_FOCAL_OUT*0.5*10:.2f} mm)",
    f"d = focal\n({PC_FOCAL_OUT*10:.1f} mm)",
    f"d = 3f/2\n({PC_FOCAL_OUT*1.5*10:.2f} mm)",
    f"d = 2f\n({PC_FOCAL_OUT*2*10:.1f} mm)",
]

N_PLANES = len(PLANE_DISTS)

# ── plotting helpers ──────────────────────────────────────────────────────────

def load_beam(fname):
    """Return (plane_arr, x_mm_arr, z_mm_arr, w_arr) or None if file missing."""
    if not os.path.exists(fname):
        return None
    data = np.loadtxt(fname, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    plane = data[:, 0].astype(int)
    x_mm  = data[:, 1] * 10.0   # cm → mm
    z_mm  = data[:, 2] * 10.0
    w     = data[:, 3]
    return plane, x_mm, z_mm, w


def beam_rms_mm(x, z):
    """RMS transverse radius [mm]."""
    if len(x) == 0:
        return 0.0
    return np.sqrt(np.mean(x**2 + z**2))


def scatter_plane(ax, x_mm, z_mm, color, label, n_max=15_000):
    """Scatter plot for one plane; downsample if needed for rendering speed."""
    n = len(x_mm)
    if n == 0:
        ax.text(0.5, 0.5, "no photons", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        return
    if n > n_max:
        idx = np.random.default_rng(0).choice(n, n_max, replace=False)
        x_mm, z_mm = x_mm[idx], z_mm[idx]
    ax.scatter(x_mm, z_mm, s=0.8, alpha=0.35, color=color,
               linewidths=0, rasterized=True)
    rms = beam_rms_mm(x_mm, z_mm)
    ax.text(0.02, 0.97, f"RMS={rms:.3f} mm\nn={n:,}",
            transform=ax.transAxes, fontsize=7, va="top",
            color="black", bbox=dict(fc="white", ec="none", alpha=0.6))


def sym_lim(vals_list, margin=0.1):
    """Symmetric ±lim covering all non-empty arrays with a margin."""
    flat = np.concatenate([v for v in vals_list if v is not None and len(v)])
    if len(flat) == 0:
        return 1.0
    p = np.percentile(np.abs(flat), 99)
    return p * (1 + margin) if p > 0 else 1.0


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    np.random.seed(0)
    saved = []

    for energy in ENERGIES:
        vt_file = f"beam_vt_E{energy:.1f}.csv"
        pc_file = f"beam_pc_E{energy:.1f}.csv"

        vt = load_beam(vt_file)
        pc = load_beam(pc_file)

        if vt is None and pc is None:
            print(f"  E={energy:.1f} keV – no CSV files found, skipping")
            continue

        fig, axes = plt.subplots(2, N_PLANES, figsize=(4 * N_PLANES, 8),
                                 squeeze=False)
        fig.suptitle(
            f"Beam cross-section  —  E = {energy:.1f} keV\n"
            f"pc-236  (L={PC_LENGTH} cm, f_out={PC_FOCAL_OUT} cm)",
            fontsize=13, y=1.01,
        )

        rows = [
            (vt, "voxTrace", "steelblue"),
            (pc, "polycap",  "firebrick"),
        ]

        # Per-column symmetric axis limits (same for both rows → fair comparison)
        col_lims = []
        for col in range(N_PLANES):
            xs, zs = [], []
            for data, _, _ in rows:
                if data is None:
                    continue
                plane_arr, x_mm, z_mm, _ = data
                mask = plane_arr == col
                xs.append(x_mm[mask])
                zs.append(z_mm[mask])
            lim = sym_lim(xs + zs)
            col_lims.append(lim)

        for row_idx, (data, label, color) in enumerate(rows):
            for col in range(N_PLANES):
                ax = axes[row_idx, col]

                if data is not None:
                    plane_arr, x_mm, z_mm, _ = data
                    mask = plane_arr == col
                    scatter_plane(ax, x_mm[mask], z_mm[mask], color, label)
                else:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="gray")

                lim = col_lims[col]
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_aspect("equal")
                ax.xaxis.set_major_locator(ticker.MaxNLocator(4, symmetric=True))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4, symmetric=True))
                ax.tick_params(labelsize=7)

                if row_idx == 0:
                    ax.set_title(PLANE_LABELS[col], fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{label}\nY [mm]", fontsize=9)
                if row_idx == 1:
                    ax.set_xlabel("X [mm]", fontsize=9)

        plt.tight_layout()
        out = f"beam_comparison_E{energy:.1f}keV.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(out)
        print(f"  Saved {out}")

    if saved:
        print(f"\nGenerated {len(saved)} figure(s): {', '.join(saved)}")
    else:
        print("\nNo beam CSV files found.  Run ./build/src/Test2 first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
