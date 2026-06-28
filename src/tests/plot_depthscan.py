"""
Plot the confocal micro-XRF depth scan produced by Test3.
Reads test-data/out/confocal_depthscan.csv  (depth profile: line intensity vs depth)
  and test-data/out/confocal_spectra.csv     (per-depth spectra).

Run from the project root after ./build/src/Test3:
    python3 src/tests/plot_depthscan.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

LINES = [("Cu-Kα", 8.05), ("Zn-Kα", 8.64), ("Cu-Kβ", 8.91),
         ("Zn-Kβ", 9.57), ("Pb-Lα", 10.55), ("elastic", 17.4)]


def draw_spectrum(ax, e, y, label=True):
    """Log-scale Si(Li) spectrum with line / Si-escape markers."""
    ymax = y.max() if np.any(y > 0) else 1.0
    yy = y.copy()
    yy[yy <= 0] = np.nan                                   # don't draw empty bins
    ax.semilogy(e, yy, lw=0.8, color="navy")
    for name, ec in LINES:
        ax.axvline(ec, color="gray", ls=":", lw=0.5)
        if label:
            ax.text(ec, ymax, name, rotation=90, fontsize=7,
                    va="top", ha="right", color="gray")
    ax.axvline(8.05 - 1.74, color="crimson", ls="--", lw=0.6)   # Cu-Kα Si escape
    if label:
        ax.text(8.05 - 1.74, ymax, "Si escape", rotation=90, fontsize=7,
                va="top", ha="right", color="crimson")
    ax.set_ylim(ymax * 1e-4, ymax * 2)
    ax.set_xlim(0, 18)
    ax.grid(alpha=0.3, which="both")


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(base, "test-data", "out")
    prof_csv = os.path.join(out, "confocal_depthscan.csv")
    spec_csv = os.path.join(out, "confocal_spectra.csv")
    if not os.path.exists(prof_csv):
        print("No depth scan found — run ./build/src/Test3 first.")
        sys.exit(1)

    # ── depth profile ─────────────────────────────────────────────────────────
    names = open(prof_csv).readline().strip().split(",")          # depth_um,detected,<lines...>
    prof = np.loadtxt(prof_csv, delimiter=",", skiprows=1, ndmin=2)
    depth = prof[:, 0]
    line_names = names[2:]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    for L, name in enumerate(line_names):
        y = prof[:, 2 + L]
        if y.max() <= 0:
            continue
        ax1.plot(depth, y, "-o", ms=4, label=name)
    ax1.axvline(0, color="gray", ls="--", lw=0.8, label="surface")
    ax1.set_xlabel("Confocal depth below surface [µm]")
    ax1.set_ylabel("Detected line intensity [arb.]")
    ax1.set_title("NIST-1107 brass — confocal depth profile")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    # ── spectra vs depth (image) + measured spectrum at the brightest depth ────
    if os.path.exists(spec_csv):
        hdr = open(spec_csv).readline().strip().split(",")        # energy_keV,d-50,...
        spec = np.loadtxt(spec_csv, delimiter=",", skiprows=1, ndmin=2)
        e = spec[:, 0]
        S = spec[:, 1:].T                                          # [depth, energy]
        dvals = [int(h[1:]) for h in hdr[1:]]
        top = S.max()
        vmax = np.log10(top)
        im = ax2.imshow(np.log10(S + top * 1e-6), aspect="auto", origin="lower",
                        cmap="inferno", extent=[e[0], e[-1], dvals[0], dvals[-1]],
                        vmin=vmax - 5, vmax=vmax)               # 5-decade range
        ax2.set_xlabel("Energy [keV]")
        ax2.set_ylabel("Confocal depth [µm]")
        ax2.set_title("Spectrum vs depth  (log scale)")
        ax2.set_xlim(0, 18)
        fig.colorbar(im, ax=ax2, label="log10 intensity")

        # 1-D Si(Li) spectrum: pick the brightest depth, draw on log scale
        di = int(np.argmax(S.sum(axis=1)))
        draw_spectrum(ax3, e, S[di])
        ax3.set_xlabel("Measured energy [keV]")
        ax3.set_ylabel("Counts [arb.]")
        ax3.set_title(f"Si(Li) spectrum at depth {dvals[di]:+d} µm")

    plt.tight_layout()
    dst = os.path.join(out, "confocal_depthscan.png")
    plt.savefig(dst, dpi=150)
    print(f"Saved {dst}")

    # ── one Si(Li) spectrum per depth (grid of small multiples) ────────────────
    if os.path.exists(spec_csv):
        nd = len(dvals)
        ncol = 4
        nrow = int(np.ceil(nd / ncol))
        fig2, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow),
                                 sharex=True, squeeze=False)
        for k in range(nrow * ncol):
            ax = axs[k // ncol][k % ncol]
            if k >= nd:
                ax.axis("off")
                continue
            draw_spectrum(ax, e, S[k], label=(k == 0))
            ax.set_title(f"depth {dvals[k]:+d} µm  "
                         f"(Σ={S[k].sum():.2e})", fontsize=9)
            if k % ncol == 0:
                ax.set_ylabel("Counts [arb.]")
            if k // ncol == nrow - 1:
                ax.set_xlabel("Energy [keV]")
        fig2.suptitle("NIST-1107 brass — Si(Li) spectrum at each confocal depth",
                      fontsize=12)
        fig2.tight_layout(rect=[0, 0, 1, 0.98])
        dst2 = os.path.join(out, "confocal_spectra_perdepth.png")
        fig2.savefig(dst2, dpi=150)
        print(f"Saved {dst2}")


if __name__ == "__main__":
    main()
