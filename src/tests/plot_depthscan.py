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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

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

    # ── spectra vs depth (image) ──────────────────────────────────────────────
    if os.path.exists(spec_csv):
        hdr = open(spec_csv).readline().strip().split(",")        # energy_keV,d-50,...
        spec = np.loadtxt(spec_csv, delimiter=",", skiprows=1, ndmin=2)
        e = spec[:, 0]
        S = spec[:, 1:].T                                          # [depth, energy]
        dvals = [int(h[1:]) for h in hdr[1:]]
        S = np.log10(S + S[S > 0].min() if np.any(S > 0) else S + 1e-12)
        im = ax2.imshow(S, aspect="auto", origin="lower", cmap="inferno",
                        extent=[e[0], e[-1], dvals[0], dvals[-1]])
        ax2.set_xlabel("Energy [keV]")
        ax2.set_ylabel("Confocal depth [µm]")
        ax2.set_title("Spectrum vs depth  (log scale)")
        ax2.set_xlim(0, 18)
        fig.colorbar(im, ax=ax2, label="log10 intensity")

    plt.tight_layout()
    dst = os.path.join(out, "confocal_depthscan.png")
    plt.savefig(dst, dpi=150)
    print(f"Saved {dst}")


if __name__ == "__main__":
    main()
