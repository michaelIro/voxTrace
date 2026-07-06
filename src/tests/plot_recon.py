"""
Plot the confocal voxel-weight reconstruction produced by Test5.
Reads test-data/out/recon_profile.csv   (layer-averaged depth profile)
      test-data/out/recon_weights.csv   (per-voxel fitted weights)
  and test-data/out/recon_spectra.csv   (measured vs fitted spectra per position)

Run from the project root after ./build/src/Test5:
    python3 src/tests/plot_recon.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

LINES = [("Cu-Kα", 8.05), ("Zn-Kα", 8.64), ("Cu-Kβ", 8.91),
         ("Zn-Kβ", 9.57), ("Pb-Lα", 10.55), ("elastic", 17.4)]


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(base, "test-data", "out")
    prof_csv = os.path.join(out, "recon_profile.csv")
    vox_csv = os.path.join(out, "recon_weights.csv")
    spec_csv = os.path.join(out, "recon_spectra.csv")
    if not os.path.exists(prof_csv):
        print("No reconstruction found — run ./build/src/Test5 first.")
        sys.exit(1)

    prof = np.loadtxt(prof_csv, delimiter=",", skiprows=1, ndmin=2)
    zp, wt, wm, ws = prof[:, 0], prof[:, 3], prof[:, 4], prof[:, 5]
    vox = np.loadtxt(vox_csv, delimiter=",", skiprows=1, ndmin=2)
    zv, mass, wtv, wfv = vox[:, 6], vox[:, 8], vox[:, 9], vox[:, 10]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # ── reconstructed depth profile vs ground truth ───────────────────────────
    zedge = np.concatenate([[0], zp + 2.5])                       # 5 µm layers
    ax1.stairs(wt, zedge, color="black", lw=1.5, label="ground truth")
    ax1.errorbar(zp, wm, yerr=ws, fmt="o-", ms=4, capsize=3, color="crimson",
                 label="reconstruction (response-weighted)")
    ax1.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel("Depth below surface [µm]")
    ax1.set_ylabel("Voxel weight")
    ax1.set_title("Reconstructed depth profile")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── per-voxel fitted weights (marker size ∝ response mass) ────────────────
    s = 4 + 60 * mass / mass.max()
    sc = ax2.scatter(zv, wfv, s=s, c=np.log10(mass), cmap="viridis", alpha=0.6)
    ax2.stairs(wt, zedge, color="black", lw=1.5)
    ax2.set_xlabel("Voxel depth [µm]")
    ax2.set_ylabel("Fitted voxel weight")
    ax2.set_title(f"Per-voxel weights ({len(wfv)} parameters)")
    ax2.set_ylim(0, min(4.0, wfv.max() * 1.05))
    fig.colorbar(sc, ax=ax2, label="log10 response mass")
    ax2.grid(alpha=0.3)

    # ── measured vs fitted spectrum at the brightest scan position ────────────
    hdr = open(spec_csv).readline().strip().split(",")
    spec = np.loadtxt(spec_csv, delimiter=",", skiprows=1, ndmin=2)
    e = spec[:, 0]
    M = spec[:, 1::2].T                                            # [pos, energy] measured
    F = spec[:, 2::2].T                                            # [pos, energy] fitted
    names = [h[len("meas_"):] for h in hdr[1::2]]
    di = int(np.argmax(M.sum(axis=1)))
    ymax = M[di].max()
    mm = M[di].copy()
    mm[mm <= 0] = np.nan
    ax3.semilogy(e, mm, drawstyle="steps-mid", lw=0.8, color="navy", label="measured")
    ff = F[di].copy()
    ff[ff <= 0] = np.nan
    ax3.semilogy(e, ff, lw=1.2, color="crimson", alpha=0.8, label="fitted model")
    for name, ec in LINES:
        ax3.axvline(ec, color="gray", ls=":", lw=0.5)
        ax3.text(ec, ymax, name, rotation=90, fontsize=7, va="top", ha="right", color="gray")
    ax3.set_xlim(0, 18)
    ax3.set_ylim(0.5, ymax * 2)
    ax3.set_xlabel("Measured energy [keV]")
    ax3.set_ylabel("Counts")
    ax3.set_title(f"Spectrum at confocal depth {names[di]}")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, which="both")

    plt.tight_layout()
    dst = os.path.join(out, "recon.png")
    plt.savefig(dst, dpi=150)
    print(f"Saved {dst}")

    # ── measured vs fitted spectrum at every scan position ────────────────────
    nd = len(names)
    ncol = 4
    nrow = int(np.ceil(nd / ncol))
    fig2, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow),
                             sharex=True, squeeze=False)
    for k in range(nrow * ncol):
        ax = axs[k // ncol][k % ncol]
        if k >= nd:
            ax.axis("off")
            continue
        mm = M[k].copy(); mm[mm <= 0] = np.nan
        ff = F[k].copy(); ff[ff <= 0] = np.nan
        ax.semilogy(e, mm, drawstyle="steps-mid", lw=0.7, color="navy")
        ax.semilogy(e, ff, lw=1.0, color="crimson", alpha=0.8)
        ax.set_xlim(0, 18)
        ax.set_title(f"depth {names[k]}  (Σ={np.nansum(M[k]):.0f} cts)", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        if k % ncol == 0:
            ax.set_ylabel("Counts")
        if k // ncol == nrow - 1:
            ax.set_xlabel("Energy [keV]")
    fig2.suptitle("Measured (blue) vs fitted (red) spectrum at each confocal depth",
                  fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.98])
    dst2 = os.path.join(out, "recon_spectra_perdepth.png")
    fig2.savefig(dst2, dpi=150)
    print(f"Saved {dst2}")


if __name__ == "__main__":
    main()
