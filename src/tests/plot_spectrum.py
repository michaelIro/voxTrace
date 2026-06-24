"""
Plot the confocal micro-XRF spectrum produced by Test3.
Reads test-data/out/confocal_spectrum.csv (energy_keV, weight).

Run from the project root after ./build/src/Test3:
    python3 src/tests/plot_spectrum.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Characteristic lines expected from NIST-1107 brass under 17.4 keV excitation.
LINES = [
    ("Sn-Lα", 3.44), ("Fe-Kα", 6.40), ("Ni-Kα", 7.48), ("Cu-Kα", 8.05),
    ("Zn-Kα", 8.64), ("Cu-Kβ", 8.91), ("Zn-Kβ", 9.57), ("Pb-Lα", 10.55),
    ("Pb-Lβ", 12.61), ("Compton", 16.8), ("Elastic", 17.4),
]


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv = os.path.join(base, "test-data", "out", "confocal_spectrum.csv")
    if not os.path.exists(csv):
        print("No spectrum found — run ./build/src/Test3 first.")
        sys.exit(1)

    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    e, w = data[:, 0], data[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(e, w, lw=0.9, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlim(0, 20)
    ax.set_ylim(max(w[w > 0].min() * 0.5, 1e-9) if np.any(w > 0) else 1e-9, w.max() * 2)
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Detected intensity [arb.]")
    ax.set_title("NIST-1107 brass — confocal micro-XRF spectrum\n"
                 "source → primary PC-236 → sample → secondary PC-236 → detector")

    ymax = w.max() if w.max() > 0 else 1.0
    for name, le in LINES:
        ax.axvline(le, color="gray", ls=":", lw=0.6, alpha=0.6)
        ax.text(le, ymax * 1.3, name, rotation=90, va="bottom", ha="center", fontsize=7)

    ax.grid(alpha=0.25, which="both")
    plt.tight_layout()
    out = os.path.join(base, "test-data", "out", "confocal_spectrum.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
