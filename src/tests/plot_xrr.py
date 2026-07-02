"""
Plot the X-ray reflectivity curves produced by Test4.
Reads test-data/out/xrr.csv (theta_deg, then one column per sample stack).

Run from the project root after ./build/src/Test4:
    python3 src/tests/plot_xrr.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv = os.path.join(base, "test-data", "out", "xrr.csv")
    if not os.path.exists(csv):
        print("No xrr.csv found — run ./build/src/Test4 first.")
        sys.exit(1)

    names = open(csv).readline().strip().split(",")
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    theta = data[:, 0]

    plt.figure(figsize=(8, 5.5))
    for k, name in enumerate(names[1:], start=1):
        plt.semilogy(theta, data[:, k], lw=1.1, label=name.replace("_", " / "))
    plt.xlabel("Grazing incidence angle θ [°]")
    plt.ylabel("Reflectivity R")
    plt.title("X-ray reflectivity (XRR) — total reflection, critical angle, Kiessig fringes")
    plt.ylim(1e-9, 2)
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    dst = os.path.join(base, "test-data", "out", "xrr.png")
    plt.savefig(dst, dpi=150)
    print(f"Saved {dst}")


if __name__ == "__main__":
    main()
