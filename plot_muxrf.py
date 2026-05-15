"""
plot_muxrf.py  — Compare simulated vs measured Cu Kα and Zn Kα profiles.

Measurement: nist-1107-depth-01-00.txt  (lateral depth scan, Cu/Zn ROI / Livetime)
Simulation:  build/muXRF_spectra.csv    (Z-scan, ±50 µm, normalised by N_RAYS)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
current_path = ROOT

N_RAYS = 1000000  # primary rays per scan point — must match Test-muXRF.cpp

# ─────────────────────────────────────────────────────────────────────────────
#  1.  Measurement data  (exactly as in the Jupyter notebook)
# ─────────────────────────────────────────────────────────────────────────────
COLNAMES = [
    "Motor-samplexraw","Motor-sampleyraw","Motor-samplex","Motor-sampley",
    "Motor-samplez","Motor-cap1x","Motor-cap1y","Motor-cap1z","Motor-cap1yaw",
    "Motor-cap1pitch","Motor-detectorx","Motor-detectory","Motor-detectorz",
    "Motor-mic","Motor-th","Motor-ml",
    "px5.Mo-Ka-Gro[1554,1628]","px5.Mo-Ka-Net[1554,1628]",
    "px5.Ni-Ka-Gro[670,695]","px5.Ni-Ka-Net[670,695]",
    "px5.Cu-Ka-Gro[708,755]","px5.Cu-Ka-Net[708,755]",
    "px5.Fe-Ka-Gro[575,595]","px5.Fe-Ka-Net[575,595]",
    "px5.Zn-Ka-Gro[770,800]","px5.Zn-Ka-Net[770,800]",
    "px5.Sn-L-Gro[306,350]","px5.Sn-L-Net[306,350]",
    "px5.Pb-L-Gro[1107,1173]","px5.Pb-L-Net[1107,1173]",
    "px5.Realtime","px5.Livetime","px5.Deadtime","px5.icr","px5.ocr",
    "miccam.AvgR","miccam.AvgG","miccam.AvgB",
    "tube.Voltage","tube.Current","px5.Filename",
]

file_loc_1 = current_path + "/test-data/measurement/nist-1107/nist-1107-depth-00-00.txt"
file_loc_2 = current_path + "/test-data/measurement/nist-1107/nist-1107-depth-00-01.txt"
file_loc_3 = current_path + "/test-data/measurement/nist-1107/nist-1107-depth-01-00.txt"

kw = dict(sep='\t', skiprows=7, header=None, names=COLNAMES)
df_1 = pd.read_csv(file_loc_1, **kw)
df_2 = pd.read_csv(file_loc_2, **kw)
df_3 = pd.read_csv(file_loc_3, **kw)

# Centre on Cu signal maximum + 2 µm offset  (as in notebook)
max_pos  = df_3["px5.Cu-Ka-Gro[708,755]"].idxmax()
max_corr = df_3["Motor-samplex"][max_pos] + 2

meas_x  = df_3["Motor-samplex"] - max_corr
meas_cu = df_3["px5.Cu-Ka-Gro[708,755]"] / df_3["px5.Livetime"]   # cps
meas_zn = df_3["px5.Zn-Ka-Gro[770,800]"] / df_3["px5.Livetime"]   # cps

# ─────────────────────────────────────────────────────────────────────────────
#  2.  Simulation spectra  (normalised by N_RAYS)
# ─────────────────────────────────────────────────────────────────────────────
sim_df   = pd.read_csv(os.path.join(ROOT, "build", "muXRF_spectra.csv"))
energies = sim_df["E_keV"].values
scan_cols = [c for c in sim_df.columns if c != "E_keV"]
scan_z   = np.array([float(c.replace("z","").replace("um","")) for c in scan_cols])

CU_KA, ZN_KA, WIN = 8.041, 8.631, 0.15

def integrate(e_center):
    mask = np.abs(energies - e_center) <= WIN
    return sim_df.loc[mask, scan_cols].sum(axis=0).values / N_RAYS

sim_cu = integrate(CU_KA)
sim_zn = integrate(ZN_KA)

# ─────────────────────────────────────────────────────────────────────────────
#  3.  Print Cu/Zn ratio statistics
# ─────────────────────────────────────────────────────────────────────────────
sim_ratio  = sim_cu / np.where(sim_zn > 0, sim_zn, np.nan)
peak_mask  = meas_cu > 0.1 * meas_cu.max()
meas_ratio = (meas_cu / meas_zn.replace(0, np.nan))[peak_mask]
print(f"Composition Cu/Zn weight ratio : 0.6119 / 0.3741 = {0.6119/0.3741:.3f}")
print(f"Simulation  Cu/Zn ratio        : {np.nanmean(sim_ratio):.3f} ± {np.nanstd(sim_ratio):.3f}")
print(f"Measurement Cu/Zn ratio (focus): {meas_ratio.mean():.3f} ± {meas_ratio.std():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
#  4.  Plot
# ─────────────────────────────────────────────────────────────────────────────
COLOR_CU = "#d62728"
COLOR_ZN = "#1f77b4"

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("NIST-1107 — Cu Kα & Zn Kα  |  simulation vs measurement",
             fontsize=13, fontweight="bold")

# ── Left: Measurement  ────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(meas_x, meas_cu, "ro", ms=4, label="Cu-Ka ROI")
ax.plot(meas_x, meas_zn, "co", ms=4, label="Zn-Ka ROI")
ax.set_ylabel("Signal [cps]")
ax.set_xlabel(r"Depth [$\mu$m]")
ax.set_title("NIST-1107 — Measurement Data")
ax.legend()

# ── Right: Simulation  ────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.plot(scan_z, sim_cu, "ro-", ms=6, lw=1.8, label="Cu Kα (sim)")
ax2.plot(scan_z, sim_zn, "co-", ms=6, lw=1.8, label="Zn Kα (sim)")
ax2.set_ylabel("Intensity / ray [arb.]")
ax2.set_xlabel(r"Z scan position [$\mu$m]")
ax2.set_title(f"Simulation  (normalised by {N_RAYS:,} rays)")
ax2.legend()

# Cu/Zn ratio box on each panel
for a, rat, err in [(axes[0], meas_ratio.mean(), meas_ratio.std()),
                     (axes[1], np.nanmean(sim_ratio), np.nanstd(sim_ratio))]:
    a.text(0.97, 0.97, f"Cu/Zn = {rat:.2f} ± {err:.2f}",
           transform=a.transAxes, ha="right", va="top", fontsize=9, color="grey",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

plt.tight_layout()
out_png = os.path.join(ROOT, "build", "muxrf_comparison.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Plot saved → {out_png}")


