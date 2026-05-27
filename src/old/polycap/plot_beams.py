#!/usr/bin/env python3
"""
plot_beams.py — Visualise the polycapillary beam at four observation planes.

Reads polycap_exit.csv and polycap_hpp_exit.csv produced by test_comparison,
propagates photons to four planes, and saves a comparison scatter-plot grid.

Usage:
    python plot_beams.py [polycap_exit.csv] [polycap_hpp_exit.csv] [--energy-idx N]

Observation planes (downstream of optic exit at z=4.03 cm):
    Plane 0: z = 4.03  cm  (exit window)
    Plane 1: z = 4.275 cm  (half focal distance, 0.49/2 = 0.245 cm beyond exit)
    Plane 2: z = 4.52  cm  (focal distance, 0.49 cm beyond exit)
    Plane 3: z = 5.01  cm  (twice focal distance, 0.98 cm beyond exit)
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Optic constants ───────────────────────────────────────────────────────────
OPTIC_LENGTH    = 4.03   # cm
FOCAL_DOWN      = 0.49   # cm

Z_EXIT   = OPTIC_LENGTH
Z_PLANES = [
    Z_EXIT,
    Z_EXIT + FOCAL_DOWN / 2,
    Z_EXIT + FOCAL_DOWN,
    Z_EXIT + 2 * FOCAL_DOWN,
]
PLANE_LABELS = [
    f"Exit window\n(z = {Z_PLANES[0]:.3f} cm)",
    f"Half focal\n(z = {Z_PLANES[1]:.3f} cm)",
    f"Focal distance\n(z = {Z_PLANES[2]:.3f} cm)",
    f"Twice focal\n(z = {Z_PLANES[3]:.3f} cm)",
]


def load_csv(path: str) -> pd.DataFrame:
    """Load exit-plane CSV and add summed weight column."""
    df = pd.read_csv(path)
    w_cols = [c for c in df.columns if c.startswith("w")]
    df["w_sum"] = df[w_cols].sum(axis=1)
    return df


def propagate(df: pd.DataFrame, z_obs: float) -> tuple[np.ndarray, np.ndarray]:
    """Propagate photon rays to observation plane at z_obs."""
    x0, y0, z0 = df["x"].values, df["y"].values, df["z"].values
    dx, dy, dz = df["dx"].values, df["dy"].values, df["dz"].values
    # Avoid division by zero for degenerate dz
    mask = np.abs(dz) > 1e-12
    dt = np.where(mask, (z_obs - z0) / dz, 0.0)
    x = x0 + dx * dt
    y = y0 + dy * dt
    return x, y


def make_figure(df_pc: pd.DataFrame, df_hpp: pd.DataFrame,
                energy_idx: int, energies: list[str]) -> plt.Figure:
    """
    Build a 4×2 comparison grid.
    Rows = observation planes; columns = [polycap library, PolyCap.hpp].
    Point colour encodes per-photon weight at the selected energy.
    """
    n_planes = len(Z_PLANES)
    fig = plt.figure(figsize=(12, 4 * n_planes), constrained_layout=True)
    fig.suptitle(
        f"Beam comparison — energy index {energy_idx+1} "
        f"({energies[energy_idx] if energies else '?'} keV)\n"
        f"polycap lib (left) vs PolyCap.hpp (right)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(n_planes, 2, figure=fig)

    # Choose weight column
    w_col = f"w{energy_idx+1}"
    if w_col not in df_pc.columns:
        w_col_pc = "w_sum"
    else:
        w_col_pc = w_col
    if w_col not in df_hpp.columns:
        w_col_hpp = "w_sum"
    else:
        w_col_hpp = w_col

    datasets = [(df_pc, w_col_pc, "polycap"), (df_hpp, w_col_hpp, "PolyCap.hpp")]

    for row, (z_obs, label) in enumerate(zip(Z_PLANES, PLANE_LABELS)):
        # Determine common axis limits across both datasets at this plane
        all_x, all_y = [], []
        for df, _, _ in datasets:
            xi, yi = propagate(df, z_obs)
            all_x.extend(xi); all_y.extend(yi)
        all_x = np.array(all_x); all_y = np.array(all_y)
        # Clip outliers to 99.5th percentile for better visualisation
        r = np.sqrt(all_x**2 + all_y**2)
        r_max = np.percentile(r, 99.5)
        lim = max(r_max * 1.1, 1e-4)

        for col, (df, wcol, title) in enumerate(datasets):
            ax = fig.add_subplot(gs[row, col])
            xi, yi = propagate(df, z_obs)
            wi = df[wcol].values
            wi_pos = np.clip(wi, 0, None)

            sc = ax.scatter(xi, yi, c=wi_pos, s=1.5, alpha=0.6,
                             cmap="inferno",
                             norm=Normalize(vmin=0, vmax=np.percentile(wi_pos, 99)))
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel("x [cm]", fontsize=8)
            ax.set_ylabel("y [cm]", fontsize=8)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")
            ax.text(0.02, 0.98, label, transform=ax.transAxes,
                    fontsize=7, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

            # Colour bar per axis
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("weight", fontsize=7)

    return fig


def make_direction_figure(df_pc: pd.DataFrame,
                           df_hpp: pd.DataFrame) -> plt.Figure:
    """
    Direction scatter plot at the exit window (angular distribution).
    Shows (dx, dy) for each photon coloured by total weight.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    fig.suptitle("Exit angular distribution (dx, dy at exit window)", fontsize=12)

    for ax, (df, title) in zip(axes, [(df_pc, "polycap"), (df_hpp, "PolyCap.hpp")]):
        wi = df["w_sum"].values
        sc = ax.scatter(df["dx"].values, df["dy"].values,
                        c=np.clip(wi, 0, None), s=1.5, alpha=0.6,
                        cmap="viridis",
                        norm=Normalize(vmin=0, vmax=np.percentile(wi, 99)))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("dx", fontsize=9)
        ax.set_ylabel("dy", fontsize=9)
        ax.set_aspect("equal")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("total weight", fontsize=8)

    return fig


def make_efficiency_figure(df_pc: pd.DataFrame,
                            df_hpp: pd.DataFrame,
                            energies: list[float]) -> plt.Figure:
    """
    Bar chart of per-energy efficiency for both implementations.
    """
    w_cols_pc  = [f"w{e+1}" for e in range(len(energies)) if f"w{e+1}" in df_pc.columns]
    w_cols_hpp = [f"w{e+1}" for e in range(len(energies)) if f"w{e+1}" in df_hpp.columns]
    if not w_cols_pc or not w_cols_hpp:
        return None

    eff_pc  = [df_pc[c].mean()  for c in w_cols_pc]
    eff_hpp = [df_hpp[c].mean() for c in w_cols_hpp]
    n = min(len(eff_pc), len(eff_hpp), len(energies))

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w/2, eff_pc[:n],  w, label="polycap",     color="steelblue",  alpha=0.8)
    ax.bar(x + w/2, eff_hpp[:n], w, label="PolyCap.hpp", color="darkorange", alpha=0.8)
    ax.set_xlabel("Energy [keV]", fontsize=10)
    ax.set_ylabel("Average transmitted weight", fontsize=10)
    ax.set_title("Per-energy transmission efficiency comparison", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:.0f}" for e in energies[:n]], fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return fig


# ─────────────────────────────────── main ────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pc_csv",   nargs="?", default="polycap_exit.csv")
    parser.add_argument("hpp_csv",  nargs="?", default="polycap_hpp_exit.csv")
    parser.add_argument("--energy-idx", type=int, default=7,
                        help="0-based energy index for beam images (default 7 → 8 keV)")
    parser.add_argument("--output", default="beam_comparison.png")
    args = parser.parse_args()

    print(f"Loading {args.pc_csv} …")
    df_pc = load_csv(args.pc_csv)
    print(f"  {len(df_pc)} photons, columns: {list(df_pc.columns[:8])} …")

    print(f"Loading {args.hpp_csv} …")
    df_hpp = load_csv(args.hpp_csv)
    print(f"  {len(df_hpp)} photons, columns: {list(df_hpp.columns[:8])} …")

    # Extract energy list from headers
    w_cols = [c for c in df_pc.columns if c.startswith("w")]
    try:
        energies_float = [float(i) for i in range(1, len(w_cols)+1)]
    except Exception:
        energies_float = list(range(1, len(w_cols)+1))
    energies_str = [f"{e:.0f}" for e in energies_float]

    ei = min(args.energy_idx, len(w_cols)-1)
    print(f"Plotting beam images at energy index {ei} ({energies_str[ei]} keV) …")

    # ── Beam image figure ────────────────────────────────────────────────────
    fig_beam = make_figure(df_pc, df_hpp, ei, energies_str)
    beam_path = args.output
    fig_beam.savefig(beam_path, dpi=150)
    plt.close(fig_beam)
    print(f"Saved: {beam_path}")

    # ── Angular distribution ─────────────────────────────────────────────────
    dir_path = beam_path.replace(".png", "_directions.png")
    fig_dir = make_direction_figure(df_pc, df_hpp)
    fig_dir.savefig(dir_path, dpi=150)
    plt.close(fig_dir)
    print(f"Saved: {dir_path}")

    # ── Efficiency comparison bar chart ──────────────────────────────────────
    eff_path = beam_path.replace(".png", "_efficiency.png")
    fig_eff = make_efficiency_figure(df_pc, df_hpp, energies_float)
    if fig_eff is not None:
        fig_eff.savefig(eff_path, dpi=150)
        plt.close(fig_eff)
        print(f"Saved: {eff_path}")


if __name__ == "__main__":
    main()
