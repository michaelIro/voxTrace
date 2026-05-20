"""                                                                              
plot_beam_profile.py                                                              
--------------------                                                              
Diagnostic: visualise the primary polycapillary beam AFTER the exit plane.       
                                                                                  
pc-236 is used as BOTH primary and secondary optic.  When used as PRIMARY it is  
turned 180°: the large end (0.3175 cm) faces the source, the small end (0.095 cm)
faces the sample.  Parameters relative to the secondary description file          
(Polycapillary.txt) are therefore swapped:                                        
                                                                                  
  Secondary (file)       Primary (turned)                                         
  rExtIn  = 0.095  cm    rExtIn  = 0.3175  cm   large end ← source               
  rExtOut = 0.3175 cm    rExtOut = 0.095   cm   small end → sample              
  rCapIn  = 9.75e-5 cm   rCapIn  = 3.25e-4 cm                                    
  rCapOut = 3.25e-4 cm   rCapOut = 9.75e-5 cm                                    
  focalIn = 0.49 cm      focalIn = 1e8 cm        source at ∞ (collimated input)  
  focalOut= 1e8 cm       focalOut= 0.49 cm       focuses 0.49 cm past exit        
                                                                                  
The exit direction for each ray is redirected toward (0, PC_EXIT+focalOut, 0),   
modelling the convergent capillary geometry of the focusing primary optic.        
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Polycapillary parameters: PRIMARY = pc-236 turned 180° ───────────────────
PC_LEN       = 4.03    # cm  — optic length
PC_REXT_IN   = 0.3175  # cm  — large end (entrance, facing source)
PC_REXT_OUT  = 0.095   # cm  — small end (exit, facing sample)
PC_RCAP_IN   = 3.25e-4 # cm  — single capillary radius at entrance
PC_RCAP_OUT  = 9.75e-5 # cm  — single capillary radius at exit
PC_FOCAL_IN  = 1e8     # cm  — source at ∞ → collimated input
PC_FOCAL_OUT = 0.49    # cm  — focuses 0.49 cm past exit
PC_NCAP      = 240000
SRC_DIST     = 100.0   # cm  — source to entrance distance (y = -SRC_DIST)

# open-area packing fraction
OPEN_AREA = PC_NCAP * PC_RCAP_IN**2 / PC_REXT_IN**2

# ── Probe planes (world Y coordinates) ───────────────────────────────────────
PC_EXIT = PC_LEN          # y of polycap exit face (entrance is at y=0)
PROBE_DY   = [0.0,  PC_FOCAL_OUT/2,  PC_FOCAL_OUT,  2*PC_FOCAL_OUT]
PROBE_NAMES = [
    f"y = exit                (y={PC_EXIT:.3f} cm)",
    f"y = exit + ½·fOut       (+{PC_FOCAL_OUT/2:.3f} cm)  half-focal",
    f"y = exit + fOut         (+{PC_FOCAL_OUT:.3f} cm)  ← FOCAL PLANE",
    f"y = exit + 2·fOut       (+{2*PC_FOCAL_OUT:.3f} cm)  past focus",
]

# ── Ray sampling ──────────────────────────────────────────────────────────────
N_RAYS = 200_000
rng = np.random.default_rng(42)

# Halton-like: use quasi-random square-root radius + uniform angle
u1 = rng.random(N_RAYS)
u2 = rng.random(N_RAYS)
r_ent  = np.sqrt(u1) * PC_REXT_IN      # uniform on disk
phi    = 2 * np.pi * u2
ex_ent = r_ent * np.cos(phi)           # entrance x
ez_ent = r_ent * np.sin(phi)           # entrance z (at y=0)

# Open-area rejection (capillary wall fraction)
accept_oa = rng.random(N_RAYS) < OPEN_AREA
ex_ent = ex_ent[accept_oa]
ez_ent = ez_ent[accept_oa]
n_oa = ex_ent.shape[0]
print(f"After open-area cut: {n_oa} / {N_RAYS} rays ({100*n_oa/N_RAYS:.1f}%)")

# Direction from source (0, -SRC_DIST, 0) to entrance point (ex, 0, ez)
ddx = ex_ent
ddy = np.full(n_oa, SRC_DIST)
ddz = ez_ent
dlen = np.sqrt(ddx**2 + ddy**2 + ddz**2)
ddx /= dlen;  ddy /= dlen;  ddz /= dlen

# ── Propagate to polycap EXIT plane (y = PC_EXIT) ────────────────────────────
# Source is at y = -SRC_DIST, entrance at y=0, exit at y=PC_LEN=PC_EXIT
# t from entrance point to exit:
t_exit = PC_EXIT / ddy        # since ray starts at y=0 (entrance)
ex_exit = ex_ent + t_exit * ddx
ez_exit = ez_ent + t_exit * ddz
r_exit  = np.sqrt(ex_exit**2 + ez_exit**2)

# Check: exit must be within rExtOut
inside = r_exit <= PC_REXT_OUT
ex_exit = ex_exit[inside];  ez_exit = ez_exit[inside]
ddx_in  = ddx[inside];      ddz_in  = ddz[inside];  ddy_in = ddy[inside]
n_pass = ex_exit.shape[0]
print(f"After exit-radius cut: {n_pass} rays ({100*n_pass/N_RAYS:.1f}%)")

# sin(theta) from entrance direction (for Fresnel weight colouring)
sinTheta = np.sqrt(ddx_in**2 + ddz_in**2)
print(f"sin(θ) range: {sinTheta.min():.4f} – {sinTheta.max():.4f}  "
      f"(mean {sinTheta.mean():.4f})")

# ── Redirect exit direction toward output focal spot ─────────────────────────
# Primary polycap focuses: each capillary exit tilts toward (0, PC_EXIT+focalOut, 0)
fdx = 0.0 - ex_exit
fdy = np.full(n_pass, PC_FOCAL_OUT)     # constant = focalOut
fdz = 0.0 - ez_exit
flen = np.sqrt(fdx**2 + fdy**2 + fdz**2)
ddx_ok = fdx / flen
ddy_ok = fdy / flen
ddz_ok = fdz / flen

# ── Collect positions at each probe plane ─────────────────────────────────────
# At probe plane y_probe = PC_EXIT + dy:
#   Δy from exit = dy
#   x_probe = ex_exit + (dy / ddy_ok) * ddx_ok
#   z_probe = ez_exit + (dy / ddy_ok) * ddz_ok

probe_data = []
for dy in PROBE_DY:
    t = dy / ddy_ok          # additional travel from exit to probe plane
    xp = ex_exit + t * ddx_ok
    zp = ez_exit + t * ddz_ok
    probe_data.append((xp, zp, ddx_ok.copy(), ddz_ok.copy()))

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(PROBE_DY), 2,
                         figsize=(11, 4 * len(PROBE_DY)),
                         squeeze=False)

fig.suptitle(f"pc-236 PRIMARY beam profile (turned optic) — {n_pass:,} rays\n"
             f"Source at y = −{SRC_DIST:.0f} cm, optic exit at y = {PC_EXIT:.2f} cm\n"
             f"Output focal distance: {PC_FOCAL_OUT*10:.0f} mm",
             fontsize=12, fontweight="bold")

# subsample for plotting (scatter is slow with many points)
NPLOT = min(n_pass, 8000)
idx = np.random.default_rng(0).choice(n_pass, NPLOT, replace=False)

# Colour by sinTheta to show angle distribution
sinT_plot = sinTheta[inside][idx]
norm = plt.Normalize(sinT_plot.min(), sinT_plot.max())
cmap = plt.cm.plasma

for row, (xp, zp, dxp, dzp) in enumerate(probe_data):
    ax_pos = axes[row, 0]
    ax_dir = axes[row, 1]

    sc = ax_pos.scatter(xp[idx]*1e4, zp[idx]*1e4,
                        c=sinT_plot, cmap=cmap, norm=norm,
                        s=2, alpha=0.5)
    ax_pos.set_xlabel("x  [µm]")
    ax_pos.set_ylabel("z  [µm]")
    ax_pos.set_title(f"{PROBE_NAMES[row]}\nTransverse position")
    ax_pos.set_aspect("equal")
    ax_pos.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_pos.axvline(0, color='k', lw=0.4, alpha=0.3)

    # RMS beam size
    x_rms = np.std(xp) * 1e4
    z_rms = np.std(zp) * 1e4
    r_rms = np.sqrt(np.mean(xp**2 + zp**2)) * 1e4
    ax_pos.set_title(f"{PROBE_NAMES[row]}\n"
                     f"Position  σ_x={x_rms:.0f}  σ_z={z_rms:.0f}  r_rms={r_rms:.0f} µm")

    ax_dir.scatter(dxp[idx]*1e3, dzp[idx]*1e3,
                   c=sinT_plot, cmap=cmap, norm=norm,
                   s=2, alpha=0.5)
    ax_dir.set_xlabel("dx  [mrad]")
    ax_dir.set_ylabel("dz  [mrad]")
    ax_dir.set_aspect("equal")
    ax_dir.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_dir.axvline(0, color='k', lw=0.4, alpha=0.3)
    dx_rms = np.std(dxp) * 1e3
    dz_rms = np.std(dzp) * 1e3
    ax_dir.set_title(f"Direction  σ_dx={dx_rms:.2f}  σ_dz={dz_rms:.2f} mrad")

plt.colorbar(sc, ax=axes[:, 1], label="sin(θ)  [transverse angle]",
             shrink=0.6, pad=0.02)

plt.tight_layout()
out = os.path.join(ROOT, "build", "beam_profile.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out}")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\nBeam size (r_rms) and divergence at each probe plane:")
print(f"  {'Plane':<35}  {'r_rms [µm]':>12}  {'σ_dx [mrad]':>12}  {'σ_dz [mrad]':>12}")
for row, (dy, name) in enumerate(zip(PROBE_DY, PROBE_NAMES)):
    xp, zp, dxp, dzp = probe_data[row]
    r = np.sqrt(np.mean(xp**2 + zp**2)) * 1e4
    sdx = np.std(dxp) * 1e3
    sdz = np.std(dzp) * 1e3
    print(f"  {name:<35}  {r:>12.1f}  {sdx:>12.3f}  {sdz:>12.3f}")
