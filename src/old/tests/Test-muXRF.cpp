// Test-muXRF.cpp  — full Monte-Carlo muXRF depth-scan simulation
// Geometry from test-data/simulation/nist-1107/
//
// Pipeline per primary ray
//   1. Point source (pc-236 entrance disk) → PolyCap optic → exit beam
//   2. Primary ray traverses voxel grid with Beer-Lambert attenuation
//   3. At each voxel: generate XRF photon (energy = fluorescence line)
//      → accept only if direction cosine with detector axis ≥ cos(detector_half_angle)
//      → weight = primary_prob × μ_photo/μ_total × fluor_yield × cos_solid_angle_factor
//   4. XRF ray Beer-Lambert attenuation from voxel to sample exit
//   5. Accumulate into spectrum histogram per scan point
//
// Optimised for Apple M4 (OpenMP, SIMD-friendly floats, Kokkos serial+OpenMP).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Kokkos first (before any macro-polluting headers)
#include "Platform.hpp"
#include "Ray.hpp"
#include "PolyCap.hpp"
#include "RNG.hpp"
#include <Kokkos_Core.hpp>

// xraylib
#include <xraylib.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Simulation constants from parameter files
// ─────────────────────────────────────────────────────────────────────────────

// ── Polycapillary (Polycapillary.txt) ─────────────────────────────────────────
static constexpr float PC_LEN       = 4.03f;        // cm
// Primary polycap = same pc-236 optic turned 180°:
//   large end (0.3175 cm) faces source, small end (0.095 cm) faces sample
static constexpr float PC_REXT_IN   = 0.3175f;      // cm  (large, source side)
static constexpr float PC_REXT_OUT  = 0.095f;        // cm  (small, sample side)
static constexpr float PC_RCAP_IN   = 3.25e-4f;      // cm
static constexpr float PC_RCAP_OUT  = 9.75e-5f;      // cm
static constexpr float PC_FOCAL_IN  = 1e8f;          // source at ∞ → collimated input
static constexpr float PC_FOCAL_OUT = 0.49f;         // cm, focuses 0.49 cm past exit
static constexpr int   PC_NELEM     = 2;
static constexpr int   PC_Z[2]      = {8, 14};
static constexpr float PC_WT[2]     = {53.f, 47.f};
static constexpr float PC_DENSITY   = 2.23f;
static constexpr float PC_ROUGHNESS = 5.f;
static constexpr int   PC_NCAP      = 240000;

// ── Source (Source.txt / Capillaries.txt) ─────────────────────────────────────
static constexpr float SRC_DIST     = 100.f;        // cm
static constexpr float EXCITATION_E = 17.4f;        // keV (from Capillaries.txt)

// ── Sample (Sample.txt) ────────────────────────────────────────────────────────
// Dimensions in µm, voxel size 5 µm → all converted to cm internally.
// The beam travels in +Y. Sample surface is at y=0 in sample coords.
// In world coords: sample surface at y_world = PC_LEN (polycap exit).
// X,Z centred: X in [-150µm, +150µm], Z in [0, 150µm].
static constexpr float SAM_DX_UM   = 300.f;         // µm total X
static constexpr float SAM_DY_UM   = 300.f;         // µm total Y (beam traversal depth)
static constexpr float SAM_DZ_UM   = 150.f;         // µm total Z
static constexpr float VOX_UM      = 5.f;           // µm voxel edge
static constexpr int   NX = (int)(SAM_DX_UM / VOX_UM);   // 60
static constexpr int   NY = (int)(SAM_DY_UM / VOX_UM);   // 60 — beam depth
static constexpr int   NZ = (int)(SAM_DZ_UM / VOX_UM);   // 30
static constexpr float VOX_CM      = VOX_UM * 1e-4f;     // 5e-4 cm
// World-space lower corner of sample box.
// Beam focuses PC_FOCAL_OUT past the polycap exit → sample surface at focal plane.
static constexpr float SAM_OX_CM   = -SAM_DX_UM * 0.5e-4f;               // -0.015 cm
static constexpr float SAM_OY_CM   =  PC_LEN + PC_FOCAL_OUT;              // 4.52 cm
static constexpr float SAM_OZ_CM   = -SAM_DZ_UM * 0.5e-4f;               // -0.0075 cm

// ── Material (Materials.txt) — NIST-1107 homogeneous ──────────────────────────
static constexpr int   MAT_NELEM   = 6;
static constexpr int   MAT_Z[6]    = {26, 28, 29, 30, 50, 82};    // Fe Ni Cu Zn Sn Pb
static constexpr float MAT_W[6]    = {0.0004f, 0.001f, 0.6119f, 0.3741f, 0.0107f, 0.0019f};
// Density of NIST SRM 1107 (Cu-Zn alloy ~brass): ~8.5 g/cm³
static constexpr float MAT_DENSITY = 8.5f;

// ── Detector geometry (Capillaries.txt) ───────────────────────────────────────
// Detector at 45° to the beam (+Y) axis, in the YZ-plane.
// Beam is +Y; detector looks at sample from the +Z side at 45°.
// Detector direction unit vector: (0, sin45°, cos45°) — sidescatter.
static constexpr float DET_ANGLE_DEG  = 45.f;        // angle from beam axis
static constexpr float DET_DIST_UM    = 5100.f;      // µm
static constexpr float DET_CAP_DIA_UM = 16.5f;       // µm single cap diameter
static constexpr float DET_HALF_ANGLE = DET_CAP_DIA_UM / (2.f * DET_DIST_UM);

// ── Scan points (Simulation.txt) — z-offsets in µm ────────────────────────────
static constexpr int   N_SCAN   = 11;
static constexpr float SCAN_Z_UM[11] = {-50,-40,-30,-20,-10,0,10,20,30,40,50};

// ── Simulation parameters ──────────────────────────────────────────────────────
static constexpr int N_RAYS     = 1000000;     // rays per scan point
static constexpr int N_CHANNELS = 2048;        // spectrum channels
static constexpr float E_MIN    = 0.5f;        // keV
static constexpr float E_MAX    = 25.f;        // keV
static constexpr float E_STEP   = (E_MAX - E_MIN) / N_CHANNELS;

// Fluorescence lines to simulate: Kα and Kβ for each element
// xraylib: KA_LINE=0, KB_LINE=1
static constexpr int N_LINES    = 2;           // Kα, Kβ per element
static constexpr int LINE_IDS[2] = {KA_LINE, KB_LINE};

// ─────────────────────────────────────────────────────────────────────────────
//  Host-side material XRF cross-section table
//  Pre-computed at excitation energy for speed inside kernel.
// ─────────────────────────────────────────────────────────────────────────────
struct XRFLine {
    float energy;        // keV
    float cs_fluor;      // cm²/g fluorescence cross-section at excitation energy
    float cs_total_exit; // cm²/g total attenuation at fluorescence energy (for exit Beer-Lambert)
    float cs_total_prim; // cm²/g total attenuation of whole material at excitation energy
};

// Per-element attenuation of the MIXTURE at a given energy [cm²/g]
static float mixCS(float e_keV) {
    float mu = 0.f;
    for (int i = 0; i < MAT_NELEM; ++i)
        mu += MAT_W[i] * (float)CS_Total(MAT_Z[i], e_keV, nullptr);
    return mu;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Spectrum accumulation (host, atomic via OpenMP reduction on Kokkos View)
// ─────────────────────────────────────────────────────────────────────────────
static inline int energyBin(float e_keV) {
    int b = (int)((e_keV - E_MIN) / E_STEP);
    return (b < 0) ? 0 : (b >= N_CHANNELS ? N_CHANNELS-1 : b);
}

// ─────────────────────────────────────────────────────────────────────────────
//  muXRF kernel: one call per scan point
//  Returns a flat N_CHANNELS spectrum array (caller owns memory).
// ─────────────────────────────────────────────────────────────────────────────
static void simulateScanPoint(
        const PolyCap&       optic,
        float                openArea,        // packing fraction
        float                scan_z_cm,       // scan point Z offset in cm
        const XRFLine*       lines,           // [MAT_NELEM * N_LINES]
        int                  nlines_total,
        float                mu_prim_cm,      // µ·ρ of mixture at excitation E [cm⁻¹]
        double*              spectrum)        // [N_CHANNELS], pre-zeroed
{
    // Detector unit-vector: 45° from beam (+Y) in YZ-plane → (0, sin45, cos45)
    const float DET_ANGLE_RAD = DET_ANGLE_DEG * VT_PI / 180.f;
    const float det_dx = 0.f;
    const float det_dy = sinf(DET_ANGLE_RAD);   // sidescatter component
    const float det_dz = cosf(DET_ANGLE_RAD);   // forward component

    // Kokkos reduction into a spectrum: use View for thread-safe accumulation
    Kokkos::View<double*> spec("spectrum", N_CHANNELS);
    Kokkos::deep_copy(spec, 0.0);

    Kokkos::parallel_for(
        "muxrf_scan", N_RAYS,
        KOKKOS_LAMBDA(int idx) {
            // ── 1. Sample uniform point on pc-236 entrance disk ──────────────
            RNG rng(775289ULL ^ (uint64_t)idx * 6364136223846793005ULL
                    ^ (uint64_t)(uintptr_t)&scan_z_cm * 2654435761ULL);

            // Halton (base2/base3) for disk sampling
            float h2 = 0.f, h3 = 0.f, fh = 0.5f, gh = 1.f/3.f;
            int n2 = idx + 1, n3 = idx + 1;
            while (n2) { h2 += fh*(float)(n2&1); fh*=0.5f; n2>>=1; }
            while (n3) { int d=n3%3; h3+=gh*(float)d; gh/=3.f; n3/=3; }
            float r   = sqrtf(h2) * PC_REXT_IN;
            float phi = 2.f * VT_PI * h3;
            float ex  = r * cosf(phi);
            float ez  = r * sinf(phi);

            // Open-area rejection (capillary wall)
            if (rng.frand() >= openArea) return;

            // Direction from point source (0,-SRC_DIST,0) to entrance point
            float ddx = ex, ddy = SRC_DIST, ddz = ez;
            float dl  = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
            ddx /= dl; ddy /= dl; ddz /= dl;

            // Build primary ray
            Ray primary(
                0.f, -SRC_DIST, 0.f,
                ddx, ddy, ddz,
                1.f, 0.f, 0.f,
                false, 0.f, idx,
                0.f, 0.f, 0.f,
                ddx, ddy, ddz,
                1.f
            );
            primary.setEnergyKeV(EXCITATION_E);

            // ── 2. Propagate through polycapillary ───────────────────────────
            optic.trace(primary);
            if (!primary.getIAFlag()) return;

            // Primary ray exits polycap at y=PC_LEN, then travels to focal plane
            // at y=PC_LEN+PC_FOCAL_OUT=SAM_OY_CM where the sample surface is.
            // The SAMPLE moves (stage scan): shift the box by -scan_z_cm in Z.
            float prob_prim = primary.getProb();
            float sx = primary.getStartX();
            float sy = primary.getStartY();        // = PC_LEN (polycap exit)
            float sz = primary.getStartZ();
            // Direction AFTER polycap: redirected toward focal spot (0,SAM_OY_CM,0).
            // Must read back from the ray, not use pre-trace ddx/ddy/ddz.
            ddx = primary.getDirX();
            ddy = primary.getDirY();
            ddz = primary.getDirZ();

            // ── 3. Ray-voxel traversal (DDA) ─────────────────────────────────
            // Beam converges from polycap exit to focal spot at y=SAM_OY_CM.

            // Find entry into sample box: advance from (sx,sy,sz) along (ddx,ddy,ddz)
            // Sample box: x in [SAM_OX_CM, SAM_OX_CM+NX*VOX_CM]
            //             y in [SAM_OY_CM, SAM_OY_CM+NY*VOX_CM]
            //             z in [SAM_OZ_CM, SAM_OZ_CM+NZ*VOX_CM]
            // Sample box shifts by -scan_z_cm (stage moves, beam is fixed)
            const float BOX_X0 = SAM_OX_CM, BOX_X1 = SAM_OX_CM + NX*VOX_CM;
            const float BOX_Y0 = SAM_OY_CM, BOX_Y1 = SAM_OY_CM + NY*VOX_CM;
            const float BOX_Z0 = SAM_OZ_CM - scan_z_cm;
            const float BOX_Z1 = BOX_Z0 + NZ*VOX_CM;

            // Slab intersection (safe: ddy ≈ 1)
            float t_enter = -1e30f, t_exit = 1e30f;
            auto slab = [&](float p, float d, float lo, float hi) {
                if (fabsf(d) < 1e-12f) {
                    if (p < lo || p > hi) { t_enter = 1e30f; }
                    return;
                }
                float t0 = (lo - p) / d, t1 = (hi - p) / d;
                if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
                t_enter = fmaxf(t_enter, t0);
                t_exit  = fminf(t_exit,  t1);
            };
            slab(sx, ddx, BOX_X0, BOX_X1);
            slab(sy, ddy, BOX_Y0, BOX_Y1);
            slab(sz, ddz, BOX_Z0, BOX_Z1);
            if (t_enter >= t_exit || t_exit <= 0.f) return;
            if (t_enter < 0.f) t_enter = 0.f;

            // Entry point
            float px = sx + t_enter * ddx;
            float py = sy + t_enter * ddy;
            float pz = sz + t_enter * ddz;

            // Current voxel indices
            int vx = (int)((px - BOX_X0) / VOX_CM);
            int vy = (int)((py - BOX_Y0) / VOX_CM);
            int vz = (int)((pz - BOX_Z0) / VOX_CM);
            vx = vx < 0 ? 0 : (vx >= NX ? NX-1 : vx);
            vy = vy < 0 ? 0 : (vy >= NY ? NY-1 : vy);
            vz = vz < 0 ? 0 : (vz >= NZ ? NZ-1 : vz);

            // DDA step signs and initial t-to-boundary
            auto sign_f = [](float v) { return v >= 0.f ? 1 : -1; };
            int  sx_ = sign_f(ddx), sy_ = sign_f(ddy), sz_ = sign_f(ddz);
            float tDx = (fabsf(ddx)>1e-12f) ? fabsf(VOX_CM/ddx) : 1e30f;
            float tDy = (fabsf(ddy)>1e-12f) ? fabsf(VOX_CM/ddy) : 1e30f;
            float tDz = (fabsf(ddz)>1e-12f) ? fabsf(VOX_CM/ddz) : 1e30f;

            // Next boundary t
            float bx = BOX_X0 + (sx_>0 ? (vx+1) : vx) * VOX_CM;
            float by = BOX_Y0 + (sy_>0 ? (vy+1) : vy) * VOX_CM;
            float bz = BOX_Z0 + (sz_>0 ? (vz+1) : vz) * VOX_CM;
            float txN = (fabsf(ddx)>1e-12f) ? (bx-px)/ddx : 1e30f;
            float tyN = (fabsf(ddy)>1e-12f) ? (by-py)/ddy : 1e30f;
            float tzN = (fabsf(ddz)>1e-12f) ? (bz-pz)/ddz : 1e30f;

            // Accumulated Beer-Lambert weight along primary path
            float acc_prob = prob_prim;
            float t_cur    = t_enter;

            // ── DDA main loop ─────────────────────────────────────────────────
            for (int step = 0; step < NX + NY + NZ + 10; ++step) {
                if (vx<0||vx>=NX||vy<0||vy>=NY||vz<0||vz>=NZ) break;
                if (t_cur >= t_exit) break;

                // Find t of next boundary
                float t_next;
                int axis;
                if (txN <= tyN && txN <= tzN) { t_next = txN; axis = 0; }
                else if (tyN <= tzN)           { t_next = tyN; axis = 1; }
                else                           { t_next = tzN; axis = 2; }
                t_next = fminf(t_next, t_exit);

                float seg = t_next - t_cur;   // path length through this voxel [cm]

                // Beer-Lambert attenuation through this voxel for primary
                float att = expf(-mu_prim_cm * seg);

                // ── XRF generation in this voxel ─────────────────────────────
                // Voxel centre position
                float vcx = BOX_X0 + (vx + 0.5f) * VOX_CM;
                float vcz = BOX_Z0 + (vz + 0.5f) * VOX_CM;

                // For each fluorescence line
                for (int li = 0; li < nlines_total; ++li) {
                    const XRFLine& L = lines[li];
                    if (L.energy <= 0.f || L.cs_fluor <= 0.f) continue;

                    // XRF intensity weight:
                    //   prob_prim × (1-att)/att_prim_coeff × cs_photo/cs_total × fluor_cs
                    // Simplified: weight = acc_prob × cs_fluor × seg × MAT_DENSITY
                    // (cs_fluor already = CS_FluorLine * weight_fraction * density implicitly)
                    float w_xrf = acc_prob * L.cs_fluor * MAT_DENSITY * seg;

                    // Check if XRF direction cosine with detector axis is within acceptance
                    // We approximate: XRF is isotropic → solid angle factor = DET_HALF_ANGLE²*π / 4π
                    // But to apply directional selection: draw a random direction and check.
                    // For speed, use analytical solid-angle fraction instead.
                    float omega_frac = 0.5f * (1.f - cosf(DET_HALF_ANGLE));  // fraction of 4π

                    // Beer-Lambert of XRF photon from voxel centre to sample surface
                    // Detector direction: (det_dx, det_dy, det_dz)
                    // Path to box boundary from voxel centre in detector direction
                    float xrf_t = 1e30f;
                    auto exit_t = [&](float p, float d, float lo, float hi) {
                        if (fabsf(d) < 1e-12f) return;
                        float ta = (lo - p) / d, tb = (hi - p) / d;
                        float tmax = fmaxf(ta, tb);
                        if (tmax > 0.f && tmax < xrf_t) xrf_t = tmax;
                    };
                    exit_t(vcx, det_dx, BOX_X0, BOX_X1);
                    // vy centre for Y
                    float vcy = BOX_Y0 + (vy + 0.5f) * VOX_CM;
                    exit_t(vcy, det_dy, BOX_Y0, BOX_Y1);
                    exit_t(vcz, det_dz, BOX_Z0, BOX_Z1);
                    float xrf_path = (xrf_t < 1e29f) ? xrf_t : 0.f;
                    float xrf_att  = expf(-L.cs_total_exit * MAT_DENSITY * xrf_path);

                    float weight = w_xrf * omega_frac * xrf_att;

                    // Accumulate into spectrum bin (use atomic add)
                    int bin = energyBin(L.energy);
                    Kokkos::atomic_add(&spec(bin), (double)weight);
                }

                // Advance primary Beer-Lambert
                acc_prob *= att;
                t_cur = t_next;

                // Step to next voxel
                if (axis == 0) { txN += tDx; vx += sx_; }
                else if (axis == 1) { tyN += tDy; vy += sy_; }
                else                { tzN += tDz; vz += sz_; }
            }
        }
    );
    Kokkos::fence();

    // Copy back to caller's array
    auto spec_h = Kokkos::create_mirror_view(spec);
    Kokkos::deep_copy(spec_h, spec);
    for (int b = 0; b < N_CHANNELS; ++b)
        spectrum[b] += spec_h(b);
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        // ── Build polycapillary optic ──────────────────────────────────────────
        int   izf[PC_NELEM]; float wtf[PC_NELEM];
        for (int i = 0; i < PC_NELEM; ++i) { izf[i] = PC_Z[i]; wtf[i] = PC_WT[i]; }

        PolyCap optic(
            0.f, PC_LEN,
            PC_REXT_IN, PC_REXT_OUT,
            PC_RCAP_IN, PC_RCAP_OUT,
            PC_FOCAL_IN, PC_FOCAL_OUT,
            PC_NELEM, izf, wtf,
            PC_DENSITY, PC_ROUGHNESS, PC_NCAP);

        // Set optic mu_rho at excitation energy
        float mu_optic = 0.f;
        for (int i = 0; i < PC_NELEM; ++i)
            mu_optic += (PC_WT[i]/100.f) * (float)CS_Total(PC_Z[i], EXCITATION_E, nullptr);
        optic.setMuRho(mu_optic * PC_DENSITY);

        float openArea = (float)PC_NCAP * PC_RCAP_IN * PC_RCAP_IN / (PC_REXT_IN * PC_REXT_IN);

        // ── Pre-compute XRF cross-section table ───────────────────────────────
        // mu_total of the mixture at excitation energy [cm²/g]
        float mu_prim_g = mixCS(EXCITATION_E);   // cm²/g
        float mu_prim_cm = mu_prim_g * MAT_DENSITY;  // cm⁻¹

        const int ntot = MAT_NELEM * N_LINES;
        XRFLine lines[MAT_NELEM * N_LINES];
        printf("XRF lines at %.1f keV excitation:\n", EXCITATION_E);
        printf("  %-4s  %-6s  %-12s  %-12s\n", "Z", "Line", "E_fluor(keV)", "CS_fluor(cm2/g)");
        for (int ei = 0; ei < MAT_NELEM; ++ei) {
            int Z = MAT_Z[ei];
            float wf = MAT_W[ei];
            for (int li = 0; li < N_LINES; ++li) {
                int idx = ei * N_LINES + li;
                float e_f = (float)LineEnergy(Z, LINE_IDS[li], nullptr);
                float cs_f = 0.f;
                if (e_f > 0.f && e_f < EXCITATION_E)
                    cs_f = wf * (float)CS_FluorLine(Z, LINE_IDS[li], EXCITATION_E, nullptr);
                float mu_exit = (e_f > 0.f) ? mixCS(e_f) : 0.f;
                lines[idx] = { e_f, cs_f, mu_exit, mu_prim_g };
                const char* lname = (LINE_IDS[li] == KA_LINE) ? "Kα" : "Kβ";
                printf("  Z=%-3d %-4s  E=%6.3f keV  cs=%.4e cm²/g\n",
                       Z, lname, e_f, cs_f);
            }
        }
        printf("\nSample: %dx%dx%d voxels (%.0fx%.0fx%.0f µm), voxel=%.0f µm\n",
               NX, NY, NZ, SAM_DX_UM, SAM_DY_UM, SAM_DZ_UM, VOX_UM);
        printf("Primary mu_total = %.3f cm²/g, mu*rho = %.3f cm⁻¹\n",
               mu_prim_g, mu_prim_cm);
        printf("Open area fraction: %.4f\n", openArea);
        printf("N_RAYS per scan point: %d\n\n", N_RAYS);

        // ── Scan loop ──────────────────────────────────────────────────────────
        // One spectrum per scan point [N_SCAN][N_CHANNELS]
        std::vector<std::vector<double>> spectra(N_SCAN, std::vector<double>(N_CHANNELS, 0.0));

        auto t0_total = std::chrono::steady_clock::now();

        for (int sp = 0; sp < N_SCAN; ++sp) {
            float scan_z_cm = SCAN_Z_UM[sp] * 1e-4f;  // µm → cm

            auto t0 = std::chrono::steady_clock::now();

            simulateScanPoint(optic, openArea, scan_z_cm,
                              lines, ntot, mu_prim_cm,
                              spectra[sp].data());

            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1-t0).count();

            // Print peak intensities for key lines (Cu Kα, Zn Kα)
            // Find Cu Kα and Zn Kα bins
            int cu_ka_bin = -1, zn_ka_bin = -1;
            for (int li = 0; li < ntot; ++li) {
                int Z = MAT_Z[li / N_LINES];
                if (Z == 29 && li % N_LINES == 0) cu_ka_bin = energyBin(lines[li].energy);
                if (Z == 30 && li % N_LINES == 0) zn_ka_bin = energyBin(lines[li].energy);
            }

            printf("Scan point %2d  z=%+6.0f µm  %.1f ms  |  Cu Kα=%.4e  Zn Kα=%.4e\n",
                   sp+1, SCAN_Z_UM[sp], ms,
                   (cu_ka_bin >= 0) ? spectra[sp][cu_ka_bin] : 0.0,
                   (zn_ka_bin >= 0) ? spectra[sp][zn_ka_bin] : 0.0);
        }

        auto t1_total = std::chrono::steady_clock::now();
        double ms_total = std::chrono::duration<double, std::milli>(t1_total-t0_total).count();
        printf("\nTotal simulation time: %.2f s\n", ms_total * 1e-3);

        // ── Write spectra to CSV ───────────────────────────────────────────────
        const char* out_path = "test-data/out/muXRF_spectra.csv";
        FILE* fp = fopen(out_path, "w");
        if (fp) {
            // Header: energy column, then one column per scan point
            fprintf(fp, "E_keV");
            for (int sp = 0; sp < N_SCAN; ++sp)
                fprintf(fp, ",z%+.0fum", SCAN_Z_UM[sp]);
            fprintf(fp, "\n");

            for (int b = 0; b < N_CHANNELS; ++b) {
                float e = E_MIN + (b + 0.5f) * E_STEP;
                fprintf(fp, "%.4f", e);
                for (int sp = 0; sp < N_SCAN; ++sp)
                    fprintf(fp, ",%.6e", spectra[sp][b]);
                fprintf(fp, "\n");
            }
            fclose(fp);
            printf("Spectra written to %s\n", out_path);
        } else {
            printf("Warning: could not open %s for writing\n", out_path);
        }

        // ── Print summary table: integrated Kα intensities vs depth ──────────
        printf("\nIntegrated Kα intensity vs scan depth:\n");
        printf("%-10s", "z(µm)");
        const char* elem_names[] = {"Fe", "Ni", "Cu", "Zn", "Sn", "Pb"};
        for (int ei = 0; ei < MAT_NELEM; ++ei) printf("  %-12s", elem_names[ei]);
        printf("\n");

        for (int sp = 0; sp < N_SCAN; ++sp) {
            printf("%-10.0f", SCAN_Z_UM[sp]);
            for (int ei = 0; ei < MAT_NELEM; ++ei) {
                // Integrate ±2 bins around the Kα peak bin
                int bin = energyBin(lines[ei * N_LINES].energy);
                double I = 0.0;
                for (int db = -2; db <= 2; ++db) {
                    int b2 = bin + db;
                    if (b2 >= 0 && b2 < N_CHANNELS) I += spectra[sp][b2];
                }
                printf("  %-12.4e", I);
            }
            printf("\n");
        }
    }
    Kokkos::finalize();
    return 0;
}
