#pragma once
#include "Platform.hpp"
#include "Ray.hpp"

#ifndef __METAL_VERSION__
#   include <cmath>
#   include <cstdio>
#   include "../api/XRayLibAPI.hpp"
#endif

// ── PolyCap_new ───────────────────────────────────────────────────────────────
// Polycapillary X-ray optic — native core class following the Ray/Material/Voxel
// design pattern. Compiles unchanged for Kokkos (CUDA/HIP/OpenMP) and Metal MSL.
//
// Usage:
//   PolyCap_new optic(posZ, length, rExtIn, rExtOut, rCapIn, rCapOut,
//                     focalIn, focalOut, PolyCap_new::ELLIPSOIDAL,
//                     nElem, atomicNumbers, weightFracs, density, roughness, nCap);
//   optic.trace(ray);   // modifies ray in-place
//
// Physics per wall reflection (in reflect()):
//   • Full complex Fresnel reflectivity R_s, R_p via precomputed refractive index
//   • Anomalous scattering f'(E) included in delta_[] table (via XRayLibAPI at
//     construction time)
//   • Debye-Waller surface-roughness attenuation
//   • s/p polarization-weighted total R; s/p frame updated geometrically
//
// Refractive-index tables delta_[100], beta_[100] are fixed-size floats on a
// 0.2 keV grid (same as ChemElement) — no dynamic allocation, MSL-compatible.

class PolyCap_new {
public:
    enum ProfileShape { CONICAL, PARABOLOIDAL, ELLIPSOIDAL };

private:
    // ── Physical constants ────────────────────────────────────────────────────
    VT_SCONSTEXPR float PC_HC     = 1.23984193e-7f;  // hc [keV·cm]
    VT_SCONSTEXPR float PC_TWOPI  = 6.28318530718f;
    VT_SCONSTEXPR float PC_NA_R0  = 1.69699e11f;     // N_A × R₀ [cm⁻²·mol]
    VT_SCONSTEXPR float PC_COSPI6 = 0.86602540378f;  // cos(π/6)

    // ── Refractive-index grid ─────────────────────────────────────────────────
    VT_SCONSTEXPR float PC_E_STEP = 0.2f;   // keV per grid step (matches ChemElement)
    VT_SCONSTEXPR int   PC_N_E    = 100;    // 0..19.8 keV

    // ── Trace resolution ──────────────────────────────────────────────────────
    VT_SCONSTEXPR int   PC_NSTEPS = 200;    // capillary frustum segments; ≥ max reflections

    // ── Geometry ──────────────────────────────────────────────────────────────
    float pos_z_    = 0.f;
    float length_   = 1.f;
    float r_ext_in_ = 0.f, r_ext_out_ = 0.f;  // outer radius at entrance / exit [cm]
    float r_cap_in_ = 0.f, r_cap_out_ = 0.f;  // inner capillary radius  [cm]
    float focal_in_ = 0.f, focal_out_ = 0.f;  // focal distances [cm]
    ProfileShape profile_ = CONICAL;
    // PARABOLOIDAL: coeff_ = {a, b, c} so ext(t) = a + b·t + c·t², t = z-pos_z_
    // ELLIPSOIDAL:  coeff_ = {b, k, ±a²}, ext(t) = √(b²-b²·t²/|a²|)+k
    //               negative a² signals collimating geometry (z reversed)
    float coeff_[3] = {};
    float n_shells_ = 0.f;    // hexagonal shell count; 0 = monocapillary

    // ── Glass ─────────────────────────────────────────────────────────────────
    float roughness_ = 0.f;   // surface roughness [Å]

    // ── Precomputed complex refractive index: n = (1−δ) + iβ ─────────────────
    float delta_[100] = {};   // δ(E) on PC_E_STEP grid
    float beta_ [100] = {};   // β(E) on PC_E_STEP grid

    // ═════════════════════════════════════════════════════════════════════════
    // Profile helpers
    // ═════════════════════════════════════════════════════════════════════════

    // Inner capillary radius at z (always linear)
    KOKKOS_INLINE_FUNCTION float capRadius(float z) const VT_DEVICE_METH {
        return r_cap_in_ + (r_cap_out_ - r_cap_in_) * (z - pos_z_) / length_;
    }

    // Outer optic radius at z (conical / paraboloidal / ellipsoidal)
    KOKKOS_INLINE_FUNCTION float extRadius(float z) const VT_DEVICE_METH {
        float t = z - pos_z_;
        if (profile_ == PARABOLOIDAL) {
            return coeff_[0] + coeff_[1]*t + coeff_[2]*t*t;
        } else if (profile_ == ELLIPSOIDAL) {
            float b    = coeff_[0], k = coeff_[1], a_sq = coeff_[2];
            float z_loc = (a_sq < 0.f) ? length_ - t : t;  // negative → collimating
            float val   = b*b * (1.f - z_loc*z_loc / fabsf(a_sq));
            return sqrtf(fmaxf(0.f, val)) + k;
        } else {  // CONICAL
            return r_ext_in_ + (r_ext_out_ - r_ext_in_) * t / length_;
        }
    }

    // Capillary axis x at z for hex index (q, r)
    KOKKOS_INLINE_FUNCTION float axisX(float q, float r, float z) const VT_DEVICE_METH {
        float sp = extRadius(z) / (2.f * PC_COSPI6 * (n_shells_ + 1.f));
        return (2.f*q + r) * PC_COSPI6 * sp;
    }

    // Capillary axis y at z for hex index r
    KOKKOS_INLINE_FUNCTION float axisY(float r, float z) const VT_DEVICE_METH {
        float sp = extRadius(z) / (2.f * PC_COSPI6 * (n_shells_ + 1.f));
        return r * 1.5f * sp;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Hex grid helpers
    // ═════════════════════════════════════════════════════════════════════════

    // Is point (x, y) inside hexagonal optic of circumradius ext?
    KOKKOS_INLINE_FUNCTION static bool withinHex(float ext, float x, float y) {
        float d = ext * PC_COSPI6;  // inradius
        return fabsf(y)                        <= d
            && fabsf(PC_COSPI6*x + 0.5f*y)    <= d
            && fabsf(PC_COSPI6*x - 0.5f*y)    <= d;
    }

    // Hex axial indices (q, r) via cube-coordinate rounding
    KOKKOS_INLINE_FUNCTION static void capillaryIndex(float ext, float n_shells,
                                                        float x, float y,
                                                        float& q, float& r) {
        float sp = ext / (2.f * PC_COSPI6 * (n_shells + 1.f));
        float qf =  (x / (2.f*PC_COSPI6) - y / 3.f) / sp;
        float rf =   y * (2.f/3.f) / sp;
        float sf = -qf - rf;
        float rq = roundf(qf), rr = roundf(rf), rs = roundf(sf);
        float dq = fabsf(rq-qf), dr = fabsf(rr-rf), ds = fabsf(rs-sf);
        if      (dq > dr && dq > ds) { q = -rr-rs;  r =  rr;      }
        else if (dr > ds)            { q =  rq;      r = -rq-rs;   }
        else                         { q =  rq;      r =  rr;      }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Wall intersection
    // ═════════════════════════════════════════════════════════════════════════

    // Intersect ray with one conical frustum segment [cap0→cap1] of radii [r0→r1].
    // px/py/pz: current ray position (used as minimum-z filter: hit must be > pz+1e-5).
    // On success: sets hit point (hx,hy,hz) and outward surface normal (nx,ny,nz).
    KOKKOS_INLINE_FUNCTION static bool wallHit(
        float c0x, float c0y, float c0z,   // cap axis start
        float c1x, float c1y, float c1z,   // cap axis end
        float r0,  float r1,               // inner radii at c0, c1
        float px,  float py,  float pz,    // current ray position
        float ddx, float ddy, float ddz,   // ray direction (unit, ddz > 0)
        float& hx, float& hy, float& hz,   // out: hit point
        float& nx, float& ny, float& nz)   // out: outward surface normal
    {
        // Propagate ray to segment start plane z = c0z
        float p0x = px + ddx * (c0z - pz) / ddz;
        float p0y = py + ddy * (c0z - pz) / ddz;

        float dz_cap = c1z - c0z;
        float ax0    = p0x - c0x,  ay0 = p0y - c0y;
        float ex     = ddx/ddz - (c1x-c0x)/dz_cap;  // relative x-slope
        float ey     = ddy/ddz - (c1y-c0y)/dz_cap;  // relative y-slope
        float eR     = (r1 - r0) / dz_cap;           // radius change per z

        // Quadratic a·t² + b·t + c = 0  (t = z-displacement from p0z)
        float qa = ex*ex + ey*ey - eR*eR;
        float qb = 2.f*(ax0*ex + ay0*ey) - 2.f*r0*eR;
        float qc = ax0*ax0 + ay0*ay0 - r0*r0;
        float discr = qb*qb - 4.f*qa*qc;
        if (discr < 0.f) return false;

        float best = 1e30f;
        if (fabsf(qa) < 1e-30f) {
            if (fabsf(qb) > 1e-30f) {
                float t  = -qc / qb;
                float zh = c0z + t;
                if (t/ddz > 1e-10f && zh > c0z && zh <= c1z && zh - pz > 1e-5f)
                    best = t;
            }
        } else {
            float sq = sqrtf(discr);
            float t1 = (-qb + sq) / (2.f*qa);
            float t2 = (-qb - sq) / (2.f*qa);
            for (int k = 0; k < 2; ++k) {
                float t  = (k == 0) ? t1 : t2;
                float zh = c0z + t;
                if (t/ddz < 1e-10f)              continue;
                if (zh <= c0z || zh > c1z)       continue;
                if (zh - pz    < 1e-5f)          continue;
                if (t < best) best = t;
            }
        }
        if (best > 1e29f) return false;

        // Hit point
        float dp = best / ddz;
        hz = c0z + best;
        hx = p0x + dp * ddx;
        hy = p0y + dp * ddy;

        // Outward surface normal (mirrors PolyCap_old capilSegment)
        float rx = hx-c0x, ry = hy-c0y, rz = hz-c0z;
        float cdx = c1x-c0x, cdy = c1y-c0y;
        float d_seg_sq = cdx*cdx + cdy*cdy + dz_cap*dz_cap;
        float pt       = (rx*cdx + ry*cdy + rz*dz_cap) / d_seg_sq;
        float radx = hx-(c0x+pt*cdx), rady = hy-(c0y+pt*cdy), radz = hz-(c0z+pt*dz_cap);
        float r_dist = sqrtf(radx*radx + rady*rady + radz*radz);
        float d_seg  = sqrtf(d_seg_sq);
        float tga    = (r0 - r1) / d_seg;
        float ih     = 1.f / sqrtf(1.f + tga*tga);  // cos(γ)
        float cga    = ih,  sga = tga * ih;           // sin(γ)
        float chx    = cdx/d_seg, chy = cdy/d_seg, chz = dz_cap/d_seg;

        nx = cga * radx/r_dist + sga * chx;
        ny = cga * rady/r_dist + sga * chy;
        nz = cga * radz/r_dist + sga * chz;
        float nl = sqrtf(nx*nx + ny*ny + nz*nz);
        nx /= nl;  ny /= nl;  nz /= nl;
        return true;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Physics helpers
    // ═════════════════════════════════════════════════════════════════════════

    // 7a. Linearly interpolated complex refractive index on PC_E_STEP grid.
    KOKKOS_INLINE_FUNCTION void refractiveIndex(float e, float& delta, float& beta) const VT_DEVICE_METH {
        int idx = (int)(e / PC_E_STEP);
        if (idx <= 0)         { delta = delta_[0];          beta = beta_[0];          return; }
        if (idx >= PC_N_E-1)  { delta = delta_[PC_N_E-1];   beta = beta_[PC_N_E-1];   return; }
        float t = e / PC_E_STEP - (float)idx;
        delta = delta_[idx] + t * (delta_[idx+1] - delta_[idx]);
        beta  = beta_ [idx] + t * (beta_ [idx+1] - beta_ [idx]);
    }

    // 7b. Full complex Fresnel reflectance for s and p polarizations.
    //
    // cos_alfa: cosine of angle between incident direction and surface normal
    //           = sin(grazing angle θ_g); same as polycap_old's cos_alfa.
    // Implements r_s = (cos_alfa − n·cos_t)/(cos_alfa + n·cos_t)
    //            r_p = (cos_t − n·cos_alfa)/(cos_t + n·cos_alfa)
    // using manual (re,im) float pairs — no std::complex required (Metal-safe).
    KOKKOS_INLINE_FUNCTION static void fresnelR(float cos_alfa, float delta, float beta,
                                                  float& Rs, float& Rp) {
        float n_re  = 1.f - delta,  n_im = beta;
        float sin2  = fmaxf(0.f, 1.f - cos_alfa*cos_alfa);

        // n² = (n_re + i·n_im)²
        float n2_re = n_re*n_re - n_im*n_im;
        float n2_im = 2.f*n_re*n_im;

        // cos²(θ_t) = 1 − sin²/n²  (complex division)
        float D       = n2_re*n2_re + n2_im*n2_im;
        float ct2_re  = 1.f - sin2*n2_re / D;
        float ct2_im  =       sin2*n2_im / D;   // > 0 for physical β > 0

        // cos(θ_t) = √(ct2) via complex square root
        float r_mag  = sqrtf(ct2_re*ct2_re + ct2_im*ct2_im);
        float ct_re  = sqrtf(fmaxf(0.f, (r_mag + ct2_re) * 0.5f));
        float ct_im  = (ct2_im >= 0.f ? 1.f : -1.f)
                     * sqrtf(fmaxf(0.f, (r_mag - ct2_re) * 0.5f));

        // n·cos(θ_t)
        float nc_re = n_re*ct_re - n_im*ct_im;
        float nc_im = n_re*ct_im + n_im*ct_re;

        // Rs = |r_s|²  where r_s = (cos_alfa − n·cos_t)/(cos_alfa + n·cos_t)
        float rs_nr = cos_alfa - nc_re,  rs_ni = -nc_im;
        float rs_dr = cos_alfa + nc_re,  rs_di =  nc_im;
        Rs = (rs_nr*rs_nr + rs_ni*rs_ni) / (rs_dr*rs_dr + rs_di*rs_di);

        // Rp = |r_p|²  where r_p = (cos_t − n·cos_alfa)/(cos_t + n·cos_alfa)
        float rp_nr = ct_re - n_re*cos_alfa,  rp_ni = ct_im - n_im*cos_alfa;
        float rp_dr = ct_re + n_re*cos_alfa,  rp_di = ct_im + n_im*cos_alfa;
        Rp = (rp_nr*rp_nr + rp_ni*rp_ni) / (rp_dr*rp_dr + rp_di*rp_di);
    }

    // 7c. Debye-Waller surface-roughness factor.
    //
    // sin_graz: sin(θ_grazing) = cos_alfa (angle between direction and normal).
    // roughness: glass surface roughness [Å].
    // Factor = exp(−(4π σ sin(θ_g) E / hc)²)  with hc/4π ≈ 1/1.01358 keV⁻¹Å⁻¹cm.
    KOKKOS_INLINE_FUNCTION static float debeyeWaller(float roughness, float sin_graz, float energy) {
        float cons = 1.01358f * energy * sin_graz * roughness;
        return expf(-cons * cons);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Reflection: update Ray at wall hit (hx,hy,hz) with outward normal (nx,ny,nz).
    // Returns false if ray is absorbed (prob fell below threshold).
    // ═════════════════════════════════════════════════════════════════════════

    KOKKOS_INLINE_FUNCTION bool reflect(Ray& ray,
                                         float hx, float hy, float hz,
                                         float nx, float ny, float nz) const VT_DEVICE_METH {
        ray.setStartCoordinates(hx, hy, hz);

        float dx = ray.getDirX(), dy = ray.getDirY(), dz = ray.getDirZ();

        // cos_alfa = cos(angle between dir and normal) = sin(grazing angle θ_g)
        float cos_alfa = nx*dx + ny*dy + nz*dz;
        if (cos_alfa < 0.f) { nx=-nx; ny=-ny; nz=-nz; cos_alfa=-cos_alfa; }  // flip toward photon

        // Physics helpers
        float energy = ray.getEnergyKeV();
        float delta_v, beta_v;
        refractiveIndex(energy, delta_v, beta_v);
        float Rs, Rp;
        fresnelR(cos_alfa, delta_v, beta_v, Rs, Rp);
        float dw = debeyeWaller(roughness_, cos_alfa, energy);  // sin_graz = cos_alfa

        // s-direction: perpendicular to plane of incidence (n̂ × d̂)
        float sx = ny*dz - nz*dy;
        float sy = nz*dx - nx*dz;
        float sz = nx*dy - ny*dx;
        float sl = sqrtf(sx*sx + sy*sy + sz*sz);
        if (sl < 1e-7f) { ray.setIAFlag(false); return false; }  // degenerate normal-incidence
        sx /= sl;  sy /= sl;  sz /= sl;

        // Polarization fraction along s
        float as_dot = ray.getSPolX()*sx + ray.getSPolY()*sy + ray.getSPolZ()*sz;
        float frac_s = as_dot * as_dot;
        float frac_p = 1.f - frac_s;

        // Total reflectance: polarization-weighted Fresnel × Debye-Waller
        float R = (Rs*frac_s + Rp*frac_p) * dw;
        ray.setProb(ray.getProb() * R);
        if (ray.getProb() < 1e-6f) { ray.setIAFlag(false); return false; }

        // Specular reflection: d′ = d − 2(d·n̂)n̂
        float d2x = dx - 2.f*cos_alfa*nx;
        float d2y = dy - 2.f*cos_alfa*ny;
        float d2z = dz - 2.f*cos_alfa*nz;
        float d2l = sqrtf(d2x*d2x + d2y*d2y + d2z*d2z);
        d2x /= d2l;  d2y /= d2l;  d2z /= d2l;
        ray.setEndCoordinates(d2x, d2y, d2z);

        // Update polarization frames: s_new = n̂ × d′,  p_new = d′ × s_new
        float snx = ny*d2z - nz*d2y;
        float sny = nz*d2x - nx*d2z;
        float snz = nx*d2y - ny*d2x;
        float snl = sqrtf(snx*snx + sny*sny + snz*snz);
        if (snl > 1e-7f) { snx/=snl; sny/=snl; snz/=snl; }
        ray.setSPol(snx, sny, snz);
        ray.setPPol(d2y*snz - d2z*sny,
                    d2z*snx - d2x*snz,
                    d2x*sny - d2y*snx);

        ray.setIANum(ray.getIANum() + 1);
        return true;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Host-only: quadratic least-squares fit y = c[0] + c[1]x + c[2]x² (3×3)
    // Used by the constructor for the PARABOLOIDAL profile.
    // ═════════════════════════════════════════════════════════════════════════

#ifndef __METAL_VERSION__
    static void fitQuadratic(const double* px, const double* py, int np, double c[3]) {
        double A[3][3] = {}, rhs[3] = {};
        for (int i = 0; i < np; ++i) {
            double xi[3] = {1.0, px[i], px[i]*px[i]};
            for (int row = 0; row < 3; ++row) {
                rhs[row] += xi[row] * py[i];
                for (int col = 0; col < 3; ++col) A[row][col] += xi[row] * xi[col];
            }
        }
        // Gaussian elimination with partial pivoting
        for (int col = 0; col < 3; ++col) {
            int pivot = col;
            for (int row = col+1; row < 3; ++row)
                if (std::abs(A[row][col]) > std::abs(A[pivot][col])) pivot = row;
            for (int s = 0; s < 3; ++s) { double tmp = A[col][s]; A[col][s]=A[pivot][s]; A[pivot][s]=tmp; }
            double tmp = rhs[col]; rhs[col]=rhs[pivot]; rhs[pivot]=tmp;
            for (int row = col+1; row < 3; ++row) {
                double f = A[row][col] / A[col][col];
                rhs[row] -= f * rhs[col];
                for (int s = col; s < 3; ++s) A[row][s] -= f * A[col][s];
            }
        }
        for (int row = 2; row >= 0; --row) {
            c[row] = rhs[row];
            for (int s = row+1; s < 3; ++s) c[row] -= A[row][s] * c[s];
            c[row] /= A[row][row];
        }
    }
#endif

public:
    KOKKOS_INLINE_FUNCTION PolyCap_new() = default;

    // ── Constructor (host-only) ───────────────────────────────────────────────
    // weightFracs may be percentages (sum ≈ 100) or fractions (sum ≈ 1) — normalised automatically.
    // Calls XRayLibAPI (Fi, CS_Tot, A) once per energy grid step to build delta_[]/beta_[].

#ifndef __METAL_VERSION__
    PolyCap_new(float posZ, float length,
                float rExtIn,  float rExtOut,
                float rCapIn,  float rCapOut,
                float focalIn, float focalOut,
                ProfileShape profile,
                int nElements, const int* atomicNumbers, const float* weightFracs,
                float density, float roughness, int nCapillaries)
        : pos_z_(posZ), length_(length),
          r_ext_in_(rExtIn), r_ext_out_(rExtOut),
          r_cap_in_(rCapIn), r_cap_out_(rCapOut),
          focal_in_(focalIn), focal_out_(focalOut),
          profile_(profile), roughness_(roughness)
    {
        n_shells_ = roundf(sqrtf(12.f * (float)nCapillaries - 3.f) / 6.f - 0.5f);

        // Normalise weight fractions (accept percentages or fractions)
        int ne = (nElements < 8) ? nElements : 8;
        float w[8] = {};
        float wsum = 0.f;
        for (int j = 0; j < ne; ++j) { w[j] = weightFracs[j]; wsum += w[j]; }
        if (wsum > 1.5f) for (int j = 0; j < ne; ++j) w[j] /= wsum;

        // ── Profile coefficients ──────────────────────────────────────────────
        if (profile == PARABOLOIDAL) {
            double px[4], py[4];
            px[0] = 0.;      py[0] = rExtIn;
            px[3] = length;  py[3] = rExtOut;
            px[1] = (focalIn  <= length) ? focalIn  / 10.0 : length / 10.0;
            py[1] = (rExtIn  / focalIn)  * px[1] + rExtIn;
            px[2] = (focalOut <= length) ? length - focalOut / 10.0 : length * 0.9;
            py[2] = (-rExtOut / focalOut) * (px[2] - length) + rExtOut;
            double c[3];
            fitQuadratic(px, py, 4, c);
            coeff_[0] = (float)c[0];
            coeff_[1] = (float)c[1];
            coeff_[2] = (float)c[2];
        } else if (profile == ELLIPSOIDAL) {
            double b, k, a_sq;
            if (rExtOut < rExtIn) {  // focusing: exit smaller than entrance
                double slope = rExtOut / focalOut;
                double dr    = rExtOut - rExtIn;
                b    = (-dr*dr - slope*length*dr) / (slope*length + 2.0*dr);
                k    = rExtIn - b;
                a_sq = (b*b * length) / (slope * (rExtOut - k));
                coeff_[2] = (float)a_sq;   // positive → no z-reversal
            } else {                  // collimating: entrance smaller
                double slope = rExtIn / focalIn;
                double dr    = rExtIn - rExtOut;
                b    = (-dr*dr - slope*length*dr) / (slope*length + 2.0*dr);
                k    = rExtOut - b;
                a_sq = std::abs((b*b * length) / (slope * (rExtIn - k)));
                coeff_[2] = -(float)a_sq;  // negative → z-reversed in extRadius()
            }
            coeff_[0] = (float)b;
            coeff_[1] = (float)k;
        }

        // ── Precompute delta_[]/beta_[] on 0.2 keV grid ───────────────────────
        float A_arr[8] = {};
        for (int j = 0; j < ne; ++j) A_arr[j] = (float)XRayLibAPI::A(atomicNumbers[j]);

        for (int i = 1; i < PC_N_E; ++i) {
            float E      = (float)i * PC_E_STEP;
            float scatf  = 0.f;   // Σ w_j (Z_j + f'_j) / A_j
            float mu_lin = 0.f;   // Σ w_j · CS_Tot(Z_j,E)  [cm²/g, before × ρ]
            for (int j = 0; j < ne; ++j) {
                float fi = (float)XRayLibAPI::Fi(atomicNumbers[j], (double)E);
                scatf  += w[j] * ((float)atomicNumbers[j] + fi) / A_arr[j];
                mu_lin += w[j] * (float)XRayLibAPI::CS_Tot(atomicNumbers[j], (double)E);
            }
            mu_lin *= density;  // linear attenuation [cm⁻¹]
            float lam   = PC_HC / E;
            delta_[i] = lam*lam * (PC_NA_R0 * density / PC_TWOPI) * scatf;
            beta_ [i] = (PC_HC / (2.f * PC_TWOPI)) * (mu_lin / E);
        }
        delta_[0] = delta_[1];
        beta_ [0] = beta_ [1];
    }
#endif  // !__METAL_VERSION__

    // ── Entry check ───────────────────────────────────────────────────────────
    KOKKOS_INLINE_FUNCTION bool isEntering(const Ray& ray) const VT_DEVICE_METH {
        if (fabsf(ray.getDirZ()) < 1e-9f) return false;
        float t = (pos_z_ - ray.getStartZ()) / ray.getDirZ();
        if (t < 0.f) return false;
        float x   = ray.getStartX() + t * ray.getDirX();
        float y   = ray.getStartY() + t * ray.getDirY();
        float ext = extRadius(pos_z_);
        if (n_shells_ == 0.f)
            return x*x + y*y <= ext*ext;
        return withinHex(ext, x, y);
    }

    // ── Main ray trace ────────────────────────────────────────────────────────
    // 1. Checks entrance; identifies capillary (hex index q, r).
    // 2. Iterates up to PC_NSTEPS times: per step scans frustum segments forward
    //    from the current position, calls wallHit() + reflect() on each hit.
    // 3. On exit (no further hit): advances ray to exit plane, sets iaFlag = true.
    KOKKOS_INLINE_FUNCTION void trace(Ray& ray) const VT_DEVICE_METH {
        if (!isEntering(ray)) { ray.setIAFlag(false); return; }

        // Entry point on entrance plane z = pos_z_
        float t_en = (pos_z_ - ray.getStartZ()) / ray.getDirZ();
        float xe   = ray.getStartX() + t_en * ray.getDirX();
        float ye   = ray.getStartY() + t_en * ray.getDirY();

        // Identify capillary (hex indices q, r); verify entry is in glass interior
        float q = 0.f, r_idx = 0.f;
        float ext0 = extRadius(pos_z_);
        if (n_shells_ > 0.f) {
            capillaryIndex(ext0, n_shells_, xe, ye, q, r_idx);
            float cx0 = axisX(q, r_idx, pos_z_);
            float cy0 = axisY(r_idx,    pos_z_);
            float dx0 = xe-cx0, dy0 = ye-cy0;
            float rc0 = capRadius(pos_z_);
            if (dx0*dx0 + dy0*dy0 > rc0*rc0) { ray.setIAFlag(false); return; }  // in glass wall
        }

        // Move ray to entrance plane
        ray.setStartCoordinates(xe, ye, pos_z_);

        float dz_seg = length_ / (float)PC_NSTEPS;
        int seg = 0;  // lower-bound segment index; never decreases

        for (int step = 0; step < PC_NSTEPS; ++step) {
            bool found = false;
            for (int i = seg; i < PC_NSTEPS; ++i) {
                float z0 = pos_z_ + (float)i       * dz_seg;
                float z1 = pos_z_ + (float)(i + 1) * dz_seg;

                float hx, hy, hz, nx, ny, nz;
                if (!wallHit(axisX(q, r_idx, z0), axisY(r_idx, z0), z0,
                             axisX(q, r_idx, z1), axisY(r_idx, z1), z1,
                             capRadius(z0), capRadius(z1),
                             ray.getStartX(), ray.getStartY(), ray.getStartZ(),
                             ray.getDirX(),   ray.getDirY(),   ray.getDirZ(),
                             hx, hy, hz, nx, ny, nz))
                    continue;

                if (!reflect(ray, hx, hy, hz, nx, ny, nz)) return;  // absorbed

                // Advance seg to the segment containing the new hit position
                int new_seg = (int)((hz - pos_z_) / dz_seg);
                seg = (new_seg > seg) ? new_seg : seg;
                if (seg >= PC_NSTEPS) seg = PC_NSTEPS - 1;
                found = true;
                break;
            }
            if (!found) break;  // no further wall hit — photon exits capillary
        }

        // Advance to exit plane z = pos_z_ + length_
        float exit_z = pos_z_ + length_;
        float dz_rem = exit_z - ray.getStartZ();
        if (dz_rem > 0.f && fabsf(ray.getDirZ()) > 1e-9f) {
            float t_ex = dz_rem / ray.getDirZ();
            ray.setStartCoordinates(ray.getStartX() + t_ex * ray.getDirX(),
                                    ray.getStartY() + t_ex * ray.getDirY(),
                                    exit_z);
        }
        ray.setIAFlag(true);
    }

    // ── Debug print ───────────────────────────────────────────────────────────
#ifndef __METAL_VERSION__
    void print() const {
        const char* pname = (profile_ == PARABOLOIDAL) ? "PARABOLOIDAL"
                          : (profile_ == ELLIPSOIDAL)  ? "ELLIPSOIDAL"
                          :                              "CONICAL";
        printf("PolyCap_new: %s  L=%.3fcm  rExt=(%.5f→%.5f)cm  rCap=(%.7f→%.7f)cm"
               "  f=(%.3f,%.3f)cm  nShells=%d\n",
               pname, length_, r_ext_in_, r_ext_out_, r_cap_in_, r_cap_out_,
               focal_in_, focal_out_, (int)n_shells_);
    }
#endif
};
