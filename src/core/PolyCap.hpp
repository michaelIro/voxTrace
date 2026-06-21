#pragma once
#include "Platform.hpp"
#include "Ray.hpp"

#ifndef __METAL_VERSION__
#   include <cmath>
#   include <cstdio>
#   include "../api/XRayLibAPI.hpp"
#endif

// ── PolyCap ───────────────────────────────────────────────────────────
// Polycapillary X-ray optic — standalone native core class in the Ray/Voxel/
// Material design pattern. Ray is the parallelization anchor: trace(ray) advances
// ONE photon through the optic in place. Self-contained: xraylib is used host-side
// at construction only (like ChemElement::discretize()), never at runtime.
//
// On return:
//   ray.getIAFlag() == true  → transmitted; ray holds exit position/direction and
//                              ray.getProb() the transmission weight.
//   ray.getIAFlag() == false → absorbed or escaped.
//
// Optic axis is the ray's +z; the photon enters at z = posZ and exits at
// z = posZ + length. Geometry/physics mirror the original polycap-1.2 library
// (https://github.com/PieterTack/polycap), so the transmitted-weight output
// matches it to ~1%.
//
// Precision note: an off-axis capillary axis sits ~0.1–0.3 cm off the optic axis
// while the photon rides ~1 µm from it and the grazing angle is ~mrad. Forming
// (photon − axis) is a catastrophic cancellation that float cannot hold (it costs
// a few % of the grazing angle and compounds over the dozens of reflections),
// so the per-photon trace state is kept in double — exactly as the reference
// library keeps its photon in double. Storage (tables, geometry) stays float;
// only the transient trace locals are double, matching the reference output.

class PolyCap {
public:
    enum Profile { CONICAL, PARABOLOIDAL, ELLIPSOIDAL };

private:
    // ── Physical constants (cgs / keV) ────────────────────────────────────────
    VT_SCONSTEXPR double PC_HC     = 1.23984193e-7;   // hc [keV·cm]
    VT_SCONSTEXPR double PC_TWOPI  = 6.283185307179586;
    VT_SCONSTEXPR double PC_NA_R0  = 1.6969912781e11; // N_A · r_e [cm⁻²·mol]
    VT_SCONSTEXPR double PC_COSPI6 = 0.8660254037844387; // cos(π/6)

    // ── Refractive-index grid (matches ChemElement spacing) ───────────────────
    VT_SCONSTEXPR double PC_E_STEP = 0.2;   // keV per grid step
    VT_SCONSTEXPR int    PC_N_E    = 100;   // 0 .. 19.8 keV

    // ── Trace resolution ──────────────────────────────────────────────────────
    // Axial frustum segments along the optic. Matches the reference library's
    // profile sampling (nmax) so reflection counts agree; also bounds the maximum
    // reflection count (≤ one reflection resolved per segment).
    VT_SCONSTEXPR int    PC_NSEG = 999;
    VT_SCONSTEXPR double PC_WMIN = 1.0e-4;  // photon survival weight threshold

    // ── Geometry (float storage) ──────────────────────────────────────────────
    float pos_z_    = 0.f;
    float length_   = 1.f;
    float r_ext_in_ = 0.f, r_ext_out_ = 0.f;  // outer optic radius in/out [cm]
    float r_cap_in_ = 0.f, r_cap_out_ = 0.f;  // single-capillary radius in/out [cm]
    float focal_in_ = 0.f, focal_out_ = 0.f;  // focal distances [cm]
    Profile profile_ = CONICAL;
    // PARABOLOIDAL: coeff_ = {a,b,c},  ext(t) = a + b·t + c·t²
    // ELLIPSOIDAL : coeff_ = {b,k,±a²}, ext(t) = √(b²−b²·t²/|a²|)+k
    //               negative a² flags the collimating branch (z reversed)
    double coeff_[3] = {};
    float  n_shells_ = 0.f;    // hexagonal shell count; 0 = monocapillary
    float  roughness_ = 0.f;   // surface roughness [Å]

    // ── Precomputed complex refractive index n = (1−δ) + iβ ───────────────────
    float delta_[PC_N_E] = {};
    float beta_ [PC_N_E] = {};

    // ═════════════════════════════════════════════════════════════════════════
    // Profile / capillary-axis geometry  (double; identical to the reference's
    // sampled profile)
    // ═════════════════════════════════════════════════════════════════════════

    // Inner capillary radius at z (always linear)
    KOKKOS_INLINE_FUNCTION double capRadius(double z) const VT_DEVICE_METH {
        return r_cap_in_ + (r_cap_out_ - r_cap_in_) * (z - pos_z_) / length_;
    }

    // Outer optic radius at z
    KOKKOS_INLINE_FUNCTION double extRadius(double z) const VT_DEVICE_METH {
        double t = z - pos_z_;
        if (profile_ == PARABOLOIDAL) {
            return coeff_[0] + coeff_[1]*t + coeff_[2]*t*t;
        } else if (profile_ == ELLIPSOIDAL) {
            double b = coeff_[0], k = coeff_[1], a_sq = coeff_[2];
            double z_loc = (a_sq < 0.0) ? (length_ - t) : t;   // collimating → reversed
            double val   = b*b * (1.0 - z_loc*z_loc / fabs(a_sq));
            return sqrt(val > 0.0 ? val : 0.0) + k;
        }
        return r_ext_in_ + (r_ext_out_ - r_ext_in_) * t / length_;   // CONICAL
    }

    // Selected capillary's axis (x,y) at z, for hex indices (q,r)
    KOKKOS_INLINE_FUNCTION double cellPitch(double z) const VT_DEVICE_METH {
        return extRadius(z) / (2.0 * PC_COSPI6 * (n_shells_ + 1.0));
    }
    KOKKOS_INLINE_FUNCTION double axisX(double q, double r, double z) const VT_DEVICE_METH {
        return (2.0*q + r) * PC_COSPI6 * cellPitch(z);
    }
    KOKKOS_INLINE_FUNCTION double axisY(double r, double z) const VT_DEVICE_METH {
        return r * 1.5 * cellPitch(z);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Optic-boundary helpers
    // ═════════════════════════════════════════════════════════════════════════

    // Point (x,y) inside hexagonal optic of circumradius ext?
    KOKKOS_INLINE_FUNCTION static bool withinHex(double ext, double x, double y) {
        double d = ext * PC_COSPI6;   // inradius
        return fabs(y)                    <= d
            && fabs(PC_COSPI6*x + 0.5*y) <= d
            && fabs(PC_COSPI6*x - 0.5*y) <= d;
    }

    // Point (x,y) inside the optic envelope at the given ext radius?
    KOKKOS_INLINE_FUNCTION bool withinOptic(double ext, double x, double y) const VT_DEVICE_METH {
        if (n_shells_ == 0.f) return x*x + y*y <= ext*ext;
        return withinHex(ext, x, y);
    }

    // Hex axial indices (q,r) for point (x,y) via cube-coordinate rounding
    KOKKOS_INLINE_FUNCTION void capillaryIndex(double x, double y, double& q, double& r) const VT_DEVICE_METH {
        double sp = cellPitch(pos_z_);
        double qf =  (x / (2.0*PC_COSPI6) - y / 3.0) / sp;
        double rf =   y * (2.0/3.0) / sp;
        double sf = -qf - rf;
        double rq = round(qf), rr = round(rf), rs = round(sf);
        double dq = fabs(rq-qf), dr = fabs(rr-rf), ds = fabs(rs-sf);
        if      (dq > dr && dq > ds) { q = -rr - rs; r =  rr;      }
        else if (dr > ds)            { q =  rq;      r = -rq - rs; }
        else                         { q =  rq;      r =  rr;      }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Ray–frustum intersection  (port of polycap_capil_segment)
    // ═════════════════════════════════════════════════════════════════════════
    //
    // Intersect the photon at (px,py,pz) travelling along unit dir (dd, ddz > 0)
    // with one conical frustum segment whose axis runs c0 → c1 and inner radii
    // r0 → r1. On success: writes hit point (h) and outward surface normal (n).
    KOKKOS_INLINE_FUNCTION static bool capilSegment(
        double c0x, double c0y, double c0z,
        double c1x, double c1y, double c1z,
        double r0,  double r1,
        double px,  double py,  double pz,
        double ddx, double ddy, double ddz,
        double& hx, double& hy, double& hz,
        double& nx, double& ny, double& nz)
    {
        double dz_cap = c1z - c0z;
        double p0x = px + ddx * (c0z - pz) / ddz;   // project onto plane z = c0z
        double p0y = py + ddy * (c0z - pz) / ddz;

        double ax0 = p0x - c0x, ay0 = p0y - c0y;
        double ex  = ddx/ddz - (c1x - c0x)/dz_cap;  // transverse drift per Δz
        double ey  = ddy/ddz - (c1y - c0y)/dz_cap;
        double eR  = (r1 - r0) / dz_cap;            // radius change per Δz

        // Quadratic a·s² + b·s + c = 0,  s = Δz from c0z
        double a = ex*ex + ey*ey - eR*eR;
        double b = 2.0*(ax0*ex + ay0*ey) - 2.0*r0*eR;
        double c = ax0*ax0 + ay0*ay0 - r0*r0;
        double discr = b*b - 4.0*a*c;
        if (discr < 0.0) return false;

        // Nearest forward intersection strictly inside the segment.
        double best = 1e30;
        if (fabs(a) < 1e-30) {
            if (fabs(b) > 1e-30) {
                double s  = -c / b;
                double zh = c0z + s;
                if (s/ddz > 1e-10 && zh > c0z && zh <= c1z && zh - pz > 1e-5) best = s;
            }
        } else {
            double sq = sqrt(discr);
            double s1 = (-b + sq) / (2.0*a);
            double s2 = (-b - sq) / (2.0*a);
            for (int k = 0; k < 2; ++k) {
                double s  = (k == 0) ? s1 : s2;
                double zh = c0z + s;
                if (s/ddz < 1e-10)         continue;
                if (zh <= c0z || zh > c1z) continue;
                if (zh - pz   < 1e-5)      continue;
                if (s < best) best = s;
            }
        }
        if (best > 1e29) return false;

        double dp = best / ddz;
        hz = c0z + best;
        hx = p0x + dp * ddx;
        hy = p0y + dp * ddy;

        // Outward surface normal: tilt the radial direction by the wall taper γ
        double rx = hx - c0x, ry = hy - c0y, rz = hz - c0z;
        double cdx = c1x - c0x, cdy = c1y - c0y;
        double d_seg_sq = cdx*cdx + cdy*cdy + dz_cap*dz_cap;
        double t_ax = (rx*cdx + ry*cdy + rz*dz_cap) / d_seg_sq;   // proj onto axis
        double radx = hx - (c0x + t_ax*cdx);
        double rady = hy - (c0y + t_ax*cdy);
        double radz = hz - (c0z + t_ax*dz_cap);
        double r_dist = sqrt(radx*radx + rady*rady + radz*radz);
        double d_seg  = sqrt(d_seg_sq);
        double tga = (r0 - r1) / d_seg;
        double ih  = 1.0 / sqrt(1.0 + tga*tga);
        double cga = ih, sga = tga * ih;            // cos γ, sin γ
        double chx = cdx/d_seg, chy = cdy/d_seg, chz = dz_cap/d_seg;

        nx = cga * radx/r_dist + sga * chx;
        ny = cga * rady/r_dist + sga * chy;
        nz = cga * radz/r_dist + sga * chz;
        double nl = sqrt(nx*nx + ny*ny + nz*nz);
        nx /= nl; ny /= nl; nz /= nl;
        return true;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Reflection physics
    // ═════════════════════════════════════════════════════════════════════════

    // Linearly interpolated complex refractive index on the PC_E_STEP grid.
    KOKKOS_INLINE_FUNCTION void refractiveIndex(double e, double& delta, double& beta) const VT_DEVICE_METH {
        int idx = (int)(e / PC_E_STEP);
        if (idx <= 0)        { delta = delta_[0];        beta = beta_[0];        return; }
        if (idx >= PC_N_E-1) { delta = delta_[PC_N_E-1]; beta = beta_[PC_N_E-1]; return; }
        double t = e / PC_E_STEP - (double)idx;
        delta = delta_[idx] + t * (delta_[idx+1] - delta_[idx]);
        beta  = beta_ [idx] + t * (beta_ [idx+1] - beta_ [idx]);
    }

    // Complex Fresnel reflectance |r_s|², |r_p|² (manual complex, MSL-safe).
    // cos_alfa = cos(angle between incident dir and surface normal) = sin θ_grazing.
    KOKKOS_INLINE_FUNCTION static void fresnelR(double cos_alfa, double delta, double beta,
                                                 double& Rs, double& Rp) {
        double n_re = 1.0 - delta, n_im = beta;
        double sin2 = 1.0 - cos_alfa*cos_alfa; if (sin2 < 0.0) sin2 = 0.0;

        double n2_re = n_re*n_re - n_im*n_im;
        double n2_im = 2.0*n_re*n_im;

        // cos²θ_t = 1 − sin²/n²
        double D      = n2_re*n2_re + n2_im*n2_im;
        double ct2_re = 1.0 - sin2*n2_re / D;
        double ct2_im =       sin2*n2_im / D;

        // cosθ_t = √(ct2)  (principal complex root)
        double r_mag = sqrt(ct2_re*ct2_re + ct2_im*ct2_im);
        double ct_re = sqrt(fmax(0.0, (r_mag + ct2_re) * 0.5));
        double ct_im = (ct2_im >= 0.0 ? 1.0 : -1.0) * sqrt(fmax(0.0, (r_mag - ct2_re) * 0.5));

        double nc_re = n_re*ct_re - n_im*ct_im;   // n · cosθ_t
        double nc_im = n_re*ct_im + n_im*ct_re;

        // r_s = (cos_alfa − n·cosθ_t)/(cos_alfa + n·cosθ_t)
        double rs_nr = cos_alfa - nc_re, rs_ni = -nc_im;
        double rs_dr = cos_alfa + nc_re, rs_di =  nc_im;
        Rs = (rs_nr*rs_nr + rs_ni*rs_ni) / (rs_dr*rs_dr + rs_di*rs_di);

        // r_p = (cosθ_t − n·cos_alfa)/(cosθ_t + n·cos_alfa)
        double rp_nr = ct_re - n_re*cos_alfa, rp_ni = ct_im - n_im*cos_alfa;
        double rp_dr = ct_re + n_re*cos_alfa, rp_di = ct_im + n_im*cos_alfa;
        Rp = (rp_nr*rp_nr + rp_ni*rp_ni) / (rp_dr*rp_dr + rp_di*rp_di);
    }

    // Debye-Waller surface-roughness attenuation.
    KOKKOS_INLINE_FUNCTION static double debyeWaller(double roughness, double sin_graz, double energy) {
        double cons = 1.01358 * energy * sin_graz * roughness;
        return exp(-cons * cons);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Transient photon trace state (double; lives only inside trace())
    // ═════════════════════════════════════════════════════════════════════════
    struct Photon {
        double x, y, z;        // position
        double dx, dy, dz;     // direction (unit)
        double ex, ey, ez;     // electric vector (unit)
        double weight;         // transmission weight
        double energy;         // [keV]
        int    n_refl;
    };

    // Apply one wall reflection to the photon: attenuate the weight by the
    // polarization-weighted Fresnel × roughness factor, mirror the direction,
    // and advance the (energy-independent) electric vector (polycap_refl_polar).
    // Returns false if the photon is absorbed (weight below threshold).
    KOKKOS_INLINE_FUNCTION bool reflect(Photon& ph,
                                         double hx, double hy, double hz,
                                         double nx, double ny, double nz) const VT_DEVICE_METH {
        ph.x = hx; ph.y = hy; ph.z = hz;
        double cos_alfa = nx*ph.dx + ny*ph.dy + nz*ph.dz;   // caller guarantees ≥ 0

        double delta_v, beta_v;
        refractiveIndex(ph.energy, delta_v, beta_v);
        double Rs, Rp;
        fresnelR(cos_alfa, delta_v, beta_v, Rs, Rp);
        double dw = debyeWaller(roughness_, cos_alfa, ph.energy);

        // s direction (⊥ plane of incidence) and p direction
        double sx = ny*ph.dz - nz*ph.dy, sy = nz*ph.dx - nx*ph.dz, sz = nx*ph.dy - ny*ph.dx;
        double sl = sqrt(sx*sx + sy*sy + sz*sz);
        if (sl < 1e-9) return false;                        // normal incidence
        sx /= sl; sy /= sl; sz /= sl;
        double pdx = ph.dy*sz - ph.dz*sy, pdy = ph.dz*sx - ph.dx*sz, pdz = ph.dx*sy - ph.dy*sx;

        double angle_a = ph.ex*sx + ph.ey*sy + ph.ez*sz;    // projection onto s
        double frac_s  = angle_a * angle_a;
        double frac_p  = 1.0 - frac_s;

        ph.weight *= (Rs*frac_s + Rp*frac_p) * dw;
        if (ph.weight < PC_WMIN) return false;

        // Specular reflection of the direction
        double d2x = ph.dx - 2.0*cos_alfa*nx;
        double d2y = ph.dy - 2.0*cos_alfa*ny;
        double d2z = ph.dz - 2.0*cos_alfa*nz;
        double d2l = sqrt(d2x*d2x + d2y*d2y + d2z*d2z);
        ph.dx = d2x/d2l; ph.dy = d2y/d2l; ph.dz = d2z/d2l;

        // Electric vector update (polycap_refl_polar)
        double angle_b = ph.ex*nx  + ph.ey*ny  + ph.ez*nz;
        double angle_c = ph.ex*pdx + ph.ey*pdy + ph.ez*pdz;
        double vx = sqrt(sq(ph.ex*angle_a*frac_s) + sq(ph.ex*angle_b*frac_p) + sq(ph.ex*angle_c*frac_p));
        double vy = sqrt(sq(ph.ey*angle_a*frac_s) + sq(ph.ey*angle_b*frac_p) + sq(ph.ey*angle_c*frac_p));
        double vz = sqrt(sq(ph.ez*angle_a*frac_s) + sq(ph.ez*angle_b*frac_p) + sq(ph.ez*angle_c*frac_p));
        double vl = sqrt(vx*vx + vy*vy + vz*vz);
        if (vl > 1e-12) { ph.ex = vx/vl; ph.ey = vy/vl; ph.ez = vz/vl; }

        ph.n_refl++;
        return true;
    }

    KOKKOS_INLINE_FUNCTION static double sq(double v) VT_DEVICE_METH { return v*v; }

    // ═════════════════════════════════════════════════════════════════════════
    // Host-only: quadratic least-squares fit y = c0 + c1·x + c2·x² (PARABOLOIDAL)
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
        for (int col = 0; col < 3; ++col) {
            int piv = col;
            for (int row = col+1; row < 3; ++row)
                if (std::abs(A[row][col]) > std::abs(A[piv][col])) piv = row;
            for (int s = 0; s < 3; ++s) { double t = A[col][s]; A[col][s]=A[piv][s]; A[piv][s]=t; }
            double t = rhs[col]; rhs[col]=rhs[piv]; rhs[piv]=t;
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
    KOKKOS_INLINE_FUNCTION PolyCap() = default;

    // ── Constructor (host-only) ───────────────────────────────────────────────
    // weightFracs may be percentages (≈100) or fractions (≈1); normalised here.
    // Fills delta_[]/beta_[] from xraylib once — no runtime dependency thereafter.
#ifndef __METAL_VERSION__
    PolyCap(float posZ, float length,
                    float rExtIn,  float rExtOut,
                    float rCapIn,  float rCapOut,
                    float focalIn, float focalOut,
                    Profile profile,
                    int nElements, const int* atomicNumbers, const float* weightFracs,
                    float density, float roughness, int nCapillaries)
        : pos_z_(posZ), length_(length),
          r_ext_in_(rExtIn), r_ext_out_(rExtOut),
          r_cap_in_(rCapIn), r_cap_out_(rCapOut),
          focal_in_(focalIn), focal_out_(focalOut),
          profile_(profile), roughness_(roughness)
    {
        n_shells_ = roundf(sqrtf(12.f * (float)nCapillaries - 3.f) / 6.f - 0.5f);

        int ne = (nElements < 8) ? nElements : 8;
        double w[8] = {}, wsum = 0.0;
        for (int j = 0; j < ne; ++j) { w[j] = weightFracs[j]; wsum += w[j]; }
        if (wsum > 1.5) for (int j = 0; j < ne; ++j) w[j] /= wsum;

        // ── Outer-profile coefficients ────────────────────────────────────────
        if (profile == PARABOLOIDAL) {
            double px[4], py[4];
            px[0] = 0.;      py[0] = rExtIn;
            px[3] = length;  py[3] = rExtOut;
            px[1] = (focalIn  <= length) ? focalIn  / 10.0 : length / 10.0;
            py[1] = (rExtIn  / focalIn)  * px[1] + rExtIn;
            px[2] = (focalOut <= length) ? length - focalOut / 10.0 : length * 0.9;
            py[2] = (-rExtOut / focalOut) * (px[2] - length) + rExtOut;
            fitQuadratic(px, py, 4, coeff_);
        } else if (profile == ELLIPSOIDAL) {
            if (rExtOut < rExtIn) {                 // focusing
                double slope = rExtOut / focalOut, dr = rExtOut - rExtIn;
                double b = (-dr*dr - slope*length*dr) / (slope*length + 2.0*dr);
                coeff_[0] = b;
                coeff_[1] = rExtIn - b;             // k
                coeff_[2] = (b*b * length) / (slope * (rExtOut - coeff_[1]));  // +a²
            } else {                                // collimating
                double slope = rExtIn / focalIn, dr = rExtIn - rExtOut;
                double b = (-dr*dr - slope*length*dr) / (slope*length + 2.0*dr);
                coeff_[0] = b;
                coeff_[1] = rExtOut - b;            // k
                coeff_[2] = -std::abs((b*b * length) / (slope * (rExtIn - coeff_[1])));  // −a²
            }
        }

        // ── Refractive-index tables on the 0.2 keV grid ───────────────────────
        double A_arr[8] = {};
        for (int j = 0; j < ne; ++j) A_arr[j] = XRayLibAPI::A(atomicNumbers[j]);

        for (int i = 1; i < PC_N_E; ++i) {
            double E = (double)i * PC_E_STEP;
            double scatf = 0.0, mu_mass = 0.0;
            for (int j = 0; j < ne; ++j) {
                double fi = XRayLibAPI::Fi(atomicNumbers[j], E);
                scatf   += w[j] * ((double)atomicNumbers[j] + fi) / A_arr[j];
                mu_mass += w[j] * XRayLibAPI::CS_Tot(atomicNumbers[j], E);
            }
            double mu_lin = mu_mass * density;      // [cm⁻¹]
            double lam    = PC_HC / E;
            delta_[i] = (float)(lam*lam * (PC_NA_R0 * density / PC_TWOPI) * scatf);
            beta_ [i] = (float)((PC_HC / (2.0 * PC_TWOPI)) * (mu_lin / E));
        }
        delta_[0] = delta_[1];
        beta_ [0] = beta_ [1];
    }
#endif  // !__METAL_VERSION__

    // ── Main trace ────────────────────────────────────────────────────────────
    // Advances ONE photon (the ray) through the optic in place. The Ray (float)
    // is read once on entry and written once on exit; the bounce loop runs in
    // double via the local Photon to preserve grazing-angle precision.
    KOKKOS_INLINE_FUNCTION void trace(Ray& ray) const VT_DEVICE_METH {
        // 1. Must be heading downstream toward the entrance plane.
        Photon ph;
        ph.dx = ray.getDirX(); ph.dy = ray.getDirY(); ph.dz = ray.getDirZ();
        double dnorm = sqrt(ph.dx*ph.dx + ph.dy*ph.dy + ph.dz*ph.dz);
        if (dnorm < 1e-12 || ph.dz <= 0.0) { ray.setIAFlag(false); return; }
        ph.dx /= dnorm; ph.dy /= dnorm; ph.dz /= dnorm;

        double t_en = (pos_z_ - ray.getStartZ()) / ph.dz;
        if (t_en < 0.0) { ray.setIAFlag(false); return; }
        double xe = ray.getStartX() + t_en * ph.dx;
        double ye = ray.getStartY() + t_en * ph.dy;

        // 2. Inside the optic aperture? Select the capillary (hex indices q,r).
        double ext0 = extRadius(pos_z_);
        double q = 0.0, r = 0.0;
        if (n_shells_ == 0.f) {
            if (xe*xe + ye*ye > ext0*ext0) { ray.setIAFlag(false); return; }
        } else {
            if (!withinHex(ext0, xe, ye)) { ray.setIAFlag(false); return; }
            capillaryIndex(xe, ye, q, r);
        }

        // 3. Entering the open channel (not the glass wall)?
        double cx0 = axisX(q, r, pos_z_), cy0 = axisY(r, pos_z_);
        double rc0 = capRadius(pos_z_);
        if ((xe-cx0)*(xe-cx0) + (ye-cy0)*(ye-cy0) > rc0*rc0) { ray.setIAFlag(false); return; }

        // 4. Seed the photon on the entrance plane; orthogonalise its E-vector.
        ph.x = xe; ph.y = ye; ph.z = pos_z_;
        ph.weight = ray.getProb();
        ph.energy = ray.getEnergyKeV();
        ph.n_refl = 0;
        {
            double ax = ray.getSPolX(), ay = ray.getSPolY(), az = ray.getSPolZ();
            double dot = ax*ph.dx + ay*ph.dy + az*ph.dz;
            ax -= dot*ph.dx; ay -= dot*ph.dy; az -= dot*ph.dz;
            double al = sqrt(ax*ax + ay*ay + az*az);
            if (al < 1e-9) {                  // E ∥ dir: pick any perpendicular
                ax = (fabs(ph.dx) < 0.9) ? 1.0 : 0.0; ay = (fabs(ph.dx) < 0.9) ? 0.0 : 1.0; az = 0.0;
                dot = ax*ph.dx + ay*ph.dy + az*ph.dz; ax -= dot*ph.dx; ay -= dot*ph.dy; az -= dot*ph.dz;
                al = sqrt(ax*ax + ay*ay + az*az);
            }
            ph.ex = ax/al; ph.ey = ay/al; ph.ez = az/al;
        }

        // 5. Reflection loop: find successive wall interactions and reflect.
        double dz_seg = length_ / (double)PC_NSEG;
        int    seg = 0;   // first segment to scan; never decreases
        for (int refl = 0; refl < PC_NSEG; ++refl) {
            bool found = false;
            for (int i = seg; i < PC_NSEG; ++i) {
                double z0 = pos_z_ + (double)i       * dz_seg;
                double z1 = pos_z_ + (double)(i + 1) * dz_seg;

                double hx, hy, hz, nx, ny, nz;
                bool hit = capilSegment(axisX(q, r, z0), axisY(r, z0), z0,
                                        axisX(q, r, z1), axisY(r, z1), z1,
                                        capRadius(z0), capRadius(z1),
                                        ph.x, ph.y, ph.z, ph.dx, ph.dy, ph.dz,
                                        hx, hy, hz, nx, ny, nz);

                // Reject a hit whose normal faces away from the photon — keep
                // scanning forward (matches the reference's acos > π/2 rejection).
                if (hit && (nx*ph.dx + ny*ph.dy + nz*ph.dz) < 0.0) hit = false;

                if (hit) {
                    // Hit must lie inside the optic envelope at its z.
                    double frac = (z1 > z0) ? (hz - z0) / (z1 - z0) : 0.0;
                    double ext  = extRadius(z0) + frac * (extRadius(z1) - extRadius(z0));
                    if (!withinOptic(ext, hx, hy)) { ray.setIAFlag(false); return; }  // escaped

                    if (!reflect(ph, hx, hy, hz, nx, ny, nz)) { ray.setIAFlag(false); return; }  // absorbed
                    seg = i + 1;
                    found = true;
                    break;
                }

                // No hit here: if the photon already left the optic laterally at
                // this z, it has escaped through the side.
                double dseg = (z0 - ph.z) / ph.dz;
                double sxp  = ph.x + dseg * ph.dx;
                double syp  = ph.y + dseg * ph.dy;
                if (!withinOptic(extRadius(z0), sxp, syp)) { ray.setIAFlag(false); return; }
            }
            if (!found) break;   // no further wall hit → photon exits the channel
        }

        // 6. Project to the exit plane and write the result back to the Ray.
        double exit_z = pos_z_ + length_;
        double dz_rem = exit_z - ph.z;
        if (dz_rem > 0.0 && ph.dz > 1e-12) {
            double t_ex = dz_rem / ph.dz;
            ph.x += t_ex * ph.dx; ph.y += t_ex * ph.dy; ph.z = exit_z;
        }
        ray.setStartCoordinates((float)ph.x, (float)ph.y, (float)ph.z);
        ray.setEndCoordinates((float)ph.dx, (float)ph.dy, (float)ph.dz);
        ray.setSPol((float)ph.ex, (float)ph.ey, (float)ph.ez);
        ray.setProb((float)ph.weight);
        ray.setIANum(ph.n_refl);
        ray.setIAFlag(true);
    }

#ifndef __METAL_VERSION__
    void print() const {
        const char* pn = (profile_ == PARABOLOIDAL) ? "PARABOLOIDAL"
                       : (profile_ == ELLIPSOIDAL)  ? "ELLIPSOIDAL" : "CONICAL";
        printf("PolyCap: %s L=%.3fcm rExt=(%.5f→%.5f)cm rCap=(%.7f→%.7f)cm "
               "f=(%.3f,%.3f)cm nShells=%d\n",
               pn, length_, r_ext_in_, r_ext_out_, r_cap_in_, r_cap_out_,
               focal_in_, focal_out_, (int)n_shells_);
    }
#endif
};
