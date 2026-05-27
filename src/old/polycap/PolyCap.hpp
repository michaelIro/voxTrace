#pragma once
// PolyCap.hpp — Header-only C++ polycapillary X-ray optic ray-tracer
// Translated from polycap-1.2 (https://github.com/PieterTack/polycap)
// Requires: xraylib (https://github.com/tschoonj/xraylib)
// Build: g++ -std=c++17 your_code.cpp -lxraylib -lm
//
// Primary tracing only (no wall-penetration leak calculation).
// Supports CONICAL, PARABOLOIDAL, and ELLIPSOIDAL optic profiles.

#include <cmath>
#include <complex>
#include <vector>
#include <random>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

#ifdef __cplusplus
extern "C" {
#endif
#include <xraylib.h>
#ifdef __cplusplus
}
#endif

// Undefine macros that may be set by polycap.h before entering our namespace
#ifdef HC
#  undef HC
#endif
#ifdef N_AVOG
#  undef N_AVOG
#endif
#ifdef R0
#  undef R0
#endif

namespace PC {

// ─────────────────────────── Physical constants (cgs / keV) ─────────────────
static constexpr double HC      = 1.23984193e-7;       // keV·cm
static constexpr double N_AVOG  = 6.022098e23;         // mol⁻¹
static constexpr double R0      = 2.8179403227e-13;    // cm (classical electron radius)
static constexpr double COSPI_6 = 0.86602540378443865; // cos(π/6)
static constexpr double WMIN    = 1.0e-4;              // weight absorption threshold

// ─────────────────────────────────── Vec3 ────────────────────────────────────
struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vec3  operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3  operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3  operator*(double s)      const { return {x*s, y*s, z*s}; }
    Vec3  operator/(double s)      const { return {x/s, y/s, z/s}; }
    Vec3& operator+=(const Vec3& o){ x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o){ x-=o.x; y-=o.y; z-=o.z; return *this; }

    double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3   cross(const Vec3& o) const {
        return { y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x };
    }
    double norm2()  const { return x*x + y*y + z*z; }
    double norm()   const { return std::sqrt(norm2()); }
    void   normalize()    { double n = norm(); x/=n; y/=n; z/=n; }
    Vec3   normalized()   const { double n = norm(); return {x/n, y/n, z/n}; }
};
inline Vec3 operator*(double s, const Vec3& v) { return v*s; }

// ───────────────────────────────── Profile ───────────────────────────────────
struct Profile {
    int nmax = 999;                    // number of segments (nmax+1 points)
    std::vector<double> z, cap, ext;   // z coords, capillary radius, exterior radius

    // Conical profile
    static Profile conical(double length,
                            double rad_ext_up,  double rad_ext_down,
                            double rad_int_up,  double rad_int_down) {
        Profile p;
        p.z.resize(p.nmax+1); p.cap.resize(p.nmax+1); p.ext.resize(p.nmax+1);
        for (int i = 0; i <= p.nmax; ++i) {
            p.z[i]   = length / p.nmax * i;
            p.cap[i] = (rad_int_down - rad_int_up) / length * p.z[i] + rad_int_up;
            p.ext[i] = (rad_ext_down - rad_ext_up) / length * p.z[i] + rad_ext_up;
        }
        return p;
    }

    // Ellipsoidal profile — exact formula from polycap_profile_new (polycap-profile.c)
    static Profile ellipsoidal(double length,
                                double rad_ext_up,  double rad_ext_down,
                                double rad_int_up,  double rad_int_down,
                                double focal_dist_up, double focal_dist_down) {
        Profile p;
        p.z.resize(p.nmax+1); p.cap.resize(p.nmax+1); p.ext.resize(p.nmax+1);

        if (rad_ext_down < rad_ext_up) {
            // Focusing branch
            double slope = rad_ext_down / focal_dist_down;
            double dr    = rad_ext_down - rad_ext_up;
            double b     = (-dr*dr - slope*length*dr) / (slope*length + 2.*dr);
            double k     = rad_ext_up - b;
            double a     = std::sqrt((b*b*length) / (slope*(rad_ext_down - k)));
            for (int i = 0; i <= p.nmax; ++i) {
                p.z[i]   = length / p.nmax * i;
                p.cap[i] = (rad_int_down - rad_int_up) / length * p.z[i] + rad_int_up;
                p.ext[i] = std::sqrt(b*b - b*b*p.z[i]*p.z[i]/(a*a)) + k;
            }
        } else {
            // Collimating / confocal branch
            double slope = rad_ext_up / focal_dist_up;
            double dr    = rad_ext_up - rad_ext_down;
            double b     = (-dr*dr - slope*length*dr) / (slope*length + 2.*dr);
            double k     = rad_ext_down - b;
            double a     = std::sqrt(std::fabs((b*b*length) / (slope*(rad_ext_up - k))));
            for (int i = 0; i <= p.nmax; ++i) {
                p.z[i]   = length / p.nmax * i;
                p.cap[i] = (rad_int_down - rad_int_up) / length * p.z[i] + rad_int_up;
            }
            for (int i = 0; i <= p.nmax; ++i)
                p.ext[i] = std::sqrt(b*b - b*b*p.z[p.nmax-i]*p.z[p.nmax-i]/(a*a)) + k;
        }
        return p;
    }

    // Paraboloidal profile — inline least-squares quadratic fit (no GSL)
    static Profile paraboloidal(double length,
                                 double rad_ext_up,  double rad_ext_down,
                                 double rad_int_up,  double rad_int_down,
                                 double focal_dist_up, double focal_dist_down) {
        Profile p;
        p.z.resize(p.nmax+1); p.cap.resize(p.nmax+1); p.ext.resize(p.nmax+1);
        // 4 control points (matching polycap_profile_new PARABOLOIDAL branch)
        double px[4], py[4];
        px[0] = 0.;        py[0] = rad_ext_up;
        px[3] = length;    py[3] = rad_ext_down;
        px[1] = (focal_dist_up <= length)   ? focal_dist_up/10.           : length/10.;
        py[1] = (rad_ext_up-0.) / (0.-(-focal_dist_up))    * (px[1]-0.)      + rad_ext_up;
        px[2] = (focal_dist_down <= length) ? length-focal_dist_down/10.  : length-length/10.;
        py[2] = (rad_ext_down-0.) / (length-(length+focal_dist_down)) * (px[2]-length) + rad_ext_down;

        double coeff[3];
        quadratic_lsq(4, px, py, coeff);
        for (int i = 0; i <= p.nmax; ++i) {
            p.z[i]   = length / p.nmax * i;
            p.cap[i] = (rad_int_down - rad_int_up) / length * p.z[i] + rad_int_up;
            p.ext[i] = coeff[0] + coeff[1]*p.z[i] + coeff[2]*p.z[i]*p.z[i];
        }
        return p;
    }

private:
    // Least-squares quadratic fit (Vandermonde normal equations, Gauss-Jordan)
    static void quadratic_lsq(int n, const double* x, const double* y, double coeff[3]) {
        double M[3][4] = {};
        for (int i = 0; i < n; ++i) {
            double p[3] = {1., x[i], x[i]*x[i]};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) M[r][c] += p[r]*p[c];
                M[r][3] += p[r]*y[i];
            }
        }
        // Gauss-Jordan elimination with partial pivoting
        for (int col = 0; col < 3; ++col) {
            int pivot = col;
            for (int r = col+1; r < 3; ++r)
                if (std::fabs(M[r][col]) > std::fabs(M[pivot][col])) pivot = r;
            for (int c = 0; c <= 3; ++c) std::swap(M[col][c], M[pivot][c]);
            double d = M[col][col];
            for (int c = col; c <= 3; ++c) M[col][c] /= d;
            for (int r = 0; r < 3; ++r) {
                if (r == col) continue;
                double f = M[r][col];
                for (int c = col; c <= 3; ++c) M[r][c] -= f*M[col][c];
            }
        }
        for (int r = 0; r < 3; ++r) coeff[r] = M[r][3];
    }
};

// ─────────────────────────────── Description ─────────────────────────────────
struct Description {
    double  sig_rough = 0.;  // surface roughness [Å]
    double  density   = 0.;  // material density [g/cm³]
    int64_t n_cap     = 0;   // total number of capillaries
    double  open_area = 0.;  // entrance open-area fraction
    std::vector<int>    iz;  // atomic numbers
    std::vector<double> wi;  // weight fractions (must sum to 1)
    Profile profile;

    Description() = default;

    Description(Profile prof, double sig_rough_, int64_t n_cap_,
                std::vector<int> iz_, std::vector<double> wi_, double density_)
        : profile(std::move(prof))
        , sig_rough(sig_rough_), density(density_), n_cap(n_cap_)
        , iz(std::move(iz_)), wi(std::move(wi_))
    {
        // Open area (same formula as polycap_source_new_from_file)
        double ns  = (std::round(std::sqrt(12.*n_cap - 3.)/6. - 0.5) + 0.5) * 6.;
        double nct = (ns*ns + 3.) / 12.;
        open_area  = (profile.cap[0]*profile.cap[0]*M_PI*nct) /
                     (3.*std::sin(M_PI/3.) * profile.ext[0]*profile.ext[0]);
    }
};

// ──────────────────────────── Source parameters ───────────────────────────────
struct SourceParams {
    double d_source   = 500.;   // source-to-optic distance [cm]
    double src_x      = 0.01;   // source half-width x [cm]
    double src_y      = 0.01;   // source half-width y [cm]
    double src_sigx   = -1.;    // angular spread x; <0 = uniform over PC entrance
    double src_sigy   = -1.;    // angular spread y; <0 = uniform over PC entrance
    double src_shiftx = 0.;     // source centre offset x [cm]
    double src_shifty = 0.;     // source centre offset y [cm]
    double hor_pol    = 0.9;    // horizontal polarisation fraction [-1, 1]
    std::vector<double> energies;  // photon energies [keV]
};

// ────────────────────────── Simulation output types ──────────────────────────
struct TransmittedPhoton {
    double x_exit, y_exit, z_exit;      // propagated to exit plane [cm]
    double dx, dy, dz;                   // exit direction (unit vector)
    std::vector<double> weights;         // per-energy weights
    int64_t n_refl  = 0;
    double  d_travel = 0.;
};

struct SimResult {
    std::vector<TransmittedPhoton> photons;
    std::vector<double> efficiencies;   // per energy = sum_weight / n_transmitted
    std::vector<double> energies;       // copy of source energies [keV]
    int64_t n_transmitted = 0;
    int64_t n_launched    = 0;          // total photons launched (all attempts)
};

// ═══════════════════════════ Internal implementation ═════════════════════════
namespace detail {

// Internal photon state (mirrors polycap_photon minus leak arrays)
struct PhotonState {
    Vec3 start_coords, start_dir, start_elecv;
    Vec3 src_start_coords;
    Vec3 exit_coords,  exit_dir,  exit_elecv;
    std::vector<double> energies, weight, amu, scatf;
    int64_t i_refl  = 0;
    double  d_travel = 0.;
};

// ── X-ray material properties via xraylib ────────────────────────────────────
inline void compute_scatf(PhotonState& ph, const Description& desc) {
    int ne = (int)ph.energies.size();
    ph.amu.resize(ne);
    ph.scatf.resize(ne);
    for (int i = 0; i < ne; ++i) {
        double totmu = 0., sf = 0.;
        for (int j = 0; j < (int)desc.iz.size(); ++j) {
            totmu += CS_Total(desc.iz[j], ph.energies[i], nullptr) * desc.wi[j];
            sf    += (desc.iz[j] + Fi(desc.iz[j], ph.energies[i], nullptr)) *
                     (desc.wi[j] / AtomicWeight(desc.iz[j], nullptr));
        }
        ph.amu[i]   = totmu * desc.density;
        ph.scatf[i] = sf;
    }
}

// ── Hexagonal boundary check (matches polycap_photon_within_pc_boundary) ─────
inline bool within_hex_boundary(double radius, const Vec3& c) {
    double d = std::sqrt(radius*radius - (radius/2.)*(radius/2.));  // inradius
    if (std::fabs(c.y)                         > d) return false;
    if (std::fabs(COSPI_6*c.x + 0.5*c.y)      > d) return false;
    if (std::fabs(COSPI_6*c.x - 0.5*c.y)      > d) return false;
    return true;
}

// ── Ray–frustum intersection (polycap_capil_segment, active version) ─────────
// cap0/cap1: capillary axis endpoints; r0/r1: radii at cap0.z and cap1.z
// phot0/phot1: photon ray at segment z-boundaries (for quadratic parametrisation)
// dir: normalised photon direction
// photon_coord: current photon position — updated to hit point on success
// surface_norm: set on success
// Returns: 1 = hit found, negative = no hit in this segment
inline int capil_segment(Vec3 cap0, Vec3 cap1, double r0, double r1,
                          Vec3 phot0, Vec3 /*phot1*/, Vec3 dir,
                          Vec3& photon_coord, Vec3& surface_norm) {
    surface_norm = {0,0,0};

    Vec3   cap_dir      = cap1 - cap0;
    double d_cap_coord  = cap_dir.norm();
    double dz_cap       = cap1.z - cap0.z;  // > 0 always (z increases along optic)

    // Relative slope terms (photon minus cap axis, per unit z)
    double dx_rel = dir.x/dir.z - cap_dir.x/dz_cap;
    double dy_rel = dir.y/dir.z - cap_dir.y/dz_cap;
    double r_dr   = (r1 - r0) / dz_cap;

    // Quadratic a·dist² + b·dist + c = 0   (dist = Δz from phot0.z)
    double a = dx_rel*dx_rel + dy_rel*dy_rel - r_dr*r_dr;
    double b = 2.*(phot0.x-cap0.x)*dx_rel + 2.*(phot0.y-cap0.y)*dy_rel - 2.*r0*r_dr;
    double c = (phot0.x-cap0.x)*(phot0.x-cap0.x) +
               (phot0.y-cap0.y)*(phot0.y-cap0.y) - r0*r0;

    double discr = b*b - 4.*a*c;
    if (discr < 0.) return -2;

    // Helper: check whether a Δz solution is valid (forward, within segment)
    auto valid = [&](double dist) {
        double iz = phot0.z + dist;
        return iz >= cap0.z && iz - photon_coord.z >= 1.e-5 && iz <= cap1.z;
    };

    double interact_z;
    if (discr == 0.) {
        double dist1 = -b / (2.*a);
        if (!valid(dist1)) return -3;
        interact_z = phot0.z + dist1;
    } else {
        double sq    = std::sqrt(discr);
        double dist1 = (-b + sq) / (2.*a);
        double dist2 = (-b - sq) / (2.*a);
        bool v1 = valid(dist1), v2 = valid(dist2);
        if (!v1 && !v2) return -3;
        double dist;
        if (v1 && v2) {
            // Pick the solution closest (but forward of) current photon_coord.z
            double iz1 = phot0.z + dist1, iz2 = phot0.z + dist2;
            dist = ((iz2-photon_coord.z) < (iz1-photon_coord.z)) ? dist2 : dist1;
        } else {
            dist = v1 ? dist1 : dist2;
        }
        interact_z = phot0.z + dist;
    }

    if (interact_z > cap1.z) return -4;
    if (interact_z < cap0.z || interact_z - photon_coord.z < 1.e-5) return -5;

    // Interaction x, y from ray
    double d_proj = (interact_z - phot0.z) / dir.z;
    if (d_proj < 1.e-10) return -6;
    Vec3 interact = { phot0.x + d_proj*dir.x,
                      phot0.y + d_proj*dir.y,
                      interact_z };

    // ── Surface normal at interaction ────────────────────────────────────────
    // Find point on capillary axis at same parametric position as interact:
    //   s = dot(interact - cap0, cap_dir) / dot(cap_dir, cap_dir)
    Vec3   phot_rel = phot0 - cap0;
    double cap_dir2 = cap_dir.dot(cap_dir);        // |cap_dir|²
    double dir_cap  = dir.dot(cap_dir);             // dot(photon_dir, cap_dir)
    double s        = (phot_rel.dot(cap_dir) + d_proj*dir_cap) / cap_dir2;
    Vec3   cap_at_z = cap0 + cap_dir * s;

    Vec3   interact_norm = interact - cap_at_z;
    double d_cap_inter   = interact_norm.norm();

    double tga = (r0 - r1) / d_cap_coord;          // wall taper (< 0 for focusing)
    double gam = std::atan(tga);
    double sga = std::sin(gam), cga = std::cos(gam);

    surface_norm.x = cga * interact_norm.x/d_cap_inter + sga * cap_dir.x/d_cap_coord;
    surface_norm.y = cga * interact_norm.y/d_cap_inter + sga * cap_dir.y/d_cap_coord;
    surface_norm.z = cga * interact_norm.z/d_cap_inter + sga * cap_dir.z/d_cap_coord;
    surface_norm.normalize();

    photon_coord = interact;
    return 1;
}

// ── Fresnel reflection: apply to all energies, update weights and elecv ──────
// cosalfa = dot(surface_norm, normalised exit_dir)  — must be >= 0
// Returns false if all weights fell below WMIN (photon absorbed)
inline bool apply_reflection(PhotonState& ph, const Description& desc,
                               const Vec3& surface_norm, double cosalfa) {
    Vec3 dir  = ph.exit_dir;  // already normalised
    Vec3 elec = ph.exit_elecv;

    // s and p directions (energy-independent)
    Vec3 s_dir = surface_norm.cross(dir); s_dir.normalize();
    Vec3 p_dir = dir.cross(s_dir);        p_dir.normalize();

    double angle_a = elec.dot(s_dir);
    double frac_s  = angle_a * angle_a;
    double frac_p  = 1. - frac_s;

    double sin2t = 1. - cosalfa*cosalfa;  // sin²θ  (θ = angle between normal and dir)

    bool any_alive = false;
    for (int e = 0; e < (int)ph.energies.size(); ++e) {
        using cplx = std::complex<double>;
        double E    = ph.energies[e];
        double alpha_f = (HC/E)*(HC/E) * (N_AVOG*R0*desc.density/(2.*M_PI)) * ph.scatf[e];
        double beta_f  = HC/(4.*M_PI) * ph.amu[e]/E;
        cplx n(1.0 - alpha_f, beta_f);

        cplx n_inv = 1.0 / n;
        // transmitted angle: sin²θ_t = sin²θ / n²
        cplx tmp     = n_inv * n_inv * cplx(sin2t, 0.);
        cplx csqrt_t = std::sqrt(cplx(1. - tmp.real(), -tmp.imag()));

        cplx r_s = (cplx(cosalfa) - n*csqrt_t) / (cplx(cosalfa) + n*csqrt_t);
        double R_s = std::norm(r_s);   // |r_s|²

        cplx r_p = (csqrt_t - n*cplx(cosalfa)) / (csqrt_t + n*cplx(cosalfa));
        double R_p = std::norm(r_p);

        double rtot = R_s*frac_s + R_p*frac_p;

        // Roughness correction (Debye-Waller)
        double cons1 = 1.01358 * E * cosalfa * desc.sig_rough;
        double r_rough = std::exp(-cons1*cons1);

        ph.weight[e] *= rtot * r_rough;
        if (ph.weight[e] >= WMIN) any_alive = true;
    }

    // Update electric vector (energy-independent, taken from polycap_refl_polar)
    double angle_b = elec.dot(surface_norm);
    double angle_c = elec.dot(p_dir);
    Vec3 new_elecv;
    new_elecv.x = std::sqrt(
        (elec.x*angle_a*frac_s)*(elec.x*angle_a*frac_s) +
        (elec.x*angle_b*frac_p)*(elec.x*angle_b*frac_p) +
        (elec.x*angle_c*frac_p)*(elec.x*angle_c*frac_p));
    new_elecv.y = std::sqrt(
        (elec.y*angle_a*frac_s)*(elec.y*angle_a*frac_s) +
        (elec.y*angle_b*frac_p)*(elec.y*angle_b*frac_p) +
        (elec.y*angle_c*frac_p)*(elec.y*angle_c*frac_p));
    new_elecv.z = std::sqrt(
        (elec.z*angle_a*frac_s)*(elec.z*angle_a*frac_s) +
        (elec.z*angle_b*frac_p)*(elec.z*angle_b*frac_p) +
        (elec.z*angle_c*frac_p)*(elec.z*angle_c*frac_p));
    new_elecv.normalize();
    ph.exit_elecv = new_elecv;

    return any_alive;
}

// ── Trace one reflection (polycap_capil_trace) ───────────────────────────────
// Returns: 1 = another reflection (still inside), 0 = absorbed, -2 = reached exit,
//          -3 = escaped optic laterally
inline int capil_trace_one(int& ix, PhotonState& ph,
                            const Description& desc,
                            const std::vector<double>& cap_x,
                            const std::vector<double>& cap_y) {
    const Profile& prof = desc.profile;

    if (ph.exit_dir.z <= 0.) return -3;   // photon going sideways or backward
    ph.exit_dir.normalize();

    double n_shells = std::round(std::sqrt(12.*desc.n_cap - 3.)/6. - 0.5);
    Vec3   photon_coord = ph.exit_coords;  // updated by capil_segment
    Vec3   surface_norm;
    double cosalfa = 0.;
    int    iesc    = 0;

    for (int i = ix; i < prof.nmax; ++i) {
        Vec3   cap0 = {cap_x[i],   cap_y[i],   prof.z[i]};
        Vec3   cap1 = {cap_x[i+1], cap_y[i+1], prof.z[i+1]};
        double cr0  = prof.cap[i], cr1 = prof.cap[i+1];

        // Photon ray projected to segment z-bounds
        double inv_dz  = 1. / ph.exit_dir.z;
        double t0      = (prof.z[i]   - ph.exit_coords.z) * inv_dz;
        double t1      = (prof.z[i+1] - ph.exit_coords.z) * inv_dz;
        Vec3 phot0 = ph.exit_coords + ph.exit_dir*t0;  phot0.z = prof.z[i];
        Vec3 phot1 = ph.exit_coords + ph.exit_dir*t1;  phot1.z = prof.z[i+1];

        iesc = capil_segment(cap0, cap1, cr0, cr1, phot0, phot1,
                              ph.exit_dir, photon_coord, surface_norm);

        if (iesc == 1) {
            cosalfa = surface_norm.dot(ph.exit_dir);
            // Reject if normal points away from photon direction
            if (cosalfa < 0. || std::acos(std::min(1., std::max(-1., cosalfa))) > M_PI/2.)
                iesc = -5;
        }

        if (iesc == 1) {
            // Verify interaction is still inside the optic
            int iz_here = i;
            double dz_seg = prof.z[i+1] - prof.z[i];
            double frac   = (dz_seg > 0.) ? (photon_coord.z - prof.z[i]) / dz_seg : 0.;
            double ext_here = prof.ext[i] + frac*(prof.ext[i+1]-prof.ext[i]);
            bool in_optic;
            if (n_shells == 0.) {
                in_optic = (photon_coord.x*photon_coord.x + photon_coord.y*photon_coord.y)
                           < ext_here*ext_here;
            } else {
                in_optic = within_hex_boundary(ext_here, photon_coord);
            }
            if (!in_optic) return -3;
            ix = i + 1;
            break;
        } else {
            // Check photon has not already escaped the optic laterally at z[i]
            Vec3 temp = ph.exit_coords + ph.exit_dir*t0;
            temp.z = prof.z[i];
            bool in_optic;
            if (n_shells == 0.) {
                in_optic = (temp.x*temp.x + temp.y*temp.y) <= prof.ext[i]*prof.ext[i];
            } else {
                in_optic = within_hex_boundary(prof.ext[i], temp);
            }
            if (!in_optic) return -3;
        }
    }

    if (iesc != 1) return -2;  // no more intersections → photon reached exit

    // ── Interaction found: update travel distance and exit coords ────────────
    ph.d_travel += (photon_coord - ph.exit_coords).norm();
    ph.exit_coords = photon_coord;

    // ── Fresnel reflection: update weights and electric vector ───────────────
    if (!apply_reflection(ph, desc, surface_norm, cosalfa))
        return 0;  // absorbed

    // ── Mirror reflection: update direction ──────────────────────────────────
    ph.exit_dir -= surface_norm * (2. * cosalfa);
    ph.exit_dir.normalize();
    ph.i_refl++;

    return 1;
}

// ── Launch a photon through the capillary (polycap_photon_launch) ─────────────
// Returns: 1 = reached exit, 0 = absorbed, 2 = started in glass wall (retry),
//         -1 = error/escaped, -2 = outside optic boundary at entrance
inline int launch_photon(PhotonState& ph, const Description& desc) {
    const Profile& prof = desc.profile;

    ph.i_refl  = 0;
    ph.d_travel = 0.;
    ph.exit_coords = ph.start_coords;
    ph.exit_dir    = ph.start_dir;
    ph.exit_elecv  = ph.start_elecv;
    ph.exit_dir.normalize();

    // Fill weight array to 1 for each energy
    int ne = (int)ph.energies.size();
    ph.weight.assign(ne, 1.);

    // Compute X-ray material properties
    compute_scatf(ph, desc);

    double n_shells = std::round(std::sqrt(12.*desc.n_cap - 3.)/6. - 0.5);

    // Find z-segment index for current photon start position
    int z_id = 0;
    for (int i = 0; i < prof.nmax; ++i)
        if (prof.z[i] <= ph.start_coords.z) z_id = i;

    // External PC radius at start z
    double ext0;
    if (prof.z[z_id] != prof.z[z_id+1]) {
        double frac = (ph.start_coords.z - prof.z[z_id]) /
                      (prof.z[z_id+1] - prof.z[z_id]);
        ext0 = prof.ext[z_id] + frac*(prof.ext[z_id+1]-prof.ext[z_id]);
    } else {
        ext0 = prof.ext[z_id];
    }

    // ── Select capillary hex indices (q, r) from start coordinates ───────────
    double q_i, r_i;
    if (n_shells == 0.) {
        q_i = r_i = 0.;
        // Monocapillary: check within circular boundary
        if (ph.start_coords.x*ph.start_coords.x +
            ph.start_coords.y*ph.start_coords.y > ext0*ext0) return -2;
    } else {
        // Polycapillary: hexagonal index from cube-coordinate rounding
        if (!within_hex_boundary(ext0, ph.start_coords)) return -2;

        double z_cell = ext0 / (2.*COSPI_6*(n_shells+1.));
        r_i = ph.start_coords.y * (2./3.) / z_cell;
        q_i = (ph.start_coords.x/(2.*COSPI_6) - ph.start_coords.y/3.) / z_cell;

        // Cube-coordinate rounding (same as polycap_photon_launch)
        double s_i = -q_i - r_i;
        double dq = std::fabs(q_i - std::round(q_i));
        double dr = std::fabs(r_i - std::round(r_i));
        double ds = std::fabs(s_i - std::round(s_i));
        if (dq > dr && dq > ds) {
            q_i = -std::round(r_i) - std::round(s_i);
            r_i = std::round(r_i);
        } else if (dr > ds) {
            r_i = -std::round(q_i) - std::round(s_i);
            q_i = std::round(q_i);
        } else {
            q_i = std::round(q_i);
            r_i = std::round(r_i);
        }
    }

    // ── Build capillary axis coordinate arrays ────────────────────────────────
    int npts = prof.nmax + 1;
    std::vector<double> cap_x(npts), cap_y(npts);
    int ix_val = 0;
    for (int i = 0; i < npts; ++i) {
        double z_cell = prof.ext[i] / (2.*COSPI_6*(n_shells+1.));
        cap_y[i] = r_i * 1.5 * z_cell;
        cap_x[i] = (2.*q_i + r_i) * COSPI_6 * z_cell;
        if (prof.z[i] <= ph.start_coords.z) ix_val = i;
    }

    // ── Check photon is inside capillary opening (not in glass wall) ──────────
    double frac_z = (prof.z[z_id+1] > prof.z[z_id])
                  ? (ph.start_coords.z - prof.z[z_id]) / (prof.z[z_id+1]-prof.z[z_id])
                  : 0.;
    double cur_cap_r = prof.cap[z_id] + frac_z*(prof.cap[z_id+1]-prof.cap[z_id]);
    double cur_cap_x = cap_x[z_id]   + frac_z*(cap_x[z_id+1]-cap_x[z_id]);
    double cur_cap_y = cap_y[z_id]   + frac_z*(cap_y[z_id+1]-cap_y[z_id]);
    double d_ph_cap  = std::sqrt((ph.start_coords.x-cur_cap_x)*(ph.start_coords.x-cur_cap_x) +
                                  (ph.start_coords.y-cur_cap_y)*(ph.start_coords.y-cur_cap_y));
    if (d_ph_cap > cur_cap_r) return 2;  // in glass wall — retry with new photon

    // ── Trace loop: find successive wall interactions ─────────────────────────
    int ix = ix_val;
    for (int i = 0; i <= prof.nmax; ++i) {
        int iesc = capil_trace_one(ix, ph, desc, cap_x, cap_y);
        if (iesc == 0)  return 0;   // absorbed
        if (iesc != 1)  break;      // -2: reached exit, -3: escaped, -1: error
    }
    return 1;  // reached exit
}

// ── Generate a source photon (polycap_source_get_photon) ─────────────────────
template<class RNG>
inline PhotonState generate_photon(const SourceParams& sp,
                                    const Description& desc, RNG& rng) {
    std::uniform_real_distribution<double> uni(0., 1.);

    // ── Source position (elliptical beam profile) ─────────────────────────────
    double r   = uni(rng);
    double phi = std::atan(sp.src_y/sp.src_x * std::tan(2.*M_PI*r/4.));
    r = uni(rng);
    if (r >= 0.25 && r < 0.50) phi = M_PI - phi;
    if (r >= 0.50 && r < 0.75) phi = M_PI + phi;
    if (r >= 0.75)              phi = -phi;

    double max_rad = sp.src_x*sp.src_y /
                     std::sqrt((sp.src_y*std::cos(phi))*(sp.src_y*std::cos(phi)) +
                                (sp.src_x*std::sin(phi))*(sp.src_x*std::sin(phi)));
    r = uni(rng);
    Vec3 src_pos = { std::sqrt(r)*max_rad*std::cos(phi) + sp.src_shiftx,
                     std::sqrt(r)*max_rad*std::sin(phi) + sp.src_shifty,
                     0. };

    // ── Entrance coordinates and direction ────────────────────────────────────
    Vec3 start_coords, start_dir;
    if (sp.src_sigx < 0. || sp.src_sigy < 0.) {
        // Uniform sampling over PC entrance window
        double ext0 = desc.profile.ext[0];
        double n_shells = std::round(std::sqrt(12.*desc.n_cap - 3.)/6. - 0.5);
        if (n_shells == 0.) {
            // Monocapillary: uniform in square, accept circle
            do {
                r = uni(rng); start_coords.x = (2.*r-1.) * desc.profile.cap[0];
                r = uni(rng); start_coords.y = (2.*r-1.) * desc.profile.cap[0];
            } while (start_coords.x*start_coords.x + start_coords.y*start_coords.y
                     > desc.profile.cap[0]*desc.profile.cap[0]);
        } else {
            // Polycapillary: accept–reject within hexagonal boundary
            do {
                r = uni(rng); start_coords.x = (2.*r-1.) * ext0;
                r = uni(rng); start_coords.y = (2.*r-1.) * ext0;
            } while (!within_hex_boundary(ext0, start_coords));
        }
        start_coords.z = 0.;
        start_dir = { start_coords.x - src_pos.x,
                      start_coords.y - src_pos.y,
                      sp.d_source };
    } else {
        // Non-uniform: direction sampled within ±sigx/sigy cone
        r = uni(rng); start_dir.x = sp.src_sigx * (1. - 2.*std::fabs(r));
        r = uni(rng); start_dir.y = sp.src_sigy * (1. - 2.*std::fabs(r));
        start_dir.z = 1.;
        start_coords.x = src_pos.x + start_dir.x * sp.d_source / start_dir.z;
        start_coords.y = src_pos.y + start_dir.y * sp.d_source / start_dir.z;
        start_coords.z = 0.;
    }
    start_dir.normalize();

    // ── Electric vector (Gram–Schmidt orthogonalisation to direction) ─────────
    double frac_hor = (1. + sp.hor_pol) / 2.;
    r = uni(rng);
    Vec3 elecv = (std::fabs(r) <= frac_hor)
               ? Vec3{1., 0., 0.}   // horizontal
               : Vec3{0., 1., 0.};  // vertical

    // Remove component along direction: e_perp = (e - dot(e,d)*d) / |e - dot(e,d)*d|
    double cosalpha = elecv.dot(start_dir);
    double alpha    = std::acos(cosalpha);
    double c_ae     = 1. / std::sin(alpha);
    double c_be     = -c_ae * cosalpha;
    elecv = elecv*c_ae + start_dir*c_be;
    elecv.normalize();

    PhotonState ph;
    ph.start_coords   = start_coords;
    ph.exit_coords    = start_coords;
    ph.start_dir      = start_dir;
    ph.exit_dir       = start_dir;
    ph.start_elecv    = elecv;
    ph.exit_elecv     = elecv;
    ph.src_start_coords = src_pos;
    ph.energies       = sp.energies;
    return ph;
}

}  // namespace detail


// ═══════════════════════════════ Public API ═══════════════════════════════════

/// Simulate n_photons transmitted photons through the polycapillary.
/// Loops until exactly n_photons reach the exit window.
/// \param seed  RNG seed; 0 = use std::random_device
inline SimResult simulate(const SourceParams& sp, const Description& desc,
                           int n_photons, uint64_t seed = 0) {
    if (seed == 0) {
        std::random_device rd;
        seed = ((uint64_t)rd() << 32) | rd();
    }
    std::mt19937_64 rng(seed);

    const Profile& prof = desc.profile;
    double z_exit = prof.z[prof.nmax];
    double n_shells = std::round(std::sqrt(12.*desc.n_cap - 3.)/6. - 0.5);

    SimResult result;
    result.energies = sp.energies;
    int ne = (int)sp.energies.size();
    result.efficiencies.assign(ne, 0.);

    std::vector<double> sum_weights(ne, 0.);

    while (result.n_transmitted < n_photons) {
        detail::PhotonState ph = detail::generate_photon(sp, desc, rng);
        int iesc = detail::launch_photon(ph, desc);
        result.n_launched++;

        // Retry photons that are absorbed or hit the wall at entrance
        if (iesc == 0 || iesc == 2 || iesc == -2 || iesc == -1) continue;

        // iesc == 1: photon reached end; check if within exit window
        Vec3 exit_pt;
        if (ph.exit_dir.z > 0.) {
            double dt = (z_exit - ph.exit_coords.z) / ph.exit_dir.z;
            exit_pt = ph.exit_coords + ph.exit_dir * dt;
            exit_pt.z = z_exit;
        } else {
            continue;  // degenerate: skip
        }

        bool in_exit;
        if (n_shells == 0.) {
            double r2 = exit_pt.x*exit_pt.x + exit_pt.y*exit_pt.y;
            double re = prof.ext[prof.nmax];
            in_exit = (r2 <= re*re);
        } else {
            in_exit = detail::within_hex_boundary(prof.ext[prof.nmax], exit_pt);
        }
        if (!in_exit) continue;

        // Record transmitted photon
        TransmittedPhoton tp;
        tp.x_exit  = exit_pt.x;
        tp.y_exit  = exit_pt.y;
        tp.z_exit  = exit_pt.z;
        tp.dx      = ph.exit_dir.x;
        tp.dy      = ph.exit_dir.y;
        tp.dz      = ph.exit_dir.z;
        tp.weights = ph.weight;
        tp.n_refl  = ph.i_refl;
        // total d_travel includes last free-flight to exit plane
        double dx_exit = exit_pt.x - ph.exit_coords.x;
        double dy_exit = exit_pt.y - ph.exit_coords.y;
        double dz_exit = exit_pt.z - ph.exit_coords.z;
        tp.d_travel = ph.d_travel +
                      std::sqrt(dx_exit*dx_exit + dy_exit*dy_exit + dz_exit*dz_exit);
        result.photons.push_back(tp);

        for (int e = 0; e < ne; ++e) sum_weights[e] += ph.weight[e];
        result.n_transmitted++;
    }

    // Efficiency = total transmitted weight / total launched photons (like polycap library)
    for (int e = 0; e < ne; ++e)
        result.efficiencies[e] = sum_weights[e] / (double)result.n_launched;

    return result;
}

}  // namespace PC
