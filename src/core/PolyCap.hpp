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

#include "Platform.hpp"
#include "Material.hpp"
#include "ChemElement.hpp"
#include "Ray.hpp"
#include "../api/XRayLibAPI.hpp"

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

namespace polycap_detail {

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
    std::vector<ChemElement> elements;
    Profile profile;

    Description() = default;

    Description(Profile prof, double sig_rough_, int64_t n_cap_,
                std::vector<int> iz_, std::vector<double> wi_, double density_)
        : profile(std::move(prof))
        , sig_rough(sig_rough_), density(density_), n_cap(n_cap_)
        , iz(std::move(iz_))
        , wi(normalize_weights(std::move(wi_)))
        , elements(build_elements(iz))
    {
        if (iz.size() != wi.size()) {
            throw std::invalid_argument("PolyCap description needs one weight per element");
        }

        // Open area (same formula as polycap_source_new_from_file)
        double ns  = (std::round(std::sqrt(12.*n_cap - 3.)/6. - 0.5) + 0.5) * 6.;
        double nct = (ns*ns + 3.) / 12.;
        open_area  = (profile.cap[0]*profile.cap[0]*M_PI*nct) /
                     (3.*std::sin(M_PI/3.) * profile.ext[0]*profile.ext[0]);
    }

private:
    static std::vector<double> normalize_weights(std::vector<double> weights) {
        double sum = 0.;
        for (double weight : weights) sum += weight;
        if (sum <= 0.) {
            throw std::invalid_argument("PolyCap weights must sum to a positive value");
        }
        for (double& weight : weights) weight /= sum;
        return weights;
    }

    static std::vector<ChemElement> build_elements(const std::vector<int>& atomic_numbers) {
        std::vector<ChemElement> elements_;
        elements_.reserve(atomic_numbers.size());
        for (int atomic_number : atomic_numbers) {
            elements_.emplace_back(atomic_number);
        }
        return elements_;
    }
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

inline Vec3 repo_to_pc(const Vec3& v) {
    return {v.x, v.z, v.y};
}

inline Vec3 pc_to_repo(const Vec3& v) {
    return {v.x, v.z, v.y};
}

inline Vec3 orthogonal_basis(const Vec3& dir) {
    Vec3 ref = (std::fabs(dir.x) < 0.9) ? Vec3{1., 0., 0.} : Vec3{0., 1., 0.};
    Vec3 perp = ref - dir * dir.dot(ref);
    if (perp.norm2() < 1.e-12) {
        perp = Vec3{0., 0., 1.} - dir * dir.z;
    }
    perp.normalize();
    return perp;
}

inline Vec3 orthogonalize(const Vec3& dir, const Vec3& candidate) {
    Vec3 perp = candidate - dir * dir.dot(candidate);
    if (perp.norm2() < 1.e-12) {
        return orthogonal_basis(dir);
    }
    perp.normalize();
    return perp;
}

inline PhotonState photon_from_ray(const Ray& ray) {
    PhotonState ph;
    ph.start_coords = repo_to_pc({ray.getStartX(), ray.getStartY(), ray.getStartZ()});

    Vec3 start_dir = repo_to_pc({ray.getDirX(), ray.getDirY(), ray.getDirZ()});
    start_dir.normalize();

    // The core tracer launches from the optic entrance plane (z = 0 in PC space).
    if (ph.start_coords.z < 0. && start_dir.z > 0.) {
        double dt = -ph.start_coords.z / start_dir.z;
        ph.start_coords += start_dir * dt;
        ph.start_coords.z = 0.;
    }

    ph.exit_coords = ph.start_coords;
    ph.start_dir = start_dir;
    ph.exit_dir = start_dir;

    Vec3 start_elecv = repo_to_pc({ray.getSPolX(), ray.getSPolY(), ray.getSPolZ()});
    start_elecv = orthogonalize(start_dir, start_elecv);
    ph.start_elecv = start_elecv;
    ph.exit_elecv = start_elecv;

    ph.src_start_coords = ph.start_coords;
    ph.energies = {ray.getEnergyKeV()};
    ph.weight = {ray.getProb()};
    return ph;
}

inline Ray ray_from_photon(const Ray& input, const PhotonState& ph,
                           const Vec3& exit_pt, double weight) {
    Ray output = input;

    Vec3 repo_exit = pc_to_repo(exit_pt);
    Vec3 repo_dir = pc_to_repo(ph.exit_dir);
    Vec3 repo_s = orthogonalize(repo_dir, pc_to_repo(ph.exit_elecv));
    Vec3 repo_p = repo_dir.cross(repo_s);
    if (repo_p.norm2() < 1.e-12) {
        repo_p = orthogonal_basis(repo_dir);
    } else {
        repo_p.normalize();
    }

    output.setStartCoordinates((float)repo_exit.x, (float)repo_exit.y, (float)repo_exit.z);
    output.setEndCoordinates((float)repo_dir.x, (float)repo_dir.y, (float)repo_dir.z);
    output.setSPol((float)repo_s.x, (float)repo_s.y, (float)repo_s.z);
    output.setPPol((float)repo_p.x, (float)repo_p.y, (float)repo_p.z);
    output.setProb((float)weight);
    output.setIAFlag(true);
    output.setIANum((int)ph.i_refl);
    return output;
}

// ── X-ray material properties via xraylib ────────────────────────────────────
inline void compute_scatf(PhotonState& ph, const Description& desc) {
    int ne = (int)ph.energies.size();
    ph.amu.resize(ne);
    ph.scatf.resize(ne);
    for (int i = 0; i < ne; ++i) {
        double totmu = 0., sf = 0.;
        for (int j = 0; j < (int)desc.elements.size(); ++j) {
            const ChemElement& element = desc.elements[j];
            int atomic_number = element.Z();
            totmu += XRayLibAPI::CS_Tot(atomic_number, ph.energies[i]) * desc.wi[j];
            sf    += (atomic_number + XRayLibAPI::Fi(atomic_number, ph.energies[i])) *
                     (desc.wi[j] / element.A());
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

inline bool project_to_exit(const PhotonState& ph, const Description& desc, Vec3& exit_pt) {
    const Profile& prof = desc.profile;
    double z_exit = prof.z[prof.nmax];
    double n_shells = std::round(std::sqrt(12.*desc.n_cap - 3.)/6. - 0.5);

    if (ph.exit_dir.z <= 0.) {
        return false;
    }

    double dt = (z_exit - ph.exit_coords.z) / ph.exit_dir.z;
    exit_pt = ph.exit_coords + ph.exit_dir * dt;
    exit_pt.z = z_exit;

    if (n_shells == 0.) {
        double r2 = exit_pt.x*exit_pt.x + exit_pt.y*exit_pt.y;
        double re = prof.ext[prof.nmax];
        return r2 <= re*re;
    }
    return within_hex_boundary(prof.ext[prof.nmax], exit_pt);
}

}  // namespace detail
}  // namespace polycap_detail

// ── PolyCapProfile ───────────────────────────────────────────────────────────
// New facade around the translated internal profile so callers can move to the
// package-style object model without exposing the translated runtime API.

class PolyCapProfile {
    polycap_detail::Profile native_profile_;

public:
    PolyCapProfile() = default;
    explicit PolyCapProfile(const polycap_detail::Profile& nativeProfile)
        : native_profile_(nativeProfile) {}

    static PolyCapProfile conical(double lengthCm,
                                  double extUpstreamCm,
                                  double extDownstreamCm,
                                  double capUpstreamCm,
                                  double capDownstreamCm) {
        return PolyCapProfile(polycap_detail::Profile::conical(lengthCm,
                                                   extUpstreamCm,
                                                   extDownstreamCm,
                                                   capUpstreamCm,
                                                   capDownstreamCm));
    }

    static PolyCapProfile ellipsoidal(double lengthCm,
                                      double extUpstreamCm,
                                      double extDownstreamCm,
                                      double capUpstreamCm,
                                      double capDownstreamCm,
                                      double focalDistanceInCm,
                                      double focalDistanceOutCm) {
        return PolyCapProfile(polycap_detail::Profile::ellipsoidal(lengthCm,
                                                       extUpstreamCm,
                                                       extDownstreamCm,
                                                       capUpstreamCm,
                                                       capDownstreamCm,
                                                       focalDistanceInCm,
                                                       focalDistanceOutCm));
    }

    static PolyCapProfile paraboloidal(double lengthCm,
                                       double extUpstreamCm,
                                       double extDownstreamCm,
                                       double capUpstreamCm,
                                       double capDownstreamCm,
                                       double focalDistanceInCm,
                                       double focalDistanceOutCm) {
        return PolyCapProfile(polycap_detail::Profile::paraboloidal(lengthCm,
                                                        extUpstreamCm,
                                                        extDownstreamCm,
                                                        capUpstreamCm,
                                                        capDownstreamCm,
                                                        focalDistanceInCm,
                                                        focalDistanceOutCm));
    }

    KOKKOS_INLINE_FUNCTION const polycap_detail::Profile& native() const {
        return native_profile_;
    }

    KOKKOS_INLINE_FUNCTION double lengthCm() const {
        return native_profile_.z.empty() ? 0. : native_profile_.z.back();
    }

    KOKKOS_INLINE_FUNCTION double entranceExtRadiusCm() const {
        return native_profile_.ext.empty() ? 0. : native_profile_.ext.front();
    }

    KOKKOS_INLINE_FUNCTION double exitExtRadiusCm() const {
        return native_profile_.ext.empty() ? 0. : native_profile_.ext.back();
    }

    KOKKOS_INLINE_FUNCTION double entranceCapRadiusCm() const {
        return native_profile_.cap.empty() ? 0. : native_profile_.cap.front();
    }

    KOKKOS_INLINE_FUNCTION double exitCapRadiusCm() const {
        return native_profile_.cap.empty() ? 0. : native_profile_.cap.back();
    }
};

// ── PolyCapWall ──────────────────────────────────────────────────────────────
// Fixed-capacity validation matches the rest of core, while the translated
// still owns the actual tracing data for now.

class PolyCapWall {
    std::vector<int> atomic_numbers_;
    std::vector<double> weight_fractions_;
    double density_g_per_cm3_ = 0.;
    double roughness_angstrom_ = 0.;

public:
    PolyCapWall() = default;

    PolyCapWall(std::vector<int> atomicNumbers,
                std::vector<double> weightFractions,
                double densityGPerCm3,
                double roughnessAngstrom)
        : atomic_numbers_(std::move(atomicNumbers))
        , weight_fractions_(normalizeWeights(std::move(weightFractions)))
        , density_g_per_cm3_(densityGPerCm3)
        , roughness_angstrom_(roughnessAngstrom) {
        validate();
    }

    const std::vector<int>& atomicNumbers() const {
        return atomic_numbers_;
    }

    const std::vector<double>& weightFractions() const {
        return weight_fractions_;
    }

    KOKKOS_INLINE_FUNCTION double densityGPerCm3() const {
        return density_g_per_cm3_;
    }

    KOKKOS_INLINE_FUNCTION double roughnessAngstrom() const {
        return roughness_angstrom_;
    }

private:
    void validate() const {
        if (atomic_numbers_.size() != weight_fractions_.size()) {
            throw std::invalid_argument("PolyCap wall needs one weight per element");
        }
        if (atomic_numbers_.size() > static_cast<std::size_t>(MAX_ELEMENTS)) {
            throw std::invalid_argument("PolyCap wall exceeds MAX_ELEMENTS");
        }
    }

    static std::vector<double> normalizeWeights(std::vector<double> weightFractions) {
        double sum = 0.;
        for (double weight : weightFractions) {
            sum += weight;
        }
        if (sum <= 0.) {
            throw std::invalid_argument("PolyCap wall weights must sum to a positive value");
        }
        for (double& weight : weightFractions) {
            weight /= sum;
        }
        return weightFractions;
    }
};

// ── PolyCapSource ────────────────────────────────────────────────────────────
// Source geometry separated from the energy list so the public API can move to
// per-energy orchestration while the translated runtime stays internal.

class PolyCapSource {
    double source_distance_cm_ = 500.;
    double source_half_width_x_cm_ = 0.01;
    double source_half_width_y_cm_ = 0.01;
    double angular_spread_x_ = -1.;
    double angular_spread_y_ = -1.;
    double source_shift_x_cm_ = 0.;
    double source_shift_y_cm_ = 0.;
    double horizontal_polarization_ = 0.9;

public:
    KOKKOS_INLINE_FUNCTION PolyCapSource() = default;

    KOKKOS_INLINE_FUNCTION void setSourceDistanceCm(double sourceDistanceCm) {
        source_distance_cm_ = sourceDistanceCm;
    }

    KOKKOS_INLINE_FUNCTION void setSourceHalfSizeCm(double halfWidthXCm, double halfWidthYCm) {
        source_half_width_x_cm_ = halfWidthXCm;
        source_half_width_y_cm_ = halfWidthYCm;
    }

    KOKKOS_INLINE_FUNCTION void setAngularSpread(double spreadX, double spreadY) {
        angular_spread_x_ = spreadX;
        angular_spread_y_ = spreadY;
    }

    KOKKOS_INLINE_FUNCTION void setSourceShiftCm(double shiftXCm, double shiftYCm) {
        source_shift_x_cm_ = shiftXCm;
        source_shift_y_cm_ = shiftYCm;
    }

    KOKKOS_INLINE_FUNCTION void setHorizontalPolarization(double horizontalPolarization) {
        horizontal_polarization_ = horizontalPolarization;
    }

    KOKKOS_INLINE_FUNCTION double sourceDistanceCm() const {
        return source_distance_cm_;
    }

    KOKKOS_INLINE_FUNCTION double sourceHalfWidthXCm() const {
        return source_half_width_x_cm_;
    }

    KOKKOS_INLINE_FUNCTION double sourceHalfWidthYCm() const {
        return source_half_width_y_cm_;
    }

    KOKKOS_INLINE_FUNCTION double angularSpreadX() const {
        return angular_spread_x_;
    }

    KOKKOS_INLINE_FUNCTION double angularSpreadY() const {
        return angular_spread_y_;
    }

    KOKKOS_INLINE_FUNCTION double sourceShiftXCm() const {
        return source_shift_x_cm_;
    }

    KOKKOS_INLINE_FUNCTION double sourceShiftYCm() const {
        return source_shift_y_cm_;
    }

    KOKKOS_INLINE_FUNCTION double horizontalPolarization() const {
        return horizontal_polarization_;
    }
};

// ── Facade results ───────────────────────────────────────────────────────────

struct PolyCapTraceResult {
    Ray ray;
    std::vector<double> weights;
    int64_t reflections = 0;
    double travel_distance_cm = 0.;
    bool transmitted = false;

    double primaryWeight() const {
        return weights.empty() ? ray.getProb() : weights.front();
    }
};

struct PolyCapExitPhoton {
    double x_exit_cm = 0.;
    double y_exit_cm = 0.;
    double z_exit_cm = 0.;
    double dx = 0.;
    double dy = 0.;
    double dz = 0.;
    std::vector<double> weights;
    int64_t reflections = 0;
    double travel_distance_cm = 0.;
};

struct PolyCapEnergySummary {
    double energy_keV = 0.;
    double efficiency = 0.;
};

struct PolyCapBatchTraceResult {
    std::vector<PolyCapTraceResult> rays;
    int64_t transmitted_count = 0;
};

struct PolyCapSimulationResult {
    std::vector<PolyCapExitPhoton> photons;
    std::vector<PolyCapEnergySummary> energies;
    int64_t launched_count = 0;
    int64_t transmitted_count = 0;
};

// ── PolyCap ──────────────────────────────────────────────────────────────────
// Facade that matches the rest of core more closely while owning the translated
// tracing runtime as an internal implementation detail.

class PolyCap {
    PolyCapProfile profile_;
    PolyCapWall wall_;
    int64_t capillary_count_ = 0;
    polycap_detail::Description description_;
    double open_area_ = 0.;

public:
    PolyCap() = default;

    PolyCap(PolyCapProfile profile, PolyCapWall wall, int64_t capillaryCount)
        : profile_(std::move(profile))
        , wall_(std::move(wall))
        , capillary_count_(capillaryCount)
        , description_(profile_.native(),
                       wall_.roughnessAngstrom(),
                       capillary_count_,
                       wall_.atomicNumbers(),
                       wall_.weightFractions(),
                       wall_.densityGPerCm3())
        , open_area_(description_.open_area) {}

    const PolyCapProfile& profile() const {
        return profile_;
    }

    const PolyCapWall& wall() const {
        return wall_;
    }

    KOKKOS_INLINE_FUNCTION int64_t capillaryCount() const {
        return capillary_count_;
    }

    double openArea() const {
        return open_area_;
    }

    PolyCapTraceResult trace(const Ray& ray) const {
        PolyCapTraceResult result;
        result.ray = ray;
        result.ray.setIAFlag(false);
        result.ray.setProb(0.f);

        polycap_detail::detail::PhotonState photon = polycap_detail::detail::photon_from_ray(ray);
        int status = polycap_detail::detail::launch_photon(photon, description_);
        if (status != 1) {
            return result;
        }

        polycap_detail::Vec3 exit_point;
        if (!polycap_detail::detail::project_to_exit(photon, description_, exit_point)) {
            return result;
        }

        result.weights = photon.weight;
        double input_weight = ray.getProb();
        for (double& weight : result.weights) {
            weight *= input_weight;
        }

        result.reflections = photon.i_refl;
        result.travel_distance_cm = photon.d_travel + (exit_point - photon.exit_coords).norm();
        result.transmitted = true;
        result.ray = polycap_detail::detail::ray_from_photon(
            ray,
            photon,
            exit_point,
            result.weights.empty() ? 0. : result.weights.front());
        return result;
    }

    PolyCapBatchTraceResult traceBatch(const std::vector<Ray>& rays) const {
        PolyCapBatchTraceResult batch;
        batch.rays.resize(rays.size());

        if (rays.empty()) {
            return batch;
        }

#if defined(VOXTRACE_HOST_ONLY) || defined(VOXTRACE_METAL) || defined(__METAL_VERSION__)
        for (std::size_t i = 0; i < rays.size(); ++i) {
            batch.rays[i] = trace(rays[i]);
            batch.transmitted_count += batch.rays[i].transmitted ? 1 : 0;
        }
#else
        const int ray_count = static_cast<int>(rays.size());
        int64_t transmitted_count = 0;
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace> policy(0, ray_count);

        Kokkos::parallel_reduce(
            "polycap_trace_batch",
            policy,
            [this, &rays, &batch](int index, int64_t& local_transmitted) {
                PolyCapTraceResult traced = trace(rays[index]);
                if (traced.transmitted) {
                    ++local_transmitted;
                }
                batch.rays[index] = std::move(traced);
            },
            transmitted_count);

        Kokkos::fence();
        batch.transmitted_count = transmitted_count;
#endif

        return batch;
    }

    PolyCapSimulationResult simulate(const PolyCapSource& source,
                                     const std::vector<double>& energiesKeV,
                                     int transmittedPhotons,
                                     uint64_t seed = 0) const {
        if (energiesKeV.empty() || transmittedPhotons <= 0) {
            return PolyCapSimulationResult();
        }

        if (energiesKeV.size() == 1) {
            return simulateSingleEnergyBatch(source, energiesKeV.front(), transmittedPhotons, seed);
        }

        return simulateMultiEnergy(source, energiesKeV, transmittedPhotons, seed);
    }

private:
    static uint64_t resolveSeed(uint64_t seed) {
        if (seed != 0) {
            return seed;
        }

        std::random_device rd;
        return (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd());
    }

    template<class RNG>
    polycap_detail::detail::PhotonState generateLaunchPhoton(const PolyCapSource& source,
                                                             const std::vector<double>& energiesKeV,
                                                             RNG& rng) const {
        std::uniform_real_distribution<double> unit(0., 1.);

        double r = unit(rng);
        double phi = std::atan(source.sourceHalfWidthYCm() / source.sourceHalfWidthXCm() *
                               std::tan(2. * M_PI * r / 4.));
        r = unit(rng);
        if (r >= 0.25 && r < 0.50) phi = M_PI - phi;
        if (r >= 0.50 && r < 0.75) phi = M_PI + phi;
        if (r >= 0.75) phi = -phi;

        double max_radius = source.sourceHalfWidthXCm() * source.sourceHalfWidthYCm() /
                            std::sqrt(
                                std::pow(source.sourceHalfWidthYCm() * std::cos(phi), 2) +
                                std::pow(source.sourceHalfWidthXCm() * std::sin(phi), 2));
        r = unit(rng);
        polycap_detail::Vec3 source_position = {
            std::sqrt(r) * max_radius * std::cos(phi) + source.sourceShiftXCm(),
            std::sqrt(r) * max_radius * std::sin(phi) + source.sourceShiftYCm(),
            0.
        };

        polycap_detail::Vec3 start_coords;
        polycap_detail::Vec3 start_dir;
        if (source.angularSpreadX() < 0. || source.angularSpreadY() < 0.) {
            double entrance_radius = description_.profile.ext[0];
            double n_shells = std::round(std::sqrt(12. * description_.n_cap - 3.) / 6. - 0.5);
            if (n_shells == 0.) {
                do {
                    r = unit(rng);
                    start_coords.x = (2. * r - 1.) * description_.profile.cap[0];
                    r = unit(rng);
                    start_coords.y = (2. * r - 1.) * description_.profile.cap[0];
                } while (start_coords.x * start_coords.x + start_coords.y * start_coords.y >
                         description_.profile.cap[0] * description_.profile.cap[0]);
            } else {
                do {
                    r = unit(rng);
                    start_coords.x = (2. * r - 1.) * entrance_radius;
                    r = unit(rng);
                    start_coords.y = (2. * r - 1.) * entrance_radius;
                } while (!polycap_detail::detail::within_hex_boundary(entrance_radius, start_coords));
            }

            start_coords.z = 0.;
            start_dir = {
                start_coords.x - source_position.x,
                start_coords.y - source_position.y,
                source.sourceDistanceCm()
            };
        } else {
            r = unit(rng);
            start_dir.x = source.angularSpreadX() * (1. - 2. * std::fabs(r));
            r = unit(rng);
            start_dir.y = source.angularSpreadY() * (1. - 2. * std::fabs(r));
            start_dir.z = 1.;
            start_coords.x = source_position.x + start_dir.x * source.sourceDistanceCm() / start_dir.z;
            start_coords.y = source_position.y + start_dir.y * source.sourceDistanceCm() / start_dir.z;
            start_coords.z = 0.;
        }
        start_dir.normalize();

        double horizontal_fraction = (1. + source.horizontalPolarization()) / 2.;
        polycap_detail::Vec3 electric_vector = (unit(rng) <= horizontal_fraction)
                                            ? polycap_detail::Vec3{1., 0., 0.}
                                            : polycap_detail::Vec3{0., 1., 0.};
        electric_vector = polycap_detail::detail::orthogonalize(start_dir, electric_vector);

        polycap_detail::detail::PhotonState photon;
        photon.start_coords = start_coords;
        photon.exit_coords = start_coords;
        photon.start_dir = start_dir;
        photon.exit_dir = start_dir;
        photon.start_elecv = electric_vector;
        photon.exit_elecv = electric_vector;
        photon.src_start_coords = source_position;
        photon.energies = energiesKeV;
        return photon;
    }

    Ray buildLaunchRay(const polycap_detail::detail::PhotonState& photon, int rayIndex) const {
        polycap_detail::Vec3 repo_start = polycap_detail::detail::pc_to_repo(photon.start_coords);
        polycap_detail::Vec3 repo_dir = polycap_detail::detail::pc_to_repo(photon.start_dir);
        repo_dir.normalize();
        polycap_detail::Vec3 repo_s = polycap_detail::detail::orthogonalize(
            repo_dir,
            polycap_detail::detail::pc_to_repo(photon.start_elecv));
        polycap_detail::Vec3 repo_p = repo_dir.cross(repo_s);
        if (repo_p.norm2() < 1.e-12) {
            repo_p = polycap_detail::detail::orthogonal_basis(repo_dir);
        } else {
            repo_p.normalize();
        }

        Ray ray;
        ray.setStartCoordinates(static_cast<float>(repo_start.x),
                                static_cast<float>(repo_start.y),
                                static_cast<float>(repo_start.z));
        ray.setEndCoordinates(static_cast<float>(repo_dir.x),
                              static_cast<float>(repo_dir.y),
                              static_cast<float>(repo_dir.z));
        ray.setSPol(static_cast<float>(repo_s.x),
                    static_cast<float>(repo_s.y),
                    static_cast<float>(repo_s.z));
        ray.setPPol(static_cast<float>(repo_p.x),
                    static_cast<float>(repo_p.y),
                    static_cast<float>(repo_p.z));
        ray.setEnergyKeV(photon.energies.empty() ? 0.f : static_cast<float>(photon.energies.front()));
        ray.setProb(1.f);
        ray.setIAFlag(false);
        ray.setIANum(rayIndex);
        ray.setOOBFlag(false);
        ray.setTIn(0.f);
        return ray;
    }

    static PolyCapExitPhoton buildExitPhoton(const PolyCapTraceResult& traced) {
        PolyCapExitPhoton photon;
        photon.x_exit_cm = traced.ray.getStartX();
        photon.y_exit_cm = traced.ray.getStartY();
        photon.z_exit_cm = traced.ray.getStartZ();
        photon.dx = traced.ray.getDirX();
        photon.dy = traced.ray.getDirY();
        photon.dz = traced.ray.getDirZ();
        photon.weights = traced.weights;
        photon.reflections = traced.reflections;
        photon.travel_distance_cm = traced.travel_distance_cm;
        return photon;
    }

    static PolyCapExitPhoton buildExitPhoton(const polycap_detail::detail::PhotonState& photonState,
                                             const polycap_detail::Vec3& exitPoint) {
        PolyCapExitPhoton photon;
        polycap_detail::Vec3 repo_exit = polycap_detail::detail::pc_to_repo(exitPoint);
        polycap_detail::Vec3 repo_dir = polycap_detail::detail::pc_to_repo(photonState.exit_dir);
        repo_dir.normalize();
        photon.x_exit_cm = repo_exit.x;
        photon.y_exit_cm = repo_exit.y;
        photon.z_exit_cm = repo_exit.z;
        photon.dx = repo_dir.x;
        photon.dy = repo_dir.y;
        photon.dz = repo_dir.z;
        photon.weights = photonState.weight;
        photon.reflections = photonState.i_refl;
        photon.travel_distance_cm = photonState.d_travel + (exitPoint - photonState.exit_coords).norm();
        return photon;
    }

    PolyCapSimulationResult simulateSingleEnergyBatch(const PolyCapSource& source,
                                                      double energyKeV,
                                                      int transmittedPhotons,
                                                      uint64_t seed) const {
        PolyCapSimulationResult result;
        PolyCapEnergySummary summary;
        summary.energy_keV = energyKeV;

        std::mt19937_64 rng(resolveSeed(seed));
        double sum_weights = 0.;
        int launched_index = 0;

        while (result.transmitted_count < transmittedPhotons) {
            int batch_size = transmittedPhotons - static_cast<int>(result.transmitted_count);
            std::vector<Ray> launch_rays;
            launch_rays.reserve(static_cast<std::size_t>(batch_size));

            for (int i = 0; i < batch_size; ++i) {
                polycap_detail::detail::PhotonState launch_photon =
                    generateLaunchPhoton(source, {energyKeV}, rng);
                launch_rays.push_back(buildLaunchRay(launch_photon, launched_index + i));
            }

            launched_index += batch_size;
            result.launched_count += batch_size;

            PolyCapBatchTraceResult traced_batch = traceBatch(launch_rays);
            for (const PolyCapTraceResult& traced : traced_batch.rays) {
                if (!traced.transmitted) {
                    continue;
                }

                result.photons.push_back(buildExitPhoton(traced));
                result.transmitted_count += 1;
                sum_weights += traced.primaryWeight();
            }
        }

        summary.efficiency = (result.launched_count > 0)
                           ? sum_weights / static_cast<double>(result.launched_count)
                           : 0.;
        result.energies.push_back(summary);
        return result;
    }

    PolyCapSimulationResult simulateMultiEnergy(const PolyCapSource& source,
                                                const std::vector<double>& energiesKeV,
                                                int transmittedPhotons,
                                                uint64_t seed) const {
        PolyCapSimulationResult result;
        std::mt19937_64 rng(resolveSeed(seed));
        std::vector<double> sum_weights(energiesKeV.size(), 0.);

        result.energies.resize(energiesKeV.size());
        for (std::size_t i = 0; i < energiesKeV.size(); ++i) {
            result.energies[i].energy_keV = energiesKeV[i];
        }

        while (result.transmitted_count < transmittedPhotons) {
            polycap_detail::detail::PhotonState photon =
                generateLaunchPhoton(source, energiesKeV, rng);
            int status = polycap_detail::detail::launch_photon(photon, description_);
            result.launched_count += 1;

            if (status == 0 || status == 2 || status == -2 || status == -1) {
                continue;
            }

            polycap_detail::Vec3 exit_point;
            if (!polycap_detail::detail::project_to_exit(photon, description_, exit_point)) {
                continue;
            }

            result.photons.push_back(buildExitPhoton(photon, exit_point));
            result.transmitted_count += 1;
            for (std::size_t i = 0; i < energiesKeV.size(); ++i) {
                sum_weights[i] += photon.weight[i];
            }
        }

        for (std::size_t i = 0; i < energiesKeV.size(); ++i) {
            result.energies[i].efficiency = (result.launched_count > 0)
                                          ? sum_weights[i] / static_cast<double>(result.launched_count)
                                          : 0.;
        }

        return result;
    }
};
