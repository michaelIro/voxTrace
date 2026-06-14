// PolyCap.hpp ─────────────────────────────────────────────────────────────────
// Self-contained C++17 single-header re-implementation of the polycap ray tracer.
//
// Traces single photon rays through a polycapillary X-ray optic with the same
// physics as the original polycap library (v1.2):
//   • Conical / paraboloidal / ellipsoidal outer profile shapes
//   • Hexagonally-packed capillary grid (cube-coordinate indexing)
//   • Fresnel reflectivity with full s/p polarisation decomposition
//   • Debye-Waller surface-roughness correction
//   • Per-energy weight tracking through successive reflections
//
// Removed (none of these affect single-ray physics):
//   • polycap_error reporting system
//   • Photon leak / halo calculation
//   • Monte-Carlo source / RNG
//   • Threading (OpenMP)
//   • Transmission-efficiency accumulator
//   • Progress monitor
//   • Profile / source loading from ASCII files
//
// External dependency: xraylib  (CS_Total, Fi, AtomicWeight)
//
// Compile example:
//   g++ -std=c++17 -O2 my_program.cpp -lxraylib -lm
//
// Quick-start:
// ─────────────────────────────────────────────────────────────────────────────
//   polycap::Profile prof(polycap::ProfileType::ELLIPSOIDAL,
//                         6.0,              // optic length [cm]
//                         0.2883, 0.07,     // outer radii upstream / downstream [cm]
//                         0.00035, 8.5e-5,  // inner radii upstream / downstream [cm]
//                         500.0, 0.25);     // focal distances [cm]
//
//   polycap::Description desc(prof,
//                             5.0,              // surface roughness [Angstrom]
//                             227701,           // number of capillaries
//                             {8, 14},          // atomic numbers  (O, Si)
//                             {53.0, 47.0},     // weight percents → auto-normalised
//                             2.23);            // glass density [g/cm^3]
//
//   polycap::PolyCap optic(desc);
//
//   std::vector<double> energies = {5.0, 10.0, 20.0};
//   auto result = optic.traceRay({0.0, 0.0, 0.0},   // start position [cm]
//                                {0.0, 0.0, 1.0},   // direction (unnormalised OK)
//                                {1.0, 0.0, 0.0},   // electric-field polarisation
//                                energies);
//
//   if (result.status == polycap::TraceResult::TRANSMITTED) {
//       for (int i = 0; i < (int)energies.size(); ++i)
//           printf("E=%.1f keV  weight=%.4f\n", energies[i], result.weights[i]);
//   }
// ─────────────────────────────────────────────────────────────────────────────

#pragma once

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <xraylib.h>
// xraylib.h defines PI and TWOPI as C macros; undefine them so we can use proper
// constexpr constants without naming conflicts.
#undef PI
#undef TWOPI

namespace polycap {

// ─── Physical constants ───────────────────────────────────────────────────────
constexpr double HC      = 1.23984193e-7;          // h·c  [keV·cm]
constexpr double N_AVOG  = 6.022098e+23;            // Avogadro constant
constexpr double R0_ELEC = 2.8179403227e-13;        // classical electron radius [cm]
constexpr double COSPI6  = 0.86602540378443864676;  // cos(π/6)
constexpr double TWOPI   = 6.28318530717958647692;  // 2π

// ─── Vec3 ─────────────────────────────────────────────────────────────────────
struct Vec3 {
    double x = 0, y = 0, z = 0;

    Vec3 operator+(const Vec3& o) const noexcept { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const noexcept { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s)      const noexcept { return {x*s,   y*s,   z*s  }; }

    double dot(const Vec3& o) const noexcept { return x*o.x + y*o.y + z*o.z; }
    double norm2()            const noexcept { return x*x + y*y + z*z; }
    double norm()             const noexcept { return std::sqrt(norm2()); }

    Vec3 cross(const Vec3& o) const noexcept {
        return {y*o.z - z*o.y,  z*o.x - x*o.z,  x*o.y - y*o.x};
    }

    void normalize() noexcept { double n = norm(); x/=n; y/=n; z/=n; }
};

// ─── Profile shape type ───────────────────────────────────────────────────────
enum class ProfileType { CONICAL, PARABOLOIDAL, ELLIPSOIDAL };

// ─── Profile ──────────────────────────────────────────────────────────────────
// Stores n+1 sample points of: z-position, single-capillary inner radius (cap),
// and polycapillary outer circumradius (ext), from entrance (z=0) to exit (z=L).
struct Profile {
    int n = 0;                        // number of segments; n+1 data points stored
    std::vector<double> z, cap, ext;  // arrays of length n+1

    // Construct an analytic profile.  All lengths in centimetres.
    // foc_up / foc_dn : upstream / downstream focal distances [cm]
    // n_pts           : number of profile segments (default 1000)
    Profile(ProfileType type,
            double length,
            double rad_ext_up,  double rad_ext_dn,
            double rad_int_up,  double rad_int_dn,
            double foc_up,      double foc_dn,
            int    n_pts = 1000)
    : n(n_pts), z(n_pts+1), cap(n_pts+1), ext(n_pts+1)
    {
        // Z positions and single-capillary shape (always conical – linear)
        for (int i = 0; i <= n; ++i) {
            z[i]   = length / n * i;
            cap[i] = rad_int_up + (rad_int_dn - rad_int_up) / length * z[i];
        }

        switch (type) {

        // ── Conical: outer radius varies linearly ─────────────────────────────
        case ProfileType::CONICAL:
            for (int i = 0; i <= n; ++i)
                ext[i] = rad_ext_up + (rad_ext_dn - rad_ext_up) / length * z[i];
            break;

        // ── Paraboloidal: quadratic fit through 4 control points ──────────────
        case ProfileType::PARABOLOIDAL: {
            double px[4], py[4];
            // Endpoints match required entrance / exit radii exactly
            px[0] = 0.;       py[0] = rad_ext_up;
            px[3] = length;   py[3] = rad_ext_dn;

            // Upstream intermediate point on tangent from upstream focus
            px[1] = (foc_up <= length) ? foc_up / 10.0 : length / 10.0;
            py[1] = (rad_ext_up / foc_up) * px[1] + rad_ext_up;

            // Downstream intermediate point on tangent from downstream focus
            px[2] = (foc_dn <= length) ? length - foc_dn / 10.0 : length * 0.9;
            py[2] = (-rad_ext_dn / foc_dn) * (px[2] - length) + rad_ext_dn;

            double c[3];
            fitQuadratic(px, py, 4, c);
            for (int i = 0; i <= n; ++i)
                ext[i] = c[0] + c[1]*z[i] + c[2]*z[i]*z[i];
            break;
        }

        // ── Ellipsoidal: ellipse with horizontal tangent on larger-radius side ─
        case ProfileType::ELLIPSOIDAL: {
            double slope, b, k, a;
            if (rad_ext_dn < rad_ext_up) {
                // Focusing geometry: exit is smaller; exit tangent aims at foc_dn
                slope      = rad_ext_dn / foc_dn;
                double dr  = rad_ext_dn - rad_ext_up;
                b = (-dr*dr - slope*length*dr) / (slope*length + 2.0*dr);
                k = rad_ext_up - b;
                a = std::sqrt((b*b * length) / (slope * (rad_ext_dn - k)));
                for (int i = 0; i <= n; ++i)
                    ext[i] = std::sqrt(b*b - b*b * z[i]*z[i] / (a*a)) + k;
            } else {
                // Collimating geometry: entrance is smaller; entrance tangent aims at foc_up
                slope      = rad_ext_up / foc_up;
                double dr  = rad_ext_up - rad_ext_dn;
                b = (-dr*dr - slope*length*dr) / (slope*length + 2.0*dr);
                k = rad_ext_dn - b;
                a = std::sqrt(std::abs((b*b * length) / (slope * (rad_ext_up - k))));
                // ext evaluated with reversed z so the flat tangent is on the exit side
                for (int i = 0; i <= n; ++i)
                    ext[i] = std::sqrt(b*b - b*b * z[n-i]*z[n-i] / (a*a)) + k;
            }
            break;
        }
        } // switch
    }

private:
    // Fit  y = c[0] + c[1]*x + c[2]*x^2  to n_pts data points.
    // Solves the 3×3 normal equations with Gaussian elimination – no external library.
    static void fitQuadratic(const double* px, const double* py, int n_pts, double c[3]) {
        double A[3][3] = {}, rhs[3] = {};
        for (int i = 0; i < n_pts; ++i) {
            double xi[3] = {1.0, px[i], px[i]*px[i]};
            for (int r = 0; r < 3; ++r) {
                rhs[r] += xi[r] * py[i];
                for (int s = 0; s < 3; ++s)
                    A[r][s] += xi[r] * xi[s];
            }
        }
        for (int col = 0; col < 3; ++col) {
            int pivot = col;
            for (int row = col+1; row < 3; ++row)
                if (std::abs(A[row][col]) > std::abs(A[pivot][col])) pivot = row;
            std::swap(A[col], A[pivot]);
            std::swap(rhs[col], rhs[pivot]);
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
};

// ─── Description ─────────────────────────────────────────────────────────────
// All material and geometry parameters defining the polycapillary optic.
struct Description {
    Profile             profile;
    double              sig_rough;  // glass surface roughness [Angstrom]
    int64_t             n_cap;      // total number of capillaries
    std::vector<int>    iz;         // atomic numbers of glass constituents
    std::vector<double> wi;         // weight fractions (auto-normalised from percent)
    double              density;    // glass density [g/cm^3]

    // wi may be supplied as weight percents (sum ≈ 100) – normalised to sum to 1.
    Description(Profile             prof,
                double              sig_rough_,
                int64_t             n_cap_,
                std::vector<int>    iz_,
                std::vector<double> wi_,
                double              density_)
    : profile(std::move(prof)), sig_rough(sig_rough_),
      n_cap(n_cap_), iz(std::move(iz_)), wi(std::move(wi_)), density(density_)
    {
        double sum = 0.0;
        for (double w : wi) sum += w;
        if (sum > 1.5) for (double& w : wi) w /= 100.0;
    }
};

// ─── TraceResult ──────────────────────────────────────────────────────────────
struct TraceResult {
    enum Status {
        TRANSMITTED =  1,  // photon reached the exit window
        ABSORBED    =  0,  // all per-energy weights fell below 1e-4 inside the optic
        MISSED      = -2   // photon did not enter any capillary at the entrance
    };
    Status              status         = MISSED;
    Vec3                exit_coords    = {};   // position at exit or last reflection
    Vec3                exit_direction = {};   // propagation direction at exit
    Vec3                exit_electric  = {};   // electric-field vector at exit
    std::vector<double> weights;               // per-energy transmission weights in [0,1]
    int64_t             n_reflections  = 0;   // total number of wall reflections
    double              d_travel       = 0.0; // total geometric path length [cm]
};

// ─── PolyCap ──────────────────────────────────────────────────────────────────
class PolyCap {
public:
    explicit PolyCap(Description desc) : desc_(std::move(desc)) {}

    // Trace a single photon ray through the optic.
    //
    // start     – photon start position [cm]; start.z = 0 for the entrance window.
    // direction – propagation direction vector (need not be unit; must have z > 0).
    // electric  – electric-field polarisation vector (need not be unit).
    // energies  – discrete photon energies [keV], each in the range [1, 100].
    //
    // Returns a TraceResult with per-energy transmission weights in [0,1],
    // the exit position/direction/polarisation, and bookkeeping counters.
    TraceResult traceRay(Vec3 start, Vec3 direction, Vec3 electric,
                         const std::vector<double>& energies) const
    {
        TraceResult result;
        const int nE = static_cast<int>(energies.size());
        result.weights.assign(nE, 0.0);

        direction.normalize();
        electric.normalize();

        if (direction.z <= 0.0)
            return result;   // photon must travel in +z direction

        const Profile& prof = desc_.profile;

        // ── Profile segment at the photon start position ───────────────────────
        int    z_id     = segmentAt(start.z);
        double ext_at_z = lerpArr(prof.ext, z_id, start.z);

        // ── Number of hexagonal shells (0 = monocapillary) ────────────────────
        double n_shells = std::round(std::sqrt(12.0*desc_.n_cap - 3.0) / 6.0 - 0.5);

        // ── Boundary check at entrance ─────────────────────────────────────────
        double q_i = 0.0, r_i = 0.0;
        if (n_shells == 0.0) {
            if (std::sqrt(start.x*start.x + start.y*start.y) > ext_at_z)
                return result;   // outside monocapillary boundary → MISSED
        } else {
            if (!withinHex(ext_at_z, start))
                return result;   // outside hexagonal optic boundary → MISSED
            hexIndex(ext_at_z, n_shells, start, q_i, r_i);
        }

        // ── Verify photon is inside the selected capillary, not in the glass wall ──
        double cap_at_z = lerpArr(prof.cap, z_id, start.z);
        double cx0      = axisX(q_i, r_i, ext_at_z, n_shells);
        double cy0      = axisY(r_i,       ext_at_z, n_shells);
        double d_to_cen = std::sqrt((start.x-cx0)*(start.x-cx0)
                                  + (start.y-cy0)*(start.y-cy0));
        if (d_to_cen > cap_at_z)
            return result;   // photon hits glass wall on entrance → MISSED

        // ── Build capillary central-axis coordinate arrays ─────────────────────
        // The selected capillary's (x,y) centre follows the optic geometry:
        // at each z[i] the centre is at (axisX, axisY) scaled by the local ext radius.
        std::vector<double> cap_cx(prof.n+1), cap_cy(prof.n+1);
        int ix = 0;
        for (int i = 0; i <= prof.n; ++i) {
            cap_cx[i] = axisX(q_i, r_i, prof.ext[i], n_shells);
            cap_cy[i] = axisY(r_i,       prof.ext[i], n_shells);
            if (prof.z[i] <= start.z) ix = i;
        }

        // ── Precompute xraylib attenuation / scattering data ───────────────────
        std::vector<double> amu(nE), scatf(nE);
        computeXrayData(energies, amu, scatf);

        // ── Photon state ───────────────────────────────────────────────────────
        Vec3 pos  = start;
        Vec3 dir  = direction;
        Vec3 elec = electric;
        std::vector<double> weights(nE, 1.0);
        int64_t n_refl = 0;
        double  d_trav = 0.0;

        // ── Main reflection loop ───────────────────────────────────────────────
        // reflectStep() advances the photon to the next wall hit and applies physics.
        // Return codes:   1 = reflected (continue)
        //                 0 = absorbed
        //                -2 = no wall hit, photon exits
        TraceResult::Status final_status = TraceResult::MISSED;
        for (int iter = 0; iter <= prof.n; ++iter) {
            int rc = reflectStep(ix, pos, dir, elec, weights, n_refl, d_trav,
                                 cap_cx, cap_cy, amu, scatf, energies);
            if (rc == 0) {
                final_status = TraceResult::ABSORBED;
                break;
            }
            if (rc == -2) {
                // Propagate remaining path to exit window z = prof.z[n]
                double dz = prof.z[prof.n] - pos.z;
                if (dz > 0.0) pos = pos + dir * (dz / dir.z);
                final_status = TraceResult::TRANSMITTED;
                break;
            }
        }

        result.status         = final_status;
        result.exit_coords    = pos;
        result.exit_direction = dir;
        result.exit_electric  = elec;
        result.weights        = weights;
        result.n_reflections  = n_refl;
        result.d_travel       = d_trav;
        return result;
    }

private:
    Description desc_;

    // ─── Profile helpers ──────────────────────────────────────────────────────

    // Index of the last profile z-node that is ≤ z_pos
    int segmentAt(double z_pos) const {
        int idx = 0;
        const auto& pz = desc_.profile.z;
        for (int i = 0; i < desc_.profile.n; ++i)
            if (pz[i] <= z_pos) idx = i;
        return idx;
    }

    // Linear interpolation in a profile array at z_pos given segment index seg
    double lerpArr(const std::vector<double>& arr, int seg, double z_pos) const {
        const auto& pz = desc_.profile.z;
        return arr[seg] + (arr[seg+1] - arr[seg])
                        / (pz[seg+1]  - pz[seg])
                        * (z_pos       - pz[seg]);
    }

    // ─── Hexagonal grid geometry ──────────────────────────────────────────────

    // True if (x,y) is inside the hexagonal polycapillary boundary of circumradius ext
    static bool withinHex(double ext, Vec3 c) noexcept {
        double d = std::sqrt(ext*ext - (ext*0.5)*(ext*0.5));  // inradius
        return std::abs(c.y)                       <= d
            && std::abs(COSPI6*c.x + 0.5*c.y)     <= d
            && std::abs(COSPI6*c.x - 0.5*c.y)     <= d;
    }

    // Convert (x,y) at a given cross-section (circumradius = ext) to hex axial
    // indices (q, r) using cube-coordinate rounding so that q + r + s = 0.
    static void hexIndex(double ext, double n_shells, Vec3 c, double& q, double& r) noexcept {
        double z  = ext / (2.0 * COSPI6 * (n_shells + 1.0));
        double qf = (c.x / (2.0*COSPI6) - c.y / 3.0) / z;
        double rf =  c.y * (2.0/3.0) / z;
        double sf = -qf - rf;
        double rq = std::round(qf), rr = std::round(rf), rs = std::round(sf);
        double dq = std::abs(rq-qf), dr = std::abs(rr-rf), ds = std::abs(rs-sf);
        if      (dq > dr && dq > ds) { q = -rr-rs;  r =  rr; }
        else if (dr > ds)            { q =  rq;      r = -rq-rs; }
        else                         { q =  rq;      r =  rr; }
    }

    // Capillary central-axis x/y from hex indices and outer radius at z
    static double axisX(double q, double r, double ext, double n_shells) noexcept {
        double z = ext / (2.0 * COSPI6 * (n_shells + 1.0));
        return (2.0*q + r) * COSPI6 * z;
    }
    static double axisY(double r, double ext, double n_shells) noexcept {
        double z = ext / (2.0 * COSPI6 * (n_shells + 1.0));
        return r * 1.5 * z;
    }

    // ─── X-ray optical constants (xraylib) ───────────────────────────────────

    void computeXrayData(const std::vector<double>& energies,
                         std::vector<double>& amu,
                         std::vector<double>& scatf) const
    {
        const int nE = static_cast<int>(energies.size());
        for (int i = 0; i < nE; ++i) {
            double mu = 0.0, sf = 0.0;
            for (int j = 0; j < static_cast<int>(desc_.iz.size()); ++j) {
                int    Z = desc_.iz[j];
                double w = desc_.wi[j];
                // Mass attenuation coefficient [cm^2/g] × weight fraction → weighted sum
                mu += XRayLib::CS_Total(Z, energies[i], nullptr) * w;
                // Scattering factor: (Z + f') × weight/AtomicWeight
                sf += (Z + XRayLib::Fi(Z, energies[i], nullptr)) * (w / XRayLib::AtomicWeight(Z, nullptr));
            }
            amu[i]   = mu * desc_.density;   // linear attenuation [cm^-1]
            scatf[i] = sf;
        }
    }

    // ─── Ray–conical-segment intersection ────────────────────────────────────
    //
    // Finds the next forward wall-hit of ray (phot0 + t·dir) against the conical
    // capillary segment [cap0→cap1] with inner radii [r0, r1].
    //
    // phot0        : photon position extrapolated to z = cap0.z
    // photon_coord : (in)  last interaction point used as minimum-z filter
    //               (out) set to the hit point on success
    // surface_norm : (out) outward unit normal at the hit point on success
    //
    // Returns true if a valid forward intersection was found inside the segment.
    static bool capilSegment(Vec3 cap0, Vec3 cap1, double r0, double r1,
                             Vec3 phot0, Vec3 dir,
                             Vec3& photon_coord,
                             Vec3& surface_norm) noexcept
    {
        dir.normalize();
        Vec3   cap_dir = cap1 - cap0;
        double dz_cap  = cap_dir.z;   // > 0 (segments always go in +z)

        // Per-unit-z differences between photon and capillary-axis slopes, and radius change
        double ddx = dir.x/dir.z        - cap_dir.x/dz_cap;
        double ddy = dir.y/dir.z        - cap_dir.y/dz_cap;
        double dR  = (r1 - r0) / dz_cap;

        double dx0 = phot0.x - cap0.x;
        double dy0 = phot0.y - cap0.y;

        // Quadratic  a·dist² + b·dist + c = 0
        // dist = z-displacement from phot0.z to the wall-hit point
        double qa = ddx*ddx + ddy*ddy - dR*dR;
        double qb = 2.0*(dx0*ddx + dy0*ddy) - 2.0*r0*dR;
        double qc = dx0*dx0 + dy0*dy0 - r0*r0;

        double discr = qb*qb - 4.0*qa*qc;
        if (discr < 0.0) return false;

        // Collect candidates
        double cand[2];
        int nc = 0;
        if (std::abs(qa) < 1e-30) {
            if (std::abs(qb) > 1e-30) cand[nc++] = -qc / qb;
        } else {
            double sq = std::sqrt(discr);
            cand[nc++] = (-qb + sq) / (2.0*qa);
            cand[nc++] = (-qb - sq) / (2.0*qa);
        }

        // Choose the smallest candidate that:
        //   • produces a forward path length  d_proj > 1e-10
        //   • lands inside the segment  [cap0.z, cap1.z]
        //   • is strictly ahead of the last interaction point
        double best = 1e30;
        for (int i = 0; i < nc; ++i) {
            double z_hit   = phot0.z + cand[i];
            double d_proj  = cand[i] / dir.z;
            if (d_proj < 1e-10)                         continue;
            if (z_hit <= cap0.z || z_hit > cap1.z)     continue;
            if (z_hit - photon_coord.z < 1e-5)         continue;
            if (cand[i] < best) best = cand[i];
        }
        if (best > 1e29) return false;

        // ── Hit point ─────────────────────────────────────────────────────────
        double d_proj = best / dir.z;
        Vec3 hit;
        hit.z = phot0.z + best;
        hit.x = phot0.x + d_proj * dir.x;
        hit.y = phot0.y + d_proj * dir.y;

        // ── Surface normal ────────────────────────────────────────────────────
        // Project hit onto capillary axis (matches polycap_capil_segment exactly):
        //   cap_coord = cap0 + [(hit - cap0)·cap_dir / |cap_dir|^2] * cap_dir
        // This gives axis_at_hit.z != hit.z in general, so radial.z != 0.
        Vec3 rel     = hit - cap0;
        double d_seg_sq  = cap_dir.dot(cap_dir);
        double proj_t    = rel.dot(cap_dir) / d_seg_sq;
        Vec3 axis_at_hit = {
            cap0.x + proj_t * cap_dir.x,
            cap0.y + proj_t * cap_dir.y,
            cap0.z + proj_t * cap_dir.z
        };

        Vec3   radial = hit - axis_at_hit;   // from capillary axis to hit point
        double r_dist = radial.norm();

        // γ = angle between capillary wall and axis (negative for confocal / r0 < r1)
        double d_seg = (cap1 - cap0).norm();
        double tga   = (r0 - r1) / d_seg;
        double gam   = std::atan(tga);
        double sga   = std::sin(gam);
        double cga   = std::cos(gam);
        Vec3   cap_hat = {cap_dir.x/d_seg, cap_dir.y/d_seg, cap_dir.z/d_seg};

        // Normal: radial component (scaled by cosγ) + axial component (scaled by sinγ)
        surface_norm.x = cga * radial.x / r_dist + sga * cap_hat.x;
        surface_norm.y = cga * radial.y / r_dist + sga * cap_hat.y;
        surface_norm.z = cga * radial.z / r_dist + sga * cap_hat.z;
        surface_norm.normalize();

        photon_coord = hit;
        return true;
    }

    // ─── Single reflection step ───────────────────────────────────────────────
    //
    // Scans profile segments from ix onwards, finds the next capillary-wall hit,
    // applies Fresnel reflectivity + roughness correction, and updates photon state.
    //
    // Return codes:
    //   1   – wall hit found; photon reflected; weights updated
    //   0   – absorbed (all weights < 1e-4 after reflection)
    //  -2   – no wall hit found in remaining segments; photon exits capillary
    int reflectStep(int& ix,
                    Vec3& pos, Vec3& dir, Vec3& elec,
                    std::vector<double>& weights,
                    int64_t& n_refl, double& d_trav,
                    const std::vector<double>& cap_cx,
                    const std::vector<double>& cap_cy,
                    const std::vector<double>& amu,
                    const std::vector<double>& scatf,
                    const std::vector<double>& energies) const
    {
        const Profile& prof = desc_.profile;
        const int nE = static_cast<int>(energies.size());
        dir.normalize();

        Vec3 photon_coord = pos;   // lower-bound filter: new hit must be ahead of this
        Vec3 hit, surf;
        bool found = false;

        // ── Scan segments for the next wall hit ────────────────────────────────
        for (int i = ix; i < prof.n; ++i) {
            Vec3 cap0 = {cap_cx[i],   cap_cy[i],   prof.z[i]  };
            Vec3 cap1 = {cap_cx[i+1], cap_cy[i+1], prof.z[i+1]};

            // Skip segments whose axis endpoints are outside the optic
            if (!withinHex(prof.ext[i],   cap0)) continue;
            if (!withinHex(prof.ext[i+1], cap1)) continue;

            // Photon position at the z-plane of cap0
            Vec3 phot0;
            phot0.z = prof.z[i];
            phot0.x = pos.x + dir.x * (prof.z[i] - pos.z) / dir.z;
            phot0.y = pos.y + dir.y * (prof.z[i] - pos.z) / dir.z;

            Vec3 temp_hit = photon_coord, temp_surf;
            if (!capilSegment(cap0, cap1, prof.cap[i], prof.cap[i+1],
                              phot0, dir, temp_hit, temp_surf))
                continue;

            // Confirm the hit is still inside the optic boundary
            double ext_hit = lerpArr(prof.ext, i, temp_hit.z);
            if (!withinHex(ext_hit, temp_hit))
                continue;

            hit   = temp_hit;
            surf  = temp_surf;
            ix    = i + 1;
            found = true;
            break;
        }

        if (!found) return -2;   // no wall hit: photon exits

        // ── Advance photon to wall hit ────────────────────────────────────────
        d_trav += (hit - pos).norm();
        pos = hit;

        // Reset ix to the last segment index where z[j] <= pos.z
        // (matches library polycap_capil_trace: *ix rescanned backward after hit)
        for (int j = 0; j < prof.n; ++j)
            if (prof.z[j] <= pos.z) ix = j;

        // ── Geometric decomposition (energy-independent) ──────────────────────
        dir.normalize();
        double cos_alfa = surf.dot(dir);
        if (cos_alfa < 0.0) {
            // Normal convention is flipped; correct it so the normal faces the photon
            surf     = surf * -1.0;
            cos_alfa = -cos_alfa;
        }

        // s-direction: perpendicular to both direction and surface normal
        Vec3 s_dir = surf.cross(dir);
        if (s_dir.norm() < 1e-10) return -2;   // normal incidence, degenerate
        s_dir.normalize();

        // p-direction: in the plane of incidence, perpendicular to direction
        Vec3 p_dir = dir.cross(s_dir);
        p_dir.normalize();

        // Decompose incoming electric-field vector into s / normal / p projections
        elec.normalize();
        double a_s  = elec.dot(s_dir);    // projection onto s  (determines frac_s)
        double frac_s = a_s * a_s;
        double frac_p = 1.0 - frac_s;
        double a_n  = elec.dot(surf);     // projection onto surface normal
        double a_p  = elec.dot(p_dir);    // projection onto p

        double cos_theta = cos_alfa;
        double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta*cos_theta));

        // ── Per-energy Fresnel reflectivity ───────────────────────────────────
        bool any_above = false;
        for (int k = 0; k < nE; ++k) {
            using cd = std::complex<double>;

            // Complex refractive index  n = 1 – δ + i·β
            //   δ = (λ²/2π) · r₀ · ρ_e  ≈  (hc/E)² · N_A·R₀·ρ / (2π)  · Σ(w/A)·(Z+f')
            //   β = (λ/4π) · μ = hc/(4π) · μ/E
            double delta = (HC/energies[k])*(HC/energies[k])
                         * (N_AVOG * R0_ELEC * desc_.density / TWOPI) * scatf[k];
            double beta  = (HC / (2.0*TWOPI)) * (amu[k] / energies[k]);
            cd n{1.0 - delta, beta};

            // Snell's law in complex form
            cd sin2_t = cd{sin_theta*sin_theta} / (n*n);
            cd cos_t  = std::sqrt(cd{1.0} - sin2_t);

            // Fresnel amplitude reflectances (intensity = |r|²)
            cd r_s = (cd{cos_theta} - n*cos_t) / (cd{cos_theta} + n*cos_t);
            cd r_p = (cos_t - n*cd{cos_theta}) / (cos_t + n*cd{cos_theta});
            double R_s = std::norm(r_s);
            double R_p = std::norm(r_p);

            // Debye-Waller roughness factor:  exp(–(4π σ sin(θ_g) / λ)²)
            //   where sin(θ_grazing) = cos_alfa  and  1/λ = E/hc
            //   → 4π/(hc) ≈ 1.01358 × 10⁷ cm⁻¹ keV⁻¹  (with σ in Angstrom = 1e-8 cm)
            double cons1   = 1.01358 * energies[k] * cos_alfa * desc_.sig_rough;
            double r_rough = std::exp(-cons1 * cons1);

            weights[k] *= (R_s * frac_s + R_p * frac_p) * r_rough;
            if (weights[k] >= 1e-4) any_above = true;
        }

        if (!any_above) return 0;   // photon absorbed

        // ── Update electric-field vector (energy-independent geometry) ────────
        // Each Cartesian component is rebuilt from its s / normal / p projections,
        // weighted by frac_s and frac_p.  (Same formula as polycap_refl_polar.)
        auto blend = [&](double ev) noexcept {
            double ts = ev * a_s * frac_s;
            double tn = ev * a_n * frac_p;
            double tp = ev * a_p * frac_p;
            return std::sqrt(ts*ts + tn*tn + tp*tp);
        };
        Vec3 new_elec = {blend(elec.x), blend(elec.y), blend(elec.z)};
        if (new_elec.norm() > 0.0) new_elec.normalize();
        elec = new_elec;

        // ── Specular reflection of propagation direction ───────────────────────
        //   d' = d – 2·(d·n̂)·n̂
        dir = dir - surf * (2.0 * cos_alfa);
        dir.normalize();
        ++n_refl;

        return 1;
    }
};

} // namespace polycap
