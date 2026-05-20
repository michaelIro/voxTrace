#pragma once
#include "Platform.hpp"
#include "Ray.hpp"

#ifndef __METAL_VERSION__
    #include <cmath>
    #include <cstdio>
#endif

// ── PolyCap ───────────────────────────────────────────────────────────────────
// Polycapillary X-ray optic.
//
// Physical model:
//  • Hexagonal close-packing: entrance hit snapped to the nearest capillary cell.
//  • Piecewise-curved capillary: N_SEG straight-cylinder segments whose local
//    axis direction is linearly interpolated between the entrance axis
//    (input-focal → entrance centre) and the exit axis (exit centre →
//    output-focal point).  Radius tapers linearly.
//  • Per-bounce specular reflection with complex Fresnel reflectivity for s
//    and p polarisations (Debye-Waller roughness applied) and full
//    polarisation-basis rotation after each reflection.
//
// All methods are KOKKOS_INLINE_FUNCTION / Metal-MSL compatible.
// No dynamic allocation — fixed-size stack arrays only.

#define POLYCAP_MAX_ELEMENTS 8

class PolyCap {

    // ── Tuning ────────────────────────────────────────────────────────────────
    static constexpr int   N_SEG          = 20;    // segments per capillary
    static constexpr int   MAX_BOUNCE_SEG =  4;    // max wall-hits per segment
    static constexpr float EPS_T          = 1e-6f; // minimum valid t
    static constexpr float NO_HIT         = 1e30f; // sentinel "no intersection"

    // ── Parameters ───────────────────────────────────────────────────────────
    float posY_, length_;
    float rExtIn_, rExtOut_;
    float rCapIn_, rCapOut_;
    float focalIn_, focalOut_;

    int   nElements_;
    int   atomicNumbers_[POLYCAP_MAX_ELEMENTS];
    float weightFracs_  [POLYCAP_MAX_ELEMENTS];
    float density_;
    float roughness_;
    int   nCapillaries_;
    float critCoeff_;
    float mu_rho_;   // linear attenuation [cm⁻¹] = CS_Total × density

    // ── Internal helpers ──────────────────────────────────────────────────────

    KOKKOS_INLINE_FUNCTION float atomicMass(int z) const {
        const float A[] = {0,1,4,7,9,11,12,14,16,19,20,
                           23,24,27,28,31,32,35,40,39,40};
        if (z > 0 && z <= 20) return A[z];
        return (float)(2 * z);
    }

    KOKKOS_INLINE_FUNCTION void computeCritCoeff() {
        float ZoverA = 0.f;
        for (int i = 0; i < nElements_; ++i)
            ZoverA += (weightFracs_[i] / 100.f)
                    * ((float)atomicNumbers_[i] / atomicMass(atomicNumbers_[i]));
        critCoeff_ = 28.8e-3f * sqrtf(density_ * ZoverA);
    }

    // ── Hexagonal snap ────────────────────────────────────────────────────────
    // Given entrance hit (hx, hz), find the nearest hexagonal-lattice capillary
    // centre (cx, cz).  Checks the 4 nearest candidates (2 rows × 2 cols) to
    // handle row-boundary ambiguity.
    // Returns false if the hit lands on a wall or the centre is outside rExtIn_.
    KOKKOS_INLINE_FUNCTION bool hexSnap(float hx, float hz,
                                         float& cx, float& cz) const
    {
        const float a = 2.f * rCapIn_;          // column pitch
        const float b = 1.7320508f * rCapIn_;   // row pitch = √3 · rCapIn

        int   row0 = (int)floorf(hz / b);
        float best_d2 = NO_HIT;
        float best_cx = 0.f, best_cz = 0.f;

        for (int ri = 0; ri < 2; ++ri) {
            int   m   = row0 + ri;
            float c_z = m * b;
            float off = ((m % 2 + 2) % 2) ? 0.5f * a : 0.f; // odd-row offset

            int   col0 = (int)floorf((hx - off) / a);
            for (int ci = 0; ci < 2; ++ci) {
                float c_x = (col0 + ci) * a + off;
                float ex  = hx - c_x, ez = hz - c_z;
                float d2  = ex*ex + ez*ez;
                if (d2 < best_d2) { best_d2 = d2; best_cx = c_x; best_cz = c_z; }
            }
        }

        if (best_cx*best_cx + best_cz*best_cz > rExtIn_*rExtIn_) return false;
        if (best_d2 > rCapIn_*rCapIn_) return false;

        cx = best_cx;  cz = best_cz;
        return true;
    }

    // ── Capillary axis direction at normalised parameter t ∈ [0,1] ───────────
    // t=0: normalise(cx_in,  focalIn_,  cz_in)   — source-focal → entrance centre
    // t=1: normalise(-cx_out, focalOut_, -cz_out) — exit centre → output-focal
    // Linearly interpolated and renormalised.
    KOKKOS_INLINE_FUNCTION void capAxis(
        float t,
        float cx_in,  float cz_in,
        float cx_out, float cz_out,
        float& ax, float& ay, float& az) const
    {
        float ix = cx_in,  iy = focalIn_,  iz = cz_in;
        float il = sqrtf(ix*ix + iy*iy + iz*iz);
        if (il > 1e-12f) { ix /= il; iy /= il; iz /= il; }
        else              { iy = 1.f; ix = iz = 0.f; }

        float ox = -cx_out, oy = focalOut_, oz = -cz_out;
        float ol = sqrtf(ox*ox + oy*oy + oz*oz);
        if (ol > 1e-12f) { ox /= ol; oy /= ol; oz /= ol; }
        else              { oy = 1.f; ox = oz = 0.f; }

        ax = ix + t*(ox-ix);  ay = iy + t*(oy-iy);  az = iz + t*(oz-iz);
        float l = sqrtf(ax*ax + ay*ay + az*az);
        if (l > 1e-12f) { ax /= l; ay /= l; az /= l; }
    }

    // ── Capillary centre point (world frame) at normalised parameter t ────────
    KOKKOS_INLINE_FUNCTION void capCenter(
        float t,
        float cx_in,  float cz_in,
        float cx_out, float cz_out,
        float& cx, float& cy, float& cz) const
    {
        cx = cx_in  + t*(cx_out - cx_in);
        cy = posY_  + t*length_;
        cz = cz_in  + t*(cz_out - cz_in);
    }

    // ── Ray – infinite cylinder intersection ──────────────────────────────────
    // rel* = ray origin minus cylinder centre; d* = ray direction (unit);
    // a*   = cylinder axis (unit); R = radius.
    // Returns smallest positive t, or NO_HIT.
    KOKKOS_INLINE_FUNCTION float cylIntersect(
        float relx, float rely, float relz,
        float dx,   float dy,   float dz,
        float ax,   float ay,   float az,
        float R) const
    {
        float dpar   = dx*ax   + dy*ay   + dz*az;
        float relpar = relx*ax + rely*ay + relz*az;

        float dpx = dx   - dpar*ax,    dpy = dy   - dpar*ay,    dpz = dz   - dpar*az;
        float rpx = relx - relpar*ax,  rpy = rely - relpar*ay,  rpz = relz - relpar*az;

        float A = dpx*dpx + dpy*dpy + dpz*dpz;
        if (A < 1e-18f) return NO_HIT;

        float B    = 2.f*(rpx*dpx + rpy*dpy + rpz*dpz);
        float C    = rpx*rpx + rpy*rpy + rpz*rpz - R*R;
        float disc = B*B - 4.f*A*C;
        if (disc < 0.f) return NO_HIT;

        float sq = sqrtf(disc);
        float t1 = (-B - sq) / (2.f*A);
        float t2 = (-B + sq) / (2.f*A);

        if (t1 > EPS_T) return t1;
        if (t2 > EPS_T) return t2;
        return NO_HIT;
    }

    // ── Distance from ray origin to the flat segment-end plane ───────────────
    // ec* = segment-end axis centre; a* = axis direction = plane normal.
    KOKKOS_INLINE_FUNCTION float segEndT(
        float px,  float py,  float pz,
        float dx,  float dy,  float dz,
        float ecx, float ecy, float ecz,
        float ax,  float ay,  float az) const
    {
        float dax = dx*ax + dy*ay + dz*az;
        if (fabsf(dax) < 1e-12f) return NO_HIT;
        float t = ((ecx-px)*ax + (ecy-py)*ay + (ecz-pz)*az) / dax;
        return (t > 0.f) ? t : NO_HIT;
    }

    // ── Circular-arc parameters for one capillary ─────────────────────────────
    // Fits the unique circle through entrance point P0 with tangent d0 that also
    // passes through exit point P1.  Gives the exact curved centreline geometry.
    //
    // Parameterisation (t ∈ [0,1]):
    //   P(t) = arcO + arcR·(−cos(t·dPhi)·perp + sin(t·dPhi)·d0)   ← centre
    //   T(t) = sin(t·dPhi)·perp + cos(t·dPhi)·d0                  ← unit tangent
    //
    // Degenerate (straight) capillary: arcR = 1e30, dPhi = 0.
    KOKKOS_INLINE_FUNCTION void buildArc(
        float  cx_in,  float  cz_in,
        float  cx_out, float  cz_out,
        float& d0x,    float& d0y,    float& d0z,
        float& perpx,  float& perpy,  float& perpz,
        float& arcOx,  float& arcOy,  float& arcOz,
        float& arcR,   float& dPhi) const
    {
        // Entrance tangent: direction from input focal point toward entrance centre
        float ix = cx_in, iy = focalIn_, iz = cz_in;
        float il = sqrtf(ix*ix + iy*iy + iz*iz);
        if (il > 1e-12f) { ix/=il; iy/=il; iz/=il; }
        else              { ix=0.f; iy=1.f; iz=0.f; }
        d0x=ix; d0y=iy; d0z=iz;

        float P0x=cx_in, P0y=posY_,          P0z=cz_in;
        float P1x=cx_out,P1y=posY_+length_,  P1z=cz_out;

        // Arc-plane normal  n̂ = normalise(chord × d0)
        float chx=P1x-P0x, chy=P1y-P0y, chz=P1z-P0z;
        float npx = chy*d0z - chz*d0y;
        float npy = chz*d0x - chx*d0z;
        float npz = chx*d0y - chy*d0x;
        float npl = sqrtf(npx*npx + npy*npy + npz*npz);
        if (npl < 1e-9f) {                      // chord ∥ d0: straight capillary
            perpx=perpy=perpz=0.f;
            arcOx=P0x; arcOy=P0y; arcOz=P0z;
            arcR=1e30f; dPhi=0.f;
            return;
        }
        npx/=npl; npy/=npl; npz/=npl;

        // perp = d0 × n̂  (unit, perpendicular to d0, pointing toward arc centre)
        perpx = d0y*npz - d0z*npy;
        perpy = d0z*npx - d0x*npz;
        perpz = d0x*npy - d0y*npx;
        float ppl = sqrtf(perpx*perpx + perpy*perpy + perpz*perpz);
        if (ppl > 1e-12f) { perpx/=ppl; perpy/=ppl; perpz/=ppl; }

        // Radius:  arcO = P0 + arcR·perp  and  |arcO − P1| = |arcR|
        // ⇒  arcR = −|P0−P1|² / (2·(P0−P1)·perp)
        float Vx=P0x-P1x, Vy=P0y-P1y, Vz=P0z-P1z;
        float V2  = Vx*Vx + Vy*Vy + Vz*Vz;
        float Vdp = Vx*perpx + Vy*perpy + Vz*perpz;
        if (fabsf(Vdp) < 1e-18f) {              // degenerate: treat as straight
            perpx=perpy=perpz=0.f;
            arcOx=P0x; arcOy=P0y; arcOz=P0z;
            arcR=1e30f; dPhi=0.f;
            return;
        }
        arcR  = -V2 / (2.f * Vdp);
        arcOx = P0x + arcR*perpx;
        arcOy = P0y + arcR*perpy;
        arcOz = P0z + arcR*perpz;

        // Arc angle: project (P1−arcO) onto the local frame (−perp, d0)
        float q1x=P1x-arcOx, q1y=P1y-arcOy, q1z=P1z-arcOz;
        float cosD = (-perpx*q1x - perpy*q1y - perpz*q1z) / arcR;
        float sinD = ( d0x  *q1x +  d0y  *q1y +  d0z  *q1z) / arcR;
        cosD = fmaxf(-1.f, fminf(1.f, cosD));
        dPhi = atan2f(sinD, cosD);
    }

    // ── Complex Fresnel intensities + Debye-Waller ────────────────────────────
    // sinTheta : grazing-angle sine  = |D · N_wall|
    // Rs, Rp   : output intensities  = |r_s|² · R_DW²,  |r_p|² · R_DW²
    KOKKOS_INLINE_FUNCTION void fresnelSP(
        float sinTheta, float energy,
        float& Rs, float& Rp) const
    {
        const float HC_CM  = 1.23984e-7f;
        const float HC_ANG = 12.398f;

        float thetaCrit = critCoeff_ / energy;
        float beta = mu_rho_ * HC_CM / (4.f * VT_PI * energy);

        // w = √(sin²θ − θc² − 2iβ)
        float xr = sinTheta*sinTheta - thetaCrit*thetaCrit;
        float xi = -2.f * beta;
        float rm = sqrtf(xr*xr + xi*xi);
        float wu = sqrtf((rm + xr) * 0.5f);
        float wv = (xi < 0.f ? -1.f : 1.f) * sqrtf(fabsf((rm - xr) * 0.5f));

        // r_s = (sinθ − w) / (sinθ + w)
        float ns_r = sinTheta-wu, ns_i = -wv;
        float ds_r = sinTheta+wu, ds_i =  wv;
        float ds2  = ds_r*ds_r + ds_i*ds_i;
        float Rs_F = (ds2 > 1e-30f) ? fminf((ns_r*ns_r + ns_i*ns_i) / ds2, 1.f) : 0.f;

        // r_p = (n²·sinθ − w) / (n²·sinθ + w);  n² ≈ 1 − 2δ − 2iβ,  δ = θc²/2
        float delta = 0.5f * thetaCrit * thetaCrit;
        float n2sr  = (1.f - 2.f*delta) * sinTheta;
        float n2si  = -2.f * beta * sinTheta;
        float np_r  = n2sr-wu, np_i = n2si-wv;
        float dp_r  = n2sr+wu, dp_i = n2si+wv;
        float dp2   = dp_r*dp_r + dp_i*dp_i;
        float Rp_F  = (dp2 > 1e-30f) ? fminf((np_r*np_r + np_i*np_i) / dp2, 1.f) : 0.f;

        // Debye-Waller intensity factor: exp(−2·(4π σ sinθ E / ħc)²)
        float expt  = (4.f * VT_PI * roughness_ * sinTheta * energy) / HC_ANG;
        float R_DW2 = expf(-2.f * expt * expt);

        Rs = Rs_F * R_DW2;
        Rp = Rp_F * R_DW2;
    }

    // ── Per-bounce: reflection + polarisation update + probability scaling ────
    // dir : incident direction (unit)  → updated to reflected direction
    // N   : outward wall normal (unit)
    // as* : s-pol direction vector     → updated to new s_hat
    // ap* : p-pol direction vector     → updated to D'× s_hat
    // wS/wP : s/p intensity fractions  → updated
    // prob  : probability weight       → scaled by effective reflectivity
    KOKKOS_INLINE_FUNCTION void bounce(
        float& dx,  float& dy,  float& dz,
        float  nx,  float  ny,  float  nz,
        float& asx, float& asy, float& asz,
        float& apx, float& apy, float& apz,
        float& wS,  float& wP,
        float& prob,
        float  energy) const
    {
        float dot_dn   = dx*nx + dy*ny + dz*nz;
        float sinTheta = fabsf(dot_dn);

        float Rs, Rp;
        fresnelSP(sinTheta, energy, Rs, Rp);

        // s_hat = normalise(D × N)  — perpendicular to plane of incidence
        float sx = dy*nz - dz*ny;
        float sy = dz*nx - dx*nz;
        float sz = dx*ny - dy*nx;
        float sl = sqrtf(sx*sx + sy*sy + sz*sz);
        if (sl > 1e-12f) { sx /= sl; sy /= sl; sz /= sl; }
        else {
            // near-normal incidence: pick arbitrary perpendicular to N
            float tx = 1.f - fabsf(nx), ty = -fabsf(ny), tz = 0.f;
            float tl = sqrtf(tx*tx + ty*ty);
            if (tl > 1e-12f) { sx = tx/tl; sy = ty/tl; sz = 0.f; }
        }

        // Fraction of intensity in new s/p directions
        float cos_a  = asx*sx + asy*sy + asz*sz;
        float wS_loc = wS*(cos_a*cos_a) + wP*(1.f - cos_a*cos_a);
        float wP_loc = 1.f - wS_loc;

        float scale = Rs*wS_loc + Rp*wP_loc;
        if (scale < 1e-30f) { prob = 0.f; return; }

        prob *= scale;
        wS    = (Rs * wS_loc) / scale;
        wP    = 1.f - wS;

        // Specular reflection: D' = D − 2(D·N)N
        dx -= 2.f*dot_dn*nx;
        dy -= 2.f*dot_dn*ny;
        dz -= 2.f*dot_dn*nz;

        // New polarisation basis: as = s_hat;  ap = normalise(D' × s_hat)
        asx = sx;  asy = sy;  asz = sz;
        apx = dy*sz - dz*sy;
        apy = dz*sx - dx*sz;
        apz = dx*sy - dy*sx;
        float apl = sqrtf(apx*apx + apy*apy + apz*apz);
        if (apl > 1e-12f) { apx /= apl; apy /= apl; apz /= apl; }
    }

public:

    KOKKOS_INLINE_FUNCTION PolyCap() {}

    KOKKOS_INLINE_FUNCTION PolyCap(
        float posY,   float length,
        float rExtIn, float rExtOut,
        float rCapIn, float rCapOut,
        float focalIn,float focalOut,
        int nElements, const int* atomicNumbers, const float* weightFracs,
        float density, float roughness, int nCapillaries)
        : posY_(posY), length_(length),
          rExtIn_(rExtIn), rExtOut_(rExtOut),
          rCapIn_(rCapIn), rCapOut_(rCapOut),
          focalIn_(focalIn), focalOut_(focalOut),
          nElements_(nElements), density_(density),
          roughness_(roughness), nCapillaries_(nCapillaries),
          mu_rho_(0.f)
    {
        for (int i = 0; i < nElements_; ++i) {
            atomicNumbers_[i] = atomicNumbers[i];
            weightFracs_[i]   = weightFracs[i];
        }
        computeCritCoeff();
    }

    // Set mass-attenuation × density [cm⁻¹].
    // Call on the host before each Kokkos kernel (PolyCap captured by value).
    KOKKOS_INLINE_FUNCTION void setMuRho(float v) { mu_rho_ = v; }

    // Quick check: does the ray's trajectory intersect the entrance aperture?
    KOKKOS_INLINE_FUNCTION bool isEntering(const Ray& ray) const {
        if (fabsf(ray.getDirY()) < 1e-9f) return false;
        float t  = (posY_ - ray.getStartY()) / ray.getDirY();
        if (t < 0.f) return false;
        float hx = ray.getStartX() + t * ray.getDirX();
        float hz = ray.getStartZ() + t * ray.getDirZ();
        return (hx*hx + hz*hz) <= rExtIn_ * rExtIn_;
    }

    // ── Main trace ────────────────────────────────────────────────────────────
    // Traces the ray through the polycapillary using per-capillary piecewise
    // curved geometry.  On success: iaFlag=true, position/direction/
    // polarisation/prob updated.  On failure: iaFlag=false.
    KOKKOS_INLINE_FUNCTION void trace(Ray& ray) const {
        if (!isEntering(ray)) { ray.setIAFlag(false); return; }

        float e = ray.getEnergyKeV();
        if (e <= 0.f) { ray.setIAFlag(false); return; }

        // Propagate to entrance face
        float t_in = (posY_ - ray.getStartY()) / ray.getDirY();
        float hx   = ray.getStartX() + t_in * ray.getDirX();
        float hz   = ray.getStartZ() + t_in * ray.getDirZ();

        // Find individual capillary cell
        float cx_in, cz_in;
        if (!hexSnap(hx, hz, cx_in, cz_in)) { ray.setIAFlag(false); return; }

        float taper  = rExtOut_ / rExtIn_;
        float cx_out = cx_in * taper;
        float cz_out = cz_in * taper;

        // Working ray state in world frame
        float px  = hx,             py  = posY_,         pz  = hz;
        float dx  = ray.getDirX(),  dy  = ray.getDirY(), dz  = ray.getDirZ();
        float asx = ray.getSPolX(), asy = ray.getSPolY(), asz = ray.getSPolZ();
        float apx = ray.getPPolX(), apy = ray.getPPolY(), apz = ray.getPPolZ();
        float prob = ray.getProb();
        float wS = 0.5f, wP = 0.5f;   // unpolarised input
        int   nRefl = 0;

        // ── Circular-arc geometry for this capillary ──────────────────────────
        // buildArc fits the unique circle through P0 (entrance) with tangent d0
        // that also passes through P1 (exit).  Each segment then uses the exact
        // arc tangent as its local cylinder axis and arc positions as centres.
        float d0x, d0y, d0z, perpx, perpy, perpz;
        float arcOx, arcOy, arcOz, arcR, arcDPhi;
        buildArc(cx_in, cz_in, cx_out, cz_out,
                 d0x, d0y, d0z, perpx, perpy, perpz,
                 arcOx, arcOy, arcOz, arcR, arcDPhi);
        bool useArc = (fabsf(arcDPhi) > 1e-7f && fabsf(arcR) < 1e20f);

        // Capillary arc length and frustum taper half-angle α:
        //   tan(α) = dR/ds  →  outward normal = cos(α)·r̂ − sin(α)·â
        // This makes reflection physically correct for tapered tubes.
        float arcLen = useArc
            ? fabsf(arcR * arcDPhi)
            : sqrtf((cx_out-cx_in)*(cx_out-cx_in) + length_*length_
                   +(cz_out-cz_in)*(cz_out-cz_in));
        if (arcLen < 1e-9f) arcLen = length_;
        float dR_ds  = (rCapOut_ - rCapIn_) / arcLen;
        float tapCos = 1.f / sqrtf(1.f + dR_ds*dR_ds);  // cos(α)
        float tapSin = dR_ds * tapCos;                    // sin(α)·sign(dR/ds)

        // ── Circular-arc segment loop ─────────────────────────────────────────
        for (int i = 0; i < N_SEG; ++i) {
            float t_s = (float)i       / (float)N_SEG;  // segment start
            float t_m = (i + 0.5f)     / (float)N_SEG;  // midpoint
            float t_e = (float)(i + 1) / (float)N_SEG;  // end

            // Axis direction = arc tangent T(t) = sin(φ)·perp + cos(φ)·d0
            float ax, ay, az;
            if (useArc) {
                float phi_m = t_m * arcDPhi;
                float sp = sinf(phi_m), cp = cosf(phi_m);
                ax = sp*perpx + cp*d0x;
                ay = sp*perpy + cp*d0y;
                az = sp*perpz + cp*d0z;
            } else {
                capAxis(t_m, cx_in, cz_in, cx_out, cz_out, ax, ay, az);
            }

            // Segment start/end centres on the arc:
            //   P(t) = arcO + arcR·(−cos(t·dPhi)·perp + sin(t·dPhi)·d0)
            float scx, scy, scz, ecx, ecy, ecz;
            if (useArc) {
                float phi_s=t_s*arcDPhi, phi_e=t_e*arcDPhi;
                float cps=cosf(phi_s), sps=sinf(phi_s);
                scx = arcOx + arcR*(-cps*perpx + sps*d0x);
                scy = arcOy + arcR*(-cps*perpy + sps*d0y);
                scz = arcOz + arcR*(-cps*perpz + sps*d0z);
                float cpe=cosf(phi_e), spe=sinf(phi_e);
                ecx = arcOx + arcR*(-cpe*perpx + spe*d0x);
                ecy = arcOy + arcR*(-cpe*perpy + spe*d0y);
                ecz = arcOz + arcR*(-cpe*perpz + spe*d0z);
            } else {
                capCenter(t_s, cx_in, cz_in, cx_out, cz_out, scx, scy, scz);
                capCenter(t_e, cx_in, cz_in, cx_out, cz_out, ecx, ecy, ecz);
            }

            float R_seg = rCapIn_ + t_s*(rCapOut_ - rCapIn_);

            for (int j = 0; j < MAX_BOUNCE_SEG; ++j) {
                float tw = cylIntersect(px-scx, py-scy, pz-scz,
                                        dx, dy, dz, ax, ay, az, R_seg);
                float te = segEndT(px, py, pz, dx, dy, dz,
                                   ecx, ecy, ecz, ax, ay, az);

                if (tw < te) {
                    // ── Wall hit ──────────────────────────────────────────────
                    px += tw*dx;  py += tw*dy;  pz += tw*dz;

                    // Radial unit vector from segment axis to hit point
                    float relx = px-scx, rely = py-scy, relz = pz-scz;
                    float par  = relx*ax + rely*ay + relz*az;
                    float rx   = relx - par*ax;
                    float ry   = rely - par*ay;
                    float rz   = relz - par*az;
                    float rl   = sqrtf(rx*rx + ry*ry + rz*rz);
                    if (rl < 1e-12f) { ray.setIAFlag(false); return; }
                    rx /= rl;  ry /= rl;  rz /= rl;

                    // Frustum-corrected outward surface normal:
                    //   N = cos(α)·r̂ − sin(α)·â
                    // where α is the taper half-angle.  For a pure cylinder
                    // (rCapIn == rCapOut) tapSin=0 and this reduces to r̂.
                    float nx = rx*tapCos - ax*tapSin;
                    float ny = ry*tapCos - ay*tapSin;
                    float nz = rz*tapCos - az*tapSin;
                    float nl = sqrtf(nx*nx + ny*ny + nz*nz);
                    if (nl > 1e-12f) { nx/=nl; ny/=nl; nz/=nl; }
                    else             { nx=rx;  ny=ry;  nz=rz;  }

                    // Full complex Fresnel (fresnelSP) + Debye-Waller + polarisation
                    // update are all handled inside bounce().
                    bounce(dx, dy, dz, nx, ny, nz,
                           asx, asy, asz, apx, apy, apz,
                           wS, wP, prob, e);

                    if (prob < 1e-10f) { ray.setIAFlag(false); return; }
                    ++nRefl;
                } else {
                    // ── Segment end: advance to next segment ──────────────────
                    if (te < NO_HIT)
                        { px += te*dx;  py += te*dy;  pz += te*dz; }
                    break;
                }
            }
        }

        // ── Write results back ────────────────────────────────────────────────
        ray.setStartCoordinates(px, py, pz);
        ray.setEndCoordinates(dx, dy, dz);
        ray.setSPol(asx, asy, asz);
        ray.setPPol(apx, apy, apz);
        ray.setProb(prob);
        ray.setIAFlag(true);
        ray.setIANum(ray.getIANum() + nRefl);
    }

#ifndef __METAL_VERSION__
    inline void print() const {
        printf("PolyCap: L=%.2fcm rExt=(%.4f->%.4f)cm rCap=(%.5f->%.5f)cm"
               " rho=%.2fg/cc rough=%.1fA ncap=%d critCoeff=%.4e\n",
               length_, rExtIn_, rExtOut_, rCapIn_, rCapOut_,
               density_, roughness_, nCapillaries_, critCoeff_);
    }
#endif
};
