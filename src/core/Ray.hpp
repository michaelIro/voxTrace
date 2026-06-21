#pragma once
/**
 * @file Ray.hpp
 * @brief The Ray — an X-ray photon and the unit of parallelism in voxTrace.
 */
#include "Platform.hpp"
#include "RNG.hpp"

#ifndef __METAL_VERSION__
    #include <cmath>
    #include <iostream>
#endif

/**
 * @brief X-ray photon with full polarization state; the trace's parallelization anchor.
 *
 * One @ref Ray is traced per thread, and every operator class (@ref Source,
 * @ref Voxel, @ref Material, @ref ChemElement, @ref PolyCap, @ref Tracer)
 * mutates a `Ray&` in place as the photon advances. The struct is trivially
 * copyable into device memory: fixed layout (~133 bytes), value semantics, no
 * pointers — so it compiles unchanged for Kokkos (CUDA/HIP/OpenMP) and Metal MSL.
 *
 * State carried per photon: start position, unit direction, s/p polarization
 * vectors and phases, wave number (energy), survival probability/weight,
 * interaction count, and the flags/indices used by the voxel walk
 * (`nextVoxel`, `tIn`, `oobFlag`, ...).
 */
class Ray {
    float x0_, y0_, z0_;          // position
    float dirX_, dirY_, dirZ_;    // direction (unit vector)
    float asX_,  asY_,  asZ_;    // s-polarization
    float apX_,  apY_,  apZ_;    // p-polarization
    bool  flag_;
    float k_;                     // wave number
    int   q_;                     // ray index
    float opd_;                   // optical path length
    float fS_, fP_;               // phases

    float prob_        = 1.0f;
    int   iaNum_       = 0;
    bool  iaFlag_      = false;
    bool  oobFlag_     = false;
    int   nextVoxel_   = 13;
    float tIn_         = 0.0f;
    int   respawnCounter_ = 0;
    bool  augerFlag_   = false;

public:

    KOKKOS_INLINE_FUNCTION Ray()
        : x0_(0), y0_(0), z0_(0),
          dirX_(0), dirY_(0), dirZ_(0),
          asX_(1), asY_(0), asZ_(0),
          apX_(0), apY_(1), apZ_(0),
          flag_(false), k_(0), q_(0), opd_(0), fS_(0), fP_(0) {}

    KOKKOS_INLINE_FUNCTION Ray(
        float startX, float startY, float startZ,
        float dirX,   float dirY,   float dirZ,
        float asX,    float asY,    float asZ,
        bool  flag,   float k,      int   q,
        float opd,    float fS,     float fP,
        float apX,    float apY,    float apZ,
        float prob)
        : x0_(startX), y0_(startY), z0_(startZ),
          dirX_(dirX),  dirY_(dirY),  dirZ_(dirZ),
          asX_(asX),    asY_(asY),    asZ_(asZ),
          apX_(apX),    apY_(apY),    apZ_(apZ),
          flag_(flag),  k_(k),        q_(q),
          opd_(opd),    fS_(fS),      fP_(fP),
          prob_(prob) {}

    // ── Setters ───────────────────────────────────────────────────────────────
    KOKKOS_INLINE_FUNCTION void setStartCoordinates(float x, float y, float z) { x0_=x; y0_=y; z0_=z; }
    KOKKOS_INLINE_FUNCTION void setEndCoordinates  (float x, float y, float z) { dirX_=x; dirY_=y; dirZ_=z; }
    KOKKOS_INLINE_FUNCTION void setSPol(float x, float y, float z) { asX_=x; asY_=y; asZ_=z; }
    KOKKOS_INLINE_FUNCTION void setPPol(float x, float y, float z) { apX_=x; apY_=y; apZ_=z; }
    KOKKOS_INLINE_FUNCTION void setEnergyKeV(float keV)  { k_ = keV * 50677300.0f; }
    KOKKOS_INLINE_FUNCTION void setFlag(bool f)           { flag_ = f; }
    KOKKOS_INLINE_FUNCTION void setIAFlag(bool f)         { iaFlag_ = f; }
    KOKKOS_INLINE_FUNCTION void setIANum(int n)           { iaNum_ = n; }
    KOKKOS_INLINE_FUNCTION void setOOBFlag(bool f)        { oobFlag_ = f; }
    KOKKOS_INLINE_FUNCTION void setNextVoxel(int v)       { nextVoxel_ = v; }
    KOKKOS_INLINE_FUNCTION void setTIn(float t)           { tIn_ = t; }
    KOKKOS_INLINE_FUNCTION void setProb(float p)           { prob_ = p; }
    KOKKOS_INLINE_FUNCTION void setAugerFlag(bool f)      { augerFlag_ = f; }
    KOKKOS_INLINE_FUNCTION void setRespawnCounter(int n)  { respawnCounter_ = n; }
    KOKKOS_INLINE_FUNCTION void raiseRespawnCounter()     { respawnCounter_++; }

    // ── Getters ───────────────────────────────────────────────────────────────
    KOKKOS_INLINE_FUNCTION float getStartX()    const { return x0_; }
    KOKKOS_INLINE_FUNCTION float getStartY()    const { return y0_; }
    KOKKOS_INLINE_FUNCTION float getStartZ()    const { return z0_; }
    KOKKOS_INLINE_FUNCTION float getDirX()      const { return dirX_; }
    KOKKOS_INLINE_FUNCTION float getDirY()      const { return dirY_; }
    KOKKOS_INLINE_FUNCTION float getDirZ()      const { return dirZ_; }
    KOKKOS_INLINE_FUNCTION float getSPolX()     const { return asX_; }
    KOKKOS_INLINE_FUNCTION float getSPolY()     const { return asY_; }
    KOKKOS_INLINE_FUNCTION float getSPolZ()     const { return asZ_; }
    KOKKOS_INLINE_FUNCTION float getPPolX()     const { return apX_; }
    KOKKOS_INLINE_FUNCTION float getPPolY()     const { return apY_; }
    KOKKOS_INLINE_FUNCTION float getPPolZ()     const { return apZ_; }
    KOKKOS_INLINE_FUNCTION float getSPhase()    const { return fS_; }
    KOKKOS_INLINE_FUNCTION float getPPhase()    const { return fP_; }
    KOKKOS_INLINE_FUNCTION int   getIndex()     const { return q_; }
    KOKKOS_INLINE_FUNCTION bool  getFlag()      const { return flag_; }
    KOKKOS_INLINE_FUNCTION float getWaveNumber()  const { return k_; }
    KOKKOS_INLINE_FUNCTION float getOpticalPath() const { return opd_; }
    KOKKOS_INLINE_FUNCTION float getEnergyKeV()   const { return k_ / 50677300.0f; }
    KOKKOS_INLINE_FUNCTION bool  getIAFlag()    const { return iaFlag_; }
    KOKKOS_INLINE_FUNCTION int   getIANum()     const { return iaNum_; }
    KOKKOS_INLINE_FUNCTION float getProb()      const { return prob_; }
    KOKKOS_INLINE_FUNCTION bool  getOOBFlag()   const { return oobFlag_; }
    KOKKOS_INLINE_FUNCTION int   getNextVoxel() const { return nextVoxel_; }
    KOKKOS_INLINE_FUNCTION float getTIn()       const { return tIn_; }
    KOKKOS_INLINE_FUNCTION bool  getAugerFlag() const { return augerFlag_; }
    KOKKOS_INLINE_FUNCTION int   getRespawnCounter() const { return respawnCounter_; }

    // ── Physics ───────────────────────────────────────────────────────────────

    /// Rotate the propagation direction by azimuth @p phi and polar angle @p theta
    /// (used to apply a sampled scattering/emission angle after an interaction).
    KOKKOS_INLINE_FUNCTION void rotate(float phi, float theta) {
        float cp = cosf(phi), sp = sinf(phi), ct = cosf(theta), st = sinf(theta);
        float dx = ct*cp*dirX_ - sp*dirY_ + st*cp*dirZ_;
        float dy = ct*sp*dirX_ + cp*dirY_ + st*sp*dirZ_;
        float dz = -st*dirX_              + ct*dirZ_;
        dirX_=dx; dirY_=dy; dirZ_=dz;
    }

    /// Map the ray from the primary-optic frame into the sample frame: translate
    /// by the focal point (@p x0, @p y0, @p z0), back off the focal distance @p d,
    /// and rotate by the optic-to-surface angle @p alpha (degrees) about X.
    KOKKOS_INLINE_FUNCTION void primaryTransform(float x0, float y0, float z0, float d, float alpha) {
        float a = alpha / 180.0f * VT_PI;
        float ca = cosf(a), sa = sinf(a);
        float nx = x0 + x0_;
        float ny = y0 - d*ca + ca*y0_ - sa*z0_;
        float nz = z0 - d*sa + sa*y0_ + ca*z0_;
        float ndx = dirX_;
        float ndy = ca*dirY_ - sa*dirZ_;
        float ndz = sa*dirY_ + ca*dirZ_;
        setStartCoordinates(nx, ny, nz);
        setEndCoordinates(ndx, ndy, ndz);
    }

    /// Map a post-sample ray into the secondary-optic frame and test acceptance:
    /// rotate by the surface-to-optic angle @p beta (degrees), project to the
    /// secondary entrance window at distance @p d, and set the interaction flag
    /// only if the ray lands within the input-window radius @p rin.
    KOKKOS_INLINE_FUNCTION void secondaryTransform(float x0, float y0, float z0, float d, float beta, float rin) {
        float b = beta / 180.0f * VT_PI;
        float cb = cosf(b), sb = sinf(b);

        if (z0_ >= 0.0f && dirZ_ < 0.0f) {
            float rx = x0_ - x0;
            float ry = cb*(y0_-y0) - sb*(z0_-z0);
            float rz = sb*(y0_-y0) + cb*(z0_-z0);
            float rdx = dirX_;
            float rdy = cb*dirY_ - sb*dirZ_;
            float rdz = sb*dirY_ + cb*dirZ_;

            float dfac = (d - ry) / rdy;
            float fx = rx + dfac*rdx;
            float fz = rz + dfac*rdz;

            if (sqrtf(fx*fx + fz*fz) < rin) {
                setStartCoordinates(fx/10000.f, 0.f, fz/10000.f);
                setEndCoordinates(rdx, rdy, rdz);
                setIAFlag(true);
            } else {
                setIAFlag(false);
            }
        } else {
            setIAFlag(false);
        }
    }

#ifndef __METAL_VERSION__
    inline void print() const {
        std::cout << "Ray " << q_ << "\t Energy: \t" << getEnergyKeV() << " keV\n";
    }
#endif
};
