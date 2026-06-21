#pragma once
/**
 * @file Voxel.hpp
 * @brief Axis-aligned sample cell with ray–box intersection and neighbour indices.
 */
#include "Platform.hpp"
#include "Ray.hpp"

#ifndef __METAL_VERSION__
    #include <cfloat>
#endif

/**
 * @brief Axis-aligned box cell of the sample grid: ray–box intersection + neighbour links.
 *
 * `intersect(Ray&)` is the geometric step of the voxel walk: it computes the
 * path length of the ray through this cell and, as a side effect, sets the
 * ray's exit face so the dispatch loop can hop to the next voxel. The 27-entry
 * neighbour table `nn_` stores the flat indices of all 26 neighbours plus self
 * (index 13); `-1` marks an out-of-bounds neighbour. Each voxel also stores the
 * index of its @ref Material.
 */
class Voxel {
    float x0_, y0_, z0_;   // min corner
    float x1_, y1_, z1_;   // extents (size, not max corner — same as original)
    int   mat_idx_ = -1;   // index into the shared Material array (-1 = OOB voxel)
    int   nn_[27]  = {};   // neighbor indices; -1 = OOB

public:

    KOKKOS_INLINE_FUNCTION Voxel() {}

    KOKKOS_INLINE_FUNCTION Voxel(float x0, float y0, float z0,
                                  float x1, float y1, float z1, int mat_idx)
        : x0_(x0), y0_(y0), z0_(z0),
          x1_(x1), y1_(y1), z1_(z1),
          mat_idx_(mat_idx) {}

    inline void setNN(const int indices[27]) {
        for (int i = 0; i < 27; ++i) nn_[i] = indices[i];
    }

    KOKKOS_INLINE_FUNCTION int getNN(int dir)       const VT_DEVICE_METH { return nn_[dir]; }
    KOKKOS_INLINE_FUNCTION int getMaterialIdx()     const VT_DEVICE_METH { return mat_idx_; }
    KOKKOS_INLINE_FUNCTION bool isOOB()             const VT_DEVICE_METH { return mat_idx_ < 0; }

    KOKKOS_INLINE_FUNCTION float getX0() const VT_DEVICE_METH { return x0_; }
    KOKKOS_INLINE_FUNCTION float getY0() const VT_DEVICE_METH { return y0_; }
    KOKKOS_INLINE_FUNCTION float getZ0() const VT_DEVICE_METH { return z0_; }
    KOKKOS_INLINE_FUNCTION float getX1() const VT_DEVICE_METH { return x0_ + x1_; }
    KOKKOS_INLINE_FUNCTION float getY1() const VT_DEVICE_METH { return y0_ + y1_; }
    KOKKOS_INLINE_FUNCTION float getZ1() const VT_DEVICE_METH { return z0_ + z1_; }

    /// Ray–box slab intersection. Sets `ray.nextVoxel` (the exit face → neighbour
    /// index) and `ray.tIn` (entry distance) in place, and returns the path
    /// length of the ray through this voxel (used for the attenuation draw).
    KOKKOS_INLINE_FUNCTION float intersect(VT_THREAD Ray& ray) const VT_DEVICE_METH {
        float t0x, t1x, t0y, t1y, t0z, t1z;
        bool  xDir = true, yDir = true, zDir = true;

        if (ray.getDirX() != 0.f) {
            t0x = (getX0() - ray.getStartX()) / ray.getDirX();
            t1x = (getX1() - ray.getStartX()) / ray.getDirX();
            if (t0x > t1x) { float tmp=t0x; t0x=t1x; t1x=tmp; xDir=false; }
        } else { t0x=FLT_MIN; t1x=FLT_MAX; }

        if (ray.getDirY() != 0.f) {
            t0y = (getY0() - ray.getStartY()) / ray.getDirY();
            t1y = (getY1() - ray.getStartY()) / ray.getDirY();
            if (t0y > t1y) { float tmp=t0y; t0y=t1y; t1y=tmp; yDir=false; }
        } else { t0y=FLT_MIN; t1y=FLT_MAX; }

        if (ray.getDirZ() != 0.f) {
            t0z = (getZ0() - ray.getStartZ()) / ray.getDirZ();
            t1z = (getZ1() - ray.getStartZ()) / ray.getDirZ();
            if (t0z > t1z) { float tmp=t0z; t0z=t1z; t1z=tmp; zDir=false; }
        } else { t0z=FLT_MIN; t1z=FLT_MAX; }

        float t0_max, t1_min;

        // If ray starts inside the voxel, entry distance = 0
        if (ray.getStartX() >= x0_ && ray.getStartX() <= x1_ &&
            ray.getStartY() >= y0_ && ray.getStartY() <= y1_ &&
            ray.getStartZ() >= z0_ && ray.getStartZ() <= z1_) {
            t0_max = 0.0f;
        } else {
            t0_max = fmaxf(fmaxf(t0x, t0y), t0z);
        }
        t1_min = fminf(fminf(t1x, t1y), t1z);

        // Determine which face the ray exits through → set nextVoxel
        if      (t1_min==t1z && t1_min!=t1y && t1_min!=t1x) ray.setNextVoxel(zDir ? 22 : 4);
        else if (t1_min!=t1z && t1_min==t1y && t1_min!=t1x) ray.setNextVoxel(yDir ? 16 : 10);
        else if (t1_min!=t1z && t1_min!=t1y && t1_min==t1x) ray.setNextVoxel(xDir ? 14 : 12);
        else if (t1_min==t1z && t1_min==t1y && t1_min!=t1x)
            ray.setNextVoxel(zDir ? (yDir?25:19) : (yDir?1:7));
        else if (t1_min==t1z && t1_min!=t1y && t1_min==t1x)
            ray.setNextVoxel(zDir ? (xDir?23:21) : (xDir?3:5));
        else if (t1_min!=t1z && t1_min==t1y && t1_min==t1x)
            ray.setNextVoxel(yDir ? (xDir?17:15) : (xDir?11:9));
        else if (t1_min==t1z && t1_min==t1y && t1_min==t1x) {
            if (zDir) ray.setNextVoxel(yDir ? (xDir?26:24) : (xDir?20:18));
            else      ray.setNextVoxel(yDir ? (xDir?8:6)   : (xDir?2:0));
        }

        ray.setTIn(t0_max);
        return t1_min - t0_max;
    }
};
