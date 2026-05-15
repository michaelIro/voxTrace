#pragma once
#include "Platform.hpp"
#include "Ray.hpp"

// ── Sample ────────────────────────────────────────────────────────────────────
// 3D voxel grid descriptor. Stores only grid parameters (no pointers).
// Voxel lookup is pure arithmetic — compatible with Kokkos and Metal MSL.
// The actual Voxel/Material arrays live in the Tracer dispatch layer.

class Sample {
    float x_, y_, z_;          // grid origin
    float xLV_, yLV_, zLV_;    // voxel extents
    int   xN_, yN_, zN_;       // voxel counts per axis

public:

    KOKKOS_INLINE_FUNCTION Sample() {}

    KOKKOS_INLINE_FUNCTION Sample(float x,  float y,  float z,
                                   float xL, float yL, float zL,
                                   float xLV, float yLV, float zLV,
                                   int xN, int yN, int zN)
        : x_(x), y_(y), z_(z),
          xLV_(xLV), yLV_(yLV), zLV_(zLV),
          xN_(xN),   yN_(yN),   zN_(zN) {}

    // Flat linear index; -1 = out of bounds
    KOKKOS_INLINE_FUNCTION int getVoxelIdx(float x, float y, float z) const {
        int xi = (int)floorf((x - x_) / xLV_);
        int yi = (int)floorf((y - y_) / yLV_);
        int zi = (int)floorf((z - z_) / zLV_);
        if (xi<0||xi>=xN_||yi<0||yi>=yN_||zi<0||zi>=zN_) return -1;
        return xi * yN_ * zN_ + yi * zN_ + zi;
    }

    // Entry voxel for a new ray (propagates to sample boundary if needed)
    KOKKOS_INLINE_FUNCTION int findStartVoxelIdx(const Ray& ray) const {
        float xi = ray.getStartX(), yi = ray.getStartY(), zi = ray.getStartZ();

        // Primary ray: propagate down to sample top face (z = z_)
        if (zi < z_ && ray.getDirZ() != 0.f) {
            float t = (z_ - zi) / ray.getDirZ();
            xi += t * ray.getDirX();
            yi += t * ray.getDirY();
            zi  = z_;
        }

        // Side-entry: propagate to y = y_ face
        if ((xi < x_ || yi < y_ || zi < z_) && ray.getDirY() != 0.f) {
            float t = (y_ - ray.getStartY()) / ray.getDirY();
            xi = ray.getStartX() + t * ray.getDirX();
            yi = ray.getStartY() + t * ray.getDirY();
            zi = ray.getStartZ() + t * ray.getDirZ();
        }

        return getVoxelIdx(xi, yi, zi);
    }

    KOKKOS_INLINE_FUNCTION int voxN() const { return xN_ * yN_ * zN_; }
    KOKKOS_INLINE_FUNCTION float getZPos() const { return z_; }
};
