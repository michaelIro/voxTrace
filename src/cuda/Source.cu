#ifndef Source_H
#define Source_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "RayGPU.cu"

// ── Simple host-side LCG RNG (no std:: — works in .cu context) ───────────────
// Avoids pulling in <random> which conflicts with CUDA headers
struct HostRNG {
    unsigned long long seed_;
    __host__ HostRNG(unsigned long long seed) : seed_(seed) {}
    __host__ float next() {
        seed_ = seed_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float)((seed_ >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    }
    // Box-Muller normal sample
    __host__ float nextNormal() {
        float u1 = fmaxf(next(), 1e-7f);
        float u2 = next();
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }
};

// ─────────────────────────────────────────────────────────────────────────────

enum class SourceType { Point, UniformDisk, Gaussian };

class Source {

    float      posX_,  posY_,  posZ_;   // source centre position
    float      dirX_,  dirY_,  dirZ_;   // mean beam direction (normalised)
    float      energyKeV_;              // photon energy
    float      sourceRadius_;           // disk radius or Gaussian sigma in cm
    float      divergence_;             // beam divergence sigma in rad
    SourceType type_;

public:

    __host__ __device__ Source() {}

    __host__ __device__ Source(
        float posX,  float posY,  float posZ,
        float dirX,  float dirY,  float dirZ,
        float energyKeV, float sourceRadius, float divergence,
        SourceType type = SourceType::Gaussian)
        : posX_(posX), posY_(posY), posZ_(posZ),
          energyKeV_(energyKeV),
          sourceRadius_(sourceRadius), divergence_(divergence),
          type_(type)
    {
        // Normalise direction
        float n = sqrtf(dirX*dirX + dirY*dirY + dirZ*dirZ);
        dirX_ = dirX/n; dirY_ = dirY/n; dirZ_ = dirZ/n;
    }

    // ── HOST ray generation ───────────────────────────────────────────────────

    __host__ RayGPU generateHost(int index, unsigned long long seed) const {
        HostRNG rng(seed ^ (unsigned long long)index * 2654435761ULL);

        float sx = 0.0f, sz = 0.0f;   // source point offset
        float dx = 0.0f, dz = 0.0f;   // direction offset

        if (type_ == SourceType::Point) {
            // no spatial extent
        }
        else if (type_ == SourceType::UniformDisk) {
            float r   = sourceRadius_ * sqrtf(rng.next());
            float phi = 2.0f * M_PI * rng.next();
            sx = r * cosf(phi);
            sz = r * sinf(phi);
        }
        else {  // Gaussian
            sx = sourceRadius_ * rng.nextNormal();
            sz = sourceRadius_ * rng.nextNormal();
        }

        dx = divergence_ * rng.nextNormal();
        dz = divergence_ * rng.nextNormal();

        return _makeRay(index, posX_+sx, posY_, posZ_+sz,
                        dirX_+dx, dirY_, dirZ_+dz);
    }

    // ── DEVICE ray generation ─────────────────────────────────────────────────

    __device__ RayGPU generateDevice(int index, curandState_t* state) const {
        float sx = 0.0f, sz = 0.0f;
        float dx = 0.0f, dz = 0.0f;

        if (type_ == SourceType::Point) {
            // no spatial extent
        }
        else if (type_ == SourceType::UniformDisk) {
            float r   = sourceRadius_ * sqrtf(curand_uniform(state));
            float phi = 2.0f * M_PI * curand_uniform(state);
            sx = r * cosf(phi);
            sz = r * sinf(phi);
        }
        else {  // Gaussian
            sx = sourceRadius_ * curand_normal(state);
            sz = sourceRadius_ * curand_normal(state);
        }

        dx = divergence_ * curand_normal(state);
        dz = divergence_ * curand_normal(state);

        return _makeRay(index, posX_+sx, posY_, posZ_+sz,
                        dirX_+dx, dirY_, dirZ_+dz);
    }

    __host__ void print() const {
        const char* names[] = {"Point", "UniformDisk", "Gaussian"};
        printf("Source: pos=(%.3f,%.3f,%.3f) E=%.2fkeV r=%.4fcm div=%.4frad type=%s\n",
               posX_, posY_, posZ_, energyKeV_,
               sourceRadius_, divergence_, names[(int)type_]);
    }

private:

    __host__ __device__ RayGPU _makeRay(int index,
        float x, float y, float z,
        float dx, float dy, float dz) const
    {
        float n = sqrtf(dx*dx + dy*dy + dz*dz);
        dx /= n; dy /= n; dz /= n;

        RayGPU ray(
            x, y, z,            // start position
            dx, dy, dz,         // direction
            1.0f, 0.0f, 0.0f,   // s-polarization
            true, 0.0f, index,  // flag, wavenumber, index
            0.0f, 0.0f, 0.0f,   // opd, phases
            0.0f, 1.0f, 0.0f,   // p-polarization
            1.0f                // probability
        );
        ray.setEnergyKeV(energyKeV_);
        return ray;
    }
};

#endif
