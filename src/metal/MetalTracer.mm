// ── MetalTracer.mm ────────────────────────────────────────────────────────────
// Objective-C++ host wrapper for the Metal compute backend.
// MTLBuffer with StorageModeShared = zero-copy unified memory (M-series).
// Compiled only on arm64 / Apple Silicon (VOXTRACE_METAL defined in Makefile).

#ifdef VOXTRACE_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstring>

#include "../core/Ray.hpp"
#include "../core/Voxel.hpp"
#include "../core/Material.hpp"
#include "../core/ChemElement.hpp"
#include "../core/Sample.hpp"
#include "../core/Source.hpp"

// ── TraceParams mirrors the Metal-side struct in Tracer.metal ─────────────────
struct TraceParams {
    float    pt[5];
    float    st[6];
    float    ox, oy, oz;
    uint32_t seed;
    uint32_t n_rays;
};

// ── MetalTracer ───────────────────────────────────────────────────────────────
// Holds the Metal device, command queue, and compiled pipeline.
// Construct once; call dispatch() per measurement point.

class MetalTracer {
    id<MTLDevice>              device_;
    id<MTLCommandQueue>        queue_;
    id<MTLComputePipelineState> pso_;

    // ── Shared (zero-copy) buffers — allocated once, reused across sim points ─
    id<MTLBuffer> buf_elements_;
    id<MTLBuffer> buf_materials_;
    id<MTLBuffer> buf_voxels_;
    id<MTLBuffer> buf_sample_;
    id<MTLBuffer> buf_source_;
    id<MTLBuffer> buf_rays_;
    id<MTLBuffer> buf_params_;

    int n_rays_ = 0;

public:

    MetalTracer() {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) { fprintf(stderr, "MetalTracer: no Metal device\n"); return; }
        queue_ = [device_ newCommandQueue];

        // Compile the .metallib at runtime (pre-built by xcrun in Makefile)
        NSError* err = nil;
        NSString* libPath = @"build/src/metal/Tracer.metallib";
        id<MTLLibrary> lib = [device_ newLibraryWithURL:[NSURL fileURLWithPath:libPath]
                                                  error:&err];
        if (!lib) {
            fprintf(stderr, "MetalTracer: library error: %s\n",
                    [[err localizedDescription] UTF8String]);
            return;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"traceNewBeam"];
        pso_ = [device_ newComputePipelineStateWithFunction:fn error:&err];
        if (!pso_)
            fprintf(stderr, "MetalTracer: PSO error: %s\n",
                    [[err localizedDescription] UTF8String]);
    }

    // Allocate / upload grid data (call once before dispatch loop)
    void uploadGrid(const ChemElement* elements, int n_el,
                    const Material*    materials, int n_mat,
                    const Voxel*       voxels,   int n_vox,
                    const Sample&      sample,
                    const Source&      source,
                    int n_rays)
    {
        n_rays_ = n_rays;
        auto _buf = [&](const void* src, size_t sz) {
            id<MTLBuffer> b = [device_ newBufferWithBytes:src
                                                   length:sz
                                                  options:MTLResourceStorageModeShared];
            return b;
        };
        buf_elements_  = _buf(elements,  sizeof(ChemElement) * n_el);
        buf_materials_ = _buf(materials, sizeof(Material)    * n_mat);
        buf_voxels_    = _buf(voxels,    sizeof(Voxel)       * n_vox);
        buf_sample_    = _buf(&sample,   sizeof(Sample));
        buf_source_    = _buf(&source,   sizeof(Source));
        buf_rays_      = [device_ newBufferWithLength: sizeof(Ray) * n_rays
                                             options: MTLResourceStorageModeShared];
        buf_params_    = [device_ newBufferWithLength: sizeof(TraceParams)
                                             options: MTLResourceStorageModeShared];
    }

    // Dispatch one measurement point; returns pointer to result ray array
    const Ray* dispatch(float ox, float oy, float oz,
                        const float pt[5], const float st[6],
                        uint32_t seed = 775289u)
    {
        if (!pso_) return nullptr;

        // Fill params buffer
        TraceParams* p = (TraceParams*)[buf_params_ contents];
        memcpy(p->pt, pt, 5 * sizeof(float));
        memcpy(p->st, st, 6 * sizeof(float));
        p->ox     = ox; p->oy = oy; p->oz = oz;
        p->seed   = seed;
        p->n_rays = (uint32_t)n_rays_;

        // Zero ray buffer
        memset([buf_rays_ contents], 0, sizeof(Ray) * n_rays_);

        id<MTLCommandBuffer>  cmd = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState: pso_];
        [enc setBuffer:buf_rays_      offset:0 atIndex:0];
        [enc setBuffer:buf_voxels_    offset:0 atIndex:1];
        [enc setBuffer:buf_materials_ offset:0 atIndex:2];
        [enc setBuffer:buf_elements_  offset:0 atIndex:3];
        [enc setBuffer:buf_sample_    offset:0 atIndex:4];
        [enc setBuffer:buf_source_    offset:0 atIndex:5];
        [enc setBuffer:buf_params_    offset:0 atIndex:6];

        // One thread per ray; threadgroup size = pipeline max
        NSUInteger tgSize  = pso_.maxTotalThreadsPerThreadgroup;
        MTLSize    threads = MTLSizeMake((NSUInteger)n_rays_, 1, 1);
        MTLSize    tg      = MTLSizeMake(tgSize, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        return (const Ray*)[buf_rays_ contents];
    }
};

#endif // VOXTRACE_METAL
