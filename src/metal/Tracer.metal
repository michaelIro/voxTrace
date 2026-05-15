// ── Tracer.metal ──────────────────────────────────────────────────────────────
// Metal Shading Language kernel for M-series GPU.
// The physics headers are included directly — MSL is C++14-based and the
// shared headers are pointer-free / fixed-size-array-only for this reason.

#include <metal_stdlib>
using namespace metal;

// Shim: make shared headers compile under MSL
#define KOKKOS_INLINE_FUNCTION inline
#define KOKKOS_FUNCTION        inline
static constexpr float VT_PI = 3.14159265358979323846f;
#define __METAL_VERSION__ 1   // guard std:: / XRayLib includes in headers

// Pull in the shared physics headers
#include "../core/RNG.hpp"
#include "../core/Ray.hpp"
#include "../core/ChemElement.hpp"
#include "../core/Material.hpp"
#include "../core/Voxel.hpp"
#include "../core/Sample.hpp"
#include "../core/Source.hpp"

// ── Packed parameter struct (matches MetalTracer.mm layout) ──────────────────
struct TraceParams {
    float pt[5];    // primaryTransform params
    float st[6];    // secondaryTransform params
    float ox, oy, oz;
    uint32_t seed;
    uint32_t n_rays;
};

// ── traceForward (identical physics to Tracer.cpp) ───────────────────────────
inline void traceForward(thread Ray& ray,
                         const device Voxel* voxels,
                         const device Material* materials,
                         const device ChemElement* elements,
                         thread RNG& rng)
{
    const device Material& mat = materials[voxels[0].getMaterialIdx()]; // placeholder
    // We need the actual voxel's material — but the current voxel is identified by index.
    // This function is called with the voxel index: see kernel below.
}

// ── Main trace kernel ─────────────────────────────────────────────────────────
kernel void traceNewBeam(
    device       Ray*          rays        [[buffer(0)]],
    const device Voxel*        voxels      [[buffer(1)]],
    const device Material*     materials   [[buffer(2)]],
    const device ChemElement*  elements    [[buffer(3)]],
    constant     Sample&       sample      [[buffer(4)]],
    constant     Source&       source      [[buffer(5)]],
    constant     TraceParams&  params      [[buffer(6)]],
    uint                       idx         [[thread_position_in_grid]])
{
    Ray ray;
    RNG rng((uint64_t)params.seed ^ ((uint64_t)idx * 6364136223846793005ULL));

    do {
        int savedCnt = ray.getRespawnCounter() + 1;
        ray = source.generate((int)idx, rng);
        ray.setRespawnCounter(savedCnt);

        ray.primaryTransform(params.pt[0], params.pt[1], params.pt[2],
                             params.pt[3], params.pt[4]);
        ray.setStartCoordinates(ray.getStartX() + params.ox,
                                ray.getStartY() + params.oy,
                                ray.getStartZ() + params.oz);
        ray.setIAFlag(false);
        ray.setAugerFlag(false);
        ray.setOOBFlag(false);

        int voxIdx = sample.findStartVoxelIdx(ray);

        for (;;) {
            if (voxIdx < 0 || ray.getEnergyKeV() < 1.f || ray.getEnergyKeV() > 19.f) {
                ray.setOOBFlag(true); break;
            }
            // ── traceForward (inlined for Metal) ─────────────────────────────
            {
                const device Material& mat = materials[voxels[voxIdx].getMaterialIdx()];
                float e     = ray.getEnergyKeV();
                float muLin = mat.CS_Tot_Lin(e, elements) / 10000.f;
                float len   = voxels[voxIdx].intersect(ray);

                if (exp(-muLin * len) < rng.frand()) {
                    int ei    = mat.getInteractingElementIdx(e, rng.frand(), elements);
                    const device ChemElement& elem = elements[ei];
                    int type  = elem.getInteractionType(e, rng.frand());

                    if (type == 0) {
                        int shell = elem.getExcitedShell(e, rng.frand());
                        if (rng.frand() < elem.Fluor_Y(shell)) {
                            float l   = len * rng.frand() + ray.getTIn();
                            int  line = elem.getTransition(shell, rng.frand());
                            float phi   = 2.f * VT_PI * rng.frand();
                            float theta = acos(2.f * rng.frand() - 1.f);
                            ray.setStartCoordinates(ray.getStartX()+ray.getDirX()*l,
                                                    ray.getStartY()+ray.getDirY()*l,
                                                    ray.getStartZ()+ray.getDirZ()*l);
                            ray.rotate(phi, theta);
                            ray.setEnergyKeV(elem.Line_Energy(line));
                        } else {
                            ray.setAugerFlag(true);
                        }
                    } else if (type == 1) {
                        float l     = len * rng.frand() + ray.getTIn();
                        float theta = elem.getThetaRayl(e, rng.frand());
                        float phi   = 2.f * VT_PI * rng.frand();
                        ray.setStartCoordinates(ray.getStartX()+ray.getDirX()*l,
                                                ray.getStartY()+ray.getDirY()*l,
                                                ray.getStartZ()+ray.getDirZ()*l);
                        ray.rotate(phi, theta);
                    } else {
                        float l     = len * rng.frand() + ray.getTIn();
                        float theta = elem.getThetaCompt(e, rng.frand());
                        float phi   = 2.f * VT_PI * rng.frand();
                        ray.setStartCoordinates(ray.getStartX()+ray.getDirX()*l,
                                                ray.getStartY()+ray.getDirY()*l,
                                                ray.getStartZ()+ray.getDirZ()*l);
                        ray.rotate(phi, theta);
                        ray.setEnergyKeV(elem.getComptEnergy(ray.getEnergyKeV(), theta));
                    }
                    ray.setIANum(ray.getIANum() + 1);
                    ray.setIAFlag(true);
                }
            }
            if (ray.getOOBFlag() || ray.getAugerFlag()) break;
            voxIdx = voxels[voxIdx].getNN(ray.getNextVoxel());
        }

        if (!ray.getAugerFlag() && ray.getEnergyKeV() < 15.f) {
            ray.setStartCoordinates(ray.getStartX() - params.ox,
                                    ray.getStartY() - params.oy,
                                    ray.getStartZ() - params.oz);
            ray.secondaryTransform(params.st[0], params.st[1], params.st[2],
                                   params.st[3], params.st[4], params.st[5]);
        } else {
            ray.setIAFlag(false);
        }
    } while (!ray.getIAFlag());

    rays[idx] = ray;
}
