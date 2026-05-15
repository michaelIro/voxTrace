#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <filesystem>
#include <cstdio>
#include <ctime>
#include <cstring>

#include "Tracer.hpp"
#include "Source.hpp"
#include "Sample.hpp"

// ── traceForward ──────────────────────────────────────────────────────────────
// Single Monte-Carlo step: absorption / Rayleigh / Compton.
// Identical physics to original TracerGPU::traceForward.

KOKKOS_INLINE_FUNCTION
void Tracer::traceForward(Ray& ray, const Voxel& voxel,
                           const Material* materials,
                           const ChemElement* elements,
                           RNG& rng)
{
    const Material& mat = materials[voxel.getMaterialIdx()];
    float e       = ray.getEnergyKeV();
    float muLin   = mat.CS_Tot_Lin(e, elements) / 10000.f;
    float len     = voxel.intersect(const_cast<Ray&>(ray));  // also sets nextVoxel/tIn

    // No interaction in this voxel?
    if (expf(-muLin * len) >= rng.frand()) return;

    // Select element and interaction type
    int   elemIdx = mat.getInteractingElementIdx(e, rng.frand(), elements);
    const ChemElement& elem = elements[elemIdx];
    int   type    = elem.getInteractionType(e, rng.frand());

    if (type == 0) {                                        // Photoelectric
        int shell = elem.getExcitedShell(e, rng.frand());
        if (rng.frand() < elem.Fluor_Y(shell)) {           // Fluorescence
            float l      = len * rng.frand() + ray.getTIn();
            int   line   = elem.getTransition(shell, rng.frand());
            float phi    = 2.f * VT_PI * rng.frand();
            float theta  = acosf(2.f * rng.frand() - 1.f);
            ray.setStartCoordinates(ray.getStartX() + ray.getDirX() * l,
                                    ray.getStartY() + ray.getDirY() * l,
                                    ray.getStartZ() + ray.getDirZ() * l);
            ray.rotate(phi, theta);
            ray.setEnergyKeV(elem.Line_Energy(line));
        } else {                                            // Auger
            ray.setAugerFlag(true);
        }
    } else if (type == 1) {                                 // Rayleigh
        float l     = len * rng.frand() + ray.getTIn();
        float theta = elem.getThetaRayl(e, rng.frand());
        float phi   = 2.f * VT_PI * rng.frand();
        ray.setStartCoordinates(ray.getStartX() + ray.getDirX() * l,
                                ray.getStartY() + ray.getDirY() * l,
                                ray.getStartZ() + ray.getDirZ() * l);
        ray.rotate(phi, theta);
    } else {                                                // Compton
        float l     = len * rng.frand() + ray.getTIn();
        float theta = elem.getThetaCompt(e, rng.frand());
        float phi   = 2.f * VT_PI * rng.frand();
        ray.setStartCoordinates(ray.getStartX() + ray.getDirX() * l,
                                ray.getStartY() + ray.getDirY() * l,
                                ray.getStartZ() + ray.getDirZ() * l);
        ray.rotate(phi, theta);
        ray.setEnergyKeV(elem.getComptEnergy(ray.getEnergyKeV(), theta));
    }

    ray.setIANum(ray.getIANum() + 1);
    ray.setIAFlag(true);
}

// ── callTraceNewBeam ──────────────────────────────────────────────────────────

void Tracer::callTraceNewBeam(SimulationParameter& simp) {

    clock_t t0 = clock();

    const int N   = simp.getNumRays();
    const float* ss  = simp.getSampleStart();
    const float* sl  = simp.getSampleLength();
    const float* slv = simp.getSampleVoxelLength();

    float x_=ss[0], y_=ss[1], z_=ss[2];
    float xL_=sl[0], yL_=sl[1], zL_=sl[2];

    int xN_ = (int)(xL_ / slv[0]);
    int yN_ = (int)(yL_ / slv[1]);
    int zN_ = (int)(zL_ / slv[2]);
    float xLV_ = xL_ / (float)xN_;
    float yLV_ = yL_ / (float)yN_;
    float zLV_ = zL_ / (float)zN_;
    int voxN = xN_ * yN_ * zN_;

    int n_el = (int)simp.getUniqueElements().size();

    // ── Host-side Views ───────────────────────────────────────────────────────
    Kokkos::View<ChemElement*> d_elements("elements", n_el);
    Kokkos::View<Material*>    d_materials("materials", voxN);
    Kokkos::View<Voxel*>       d_voxels("voxels", voxN);
    Kokkos::View<Ray*>         d_rays("rays", N);

    auto h_elements  = Kokkos::create_mirror_view(d_elements);
    auto h_materials = Kokkos::create_mirror_view(d_materials);
    auto h_voxels    = Kokkos::create_mirror_view(d_voxels);
    auto h_rays      = Kokkos::create_mirror_view(d_rays);

    // Discretise unique elements
    for (int i = 0; i < n_el; ++i)
        h_elements(i) = ChemElement(simp.getUniqueElements()[i]);

    // Build materials and voxels from material points
    for (auto& pt : simp.getMaterialPoints()) {
        int i = (int)pt.x, j = (int)pt.y, k = (int)pt.z;
        int vidx = i * yN_ * zN_ + j * zN_ + k;

        float w[MAX_ELEMENTS] = {};
        for (int l = 0; l < n_el; ++l)
            for (int m = 0; m < pt.n_elements; ++m)
                if (l == pt.elements[m]) { w[l] = pt.mass_fractions[m]; break; }

        h_materials(vidx) = Material(n_el, w, h_elements.data());
        h_voxels(vidx)    = Voxel(x_ + i*xLV_, y_ + j*yLV_, z_ + k*zLV_,
                                   xLV_, yLV_, zLV_, vidx);
    }

    // Build 27-neighbor connectivity (same loop order as original)
    for (int i = 0; i < xN_; ++i)
    for (int j = 0; j < yN_; ++j)
    for (int k = 0; k < zN_; ++k) {
        int nn[27]; int cnt = 0;
        for (int l = -1; l < 2; ++l)
        for (int m = -1; m < 2; ++m)
        for (int n = -1; n < 2; ++n) {
            int ni=i+n, nj=j+m, nk=k+l;
            nn[cnt++] = (ni<0||ni>=xN_||nj<0||nj>=yN_||nk<0||nk>=zN_)
                        ? -1 : ni*yN_*zN_ + nj*zN_ + nk;
        }
        h_voxels(i*yN_*zN_ + j*zN_ + k).setNN(nn);
    }

    Kokkos::deep_copy(d_elements,  h_elements);
    Kokkos::deep_copy(d_materials, h_materials);
    Kokkos::deep_copy(d_voxels,    h_voxels);

    // Grid descriptor (pure value, captured by lambda)
    Sample sample(x_, y_, z_, xL_, yL_, zL_, xLV_, yLV_, zLV_, xN_, yN_, zN_);

    // Source from primary capillary geometry
    const float* pg = simp.getPrimCapGeom();   // [r_out, f, r_f, energy_keV]
    Source source = Source::fromCapGeom(pg[0], pg[1], pg[2], pg[3]);

    // Flatten transform params for lambda capture
    const float* pt_ = simp.getPrimTransParam();
    const float* st_ = simp.getSecTransParam();
    float pt0=pt_[0],pt1=pt_[1],pt2=pt_[2],pt3=pt_[3],pt4=pt_[4];
    float st0=st_[0],st1=st_[1],st2=st_[2],st3=st_[3],st4=st_[4],st5=st_[5];

    printf("Init: %.2f s\n", (double)(clock()-t0)/CLOCKS_PER_SEC);

    // ── Per-measurement-point loop ────────────────────────────────────────────
    const auto& mpts = simp.getMeasurementPoints();
    for (int si = 0; si < (int)mpts.size(); ++si) {

        float ox = mpts[si][0], oy = mpts[si][1], oz = mpts[si][2];

        // Init ray array
        for (int j = 0; j < N; ++j) h_rays(j) = Ray();
        Kokkos::deep_copy(d_rays, h_rays);

        // Raw pointers — valid on active device (unified memory or device view)
        ChemElement* elems   = d_elements.data();
        Material*    mats    = d_materials.data();
        Voxel*       voxels  = d_voxels.data();
        Ray*         rays    = d_rays.data();

        clock_t t1 = clock();
        printf("Tracing (x=%.1f y=%.1f z=%.1f) ... ", ox, oy, oz);

        Kokkos::parallel_for("traceNewBeam", N,
        KOKKOS_LAMBDA(int idx) {

            Ray ray;
            // Unique seed per ray — preserves statistical independence
            RNG rng(775289ULL ^ (uint64_t)idx * 6364136223846793005ULL);

            do {
                int savedCnt = ray.getRespawnCounter() + 1;
                ray = source.generate(idx, rng);
                ray.setRespawnCounter(savedCnt);

                ray.primaryTransform(pt0, pt1, pt2, pt3, pt4);
                ray.setStartCoordinates(ray.getStartX() + ox,
                                        ray.getStartY() + oy,
                                        ray.getStartZ() + oz);
                ray.setIAFlag(false);
                ray.setAugerFlag(false);
                ray.setOOBFlag(false);

                int voxIdx = sample.findStartVoxelIdx(ray);

                for (;;) {
                    if (voxIdx < 0 || ray.getEnergyKeV() < 1.f || ray.getEnergyKeV() > 19.f) {
                        ray.setOOBFlag(true); break;
                    }
                    Tracer::traceForward(ray, voxels[voxIdx], mats, elems, rng);
                    if (ray.getOOBFlag() || ray.getAugerFlag()) break;
                    voxIdx = voxels[voxIdx].getNN(ray.getNextVoxel());
                }

                if (!ray.getAugerFlag() && ray.getEnergyKeV() < 15.f) {
                    ray.setStartCoordinates(ray.getStartX() - ox,
                                            ray.getStartY() - oy,
                                            ray.getStartZ() - oz);
                    ray.secondaryTransform(st0, st1, st2, st3, st4, st5);
                } else {
                    ray.setIAFlag(false);
                }
            } while (!ray.getIAFlag());

            rays[idx] = ray;
        });

        Kokkos::fence();
        printf("%.2f s\n", (double)(clock()-t1)/CLOCKS_PER_SEC);

        // Copy back to host
        Kokkos::deep_copy(h_rays, d_rays);

        // Save results
        char buf[64];
        std::sprintf(buf, "(x-%.1f--y%.1f--z-%.1f)", ox, oy, oz);
        std::string outPath = simp.getDirectory() + "/post-sample/ps-"
                            + std::string(buf) + ".h5";

        arma::Mat<double> out(N, 21);
        int totalRespawns = 0;
        for (int i = 0; i < N; ++i) {
            const Ray& r = h_rays(i);
            out.row(i) = arma::rowvec({
                r.getStartX(), r.getStartY(), r.getStartZ(),
                r.getDirX(),   r.getDirY(),   r.getDirZ(),
                r.getSPolX(),  r.getSPolY(),  r.getSPolZ(),
                (double)r.getFlag(),    r.getWaveNumber(), (double)r.getIndex(),
                r.getOpticalPath(),     r.getSPhase(),     r.getPPhase(),
                r.getPPolX(),  r.getPPolY(),  r.getPPolZ(),
                r.getProb(), (double)r.getIANum(), (double)r.getRespawnCounter()
            });
            totalRespawns += r.getRespawnCounter();
        }
        out.save(arma::hdf5_name(outPath, "my_data"));
        printf("Generated Rays: %d\n", totalRespawns);
    }
}

// ── callTracePreBeam ──────────────────────────────────────────────────────────
// Processes a pre-computed incident beam (HDF5 input), traces through a fixed
// sample geometry, and saves surviving rays.

void Tracer::callTracePreBeam() {

    static constexpr int n_el = 6;
    static constexpr float x_=0.f, y_=0.f, z_=0.f;
    static constexpr float xL_=600000.f, yL_=600000.f, zL_=3000.f;
    static constexpr float xLV_=60000.f, yLV_=60000.f, zLV_=10.f;

    int xN_ = (int)(xL_/xLV_), yN_ = (int)(yL_/yLV_), zN_ = (int)(zL_/zLV_);
    float xLVf = xL_/(float)xN_, yLVf = yL_/(float)yN_, zLVf = zL_/(float)zN_;
    int voxN = xN_*yN_*zN_;

    Kokkos::View<ChemElement*> d_elements("elements", n_el);
    Kokkos::View<Material*>    d_materials("materials", voxN);
    Kokkos::View<Voxel*>       d_voxels("voxels", voxN);

    auto h_el  = Kokkos::create_mirror_view(d_elements);
    auto h_mat = Kokkos::create_mirror_view(d_materials);
    auto h_vox = Kokkos::create_mirror_view(d_voxels);

    int Zs[n_el]    = {29, 26, 82, 28, 50, 30};
    float wts[n_el] = {0.6119f, 0.0004f, 0.0019f, 0.0010f, 0.0107f, 0.3741f};
    for (int i = 0; i < n_el; ++i) h_el(i) = ChemElement(Zs[i]);

    for (int i=0; i<xN_; ++i) for (int j=0; j<yN_; ++j) for (int k=0; k<zN_; ++k) {
        int vidx = i*yN_*zN_ + j*zN_ + k;
        h_mat(vidx) = Material(n_el, wts, h_el.data());
        h_vox(vidx) = Voxel(x_+i*xLVf, y_+j*yLVf, z_+k*zLVf, xLVf, yLVf, zLVf, vidx);
    }
    for (int i=0; i<xN_; ++i) for (int j=0; j<yN_; ++j) for (int k=0; k<zN_; ++k) {
        int nn[27]; int cnt=0;
        for (int l=-1;l<2;++l) for (int m=-1;m<2;++m) for (int n=-1;n<2;++n) {
            int ni=i+n, nj=j+m, nk=k+l;
            nn[cnt++] = (ni<0||ni>=xN_||nj<0||nj>=yN_||nk<0||nk>=zN_)
                        ? -1 : ni*yN_*zN_+nj*zN_+nk;
        }
        h_vox(i*yN_*zN_+j*zN_+k).setNN(nn);
    }
    Kokkos::deep_copy(d_elements,  h_el);
    Kokkos::deep_copy(d_materials, h_mat);
    Kokkos::deep_copy(d_voxels,    h_vox);

    Sample sample(x_, y_, z_, xL_, yL_, zL_, xLVf, yLVf, zLVf, xN_, yN_, zN_);

    const std::string inPath  = "/tank/data/";
    const std::string outBase = "/media/miro/Data/";

    for (const auto& entry : std::filesystem::directory_iterator(inPath)) {
        std::string pathname = entry.path().string();
        std::string outPath  = outBase + entry.path().filename().string() + "-pos-5.h5";
        if (std::filesystem::exists(outPath)) { printf("Exists: skipping %s\n", outPath.c_str()); continue; }

        arma::Mat<double> beam;
        beam.load(arma::hdf5_name(pathname, "my_data"));
        int N = (int)beam.n_rows;

        Kokkos::View<Ray*> d_rays("rays", N);
        auto h_rays = Kokkos::create_mirror_view(d_rays);
        for (int i = 0; i < N; ++i) {
            h_rays(i) = Ray(
                (float)beam(i,0),(float)beam(i,1),(float)beam(i,2),
                (float)beam(i,3),(float)beam(i,4),(float)beam(i,5),
                (float)beam(i,6),(float)beam(i,7),(float)beam(i,8),
                (bool) beam(i,9),(float)beam(i,10),(int)beam(i,11),
                (float)beam(i,12),(float)beam(i,13),(float)beam(i,14),
                (float)beam(i,15),(float)beam(i,16),(float)beam(i,17),
                (float)beam(i,18));
        }
        Kokkos::deep_copy(d_rays, h_rays);

        ChemElement* elems  = d_elements.data();
        Material*    mats   = d_materials.data();
        Voxel*       voxels = d_voxels.data();
        Ray*         rays   = d_rays.data();

        clock_t t0 = clock();
        Kokkos::parallel_for("tracePreBeam", N,
        KOKKOS_LAMBDA(int idx) {
            Ray& ray = rays[idx];
            ray.primaryTransform(300000.f, 300000.f, 0.f, 5100.f, 45.f);

            RNG rng(775289ULL ^ (uint64_t)idx * 6364136223846793005ULL);
            int voxIdx = sample.findStartVoxelIdx(ray);
            for (;;) {
                if (voxIdx < 0 || ray.getEnergyKeV() < 1.f || ray.getEnergyKeV() > 19.f) {
                    ray.setOOBFlag(true); break;
                }
                Tracer::traceForward(ray, voxels[voxIdx], mats, elems, rng);
                if (ray.getOOBFlag() || ray.getAugerFlag()) break;
                voxIdx = voxels[voxIdx].getNN(ray.getNextVoxel());
            }
            ray.secondaryTransform(300000.f, 300000.f, 0.f, 4900.f, 45.f, 950.f);
        });
        Kokkos::fence();
        printf("Traced %s in %.2f s\n", pathname.c_str(),
               (double)(clock()-t0)/CLOCKS_PER_SEC);

        Kokkos::deep_copy(h_rays, d_rays);
        int success = 0;
        for (int i = 0; i < N; ++i) if (h_rays(i).getIAFlag()) ++success;

        arma::Mat<double> out(success + 1, 19);
        success = 0;
        for (int i = 0; i < N; ++i) {
            if (!h_rays(i).getIAFlag()) continue;
            const Ray& r = h_rays(i);
            out.row(success++) = arma::rowvec({
                r.getStartX(), r.getStartY(), r.getStartZ(),
                r.getDirX(),   r.getDirY(),   r.getDirZ(),
                r.getSPolX(),  r.getSPolY(),  r.getSPolZ(),
                (double)r.getFlag(), r.getWaveNumber(), (double)r.getIndex(),
                r.getOpticalPath(),  r.getSPhase(),     r.getPPhase(),
                r.getPPolX(),  r.getPPolY(),  r.getPPolZ(), r.getProb()
            });
        }
        out.save(arma::hdf5_name(outPath, "my_data"));
        printf("Saved %d rays to %s\n", success, outPath.c_str());
    }
}
