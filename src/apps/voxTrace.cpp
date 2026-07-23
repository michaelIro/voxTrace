// voxTrace — the configurable µXRF beamline simulator.
//
// One executable, no hardcoded experiment: a simulation directory describes
// the instrument (Primary_/Secondary_Polycapillary.txt, Source.txt,
// Placement.txt, Sample.txt, Materials.txt, Simulation.txt — legacy
// Polycapillary.txt/Capillaries.txt still load) and the run (Setup.txt —
// statistics, physics
// switches like variance_reduction/polarization, loss, optimizer, detector
// response, and WHICH STAGES ARE MOUNTED):
//
//   chain = source,primary,sample,secondary,detector
//
// The chain decides the task:
//   source[,primary]                       → beam characterisation: exit rays,
//                                            transmission, divergence, focal spot
//   source[,primary],sample[,secondary],detector
//                                          → spectra at every scan position;
//     with fit=1                           → additionally optimize the per-voxel
//                                            sample weights so the simulated
//                                            spectra match the measured ones
//                                            (measured=<csv> loads real data;
//                                            without it a synthetic phantom
//                                            measurement is generated)
//
// Without a secondary optic the emitted photons are collected by a bare
// detector aperture (det_distance_cm / det_radius_cm) — classic scanning µXRF
// instead of confocal. Without a primary optic the bare parallel source beam
// hits the sample directly.
//
// Usage:  ./build/src/voxTrace [sim-dir] [key=value ...]
//   ./build/src/voxTrace test-data/simulation/beam-pc236
//   ./build/src/voxTrace test-data/simulation/nist-1107-recon loss=1
//   ./build/src/voxTrace test-data/simulation/nist-1107-recon chain=source,primary,sample,detector
//
// Build:  make voxtrace (host serial) | make voxtrace-kokkos (parallel/GPU-ready)
// The trace kernels live in core/Beamline.hpp and run the whole photon chain
// in device code; the fit (ensmallen L-BFGS) and IO stay on the host.

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "Beamline.hpp"
#include "Spectrum.hpp"
#include "SpectrumLoss.hpp"
#include "ResponseMatrix.hpp"
#include "DeviceBuffer.hpp"
#include "Profiler.hpp"
#include "io/SetupIO.hpp"
#include "../api/OptimizerAPI.hpp"

namespace {

// ── configuration: the simulation directory + the mounted stages ─────────────
struct Config {
    std::string        dir;
    vtio::SetupDescr   run;
    vtio::SourceDescr  so;
    vtio::BeamDescr    bm;
    vtio::PolyCapDescr pcPrim;                // valid if primary
    vtio::PolyCapDescr pcSec;                 // valid if secondary
    vtio::SampleDescr  sd;                    // valid if sample
    vtio::ScanDescr    sc;                    // valid if sample
    double energyKeV;                         // beam energy (Source.txt energy grid)
    bool primary, sample, secondary, detector;
};

// First existing file among @p names in @p dir ("" if none) — lets the new
// per-optic descriptors coexist with the legacy shared files.
std::string pickFile(const std::string& dir, std::initializer_list<const char*> names) {
    for (const char* n : names)
        if (std::filesystem::exists(dir + "/" + n)) return dir + "/" + n;
    return "";
}

Config loadConfig(int argc, char* argv[]) {
    VT_PROFILE("load-config");
    Config C;
    C.dir = (argc > 1 && !std::strchr(argv[1], '=')) ? argv[1]
                                                     : "test-data/simulation/nist-1107-recon";
    C.run = vtio::loadSetup(C.dir + "/Setup.txt");
    for (int a = 1; a < argc; ++a)
        if (std::strchr(argv[a], '=')) vtio::applyOverride(C.run, argv[a]);

    C.primary   = C.run.has("primary");
    C.sample    = C.run.has("sample");
    C.secondary = C.run.has("secondary");
    C.detector  = C.run.has("detector");
    if (!C.run.has("source"))       throw std::runtime_error("chain needs a source");
    if (C.sample && !C.detector)    throw std::runtime_error("a sample needs a detector");
    if (C.secondary && !C.sample)   throw std::runtime_error("a secondary optic needs a sample");

    C.so = vtio::loadSource(C.dir + "/Source.txt");

    // bench geometry: Placement.txt, or the legacy Capillaries.txt
    std::string plc = pickFile(C.dir, {"Placement.txt", "Capillaries.txt"});
    if (plc.empty()) throw std::runtime_error("no Placement.txt / Capillaries.txt in " + C.dir);
    C.bm = plc.find("Placement.txt") != std::string::npos ? vtio::loadPlacement(plc)
                                                          : vtio::loadBeam(plc);

    // beam energy: an explicit legacy Capillaries.txt value wins (it is the
    // tube line); otherwise the maximum of the Source.txt energy grid
    C.energyKeV = C.bm.energyKeV > 0 ? C.bm.energyKeV : C.so.energyKeV;
    if (C.energyKeV <= 0) throw std::runtime_error("no beam energy in Source.txt");

    // per-optic descriptors, falling back to one shared Polycapillary.txt
    if (C.primary) {
        std::string p = pickFile(C.dir, {"Primary_Polycapillary.txt", "Polycapillary.txt"});
        if (p.empty()) throw std::runtime_error("no primary optic descriptor in " + C.dir);
        C.pcPrim = vtio::loadPolyCap(p);
    }
    if (C.secondary) {
        std::string p = pickFile(C.dir, {"Secondary_Polycapillary.txt", "Polycapillary.txt"});
        if (p.empty()) throw std::runtime_error("no secondary optic descriptor in " + C.dir);
        C.pcSec = vtio::loadPolyCap(p);
    }
    if (C.sample) {
        C.sd = vtio::loadSample(C.dir);
        C.sc = vtio::loadScan(C.dir + "/Simulation.txt");
    }
    return C;
}

// ── sample stage: voxel grid + materials over the union of elements ──────────
struct SampleStage {
    std::vector<int>          uz;             // union of element Z's
    DeviceBuffer<Voxel>       vox;
    DeviceBuffer<Material>    mat;
    DeviceBuffer<ChemElement> elem;
    vt::Grid grid;
    int xN, yN, zN;
    double x0, y0, z0, VOX;

    double zOfVox(int v) const { return z0 + ((v % zN) + 0.5) * VOX; }
};

SampleStage buildSample(const vtio::SampleDescr& sd) {
    std::vector<int> uz;
    for (const auto& zs : sd.matZ)
        for (int z : zs)
            if (std::find(uz.begin(), uz.end(), z) == uz.end()) uz.push_back(z);
    if ((int)uz.size() > MAX_ELEMENTS)
        throw std::runtime_error("materials use more elements than Material supports");

    std::vector<ChemElement> elems;
    for (int z : uz) elems.emplace_back(z);
    std::vector<Material> mats;
    for (size_t m = 0; m < sd.matZ.size(); ++m) {
        float w[MAX_ELEMENTS] = {};
        for (size_t i = 0; i < sd.matZ[m].size(); ++i)
            w[std::find(uz.begin(), uz.end(), sd.matZ[m][i]) - uz.begin()] = sd.matW[m][i];
        mats.emplace_back((int)uz.size(), w, elems.data());
    }

    const int xN = sd.xN, yN = sd.yN, zN = sd.zN;
    std::vector<Voxel> voxels((size_t)xN*yN*zN);
    for (int i = 0; i < xN; ++i)
    for (int j = 0; j < yN; ++j)
    for (int k = 0; k < zN; ++k) {
        size_t idx = (size_t)i*yN*zN + j*zN + k;
        voxels[idx] = Voxel((float)(sd.x0+i*sd.vx), (float)(sd.y0+j*sd.vy), (float)(sd.z0+k*sd.vz),
                            (float)sd.vx, (float)sd.vy, (float)sd.vz, sd.matOf(idx));
    }
    for (int i = 0; i < xN; ++i)
    for (int j = 0; j < yN; ++j)
    for (int k = 0; k < zN; ++k) {
        int nn[27], c = 0;                                     // 27-neighbour table
        for (int l = -1; l < 2; ++l) for (int m = -1; m < 2; ++m) for (int n = -1; n < 2; ++n) {
            int ni=i+n, nj=j+m, nk=k+l;
            nn[c++] = (ni<0||ni>=xN||nj<0||nj>=yN||nk<0||nk>=zN) ? -1 : ni*yN*zN + nj*zN + nk;
        }
        voxels[(size_t)i*yN*zN + j*zN + k].setNN(nn);
    }

    SampleStage S{ uz,
                   {"voxels", voxels}, {"materials", mats}, {"elements", elems},
                   {}, xN, yN, zN, sd.x0, sd.y0, sd.z0, sd.vz };
    S.grid = vt::Grid{ Sample((float)sd.x0,(float)sd.y0,(float)sd.z0,
                              (float)sd.LX,(float)sd.LY,(float)sd.LZ,
                              (float)sd.vx,(float)sd.vy,(float)sd.vz, xN,yN,zN),
                       S.vox.device(), S.mat.device(), S.elem.device() };
    return S;
}

// ── source [+ primary optic] → focused / parallel beam ───────────────────────
std::vector<vt::ExitRay> traceBeam(const Config& C, const PolyCap& primary) {
    VT_PROFILE("beam-trace");
    const double srcR = C.primary ? std::min(C.pcPrim.rExtUp, C.so.radiusX) : C.so.radiusX;

    // spectrum source: normalized cumulative probabilities for the dice throw
    const int specN = (int)C.so.specE.size();
    DeviceBuffer<double> specE("spec-energies", std::max(specN, 1));
    DeviceBuffer<double> specCdf("spec-cdf", std::max(specN, 1));
    if (specN > 1) {
        double sum = 0;
        for (double w : C.so.specW) sum += std::max(w, 0.0);
        double acc = 0;
        for (int i = 0; i < specN; ++i) {
            acc += std::max(C.so.specW[i], 0.0) / std::fmax(sum, 1e-300);
            specE.host()[i]   = C.so.specE[i];
            specCdf.host()[i] = acc;
        }
        specCdf.host()[specN - 1] = 1.0;
        specE.toDevice();
        specCdf.toDevice();
    }

    std::vector<vt::ExitRay> beam;
    const long CHUNK = 2000000;
    DeviceBuffer<vt::ExitRay> slots("beam-chunk", (size_t)std::min(CHUNK, (long)C.run.nPrimary));
    for (long base = 0; base < C.run.nPrimary; base += CHUNK) {
        long n = std::min(CHUNK, C.run.nPrimary - base);
        vt::BeamKernel bk{
            .primary = primary, .hasOptic = C.primary, .srcR = srcR,
            .energy = C.energyKeV,
            .specE = specN > 1 ? specE.device() : nullptr,
            .specCdf = specN > 1 ? specCdf.device() : nullptr,
            .specN = specN > 1 ? specN : 0,
            .seed = C.run.seed, .base = base,
            .slots = slots.device(), .dbg = vtdbg::cfg,
        };
        vt::forRange("voxTrace::beam", n, bk);
        slots.toHost();
        for (long i = 0; i < n; ++i)
            if (slots.host()[i].w >= 0) beam.push_back(slots.host()[i]);
    }
    vtprof::count("primaries", C.run.nPrimary);
    vtprof::count("beam-rays", (long)beam.size());
    return beam;
}

// ── task: beam characterisation (no sample mounted) ──────────────────────────
int beamReport(const Config& C, const std::vector<vt::ExitRay>& beam) {
    double wSum = 0, div2 = 0;
    for (const vt::ExitRay& r : beam) {
        wSum += r.w;
        div2 += r.w * (r.dir.x*r.dir.x + r.dir.y*r.dir.y);
    }
    std::printf("  transmitted      = %zu of %ld (%.2f%%)   efficiency (weighted) = %.3f%%\n",
                beam.size(), C.run.nPrimary, 100.0*beam.size()/C.run.nPrimary,
                100.0*wSum/C.run.nPrimary);
    std::printf("  RMS divergence   = %.3f mrad\n", 1e3*std::sqrt(div2/std::fmax(wSum,1e-30)));

    // weighted radial profile in the focal plane (or the exit plane, bare source)
    const double zProf = C.primary ? C.pcPrim.focalDown : 0.0;
    std::vector<double> prof(200, 0.0);
    const double RMAX = C.primary ? 4.0*C.pcPrim.rCapDown + 20e-4 : 1.2*C.so.radiusX;
    double r50 = 0, wTot = 0;
    std::vector<std::pair<double,double>> rw;
    rw.reserve(beam.size());
    for (const vt::ExitRay& r : beam) {
        double x = r.pos.x + r.dir.x/r.dir.z*zProf;
        double y = r.pos.y + r.dir.y/r.dir.z*zProf;
        double rad = std::sqrt(x*x + y*y);
        rw.push_back({rad, r.w});
        wTot += r.w;
        int b = (int)(rad / (RMAX/200));
        if (b >= 0 && b < 200) prof[b] += r.w;
    }
    std::sort(rw.begin(), rw.end());
    double acc = 0;
    for (const auto& p : rw) { acc += p.second; if (acc >= 0.5*wTot) { r50 = p.first; break; } }
    std::printf("  focal plane      = %.2f cm after exit   spot r50 = %.2f µm\n", zProf, r50*1e4);

    std::ofstream fp("test-data/out/beam_profile.csv");
    fp << "r_um,weight\n";
    for (int b = 0; b < 200; ++b)
        fp << (b + 0.5)*(RMAX/200)*1e4 << "," << prof[b] << "\n";
    std::ofstream fr("test-data/out/beam_rays.csv");
    fr << "x_um,y_um,dir_x,dir_y,dir_z,weight,energy_keV\n";
    for (size_t i = 0; i < beam.size() && i < 200000; ++i)
        fr << beam[i].pos.x*1e4 << "," << beam[i].pos.y*1e4 << ","
           << beam[i].dir.x << "," << beam[i].dir.y << "," << beam[i].dir.z << ","
           << beam[i].w << "," << beam[i].e << "\n";
    std::printf("\nBeam profile → test-data/out/beam_profile.csv\n");
    std::printf("Exit rays    → test-data/out/beam_rays.csv\n");
    return 0;
}

// ── one confocal/µXRF scan: ScanKernel over every (position, beam ray) ───────
struct ScanRunner {
    const Config& C;
    const SampleStage& S;
    PolyCap secondary;
    Detector detector;
    DeviceBuffer<ChemElement>& detBuf;
    DeviceBuffer<vt::ExitRay>& beamBuf;
    long beamN;
    DeviceBuffer<vt::Event> slots{"event-slots", (size_t)beamN};
    vt::Vec3 d_prim, d_sec, C0;
    double zFocPrim, aimDist, rWin;
    bool usePol;

    std::vector<vt::Event> run(uint64_t seed, const char* phaseTag, const char* countTag) {
        return C.run.varianceReduction ? runVR(seed, phaseTag, countTag)
                                       : runAnalog(seed, phaseTag, countTag);
    }

    // variance reduction: one weighted, aimed sample per (position, beam ray)
    std::vector<vt::Event> runVR(uint64_t seed, const char* phaseTag, const char* countTag) {
        VT_PROFILE(phaseTag);
        std::vector<vt::Event> events;
        for (int di = 0; di < (int)C.sc.points.size(); ++di) {
            vt::Vec3 P0 = C0 + vt::Vec3{C.sc.points[di][0], C.sc.points[di][1], C.sc.points[di][2]};
            vt::OpticFrame primFrame(P0, d_prim, zFocPrim);
            vt::OpticFrame secFrame (P0, d_sec, -aimDist);
            vt::Vec3 srcPol = {1, 0, 0};              // source E-field in the optic plane
            vt::Vec3 Epol = primFrame.u*srcPol.x + primFrame.v*srcPol.y + primFrame.axis*srcPol.z;

            vt::ScanKernel k{
                .grid = S.grid, .secondary = secondary, .hasSecondary = C.secondary,
                .detector = detector, .detElems = detBuf.device(),
                .beam = beamBuf.device(), .slots = slots.device(),
                .primFrame = primFrame, .secFrame = secFrame,
                .aimCtr = P0 + d_sec*aimDist, .Epol = Epol, .d_sec = d_sec,
                .energy = C.energyKeV, .rWin = rWin, .eBin = C.run.eBin,
                .x0 = C.sd.x0, .y0 = C.sd.y0, .z0 = C.sd.z0, .LX = C.sd.LX, .LY = C.sd.LY,
                .nBin = C.run.nBin, .di = di,
                .usePol = usePol, .scanSeed = seed, .beamN = beamN, .dbg = vtdbg::cfg,
            };
            vt::forRange("voxTrace::scan", beamN, k);
            slots.toHost();
            for (long bi = 0; bi < beamN; ++bi)
                if (slots.host()[bi].ch >= 0) events.push_back(slots.host()[bi]);
        }
        vtprof::count(countTag, (long)events.size());
        return events;
    }

    // brute force: every thread respawns candidates until it has ONE detected
    // photon (full analog transport, any interaction order). Events carry
    // weight 1/attempts(position), so positions stay mutually normalised.
    std::vector<vt::Event> runAnalog(uint64_t seed, const char* phaseTag, const char* countTag) {
        VT_PROFILE(phaseTag);
        const long target = C.run.nDetected > 0 ? C.run.nDetected
                          : (C.sc.nRays > 0 ? C.sc.nRays : 30000);
        DeviceBuffer<vt::Event> eslots("analog-slots", (size_t)target);
        DeviceBuffer<long>      att   ("analog-attempts", (size_t)target);
        std::vector<vt::Event> events;
        long attAll = 0, missAll = 0;
        for (int di = 0; di < (int)C.sc.points.size(); ++di) {
            vt::Vec3 P0 = C0 + vt::Vec3{C.sc.points[di][0], C.sc.points[di][1], C.sc.points[di][2]};
            vt::OpticFrame primFrame(P0, d_prim, zFocPrim);
            vt::OpticFrame secFrame (P0, d_sec, -aimDist);

            vt::AnalogKernel k{
                .grid = S.grid, .secondary = secondary, .hasSecondary = C.secondary,
                .detector = detector, .detElems = detBuf.device(),
                .beam = beamBuf.device(), .slots = eslots.device(), .attempts = att.device(),
                .primFrame = primFrame, .secFrame = secFrame,
                .aimCtr = P0 + d_sec*aimDist, .d_sec = d_sec,
                .energy = C.energyKeV, .rWin = rWin, .eBin = C.run.eBin,
                .x0 = C.sd.x0, .y0 = C.sd.y0, .z0 = C.sd.z0, .LX = C.sd.LX, .LY = C.sd.LY,
                .nBin = C.run.nBin, .di = di, .maxGen = C.run.maxGenerations,
                .beamN = beamN, .nTarget = target, .maxAttempts = C.run.maxAttempts,
                .scanSeed = seed, .dbg = vtdbg::cfg,
            };
            vt::forRange("voxTrace::analog", target, k);
            eslots.toHost();
            att.toHost();
            long attSum = 0;
            for (long s = 0; s < target; ++s) attSum += att.host()[s];
            const float wEv = (float)(1.0 / (double)attSum);   // per-candidate rate
            for (long s = 0; s < target; ++s) {
                vt::Event e = eslots.host()[s];
                if (e.ch >= 0) { e.w = wEv; events.push_back(e); }
                else ++missAll;
            }
            attAll += attSum;
        }
        if (missAll)
            std::printf("  WARNING: %ld slots hit max_attempts=%ld undetected — raise it "
                        "or lower n_detected\n", missAll, C.run.maxAttempts);
        vtprof::count(countTag, (long)events.size());
        vtprof::count("analog-attempts", attAll);
        return events;
    }
};

// ── the measurement: loaded real spectra, or a synthetic phantom trace ───────
// Fills @p meas (per position) and returns the exposure factor that scales the
// unit-weight simulation onto the measurement's counts.
double makeMeasurement(const Config& C, const SampleStage& S, ScanRunner& T,
                       double fitEventSum, std::vector<Spectrum>& meas, long& nChan) {
    const int NPOS = (int)C.sc.points.size(), NBIN = C.run.nBin;
    std::vector<std::vector<double>> raw(NPOS, std::vector<double>(NBIN, 0.0));
    double expo;

    if (!C.run.measured.empty()) {                       // real data from disk
        raw = vtio::loadMeasured(C.dir + "/" + C.run.measured, NPOS, NBIN, C.run.eBin);
        double total = 0;
        for (const auto& s : raw) for (double v : s) total += v;
        expo = total / fitEventSum;                      // absolute-scale calibration
        std::printf("  measured spectra = %s (%.0f counts)\n", C.run.measured.c_str(), total);
    } else {                                             // synthetic phantom measurement
        auto wTrueOfZ = [&](double z) {
            double w = 1.0;
            for (const auto& L : C.run.truth)
                if (z >= L.z0um*1e-4 && z < L.z1um*1e-4) w = L.w;
            return w;
        };
        std::vector<vt::Event> ev = T.run(C.run.seed ^ 0x6D656173ULL,
                                          "scan-measurement", "events-measurement");
        double rawTotal = 0;
        for (const vt::Event& e : ev) {
            double w = e.w * wTrueOfZ(S.zOfVox(e.vox));
            raw[e.pos][e.ch] += w;
            rawTotal += w;
        }
        expo = C.run.expoPerEvent * ev.size() / rawTotal;
        for (auto& s : raw) for (double& v : s) v *= expo;
        std::mt19937_64 noise(C.run.seed ^ 0x6E6F697365ULL);         // Poisson counts
        for (auto& s : raw)
            for (double& v : s)
                v = (v > 0) ? (double)std::poisson_distribution<long>(v)(noise) : 0.0;
        std::printf("  synthetic measurement: phantom + Poisson (%zu events, ×%.1f per event)\n",
                    ev.size(), C.run.expoPerEvent);
    }

    nChan = 0;
    meas.reserve(NPOS);
    for (int d = 0; d < NPOS; ++d) {
        meas.emplace_back(NBIN, 0.f, (float)C.run.eBin);
        for (int ch = 0; ch < NBIN; ++ch)
            if (raw[d][ch] > 0) { meas[d].setCount(ch, (float)raw[d][ch]); ++nChan; }
    }
    return expo;
}

void writeSpectra(const Config& C, const std::vector<Spectrum>& meas,
                  const std::vector<std::vector<float>>& sim) {
    std::ofstream fs("test-data/out/recon_spectra.csv");
    fs << "energy_keV";
    for (size_t d = 0; d < meas.size(); ++d) {
        int um = (int)std::lround(C.sc.points[d][2]*1e4);
        fs << ",meas_" << um << "um,fit_" << um << "um";
    }
    fs << "\n";
    for (int ch = 0; ch < C.run.nBin; ++ch) {
        fs << (ch + 0.5)*C.run.eBin;
        for (size_t d = 0; d < meas.size(); ++d)
            fs << "," << meas[d].count(ch) << "," << sim[d][ch];
        fs << "\n";
    }
    std::printf("Spectra       → test-data/out/recon_spectra.csv\n");
}

// ── the fit: per-voxel weights minimise χ²(simulated, measured) ───────────────
// Linear forward model S(w) = B + Σ w_v R_v from the response trace; optimizer
// runs over x with w = x² (weights are physical, ≥ 0) plus a mild Tikhonov
// pull λ Σ m_v (w_v − 1)² that resolves laterally degenerate voxels. Truth
// columns are only meaningful for synthetic phantom runs.
void fitWeights(const Config& C, const SampleStage& S, const Detector& detector,
                const std::vector<vt::Event>& evFit, double expo,
                std::vector<Spectrum>& meas, long nChan) {
    vtprof::Phase phase("build-model");
    const int NPOS = (int)C.sc.points.size(), NBIN = C.run.nBin;
    auto wTrueOfZ = [&](double z) {
        double w = 1.0;
        for (const auto& L : C.run.truth)
            if (z >= L.z0um*1e-4 && z < L.z1um*1e-4) w = L.w;
        return w;
    };

    // parameters: voxels with enough events; the rest are frozen into baseline B
    std::unordered_map<int, int> evCount, paramOf;
    for (const vt::Event& e : evFit) ++evCount[e.vox];
    std::vector<int> voxOfParam;
    for (const vt::Event& e : evFit)
        if (evCount[e.vox] >= C.run.minEvents && !paramOf.count(e.vox)) {
            paramOf[e.vox] = (int)voxOfParam.size();
            voxOfParam.push_back(e.vox);
        }
    const int nP = (int)voxOfParam.size();
    if (nP == 0) { std::printf("No voxel reached %d events — increase n_primary.\n", C.run.minEvents); return; }
    vtprof::count("parameters", nP);

    std::vector<ResponseMatrix> R;
    R.reserve(NPOS);
    for (int d = 0; d < NPOS; ++d) R.emplace_back(nP, NBIN);
    for (const vt::Event& e : evFit) {
        auto it = paramOf.find(e.vox);
        if (it != paramOf.end()) R[e.pos].add(it->second, e.ch, e.w);
        else                     R[e.pos].addBaseline(e.ch, e.w);
    }
    size_t nnz = 0;
    for (int d = 0; d < NPOS; ++d) { R[d].finalize(); R[d].scale(expo); nnz += R[d].nnz(); }
    std::vector<double> mass(nP, 0.0);
    for (int d = 0; d < NPOS; ++d) R[d].addParamMass(mass);
    std::printf("  parameters       = %d voxels (≥%d events)   response = %zu non-zeros   exposure ×%.3e\n",
                nP, C.run.minEvents, nnz, expo);

    // the loss: plain χ² or ROI-weighted χ² that emphasises chosen lines
    SpectrumLoss loss(C.run.lossMode == 1 ? SpectrumLoss::CHI2_WEIGHTED : SpectrumLoss::CHI2);
    if (C.run.lossMode == 1) {
        std::printf("  loss = weighted χ²:");
        for (const auto& L : C.run.rois) {
            double half = std::fmax(0.10, 2.5 * detector.resolutionSigma((float)L.energyKeV));
            int c0 = std::max(0,        (int)((L.energyKeV - half) / C.run.eBin));
            int c1 = std::min(NBIN - 1, (int)((L.energyKeV + half) / C.run.eBin));
            loss.addROI(c0, c1, L.weight);
            std::printf("  %s ×%.0f", L.name.c_str(), L.weight);
        }
        std::printf("\n");
    } else {
        std::printf("  loss = plain χ² (loss=1 for ROI-weighted)   λ = %.3g\n", C.run.lambda);
    }

    // objective for ensmallen: χ² + Tikhonov, exact gradient via chain rule
    Spectrum sim(NBIN, 0.f, (float)C.run.eBin);
    std::vector<double> wbuf(nP), gw(nP);
    ObjGradFn objGrad = [&](const std::vector<double>& x, std::vector<double>& g) {
        for (int p = 0; p < nP; ++p) wbuf[p] = x[p]*x[p];
        std::fill(gw.begin(), gw.end(), 0.0);
        double L = 0;
        for (int d = 0; d < NPOS; ++d) {
            R[d].assemble(wbuf, sim);
            L += loss.value(sim, meas[d]);
            R[d].accumulateGrad(loss, sim, meas[d], gw);
        }
        for (int p = 0; p < nP; ++p) {
            L     += C.run.lambda * mass[p] * (wbuf[p] - 1.0) * (wbuf[p] - 1.0);
            gw[p] += 2.0 * C.run.lambda * mass[p] * (wbuf[p] - 1.0);
        }
        for (int p = 0; p < nP; ++p) g[p] = 2.0 * x[p] * gw[p];
        return L;
    };
    std::vector<double> g0(nP), x1(nP, 1.0);
    const double L_init = objGrad(x1, g0);

    OptimizerConfig ocfg;
    ocfg.maxIter   = (size_t)C.run.maxIter;
    ocfg.tolerance = 1e-8;
    OptimizerResult res;
    {
        VT_PROFILE("optimize");
        res = OptimizerAPI::Minimize(Algorithm::LBFGS, objGrad, x1, ocfg);
    }
    std::vector<double> wFit(nP);
    for (int p = 0; p < nP; ++p) wFit[p] = res.x[p] * res.x[p];

    SpectrumLoss plain(SpectrumLoss::CHI2);
    double chi2Fit = 0;
    for (int d = 0; d < NPOS; ++d) { R[d].assemble(wFit, sim); chi2Fit += plain.value(sim, meas[d]); }
    std::printf("%s: L %.1f → %.1f   plain χ²/channel = %.2f  (%ld channels, %d parameters)\n",
                res.algorithm.c_str(), L_init, res.fval, chi2Fit/nChan, nChan, nP);

    // depth profile (response-weighted layer averages) + per-voxel weights
    phase.next("outputs");
    const bool synthetic = C.run.measured.empty();
    std::printf("  %-12s %-6s %-7s %-8s %-10s %s\n", "layer [µm]", "nvox", "mass%",
                synthetic ? "w_true" : "-", "<w_fit>", "sd");
    std::vector<double> lm(S.zN, 0), lw(S.zN, 0), lw2(S.zN, 0);
    std::vector<int>    ln(S.zN, 0);
    double mTot = 0;
    for (int p = 0; p < nP; ++p) {
        int k = voxOfParam[p] % S.zN;
        lm[k] += mass[p]; lw[k] += mass[p]*wFit[p]; lw2[k] += mass[p]*wFit[p]*wFit[p];
        ++ln[k]; mTot += mass[p];
    }
    std::ofstream fpz("test-data/out/recon_profile.csv");
    fpz << "z_um,n_vox,mass,w_true,w_fit_mean,w_fit_sd\n";
    for (int k = 0; k < S.zN; ++k) {
        if (ln[k] == 0) continue;
        double m = lw[k]/lm[k], sd = std::sqrt(std::fmax(0.0, lw2[k]/lm[k] - m*m));
        double wt = synthetic ? wTrueOfZ(S.z0 + (k+0.5)*S.VOX) : 1.0;
        std::printf("  %3.0f – %-6.0f %-6d %-7.2f %-8.2f %-10.3f %.3f\n",
                    k*S.VOX*1e4, (k+1)*S.VOX*1e4, ln[k], 100.0*lm[k]/mTot, wt, m, sd);
        fpz << (k+0.5)*S.VOX*1e4 << "," << ln[k] << "," << lm[k] << ","
            << wt << "," << m << "," << sd << "\n";
    }

    std::ofstream fw("test-data/out/recon_weights.csv");
    fw << "vox,i,j,k,x_um,y_um,z_um,events,mass,w_true,w_fit\n";
    for (int p = 0; p < nP; ++p) {
        int vox = voxOfParam[p];
        int i = vox/(S.yN*S.zN), j = (vox/S.zN)%S.yN, k = vox%S.zN;
        fw << vox << "," << i << "," << j << "," << k << ","
           << (S.x0+(i+0.5)*C.sd.vx)*1e4 << "," << (S.y0+(j+0.5)*C.sd.vy)*1e4 << ","
           << (S.z0+(k+0.5)*C.sd.vz)*1e4 << "," << evCount[vox] << "," << mass[p] << ","
           << (synthetic ? wTrueOfZ(S.zOfVox(vox)) : 1.0) << "," << wFit[p] << "\n";
    }
    std::printf("Voxel weights → test-data/out/recon_weights.csv\n");
    std::printf("Depth profile → test-data/out/recon_profile.csv\n");

    std::vector<std::vector<float>> fitSpec(NPOS, std::vector<float>(NBIN));
    for (int d = 0; d < NPOS; ++d) {
        R[d].assemble(wFit, sim);
        for (int ch = 0; ch < NBIN; ++ch) fitSpec[d][ch] = sim.count(ch);
    }
    writeSpectra(C, meas, fitSpec);
}

}  // namespace

int main(int argc, char* argv[]) {
#ifndef VOXTRACE_HOST_ONLY
    Kokkos::ScopeGuard kokkos(argc, argv);
#endif
    Config C;
    try {
        C = loadConfig(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    vtdbg::cfg.level = C.run.debug;
    vtdbg::cfg.only  = C.run.debugRay;
    std::filesystem::create_directories("test-data/out");

    std::printf("voxTrace — %s\n", C.dir.c_str());
    std::printf("  chain            = source%s%s%s%s\n",
                C.primary ? " → primary polycap" : "",
                C.sample ? " → sample" : "",
                C.secondary ? " → secondary polycap" : (C.sample ? " → aperture" : ""),
                C.detector ? " → detector" : "");
    if (C.so.specE.size() > 1)
        std::printf("  source           = spectrum (%zu energies, %.1f keV max, dice-sampled)\n",
                    C.so.specE.size(), C.energyKeV);
    else
        std::printf("  source           = monochromatic %.1f keV\n", C.energyKeV);
    std::printf("  switches         = polarization %s | %s\n",
                C.run.polarization ? "ON" : "OFF",
                C.run.varianceReduction ? "importance sampling (variance reduction)"
                                        : "BRUTE FORCE analog MC");
    if (!C.run.varianceReduction && C.sample) {
        std::printf("  analog mode      = respawn until %ld detected/position (≤%ld attempts, "
                    "≤%d interaction orders)%s\n",
                    C.run.nDetected > 0 ? C.run.nDetected : (long)(C.sc.nRays > 0 ? C.sc.nRays : 30000),
                    C.run.maxAttempts, C.run.maxGenerations,
                    C.run.polarization ? " — polarization azimuth not modulated in analog transport" : "");
    }

    // ── source [→ primary optic]: the beam, traced once and reused ───────────
    PolyCap primary   = C.primary   ? C.pcPrim.build() : PolyCap();
    PolyCap secondary = C.secondary ? C.pcSec.reversed().build() : PolyCap();
    std::vector<vt::ExitRay> beam = traceBeam(C, primary);
    std::printf("  primary photons  = %ld   beam rays = %zu (%.2f%%)\n",
                C.run.nPrimary, beam.size(), 100.0*beam.size()/C.run.nPrimary);
    if (beam.empty()) { std::printf("No beam — check the optic descriptor.\n"); return 1; }

    // ── task: beam characterisation only? ────────────────────────────────────
    if (!C.sample) {
        int rc = beamReport(C, beam);
        if (C.run.profile) vtprof::report();
        return rc;
    }

    // ── sample + detector stages ──────────────────────────────────────────────
    SampleStage S = buildSample(C.sd);
    std::vector<ChemElement> detElems{ ChemElement(14), ChemElement(4) };   // Si, Be
    DeviceBuffer<ChemElement> detBuf("det-elements", detElems);
    Detector detector = Detector::make(0, 1, (float)C.run.detThickness, (float)C.run.detBeWindow,
                                       (float)C.run.detDeadLayer, (float)C.run.detFano,
                                       (float)C.run.detNoiseFWHM);
    DeviceBuffer<vt::ExitRay> beamBuf("beam", beam);
    const double A  = C.bm.angleDeg    * vt::PI_D / 180.0;
    const double A2 = C.bm.angleSecDeg * vt::PI_D / 180.0;
    ScanRunner T{
        .C = C, .S = S, .secondary = secondary, .detector = detector,
        .detBuf = detBuf, .beamBuf = beamBuf, .beamN = (long)beam.size(),
        .d_prim = vt::norm({std::cos(A), 0,  std::sin(A)}),
        .d_sec  = vt::norm({std::cos(A2), 0, -std::sin(A2)}),
        .C0 = {C.bm.posX, C.bm.posY, 0},
        .zFocPrim = C.primary ? C.pcPrim.length + C.pcPrim.focalDown : 1.0,   // bare beam: 1 cm standoff
        .aimDist = C.secondary ? C.pcSec.focalDown : C.run.detDistance,
        .rWin    = C.secondary ? C.pcSec.rExtDown  : C.run.detRadius,
        .usePol  = C.run.polarization && C.so.polFactor > 0.0,
    };
    std::printf("  sample           = %dx%dx%d voxels   scan = %zu positions   E = %.1f keV\n",
                S.xN, S.yN, S.zN, C.sc.points.size(), C.energyKeV);

    // ── simulate: the response trace (and the measurement) ───────────────────
    std::vector<vt::Event> evFit = T.run(C.run.seed ^ 0x7265636FULL, "scan-response", "events-response");
    std::printf("  detected events  = %zu (response)\n", evFit.size());
    if (evFit.empty()) { std::printf("No detected events — increase n_primary.\n"); return 1; }
    double fitEventSum = 0;
    for (const vt::Event& e : evFit) fitEventSum += e.w;

    std::vector<Spectrum> meas;
    long nChan = 0;
    double expo = makeMeasurement(C, S, T, fitEventSum, meas, nChan);

    // ── optionally fit the per-voxel sample weights, else just write spectra ─
    if (C.run.fit) {
        fitWeights(C, S, detector, evFit, expo, meas, nChan);
    } else {
        std::vector<std::vector<float>> sim(meas.size(), std::vector<float>(C.run.nBin, 0.f));
        for (const vt::Event& e : evFit) sim[e.pos][e.ch] += (float)(expo * e.w);
        writeSpectra(C, meas, sim);
    }

    if (C.run.profile) vtprof::report();
    return 0;
}
