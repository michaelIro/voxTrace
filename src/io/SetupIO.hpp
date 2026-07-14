#pragma once
/**
 * @file SetupIO.hpp
 * @brief Host-only loaders for the txt experiment descriptions in test-data/simulation.
 *
 * A simulation directory describes one experiment; the loaders here turn its
 * txt files into plain structs the tests build the trace chain from:
 *
 *   Primary_Polycapillary.txt    optic descriptor (polycap-library format
 *   Secondary_Polycapillary.txt  `value; //…`); legacy fallback: one shared
 *                                Polycapillary.txt used for both optics
 *   Source.txt         source geometry / polarisation (same format); the beam
 *                      energy is the maximum of its energy grid
 *   Placement.txt      bench geometry per optic, µm/° (`value   # …`);
 *                      legacy fallback: Capillaries.txt (which also carried
 *                      the beam energy)
 *   Sample.txt         voxel-grid geometry, µm        (`value   # …`)
 *   Materials.txt      per-voxel element composition
 *   Simulation.txt     scan path: confocal positions, µm offsets
 *   Setup.txt          run parameters: statistics, binning, loss/ROIs,
 *                      optimizer, physics switches, phantom (`key = value`)
 *
 * The parsers are deliberately tolerant: `#` and `//` comments and blank lines
 * are ignored everywhere; single numbers, `{…}` arrays and comma tuples are
 * collected in file order, so extra commentary or re-alignment never breaks a
 * file. Setup.txt keys can be overridden from the command line with the same
 * `key=value` syntax (see applyOverride).
 */

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PolyCap.hpp"

namespace vtio {

// ── low-level txt scanning ────────────────────────────────────────────────────

inline std::string stripComment(std::string s) {
    size_t p = s.find('#');
    if (p != std::string::npos) s.erase(p);
    p = s.find("//");
    if (p != std::string::npos) s.erase(p);
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

// All numbers on a line, split on commas/semicolons/whitespace; a trailing
// 'f' after a literal (legacy "0.0f") is tolerated by strtod.
inline std::vector<double> numbersOn(const std::string& line) {
    std::vector<double> out;
    const char* p = line.c_str();
    while (*p) {
        char* end = nullptr;
        double v = std::strtod(p, &end);
        if (end == p) { ++p; continue; }
        out.push_back(v);
        p = (*end == 'f') ? end + 1 : end;
    }
    return out;
}

/// A whole file reduced to three ordered streams: single scalars, `{…}`
/// arrays, and comma tuples (≥2 numbers on one line).
struct Txt {
    std::vector<double>              scalars;
    std::vector<std::vector<double>> arrays;
    std::vector<std::vector<double>> tuples;
};

inline Txt scanTxt(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("SetupIO: cannot open " + path);
    Txt t;
    std::string raw;
    while (std::getline(in, raw)) {
        std::string line = stripComment(raw);
        if (line.empty()) continue;
        size_t b0 = line.find('{'), b1 = line.find('}');
        if (b0 != std::string::npos && b1 != std::string::npos && b1 > b0) {
            t.arrays.push_back(numbersOn(line.substr(b0 + 1, b1 - b0 - 1)));
            continue;
        }
        // only numeric-leading lines carry values — titles and prose (which may
        // contain digits, e.g. "PC-236") are ignored
        char c = line[0];
        if (!(std::isdigit((unsigned char)c) || c == '+' || c == '-' || c == '.')) continue;
        std::vector<double> v = numbersOn(line);
        if (v.size() == 1) t.scalars.push_back(v[0]);
        else if (v.size() > 1) t.tuples.push_back(std::move(v));
    }
    return t;
}

// ── Polycapillary.txt — optic descriptor (polycap library format) ────────────

struct PolyCapDescr {
    double length     = 0;   // [cm]
    double rExtUp     = 0, rExtDown = 0;    // external radius at entrance/exit
    double rCapUp     = 0, rCapDown = 0;    // single-capillary radius
    double focalUp    = 0, focalDown = 0;   // focal distances
    std::vector<int>   iz;                  // wall material
    std::vector<float> wt;
    double density    = 0;   // [g/cm³]
    double roughness  = 0;   // [Å]
    double nCap       = 0;

    /// Same optic mounted the other way round (entrance ↔ exit).
    PolyCapDescr reversed() const {
        PolyCapDescr r = *this;
        std::swap(r.rExtUp, r.rExtDown);
        std::swap(r.rCapUp, r.rCapDown);
        std::swap(r.focalUp, r.focalDown);
        return r;
    }

    /// Construct the optic with its entrance window at z = 0.
    PolyCap build() const {
        std::vector<int>   z = iz;
        std::vector<float> w = wt;
        return PolyCap(0.f, (float)length, (float)rExtUp, (float)rExtDown,
                       (float)rCapUp, (float)rCapDown, (float)focalUp, (float)focalDown,
                       PolyCap::ELLIPSOIDAL, (int)z.size(), z.data(), w.data(),
                       (float)density, (float)roughness, (int)nCap);
    }
};

inline PolyCapDescr loadPolyCap(const std::string& path) {
    Txt t = scanTxt(path);
    if (t.scalars.size() < 11 || t.arrays.size() < 2)
        throw std::runtime_error("SetupIO: bad polycap descriptor " + path);
    PolyCapDescr d;
    d.length  = t.scalars[0];
    d.rExtUp  = t.scalars[1];  d.rExtDown = t.scalars[2];
    d.rCapUp  = t.scalars[3];  d.rCapDown = t.scalars[4];
    d.focalUp = t.scalars[5];  d.focalDown = t.scalars[6];
    // scalars[7] = number of elements (implied by the arrays)
    d.density = t.scalars[8]; d.roughness = t.scalars[9]; d.nCap = t.scalars[10];
    for (double z : t.arrays[0]) d.iz.push_back((int)z);
    for (double w : t.arrays[1]) d.wt.push_back((float)w);
    return d;
}

// ── Source.txt — source geometry (polycap library format) ────────────────────

struct SourceDescr {
    double distance = 0, radiusX = 0, radiusY = 0;
    double divX = 0, divY = 0, shiftX = 0, shiftY = 0;
    double polFactor = 1.0;
    double energyKeV = 0;   // beam energy = max of the source energy grid (0 = absent)

    // spectrum source: a second `{…}` array of per-energy weights turns the
    // energy grid into a sampled emission spectrum (mono/uniform otherwise)
    std::vector<double> specE, specW;
};

inline SourceDescr loadSource(const std::string& path) {
    Txt t = scanTxt(path);
    if (t.scalars.size() < 8)
        throw std::runtime_error("SetupIO: bad source descriptor " + path);
    SourceDescr s;
    s.distance = t.scalars[0];
    s.radiusX  = t.scalars[1]; s.radiusY = t.scalars[2];
    s.divX     = t.scalars[3]; s.divY    = t.scalars[4];
    s.shiftX   = t.scalars[5]; s.shiftY  = t.scalars[6];
    s.polFactor = t.scalars[7];
    if (!t.arrays.empty())
        for (double e : t.arrays[0]) s.energyKeV = std::max(s.energyKeV, e);
    if (t.arrays.size() >= 2 && t.arrays[1].size() == t.arrays[0].size() &&
        t.arrays[0].size() > 1) {
        s.specE = t.arrays[0];
        s.specW = t.arrays[1];
    }
    return s;
}

// ── Placement.txt — bench geometry (legacy: Capillaries.txt, incl. energy) ────

struct BeamDescr {
    double energyKeV = 0;                  // legacy Capillaries.txt only (else 0)
    double posX = 0, posY = 0, posZ = 0;   // scan origin on the sample surface [cm]
    double angleDeg    = 45.0;             // excitation-arm tilt (x-z plane)
    double angleSecDeg = 45.0;             // detection-arm tilt
};

/// Placement.txt: primary block x,y,z,distToDetector,angle — secondary block
/// x,y,z,distToDetector,angle,distToPrimary. All µm / °, no energy.
inline BeamDescr loadPlacement(const std::string& path) {
    Txt t = scanTxt(path);
    if (t.scalars.size() < 11)
        throw std::runtime_error("SetupIO: bad placement file " + path);
    BeamDescr b;
    const double UM = 1e-4;
    b.posX = t.scalars[0]*UM;
    b.posY = t.scalars[1]*UM;
    b.posZ = t.scalars[2]*UM;
    b.angleDeg    = t.scalars[4];
    b.angleSecDeg = t.scalars[9];
    return b;
}

inline BeamDescr loadBeam(const std::string& path) {
    Txt t = scanTxt(path);
    if (t.scalars.size() < 9)
        throw std::runtime_error("SetupIO: bad capillaries file " + path);
    BeamDescr b;
    const double UM = 1e-4;
    b.energyKeV = t.scalars[3];      // primary block: radius, focal, cap-diameter, energy
    b.posX = t.scalars[4]*UM;        // transformation block: x, y, z, distance, angle
    b.posY = t.scalars[5]*UM;
    b.posZ = t.scalars[6]*UM;
    b.angleDeg = t.scalars[8];
    b.angleSecDeg = t.scalars.size() > 13 ? t.scalars[13] : b.angleDeg;
    return b;
}

// ── Sample.txt + Materials.txt — voxel grid and composition ───────────────────

struct SampleDescr {
    double x0 = 0, y0 = 0, z0 = 0;   // grid origin [cm]
    double LX = 0, LY = 0, LZ = 0;   // sample extents [cm]
    double vx = 0, vy = 0, vz = 0;   // voxel extents [cm]
    int    type = 0;                 // 0 homogeneous, 1 layered, 2 heterogeneous
    int    xN = 0, yN = 0, zN = 0;

    std::vector<std::vector<int>>   matZ;      // unique compositions
    std::vector<std::vector<float>> matW;
    std::vector<int> voxelMat;                 // flat voxel → material (empty ⇒ all 0)

    int matOf(size_t vox) const { return voxelMat.empty() ? 0 : voxelMat[vox]; }
};

/// Reads @p dir/Sample.txt (geometry, µm → cm) and @p dir/Materials.txt
/// (composition). For a homogeneous sample (type 0) only the first material
/// point is read; otherwise every voxel's point is parsed and identical
/// compositions are collapsed into shared materials.
inline SampleDescr loadSample(const std::string& dir) {
    Txt t = scanTxt(dir + "/Sample.txt");
    if (t.scalars.size() < 10)
        throw std::runtime_error("SetupIO: bad sample file " + dir + "/Sample.txt");
    SampleDescr s;
    const double UM = 1e-4;          // file is in µm, the trace runs in cm
    s.x0 = t.scalars[0]*UM; s.y0 = t.scalars[1]*UM; s.z0 = t.scalars[2]*UM;
    s.LX = t.scalars[3]*UM; s.LY = t.scalars[4]*UM; s.LZ = t.scalars[5]*UM;
    s.vx = t.scalars[6]*UM; s.vy = t.scalars[7]*UM; s.vz = t.scalars[8]*UM;
    s.type = (int)t.scalars[9];
    s.xN = (int)std::lround(s.LX / s.vx);
    s.yN = (int)std::lround(s.LY / s.vy);
    s.zN = (int)std::lround(s.LZ / s.vz);

    // Materials.txt: repeating blocks of (i,j,k) tuple, element count scalar,
    // Z tuple, weight tuple. Parsed with a small state machine so the 648k-line
    // heterogeneous map stays a single cheap pass.
    std::ifstream in(dir + "/Materials.txt");
    if (!in) throw std::runtime_error("SetupIO: cannot open " + dir + "/Materials.txt");
    if (s.type != 0) s.voxelMat.assign((size_t)s.xN * s.yN * s.zN, 0);

    std::string raw;
    std::vector<double> ijk;
    std::vector<std::vector<double>> block;      // Z tuple, then W tuple
    auto flush = [&]() {
        if (ijk.size() < 3 || block.size() < 2) return;
        std::vector<int>   z(block[0].begin(), block[0].end());
        std::vector<float> w(block[1].begin(), block[1].end());
        int mi = -1;
        for (size_t m = 0; m < s.matZ.size(); ++m)
            if (s.matZ[m] == z && s.matW[m] == w) { mi = (int)m; break; }
        if (mi < 0) { s.matZ.push_back(z); s.matW.push_back(w); mi = (int)s.matZ.size()-1; }
        if (!s.voxelMat.empty()) {
            long i = std::lround(ijk[0]), j = std::lround(ijk[1]), k = std::lround(ijk[2]);
            if (i >= 0 && i < s.xN && j >= 0 && j < s.yN && k >= 0 && k < s.zN)
                s.voxelMat[(size_t)i*s.yN*s.zN + j*s.zN + k] = mi;
        }
        ijk.clear(); block.clear();
    };
    while (std::getline(in, raw)) {
        std::string line = stripComment(raw);
        if (line.empty()) continue;
        std::vector<double> v = numbersOn(line);
        if (v.size() >= 3 && ijk.empty()) ijk = v;                 // new point
        else if (v.size() >= 1 && !ijk.empty()) {
            if (v.size() > 1) block.push_back(v);                  // Z or W tuple
            if (block.size() == 2) {
                flush();
                if (s.type == 0) break;                            // one material is enough
            }
        }
    }
    flush();
    if (s.matZ.empty())
        throw std::runtime_error("SetupIO: no material points in " + dir + "/Materials.txt");
    return s;
}

// ── Simulation.txt — scan path ────────────────────────────────────────────────

struct ScanDescr {
    int nRays = 0;                                  // legacy per-point ray budget
    std::vector<std::array<double,3>> points;       // confocal offsets [cm]
};

inline ScanDescr loadScan(const std::string& path) {
    Txt t = scanTxt(path);
    ScanDescr s;
    if (t.scalars.size() >= 2) s.nRays = (int)t.scalars[1];
    const double UM = 1e-4;
    for (const auto& p : t.tuples)
        if (p.size() >= 3) s.points.push_back({p[0]*UM, p[1]*UM, p[2]*UM});
    if (s.points.empty())
        throw std::runtime_error("SetupIO: no scan points in " + path);
    return s;
}

// ── Setup.txt — run parameters, switches, phantom (key = value) ───────────────

struct SetupDescr {
    long   nPrimary = 10000000;
    unsigned long long seed = 1;
    int    nBin = 400;
    double eBin = 0.05;             // [keV]
    double expoPerEvent = 2.0;
    int    minEvents = 10;
    int    lossMode = 0;            // 0 = χ², 1 = ROI-weighted χ²
    int    maxIter = 300;
    double lambda = 0.05;           // Tikhonov strength
    bool   polarization = true;     // scatter polarization weights on/off
    bool   varianceReduction = true;// importance sampling vs brute-force analog MC
    int    debug = 0;               // vtdbg level (0 = off)
    long   debugRay = -1;           // restrict debug to one beam-ray index
    int    profile = 1;             // print the profiler report

    // the mounted stages: source[,primary][,sample][,secondary][,detector]
    std::vector<std::string> chain = {"source","primary","sample","secondary","detector"};
    int    fit = 1;                 // 0 = simulate spectra only, 1 = also optimize
    double detDistance = 1.0;       // [cm] bare detector aperture (chains w/o secondary)
    double detRadius   = 0.3;       // [cm]
    std::string measured;           // CSV of measured spectra ("" = synthetic phantom)

    // Si(Li) detector response (Detector::make)
    double detThickness = 0.30;     // active Si [cm]
    double detBeWindow  = 0.0025;   // Be window [cm]
    double detDeadLayer = 1e-4;     // Si dead layer [cm]
    double detFano      = 0.114;
    double detNoiseFWHM = 0.080;    // electronic noise FWHM [keV]

    bool has(const char* stage) const {
        for (const auto& s : chain) if (s == stage) return true;
        return false;
    }

    struct Roi   { std::string name; double energyKeV; float weight; };
    struct Layer { double z0um, z1um, w; };
    std::vector<Roi>   rois;        // for lossMode 1
    std::vector<Layer> truth;       // phantom layers (w = 1 elsewhere; later lines win)
};

/// Apply one `key=value` assignment (used for both file lines and CLI
/// overrides). Unknown keys throw, so typos surface immediately.
inline void applyKey(SetupDescr& c, const std::string& key, const std::string& val) {
    std::istringstream vs(val);
    if      (key == "n_primary")          c.nPrimary = std::atol(val.c_str());
    else if (key == "seed")               c.seed = std::strtoull(val.c_str(), nullptr, 10);
    else if (key == "nbin")               c.nBin = std::atoi(val.c_str());
    else if (key == "ebin_keV")           c.eBin = std::atof(val.c_str());
    else if (key == "expo_per_event")     c.expoPerEvent = std::atof(val.c_str());
    else if (key == "min_events")         c.minEvents = std::atoi(val.c_str());
    else if (key == "loss")               c.lossMode = std::atoi(val.c_str());
    else if (key == "max_iter")           c.maxIter = std::atoi(val.c_str());
    else if (key == "lambda")             c.lambda = std::atof(val.c_str());
    else if (key == "polarization")       c.polarization = std::atoi(val.c_str()) != 0;
    else if (key == "variance_reduction") c.varianceReduction = std::atoi(val.c_str()) != 0;
    else if (key == "debug")              c.debug = std::atoi(val.c_str());
    else if (key == "debug_ray")          c.debugRay = std::atol(val.c_str());
    else if (key == "profile")            c.profile = std::atoi(val.c_str());
    else if (key == "chain") {                       // comma- or space-separated stages
        c.chain.clear();
        std::string tok;
        for (char ch : val) {
            if (ch == ',' || ch == ' ' || ch == '\t') {
                if (!tok.empty()) c.chain.push_back(tok);
                tok.clear();
            } else tok += ch;
        }
        if (!tok.empty()) c.chain.push_back(tok);
    }
    else if (key == "fit")                c.fit = std::atoi(val.c_str());
    else if (key == "det_distance_cm")    c.detDistance = std::atof(val.c_str());
    else if (key == "det_radius_cm")      c.detRadius = std::atof(val.c_str());
    else if (key == "measured")           c.measured = val;
    else if (key == "det_thickness_cm")   c.detThickness = std::atof(val.c_str());
    else if (key == "det_be_window_cm")   c.detBeWindow = std::atof(val.c_str());
    else if (key == "det_dead_layer_cm")  c.detDeadLayer = std::atof(val.c_str());
    else if (key == "det_fano")           c.detFano = std::atof(val.c_str());
    else if (key == "det_noise_fwhm_keV") c.detNoiseFWHM = std::atof(val.c_str());
    else if (key == "roi") {
        SetupDescr::Roi r; vs >> r.name >> r.energyKeV >> r.weight;
        c.rois.push_back(r);
    }
    else if (key == "truth") {
        SetupDescr::Layer l; vs >> l.z0um >> l.z1um >> l.w;
        c.truth.push_back(l);
    }
    else throw std::runtime_error("SetupIO: unknown setup key '" + key + "'");
}

/// Apply a CLI override token of the form `key=value` (value may hold spaces
/// as separate argv words joined by the caller).
inline void applyOverride(SetupDescr& c, const std::string& token) {
    size_t eq = token.find('=');
    if (eq == std::string::npos)
        throw std::runtime_error("SetupIO: override must be key=value, got '" + token + "'");
    std::string key = stripComment(token.substr(0, eq));
    std::string val = stripComment(token.substr(eq + 1));
    applyKey(c, key, val);
}

/// Measured spectra CSV: header line, then one row per energy with the energy
/// [keV] in column 0 and one counts column per scan position. Counts are
/// rebinned onto the (nBin, eBin) grid, so any calibration works.
inline std::vector<std::vector<double>> loadMeasured(const std::string& path,
                                                     int nPos, int nBin, double eBin) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("SetupIO: cannot open " + path);
    std::vector<std::vector<double>> spec(nPos, std::vector<double>(nBin, 0.0));
    std::string raw;
    std::getline(in, raw);                                       // header
    while (std::getline(in, raw)) {
        std::vector<double> v = numbersOn(stripComment(raw));
        if (v.size() < 2) continue;
        int ch = (int)(v[0] / eBin);
        if (ch < 0 || ch >= nBin) continue;
        int cols = (int)v.size() - 1 < nPos ? (int)v.size() - 1 : nPos;
        for (int d = 0; d < cols; ++d) spec[d][ch] += v[1 + d];
    }
    return spec;
}

inline SetupDescr loadSetup(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("SetupIO: cannot open " + path);
    SetupDescr c;
    std::string raw;
    while (std::getline(in, raw)) {
        std::string line = stripComment(raw);
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = stripComment(line.substr(0, eq));
        std::string val = stripComment(line.substr(eq + 1));
        applyKey(c, key, val);
    }
    return c;
}

}  // namespace vtio
