#pragma once
#include "Platform.hpp"

#if !defined(__METAL_VERSION__) && !defined(VOXTRACE_METAL)
    #include <cmath>
    #include <cstdio>
    #include "../api/XRayLibAPI.hpp"
#endif

// ── ChemElement ───────────────────────────────────────────────────────────────
// Per-element X-ray physics database with precomputed interpolation grids.
// All accessor methods compile for Kokkos (CUDA/HIP/OpenMP) and Metal MSL.
// Fixed-size arrays are mandatory — MSL cannot use dynamic allocation.

class ChemElement {
    int   z_;
    float a_;
    float rho_;

    VT_SCONSTEXPR float energy_resolution = 0.2f;
    VT_SCONSTEXPR float max_energy        = 20.0f;
    VT_SCONSTEXPR int   energy_entries    = 100;   // max_energy / energy_resolution

    VT_SCONSTEXPR float angle_resolution  = 0.005f;
    VT_SCONSTEXPR float max_angle         = 3.14159265358979323846f;
    VT_SCONSTEXPR int   angle_entries     = 628;   // (int)(VT_PI / 0.005)

    VT_SCONSTEXPR int line_entries  = 382;
    VT_SCONSTEXPR int shell_entries = 25;

    float cs_tot         [energy_entries];
    float cs_phot_prob   [energy_entries];
    float cs_ray_prob    [energy_entries];
    float cs_compt_prob  [energy_entries];
    float cs_phot_part   [shell_entries][energy_entries];
    float dcs_rayl       [energy_entries][angle_entries];
    float dcs_comp       [energy_entries][angle_entries];
    float line_energies  [line_entries];
    float rad_rate       [line_entries];
    float fluor_yield    [shell_entries];

#if !defined(__METAL_VERSION__) && !defined(VOXTRACE_METAL)
    inline void discretize() {
        a_   = XRayLibAPI::A(z_);
        rho_ = XRayLibAPI::Rho(z_);

        for (int i = 1; i < energy_entries; ++i) {
            float e = i * energy_resolution;
            cs_tot[i]        = XRayLibAPI::CS_Tot(z_, e);
            float tot        = XRayLibAPI::CS_Tot(z_, e);
            cs_phot_prob[i]  = XRayLibAPI::CS_Phot(z_, e)  / tot;
            cs_ray_prob[i]   = XRayLibAPI::CS_Ray(z_, e)   / tot;
            cs_compt_prob[i] = XRayLibAPI::CS_Compt(z_, e) / tot;

            float rsum = 0.f, csum = 0.f;
            for (int j = 0; j < angle_entries; ++j) {
                float a = j * angle_resolution;
                dcs_rayl[i][j] = XRayLibAPI::DCS_Rayl(z_, e, a);
                dcs_comp[i][j] = XRayLibAPI::DCS_Compt(z_, e, a);
                rsum += dcs_rayl[i][j];
                csum += dcs_comp[i][j];
            }
            for (int j = 0; j < angle_entries; ++j) {
                dcs_rayl[i][j] /= rsum;
                dcs_comp[i][j] /= csum;
            }
        }

        for (int i = 0; i < shell_entries; ++i) {
            fluor_yield[i] = XRayLibAPI::FluorY(z_, i);
            for (int j = 1; j < energy_entries; ++j) {
                float e = j * energy_resolution;
                cs_phot_part[i][j] = XRayLibAPI::CS_Phot_Part(z_, i, e)
                                   / XRayLibAPI::CS_Phot(z_, e);
            }
        }

        for (int i = 0; i < line_entries; ++i) {
            line_energies[i] = XRayLibAPI::LineE(z_, i * -1 - 1);
            rad_rate[i]      = XRayLibAPI::RadRate(z_, i * -1 - 1);
        }
    }
#endif

public:

    KOKKOS_INLINE_FUNCTION ChemElement() {}

#if !defined(__METAL_VERSION__) && !defined(VOXTRACE_METAL)
    inline ChemElement(int z) : z_(z) { discretize(); }
#endif

    // ── Basic properties ──────────────────────────────────────────────────────
    KOKKOS_INLINE_FUNCTION float A()   const VT_DEVICE_METH { return a_; }
    KOKKOS_INLINE_FUNCTION int   Z()   const VT_DEVICE_METH { return z_; }
    KOKKOS_INLINE_FUNCTION float Rho() const VT_DEVICE_METH { return rho_; }

    // ── DB accessors ──────────────────────────────────────────────────────────
    KOKKOS_INLINE_FUNCTION float Fluor_Y      (int shell) const VT_DEVICE_METH { return fluor_yield[shell]; }
    KOKKOS_INLINE_FUNCTION float Rad_Rate     (int line)  const VT_DEVICE_METH { return rad_rate[line]; }
    KOKKOS_INLINE_FUNCTION float Line_Energy  (int line)  const VT_DEVICE_METH { return line_energies[line]; }

    KOKKOS_INLINE_FUNCTION float CS_Tot        (float e) const VT_DEVICE_METH { return interpolate(e, energy_resolution, cs_tot); }
    KOKKOS_INLINE_FUNCTION float CS_Phot_Prob  (float e) const VT_DEVICE_METH { return interpolate(e, energy_resolution, cs_phot_prob); }
    KOKKOS_INLINE_FUNCTION float CS_Rayl_Prob  (float e) const VT_DEVICE_METH { return interpolate(e, energy_resolution, cs_ray_prob); }
    KOKKOS_INLINE_FUNCTION float CS_Compt_Prob (float e) const VT_DEVICE_METH { return interpolate(e, energy_resolution, cs_compt_prob); }
    KOKKOS_INLINE_FUNCTION float CS_Phot_Part_Prob(int shell, float e) const VT_DEVICE_METH {
        return interpolate(e, energy_resolution, cs_phot_part[shell]);
    }
    KOKKOS_INLINE_FUNCTION float DCS_Rayl(float e, float a) const VT_DEVICE_METH {
        return interpolate(a, angle_resolution, dcs_rayl[(int)lroundf(e / energy_resolution)]);
    }
    KOKKOS_INLINE_FUNCTION float DCS_Compt(float e, float a) const VT_DEVICE_METH {
        return interpolate(a, angle_resolution, dcs_comp[(int)lroundf(e / energy_resolution)]);
    }

    // ── Sampling decisions ────────────────────────────────────────────────────

    KOKKOS_INLINE_FUNCTION int getInteractionType(float e, float r) const VT_DEVICE_METH {
        if (r <= CS_Phot_Prob(e))                          return 0;
        if (r <= CS_Phot_Prob(e) + CS_Rayl_Prob(e))       return 1;
        return 2;
    }

    KOKKOS_INLINE_FUNCTION int getExcitedShell(float e, float r) const VT_DEVICE_METH {
        float sum = 0.f;
        int s = 0;
        for (; s < shell_entries; ++s) {
            sum += CS_Phot_Part_Prob(s, e);
            if (sum > r) break;
        }
        return s;
    }

    KOKKOS_INLINE_FUNCTION float getThetaRayl(float e, float r) const VT_DEVICE_METH {
        return _sampleAngle(e, r, dcs_rayl);
    }

    KOKKOS_INLINE_FUNCTION float getThetaCompt(float e, float r) const VT_DEVICE_METH {
        return _sampleAngle(e, r, dcs_comp);
    }

    KOKKOS_INLINE_FUNCTION float getComptEnergy(float e, float theta) const VT_DEVICE_METH {
        return e / (1.0f + (e / 510.998928f) * (1.0f - cosf(theta)));
    }

    KOKKOS_INLINE_FUNCTION int getTransition(int shell, float r) const VT_DEVICE_METH {
        const int shell_lines[shell_entries][2] = {
            {0,28},
            {29,57},{85,112},{113,135},
            {136,157},{158,179},{180,199},{200,218},
            {219,236},{237,253},{254,269},{270,284},{285,298},{299,311},{312,323},
            {321,334},{335,344},{345,353},{354,361},{362,368},{369,371},{372,373},
            {374,377},{378,380},{381,382}
        };
        float sum = 0.f;
        int line = shell_lines[shell][0];
        for (int i = shell_lines[shell][0]; i <= shell_lines[shell][1]; ++i) {
            sum += Rad_Rate(i);
            line = i;
            if (sum > r) break;
        }
        return line;
    }

    // ── Helper ────────────────────────────────────────────────────────────────
    KOKKOS_INLINE_FUNCTION float interpolate(float arg, float step, const VT_DEVICE float* vec) const VT_DEVICE_METH {
        float x = arg / step;
        int   i = (int)ceilf(x);
        if (i < 1) return vec[0];
        float x1 = (i - 1) * step, x2 = i * step;
        return vec[i-1] + (vec[i] - vec[i-1]) / (x2 - x1) * (arg - x1);
    }

private:
    KOKKOS_INLINE_FUNCTION float _sampleAngle(float e, float r,
                                               const VT_DEVICE float tbl[energy_entries][angle_entries]) const VT_DEVICE_METH {
        int ei = (int)lroundf(e / energy_resolution);
        int i = 0;
        float sum = 0.f;
        for (; i < angle_entries; ++i) {
            sum += interpolate((float)i * angle_resolution, angle_resolution, tbl[ei]);
            if (sum > r) break;
        }
        if (i < angle_entries - 1) {
            float d1 = interpolate((float)i       * angle_resolution, angle_resolution, tbl[ei]);
            float d2 = interpolate((float)(i + 1) * angle_resolution, angle_resolution, tbl[ei]);
            return (float)i * angle_resolution
                 + angle_resolution / (d2 - d1) * (sum - (float)i * d1);
        }
        return (float)angle_entries * angle_resolution;
    }
};
