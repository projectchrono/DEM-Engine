//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <DEM/utils/ClumpMassProperties.hpp>

#include <DEM/BdrsAndObjs.h>
#include <core/utils/Logger.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace deme {

bool computeClumpUnionMassPropsApprox(const DEMClumpTemplate& clump, double& volume, float3& center, float3& inertia) {
    volume = 0.0;
    center = make_float3(0, 0, 0);
    inertia = make_float3(0, 0, 0);
    if (clump.nComp == 0 || clump.radii.size() != clump.nComp || clump.relPos.size() != clump.nComp) {
        return false;
    }

    float3 bb_min = make_float3(std::numeric_limits<float>::infinity());
    float3 bb_max = make_float3(-std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < clump.nComp; i++) {
        const float r = clump.radii[i];
        if (!(r > DEME_TINY_FLOAT) || !std::isfinite(r)) {
            continue;
        }
        const float3 c = clump.relPos[i];
        bb_min.x = std::min(bb_min.x, c.x - r);
        bb_min.y = std::min(bb_min.y, c.y - r);
        bb_min.z = std::min(bb_min.z, c.z - r);
        bb_max.x = std::max(bb_max.x, c.x + r);
        bb_max.y = std::max(bb_max.y, c.y + r);
        bb_max.z = std::max(bb_max.z, c.z + r);
    }

    const double lx = static_cast<double>(bb_max.x) - bb_min.x;
    const double ly = static_cast<double>(bb_max.y) - bb_min.y;
    const double lz = static_cast<double>(bb_max.z) - bb_min.z;
    if (!(lx > 0.0) || !(ly > 0.0) || !(lz > 0.0) || !std::isfinite(lx) || !std::isfinite(ly) || !std::isfinite(lz)) {
        return false;
    }

    const double bbox_vol = lx * ly * lz;

    auto radical_inverse = [](uint64_t n, uint32_t base) {
        double inv_base = 1.0 / static_cast<double>(base);
        double inv_bi = inv_base;
        double val = 0.0;
        while (n) {
            const uint32_t d = static_cast<uint32_t>(n % base);
            val += static_cast<double>(d) * inv_bi;
            n /= base;
            inv_bi *= inv_base;
        }
        return val;
    };

    struct RunningStat {
        double mean = 0.0;
        double m2 = 0.0;
        size_t n = 0;
        void add(double x) {
            n++;
            const double d = x - mean;
            mean += d / static_cast<double>(n);
            const double d2 = x - mean;
            m2 += d * d2;
        }
        double var() const { return (n > 1) ? (m2 / static_cast<double>(n - 1)) : 0.0; }
    };

    // E[I], E[I x], E[I y], E[I z], E[I xx], E[I yy], E[I zz], E[I xy], E[I yz], E[I zx]
    std::array<RunningStat, 10> stats;
    constexpr double z99 = 2.576;            // 99% confidence interval factor
    constexpr double target_rel_err = 0.01;  // 1% target relative error
    constexpr size_t min_samples = 100000;
    constexpr size_t max_samples = 4000000;
    constexpr size_t check_stride = 10000;

    bool converged = false;
    size_t n_total = 0;
    for (size_t i = 0; i < max_samples; i++) {
        const uint64_t idx = static_cast<uint64_t>(i + 1);
        const double u = radical_inverse(idx, 2);
        const double v = radical_inverse(idx, 3);
        const double w = radical_inverse(idx, 5);
        const double x = static_cast<double>(bb_min.x) + u * lx;
        const double y = static_cast<double>(bb_min.y) + v * ly;
        const double z = static_cast<double>(bb_min.z) + w * lz;

        bool inside = false;
        for (size_t s = 0; s < clump.nComp; s++) {
            const double dx = x - clump.relPos[s].x;
            const double dy = y - clump.relPos[s].y;
            const double dz = z - clump.relPos[s].z;
            const double rr = static_cast<double>(clump.radii[s]) * clump.radii[s];
            if (dx * dx + dy * dy + dz * dz <= rr) {
                inside = true;
                break;
            }
        }

        const double I = inside ? 1.0 : 0.0;
        stats[0].add(I);
        stats[1].add(I * x);
        stats[2].add(I * y);
        stats[3].add(I * z);
        stats[4].add(I * x * x);
        stats[5].add(I * y * y);
        stats[6].add(I * z * z);
        stats[7].add(I * x * y);
        stats[8].add(I * y * z);
        stats[9].add(I * z * x);
        n_total++;

        if (n_total < min_samples || (n_total % check_stride) != 0) {
            continue;
        }
        const double mean_I = stats[0].mean;
        if (!(mean_I > 1e-14)) {
            continue;
        }
        const double stderr_I = std::sqrt(std::max(0.0, stats[0].var()) / static_cast<double>(n_total));
        const double rel_err_I = z99 * stderr_I / mean_I;

        const double m2_sum = stats[4].mean + stats[5].mean + stats[6].mean;
        double rel_err_m2 = 0.0;
        if (m2_sum > 1e-14) {
            const double var_sum =
                std::max(0.0, stats[4].var()) + std::max(0.0, stats[5].var()) + std::max(0.0, stats[6].var());
            const double stderr_sum = std::sqrt(var_sum / static_cast<double>(n_total));
            rel_err_m2 = z99 * stderr_sum / m2_sum;
        }

        if (rel_err_I <= target_rel_err && rel_err_m2 <= target_rel_err) {
            converged = true;
            break;
        }
    }

    const double inside_frac = stats[0].mean;
    if (!(inside_frac > 0.0) || !std::isfinite(inside_frac)) {
        return false;
    }
    volume = bbox_vol * inside_frac;
    if (!(volume > 0.0) || !std::isfinite(volume)) {
        return false;
    }

    const double cx = stats[1].mean / inside_frac;
    const double cy = stats[2].mean / inside_frac;
    const double cz = stats[3].mean / inside_frac;
    center = make_float3(static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(cz));

    const double int_xx = bbox_vol * stats[4].mean;
    const double int_yy = bbox_vol * stats[5].mean;
    const double int_zz = bbox_vol * stats[6].mean;
    const double int_xy = bbox_vol * stats[7].mean;
    const double int_yz = bbox_vol * stats[8].mean;
    const double int_zx = bbox_vol * stats[9].mean;

    double Ixx = int_yy + int_zz;
    double Iyy = int_xx + int_zz;
    double Izz = int_xx + int_yy;
    double Ixy = -int_xy;
    double Iyz = -int_yz;
    double Izx = -int_zx;

    Ixx -= volume * (cy * cy + cz * cz);
    Iyy -= volume * (cx * cx + cz * cz);
    Izz -= volume * (cx * cx + cy * cy);
    Ixy += volume * cx * cy;
    Iyz += volume * cy * cz;
    Izx += volume * cz * cx;

    inertia = make_float3(static_cast<float>(Ixx), static_cast<float>(Iyy), static_cast<float>(Izz));
    if (!converged) {
        DEME_WARNING(
            "Clump union mass/MOI estimator hit sample cap (%zu) before 99%%-confidence 1%%-error target; using best "
            "estimate.",
            n_total);
    }
    return std::isfinite(Ixx) && std::isfinite(Iyy) && std::isfinite(Izz) && Ixx > 0.0 && Iyy > 0.0 && Izz > 0.0;
}

}  // namespace deme
