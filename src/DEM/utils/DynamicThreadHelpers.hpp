//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_DYNAMIC_THREAD_HELPERS_HPP
#define DEME_DYNAMIC_THREAD_HELPERS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <DEM/Defines.h>
#include <DEM/VariableTypes.h>

namespace deme {

// dT-side future-drift tuner.
//
// The scheduler's correctness-sensitive rule is that dT consumes a kT result and immediately posts the next kT work
// order. This helper deliberately does not decide *when* to send. It only chooses the non-negative drift/margin command
// attached to the next work order, using the measured age of the contact set that just got replaced.
struct FutureDriftRegulator {
    double covered_drift_ema = 0.0;
    bool has_covered_drift_sample = false;
    unsigned int last_covered_drift = 0;
    bool force_non_decreasing_once = false;
    unsigned int cache_size = 200;
    unsigned int samples_since_reset = 0;

    void Clear() {
        const unsigned int preserved_cache_size = cache_size;
        *this = FutureDriftRegulator{};
        cache_size = preserved_cache_size;
    }

    void SetCacheSize(unsigned int n) { cache_size = std::max(1u, n); }

    void RecordContactSetCoverage(unsigned int covered_drift) {
        last_covered_drift = covered_drift;
        const double sample = static_cast<double>(covered_drift);
        if (!has_covered_drift_sample || samples_since_reset >= cache_size) {
            covered_drift_ema = sample;
            has_covered_drift_sample = true;
            samples_since_reset = 1;
            return;
        }
        samples_since_reset++;

        // React faster when the actual required coverage grows than when it falls. Undershooting can make dT wait or
        // miss contacts; overshooting mostly costs extra kT work, so we decay conservatively.
        constexpr double EMA_UP = 0.35;
        constexpr double EMA_DOWN = 0.08;
        const double alpha = (sample > covered_drift_ema) ? EMA_UP : EMA_DOWN;
        covered_drift_ema = (1.0 - alpha) * covered_drift_ema + alpha * sample;
    }

    bool HasContactSetCoverageSample() const { return has_covered_drift_sample; }

    unsigned int LastObservedContactSetCoverage() const { return last_covered_drift; }

    unsigned int Recommend(unsigned int current_cmd,
                           float target_multiple,
                           float target_more,
                           unsigned int upper_bound,
                           unsigned int tweak_step) {
        if (upper_bound == 0) {
            return 0;
        }

        current_cmd = std::min(current_cmd, upper_bound);
        if (!has_covered_drift_sample) {
            return current_cmd;
        }

        double target = covered_drift_ema * static_cast<double>(target_multiple) + static_cast<double>(target_more);
        if (!std::isfinite(target) || target < 0.0) {
            target = static_cast<double>(current_cmd);
        }
        const unsigned int target_cmd =
            std::min(upper_bound, static_cast<unsigned int>(std::ceil(std::max(0.0, target))));

        unsigned int next_cmd = current_cmd;
        const unsigned int step = std::max(1u, tweak_step);
        if (target_cmd > current_cmd) {
            next_cmd = std::min(upper_bound, current_cmd + step);
        } else if (target_cmd < current_cmd) {
            if (force_non_decreasing_once) {
                next_cmd = current_cmd;
            } else {
                next_cmd = (current_cmd > step) ? (current_cmd - step) : 0;
            }
        }

        force_non_decreasing_once = false;
        return std::min(next_cmd, upper_bound);
    }

    unsigned int BumpAfterWait(unsigned int current_cmd, unsigned int upper_bound, unsigned int tweak_step) {
        if (upper_bound == 0) {
            return 0;
        }
        force_non_decreasing_once = true;
        const unsigned int step = std::max(1u, tweak_step);
        return std::min(upper_bound, current_cmd + step);
    }
};

// Host-side snapshot of dT's old contact arrays, used only for DEBUG-verbosity lost-contact diagnostics.
struct LostContactDebugSnapshot {
    size_t nPatchContacts = 0;
    size_t nPrimitiveContacts = 0;
    std::vector<bodyID_t> idPatchA;
    std::vector<bodyID_t> idPatchB;
    std::vector<contact_t> contactTypePatch;
    std::vector<bodyID_t> contactPatchIsland;
    std::vector<bodyID_t> idPrimitiveA;
    std::vector<bodyID_t> idPrimitiveB;
    std::vector<contact_t> contactTypePrimitive;
    std::vector<contactPairs_t> geomToPatchMap;
    std::vector<float3> primitivePenetrationStorage;
    std::vector<float3> primitiveAreaStorage;
};

}  // namespace deme

#endif
