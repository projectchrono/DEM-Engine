//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Combined-owner modular test.
//
// This test verifies:
// 1. Combined instances can be instantiated from a combined clump template.
// 2. Track(combined_instance) returns one tracker per member owner and reports
//    the instantiated member positions correctly.
// 3. Contacts between owners in the same combined group are suppressed by
//    default.
// 4. SetAllowIntraCombinedOwnerContacts(true) re-enables those internal
//    contacts.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

using namespace deme;

namespace {

constexpr float kRadius = 0.5f;
constexpr float kMass = 1.0f;
constexpr float kMemberSpacing = 0.75f;  // < 2 * kRadius, so internal contact exists if allowed
constexpr float kStepSize = 1e-5f;
constexpr float kPosTol = 1e-5f;
const float3 kInitPos = make_float3(1.0f, -0.5f, 0.25f);
const float4 kIdentityQ = make_float4(0, 0, 0, 1);

bool approxEqual(const float3& a, const float3& b, float tol = kPosTol) {
    return std::abs(a.x - b.x) < tol && std::abs(a.y - b.y) < tol && std::abs(a.z - b.z) < tol;
}

std::pair<bodyID_t, bodyID_t> normalizePair(bodyID_t a, bodyID_t b) {
    if (b < a) {
        std::swap(a, b);
    }
    return {a, b};
}

bool containsPair(const std::vector<std::pair<bodyID_t, bodyID_t>>& contacts, bodyID_t a, bodyID_t b) {
    const auto target = normalizePair(a, b);
    return std::any_of(contacts.begin(), contacts.end(), [&target](const auto& pair) {
        return normalizePair(pair.first, pair.second) == target;
    });
}

struct ScenarioResult {
    size_t combined_count_before_init = 0;
    size_t tracker_count = 0;
    std::vector<bodyID_t> tracker_owner_ids;
    std::vector<float3> tracker_positions;
    std::vector<std::pair<bodyID_t, bodyID_t>> contacts;
};

ScenarioResult runScenario(bool allow_intra_combined_contacts) {
    ScenarioResult result;

    DEMSolver DEMSim;
    DEMSim.SetVerbosity("ERROR");
    DEMSim.InstructBoxDomainDimension(10, 10, 10);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, 0));
    DEMSim.SetCDUpdateFreq(1);

    auto mat = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.2}, {"mu", 0.5}, {"Crr", 0.0}});
    auto sphere = DEMSim.LoadSphereType(kMass, kRadius, mat);

    std::vector<std::shared_ptr<DEMClumpTemplate>> component_templates = {sphere, sphere};
    std::vector<float3> component_rel_pos = {make_float3(0, 0, 0), make_float3(kMemberSpacing, 0, 0)};

    auto combined_type = DEMSim.LoadCombinedClumpType(component_templates, component_rel_pos, {}, 0);
    auto combined_inst = DEMSim.AddCombinedFromTemplate(combined_type, kInitPos, kIdentityQ);

    result.combined_count_before_init = DEMSim.GetNumCombinedInstances();
    auto trackers = DEMSim.Track(combined_inst);
    result.tracker_count = trackers.size();

    if (allow_intra_combined_contacts) {
        DEMSim.SetAllowIntraCombinedOwnerContacts(true);
    }

    DEMSim.SetInitTimeStep(kStepSize);
    DEMSim.Initialize();

    for (const auto& tracker : trackers) {
        result.tracker_owner_ids.push_back(tracker->GetOwnerID());
        result.tracker_positions.push_back(tracker->Pos());
    }

    result.contacts = DEMSim.GetClumpContacts();
    return result;
}

void printContacts(const std::vector<std::pair<bodyID_t, bodyID_t>>& contacts) {
    if (contacts.empty()) {
        std::cout << "none";
        return;
    }
    for (size_t i = 0; i < contacts.size(); i++) {
        std::cout << "(" << contacts[i].first << ", " << contacts[i].second << ")";
        if (i + 1 < contacts.size()) {
            std::cout << ", ";
        }
    }
}

}  // namespace

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "DEM Combined Owner Test" << std::endl;
    std::cout << "========================================" << std::endl;

    int test_failures = 0;

    const auto suppressed = runScenario(false);
    const auto allowed = runScenario(true);

    std::cout << "\n--- Test 1: Combined instance is cached pre-initialize ---" << std::endl;
    std::cout << "Combined count before init (suppressed case): " << suppressed.combined_count_before_init << std::endl;
    if (suppressed.combined_count_before_init == 1) {
        std::cout << "✓ PASS: One combined instance is cached before Initialize()" << std::endl;
    } else {
        std::cout << "✗ FAIL: Expected one cached combined instance before Initialize()" << std::endl;
        test_failures++;
    }

    std::cout << "\n--- Test 2: Track(combined_instance) returns member trackers ---" << std::endl;
    std::cout << "Trackers returned: " << suppressed.tracker_count << std::endl;
    if (suppressed.tracker_count == 2 && suppressed.tracker_owner_ids.size() == 2 &&
        suppressed.tracker_owner_ids[0] != suppressed.tracker_owner_ids[1]) {
        std::cout << "✓ PASS: Combined instance produced one tracker per distinct member owner" << std::endl;
    } else {
        std::cout << "✗ FAIL: Expected two distinct member trackers from Track(combined_instance)" << std::endl;
        test_failures++;
    }

    std::cout << "\n--- Test 3: Member tracker positions match instantiated combined layout ---" << std::endl;
    const float3 expected_pos_0 = kInitPos;
    const float3 expected_pos_1 = kInitPos + make_float3(kMemberSpacing, 0, 0);
    if (suppressed.tracker_positions.size() >= 2) {
        std::cout << "Tracker 0 position: (" << suppressed.tracker_positions[0].x << ", "
                  << suppressed.tracker_positions[0].y << ", " << suppressed.tracker_positions[0].z << ")" << std::endl;
        std::cout << "Tracker 1 position: (" << suppressed.tracker_positions[1].x << ", "
                  << suppressed.tracker_positions[1].y << ", " << suppressed.tracker_positions[1].z << ")" << std::endl;
    }
    if (suppressed.tracker_positions.size() >= 2 && approxEqual(suppressed.tracker_positions[0], expected_pos_0) &&
        approxEqual(suppressed.tracker_positions[1], expected_pos_1)) {
        std::cout << "✓ PASS: Member trackers report the expected instantiated positions" << std::endl;
    } else {
        std::cout << "✗ FAIL: Member tracker positions do not match the combined template layout" << std::endl;
        test_failures++;
    }

    std::cout << "\n--- Test 4: Intra-combined contacts are suppressed by default ---" << std::endl;
    std::cout << "Clump contacts with default setting: ";
    printContacts(suppressed.contacts);
    std::cout << std::endl;
    if (suppressed.tracker_owner_ids.size() >= 2 &&
        !containsPair(suppressed.contacts, suppressed.tracker_owner_ids[0], suppressed.tracker_owner_ids[1])) {
        std::cout << "✓ PASS: Internal combined-member contact pair is absent by default" << std::endl;
    } else {
        std::cout << "✗ FAIL: Internal combined-member contact pair should be suppressed by default" << std::endl;
        test_failures++;
    }

    std::cout << "\n--- Test 5: Intra-combined contacts can be enabled explicitly ---" << std::endl;
    std::cout << "Clump contacts with allow=true: ";
    printContacts(allowed.contacts);
    std::cout << std::endl;
    if (allowed.tracker_owner_ids.size() >= 2 &&
        containsPair(allowed.contacts, allowed.tracker_owner_ids[0], allowed.tracker_owner_ids[1])) {
        std::cout << "✓ PASS: Internal combined-member contact pair appears when explicitly enabled" << std::endl;
    } else {
        std::cout << "✗ FAIL: Expected internal combined-member contact pair when allow=true" << std::endl;
        test_failures++;
    }

    std::cout << "\n========================================" << std::endl;
    if (test_failures == 0) {
        std::cout << "All tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } else {
        std::cout << "FAILED " << test_failures << " test(s)" << std::endl;
        std::cout << "========================================" << std::endl;
        return 1;
    }
}
