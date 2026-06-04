//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Combined-owner modular test.
//
// This test verifies:
// 1. Combined instances can be instantiated from a combined clump template
//    (single and batch).
// 2. Track(combined_instances) returns a single tracker that covers all member
//    owners and reports the instantiated member positions correctly.
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
constexpr float kVelTol = 1e-5f;
constexpr float kQuatTol = 1e-5f;
const float3 kInitPos = make_float3(1.0f, -0.5f, 0.25f);
const float3 kInitPos2 = make_float3(5.0f, 0.0f, 0.0f);
const float4 kIdentityQ = make_float4(0, 0, 0, 1);

bool approxEqual(const float3& a, const float3& b, float tol = kPosTol) {
    return std::abs(a.x - b.x) < tol && std::abs(a.y - b.y) < tol && std::abs(a.z - b.z) < tol;
}

bool approxEqualQuat(const float4& a, const float4& b, float tol = kQuatTol) {
    const float direct = std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z) + std::abs(a.w - b.w);
    const float negated = std::abs(a.x + b.x) + std::abs(a.y + b.y) + std::abs(a.z + b.z) + std::abs(a.w + b.w);
    return direct < tol || negated < tol;
}

std::pair<bodyID_t, bodyID_t> normalizePair(bodyID_t a, bodyID_t b) {
    if (b < a) {
        std::swap(a, b);
    }
    return {a, b};
}

bool containsPair(const std::vector<std::pair<bodyID_t, bodyID_t>>& contacts, bodyID_t a, bodyID_t b) {
    const auto target = normalizePair(a, b);
    return std::any_of(contacts.begin(), contacts.end(),
                       [&target](const auto& pair) { return normalizePair(pair.first, pair.second) == target; });
}

struct ScenarioResult {
    size_t combined_count_before_init = 0;
    // Single tracker covers all member owners in the batch
    std::shared_ptr<DEMTracker> tracker;
    size_t total_member_owners = 0;
    std::vector<float3> tracker_positions;
    std::vector<bodyID_t> tracker_owner_ids;
    std::vector<std::pair<bodyID_t, bodyID_t>> contacts;
};

struct MotionResult {
    std::vector<float3> initial_positions;
    std::vector<float3> final_positions;
    std::vector<float3> final_velocities;
    std::vector<float3> final_ang_velocities;
    std::vector<float4> final_orientations;
};

ScenarioResult runScenario(bool allow_intra_combined_contacts, bool use_batch = false) {
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

    std::shared_ptr<DEMCombinedInstances> combined_inst;
    if (use_batch) {
        // Batch instantiation with 2 poses
        std::vector<float3> positions = {kInitPos, kInitPos2};
        combined_inst = DEMSim.AddCombinedFromTemplate(combined_type, positions);
    } else {
        // Single-pose instantiation
        combined_inst = DEMSim.AddCombinedFromTemplate(combined_type, kInitPos, kIdentityQ);
    }

    result.combined_count_before_init = DEMSim.GetNumCombinedInstances();
    result.total_member_owners = combined_inst->member_objs.size();
    auto tracker = DEMSim.Track(combined_inst);
    result.tracker = tracker;

    if (allow_intra_combined_contacts) {
        DEMSim.SetAllowIntraCombinedOwnerContacts(true);
    }

    DEMSim.SetInitTimeStep(kStepSize);
    DEMSim.Initialize(true);  // dry-run to establish contacts

    for (size_t i = 0; i < result.total_member_owners; i++) {
        result.tracker_owner_ids.push_back(tracker->GetOwnerID(i));
        result.tracker_positions.push_back(tracker->Pos(i));
    }

    result.contacts = DEMSim.GetClumpContacts();
    return result;
}

MotionResult runMemberAccelerationScenario(bool angular) {
    MotionResult result;

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
    auto tracker = DEMSim.Track(combined_inst);

    DEMSim.SetInitTimeStep(kStepSize);
    DEMSim.Initialize();

    result.initial_positions = tracker->Positions();
    if (angular) {
        tracker->AddAngAcc(make_float3(0, 0, 2.0e6f), 1);
    } else {
        tracker->AddAcc(make_float3(1000.f, -2000.f, 500.f), 1);
    }
    DEMSim.DoDynamicsThenSync(kStepSize);

    result.final_positions = tracker->Positions();
    result.final_velocities = tracker->Velocities();
    result.final_ang_velocities = tracker->AngularVelocitiesGlobal();
    result.final_orientations = tracker->OrientationQuaternions();
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
    const auto batch = runScenario(false, true);
    const auto linear_acc = runMemberAccelerationScenario(false);
    const auto angular_acc = runMemberAccelerationScenario(true);

    std::cout << "\n--- Test 1: Combined instance is cached pre-initialize ---" << std::endl;
    std::cout << "Combined count before init (suppressed case): " << suppressed.combined_count_before_init << std::endl;
    if (suppressed.combined_count_before_init == 1) {
        std::cout << "✓ PASS: One combined instance is cached before Initialize()" << std::endl;
    } else {
        std::cout << "✗ FAIL: Expected one cached combined instance before Initialize()" << std::endl;
        test_failures++;
    }

    std::cout << "\n--- Test 2: Track(combined_instances) returns a single tracker covering all members ---"
              << std::endl;
    std::cout << "Total member owners: " << suppressed.total_member_owners << std::endl;
    if (suppressed.tracker != nullptr && suppressed.total_member_owners == 2 &&
        suppressed.tracker_owner_ids.size() == 2 &&
        suppressed.tracker_owner_ids[0] != suppressed.tracker_owner_ids[1]) {
        std::cout << "✓ PASS: Single tracker covers two distinct member owners" << std::endl;
    } else {
        std::cout << "✗ FAIL: Expected single tracker covering two distinct member owners" << std::endl;
        test_failures++;
    }

    std::cout << "\n--- Test 3: Member tracker positions match instantiated combined layout ---" << std::endl;
    const float3 expected_pos_0 = kInitPos;
    const float3 expected_pos_1 = kInitPos + make_float3(kMemberSpacing, 0, 0);
    if (suppressed.tracker_positions.size() >= 2) {
        std::cout << "Tracker pos[0]: (" << suppressed.tracker_positions[0].x << ", "
                  << suppressed.tracker_positions[0].y << ", " << suppressed.tracker_positions[0].z << ")" << std::endl;
        std::cout << "Tracker pos[1]: (" << suppressed.tracker_positions[1].x << ", "
                  << suppressed.tracker_positions[1].y << ", " << suppressed.tracker_positions[1].z << ")" << std::endl;
    }
    if (suppressed.tracker_positions.size() >= 2 && approxEqual(suppressed.tracker_positions[0], expected_pos_0) &&
        approxEqual(suppressed.tracker_positions[1], expected_pos_1)) {
        std::cout << "✓ PASS: Member positions match the combined template layout" << std::endl;
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

    std::cout << "\n--- Test 6: Batch instantiation creates correct number of member owners ---" << std::endl;
    std::cout << "Batch total member owners: " << batch.total_member_owners << std::endl;
    // 2 instances * 2 members each = 4 total member owners
    if (batch.total_member_owners == 4) {
        std::cout << "✓ PASS: Batch of 2 combined instances produced 4 member owners (2 members each)" << std::endl;
    } else {
        std::cout << "✗ FAIL: Expected 4 member owners from batch of 2 combined instances with 2 members each"
                  << std::endl;
        test_failures++;
    }

    std::cout << "\n--- Test 7: Batch instantiation positions are correct ---" << std::endl;
    const float3 batch_expected_0 = kInitPos;
    const float3 batch_expected_1 = kInitPos + make_float3(kMemberSpacing, 0, 0);
    const float3 batch_expected_2 = kInitPos2;
    const float3 batch_expected_3 = kInitPos2 + make_float3(kMemberSpacing, 0, 0);
    if (batch.tracker_positions.size() >= 4 && approxEqual(batch.tracker_positions[0], batch_expected_0) &&
        approxEqual(batch.tracker_positions[1], batch_expected_1) &&
        approxEqual(batch.tracker_positions[2], batch_expected_2) &&
        approxEqual(batch.tracker_positions[3], batch_expected_3)) {
        std::cout << "✓ PASS: All batch member positions match expected combined layouts" << std::endl;
    } else {
        std::cout << "✗ FAIL: Batch member positions do not match expected combined layouts" << std::endl;
        if (batch.tracker_positions.size() >= 4) {
            for (size_t i = 0; i < 4; i++) {
                std::cout << "  pos[" << i << "]: (" << batch.tracker_positions[i].x << ", "
                          << batch.tracker_positions[i].y << ", " << batch.tracker_positions[i].z << ")" << std::endl;
            }
        }
        test_failures++;
    }

    std::cout << "\n--- Test 8: AddAcc on one combined member preserves rigid linear motion ---" << std::endl;
    bool linear_motion_ok = false;
    if (linear_acc.initial_positions.size() >= 2 && linear_acc.final_positions.size() >= 2 &&
        linear_acc.final_velocities.size() >= 2) {
        const float3 initial_rel = linear_acc.initial_positions[1] - linear_acc.initial_positions[0];
        const float3 final_rel = linear_acc.final_positions[1] - linear_acc.final_positions[0];
        const float3 displacement_0 = linear_acc.final_positions[0] - linear_acc.initial_positions[0];
        const float3 displacement_1 = linear_acc.final_positions[1] - linear_acc.initial_positions[1];
        std::cout << "Final vel[0]: (" << linear_acc.final_velocities[0].x << ", " << linear_acc.final_velocities[0].y
                  << ", " << linear_acc.final_velocities[0].z << ")" << std::endl;
        std::cout << "Final vel[1]: (" << linear_acc.final_velocities[1].x << ", " << linear_acc.final_velocities[1].y
                  << ", " << linear_acc.final_velocities[1].z << ")" << std::endl;
        linear_motion_ok = approxEqual(final_rel, initial_rel) && approxEqual(displacement_0, displacement_1) &&
                           approxEqual(linear_acc.final_velocities[0], linear_acc.final_velocities[1], kVelTol) &&
                           length(linear_acc.final_velocities[0]) > kVelTol;
    }
    if (linear_motion_ok) {
        std::cout << "✓ PASS: AddAcc on one member moves the combined owner as one rigid body" << std::endl;
    } else {
        std::cout << "✗ FAIL: AddAcc on one member introduced relative linear motion" << std::endl;
        test_failures++;
    }

    std::cout << "\n--- Test 9: AddAngAcc on one combined member preserves rigid rotational motion ---" << std::endl;
    bool angular_motion_ok = false;
    if (angular_acc.initial_positions.size() >= 2 && angular_acc.final_positions.size() >= 2 &&
        angular_acc.final_velocities.size() >= 2 && angular_acc.final_ang_velocities.size() >= 2 &&
        angular_acc.final_orientations.size() >= 2) {
        const float3 initial_rel = angular_acc.initial_positions[1] - angular_acc.initial_positions[0];
        const float3 final_rel = angular_acc.final_positions[1] - angular_acc.final_positions[0];
        const float3 expected_member_vel =
            angular_acc.final_velocities[0] + cross(angular_acc.final_ang_velocities[0], final_rel);
        std::cout << "Final ang vel[0]: (" << angular_acc.final_ang_velocities[0].x << ", "
                  << angular_acc.final_ang_velocities[0].y << ", " << angular_acc.final_ang_velocities[0].z << ")"
                  << std::endl;
        std::cout << "Final ang vel[1]: (" << angular_acc.final_ang_velocities[1].x << ", "
                  << angular_acc.final_ang_velocities[1].y << ", " << angular_acc.final_ang_velocities[1].z << ")"
                  << std::endl;
        angular_motion_ok =
            std::abs(length(final_rel) - length(initial_rel)) < kPosTol &&
            approxEqualQuat(angular_acc.final_orientations[0], angular_acc.final_orientations[1]) &&
            approxEqual(angular_acc.final_ang_velocities[0], angular_acc.final_ang_velocities[1], kVelTol) &&
            approxEqual(angular_acc.final_velocities[1], expected_member_vel, kVelTol) &&
            length(angular_acc.final_ang_velocities[0]) > kVelTol;
    }
    if (angular_motion_ok) {
        std::cout << "✓ PASS: AddAngAcc on one member rotates the combined owner as one rigid body" << std::endl;
    } else {
        std::cout << "✗ FAIL: AddAngAcc on one member introduced non-rigid rotational motion" << std::endl;
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
