// Copyright (c) 2021, SBEL GPU Development Team
// Copyright (c) 2021, University of Wisconsin - Madison
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "DEM/API.h"
#include "core/ApiVersion.h"
#include "core/utils/DataMigrationHelper.hpp"

using namespace deme;

namespace {

constexpr float kTolerance = 2e-5f;

bool close(float a, float b) {
    return std::abs(a - b) <= kTolerance;
}

bool close(const float3& a, const float3& b) {
    return close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z);
}

bool close(const float4& a, const float4& b) {
    return close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z) && close(a.w, b.w);
}

// The direct wrench reduction uses owner-local contact arms, while its reference below reconstructs those arms by
// subtracting packed global float positions. Compare resultants with scale-aware tolerances so coordinate packing and
// atomic accumulation order do not turn an algebraically equivalent result into a brittle test.
bool closeWrenchComponent(float a, float b) {
    constexpr float absolute_tolerance = 2e-4f;
    constexpr float relative_tolerance = 2e-5f;
    return std::abs(a - b) <= absolute_tolerance + relative_tolerance * std::max(std::abs(a), std::abs(b));
}

bool compareWrenchVectors(const std::vector<float3>& actual,
                          const std::vector<float3>& expected,
                          const std::string& label) {
    if (actual.size() != expected.size()) {
        std::cerr << "FAIL: " << label << " size mismatch." << std::endl;
        return false;
    }
    for (size_t i = 0; i < actual.size(); i++) {
        const float3& a = actual[i];
        const float3& b = expected[i];
        if (!closeWrenchComponent(a.x, b.x) || !closeWrenchComponent(a.y, b.y) || !closeWrenchComponent(a.z, b.z)) {
            std::cerr << "FAIL: " << label << " differs at index " << i << ": actual=(" << a.x << ", " << a.y << ", "
                      << a.z << "), expected=(" << b.x << ", " << b.y << ", " << b.z << ")." << std::endl;
            return false;
        }
    }
    return true;
}

// Owner positions are stored as a voxel ID plus uint16 sub-voxel coordinates. A device set/get round trip therefore
// has a domain-dependent quantization error even though the API input and output are float3 values.
bool comparePositionVectors(const std::vector<float3>& actual,
                            const std::vector<float3>& expected,
                            const std::string& label) {
    constexpr float position_tolerance = 2e-4f;
    if (actual.size() != expected.size()) {
        std::cerr << "FAIL: " << label << " size mismatch." << std::endl;
        return false;
    }
    for (size_t i = 0; i < actual.size(); i++) {
        const float3& a = actual[i];
        const float3& b = expected[i];
        if (std::abs(a.x - b.x) > position_tolerance || std::abs(a.y - b.y) > position_tolerance ||
            std::abs(a.z - b.z) > position_tolerance) {
            std::cerr << "FAIL: " << label << " differs at index " << i << ": actual=(" << a.x << ", " << a.y << ", "
                      << a.z << "), expected=(" << b.x << ", " << b.y << ", " << b.z << ")." << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T>
class TestDeviceBuffer {
  public:
    TestDeviceBuffer(size_t count, int device) : m_count(count), m_device(device) {
        ScopedCudaDevice scope(device);
        DEME_GPU_CALL(cudaMalloc(reinterpret_cast<void**>(&m_pointer), count * sizeof(T)));
    }

    ~TestDeviceBuffer() {
        ScopedCudaDevice scope(m_device);
        (void)cudaFree(m_pointer);
    }

    T* data() { return m_pointer; }
    size_t size() const { return m_count; }
    int device() const { return m_device; }

    void FromHost(const std::vector<T>& input) {
        if (input.size() > m_count) {
            DEME_ERROR("Test device input has %zu elements but its buffer holds only %zu.", input.size(), m_count);
        }
        ScopedCudaDevice scope(m_device);
        DEME_GPU_CALL(cudaMemcpy(m_pointer, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    std::vector<T> ToHost(size_t count = 0) const {
        const size_t output_count = count == 0 ? m_count : count;
        std::vector<T> output(output_count);
        ScopedCudaDevice scope(m_device);
        DEME_GPU_CALL(cudaMemcpy(output.data(), m_pointer, output_count * sizeof(T), cudaMemcpyDeviceToHost));
        return output;
    }

  private:
    T* m_pointer = nullptr;
    size_t m_count;
    int m_device;
};

template <typename T>
bool compareVectors(const std::vector<T>& actual, const std::vector<T>& expected, const std::string& label) {
    if (actual.size() != expected.size()) {
        std::cerr << "FAIL: " << label << " size mismatch." << std::endl;
        return false;
    }
    for (size_t i = 0; i < actual.size(); i++) {
        if (!close(actual[i], expected[i])) {
            std::cerr << "FAIL: " << label << " differs at index " << i << "." << std::endl;
            return false;
        }
    }
    return true;
}

template <>
bool compareVectors<unsigned int>(const std::vector<unsigned int>& actual,
                                  const std::vector<unsigned int>& expected,
                                  const std::string& label) {
    if (actual != expected) {
        std::cerr << "FAIL: " << label << " differs." << std::endl;
        return false;
    }
    return true;
}

bool testFixedOwnerData(int visible_devices) {
    DEMSolver solver(1);
    solver.SetVerbosity("ERROR");
    solver.InstructBoxDomainDimension(8, 8, 8);
    solver.SetGravitationalAcceleration(make_float3(0, 0, 0));
    solver.SetCDUpdateFreq(1);

    // Use a deterministic force-only torque contribution so the wrench comparison covers extra torque without
    // depending on the Hertz model's collision-age gate for rolling resistance.
    auto force_model = solver.DefineContactForceModel(R"(
if (overlapDepth > 0.0) {
    force = B2A * (1.0e3f * static_cast<float>(overlapDepth));
    torque_only_force = make_float3(0.f, 17.f, 0.f);
}
)");
    force_model->SetPerOwnerWildcards({"retrieval_tag"});
    auto material = solver.LoadMaterial({{"E", 1e7}, {"nu", 0.3}, {"CoR", 0.2}, {"mu", 0.4}, {"Crr", 0.02}});
    auto sphere = solver.LoadSphereType(2.5f, 0.5f, material);
    sphere->SetMOI(make_float3(0.21f, 0.37f, 0.52f));

    const std::vector<float3> positions = {make_float3(1.0f, 1.0f, 1.0f), make_float3(1.8f, 1.0f, 1.0f),
                                           make_float3(4.0f, 2.0f, 1.5f)};
    const std::vector<float3> velocities = {make_float3(1, 2, 3), make_float3(-1, 0.5f, 2), make_float3(0, -2, 1)};
    const std::vector<float3> angular_velocities = {make_float3(1, 0, 0), make_float3(0, 2, 0), make_float3(0, 0, 3)};
    const float half_sqrt_two = std::sqrt(0.5f);
    const std::vector<float4> orientations = {make_float4(0, 0, half_sqrt_two, half_sqrt_two),
                                              make_float4(half_sqrt_two, 0, 0, half_sqrt_two), make_float4(0, 0, 0, 1)};
    const std::vector<unsigned int> families = {2, 3, 4};
    const std::vector<float> wildcards = {10.5f, -2.0f, 7.25f};

    DEMClumpBatch input_batch(3);
    input_batch.SetTypes(sphere);
    input_batch.SetPos(positions);
    input_batch.SetVel(velocities);
    input_batch.SetAngVel(angular_velocities);
    input_batch.SetOriQ(orientations);
    input_batch.SetFamilies(families);
    input_batch.AddOwnerWildcard("retrieval_tag", wildcards);
    auto batch = solver.AddClumps(input_batch);
    auto tracker = solver.Track(batch);

    solver.SetTimeStepSize(1e-5);
    solver.Initialize();

    const size_t count = positions.size();
    TestDeviceBuffer<float3> float3_output(count, 0);
    TestDeviceBuffer<float4> float4_output(count, 0);
    TestDeviceBuffer<float> float_output(count, 0);
    TestDeviceBuffer<unsigned int> uint_output(count, 0);

    // Set all coupling state directly from CUDA memory, then round-trip it through the independent output kernels.
    const std::vector<float3> prescribed_positions = {make_float3(1.1f, 1.2f, 1.3f), make_float3(1.85f, 1.1f, 1.0f),
                                                      make_float3(4.1f, 2.2f, 1.7f)};
    const std::vector<float3> prescribed_velocities = {make_float3(2, 3, 4), make_float3(-2, 1, 0.5f),
                                                       make_float3(0.25f, -1, 2)};
    const std::vector<float3> prescribed_global_angular_velocities = {
        make_float3(0.5f, 1.5f, -2), make_float3(-1, 0.25f, 2.5f), make_float3(3, -2, 1)};
    const std::vector<float4> prescribed_orientations = {make_float4(0, 0, 0, 1),
                                                         make_float4(0, half_sqrt_two, 0, half_sqrt_two),
                                                         make_float4(0.182574f, 0.365148f, 0.547723f, 0.730297f)};
    const std::vector<float4> unnormalized_orientations = {
        prescribed_orientations[0] * 2.f, prescribed_orientations[1] * 3.f, prescribed_orientations[2] * 4.f};
    TestDeviceBuffer<float3> float3_input(count, 0);
    TestDeviceBuffer<float4> float4_input(count, 0);
    float3_input.FromHost(prescribed_positions);
    tracker->SetPositionsFromDevice(float3_input.data(), 0, false);
    float4_input.FromHost(unnormalized_orientations);
    tracker->SetOrientationQuaternionsFromDevice(float4_input.data(), 0, false);
    float3_input.FromHost(prescribed_velocities);
    tracker->SetVelocitiesFromDevice(float3_input.data(), 0, false);
    float3_input.FromHost(prescribed_global_angular_velocities);
    tracker->SetAngularVelocitiesGlobalFromDevice(float3_input.data(), 0, false);

    solver.GetOwnerPositionToDevice(float3_output.data(), count, 0, tracker->GetOwnerID(), count);
    if (!compareVectors(float3_output.ToHost(), prescribed_positions, "device-input positions"))
        return false;
    solver.GetOwnerOriQToDevice(float4_output.data(), count, 0, tracker->GetOwnerID(), count);
    if (!compareVectors(float4_output.ToHost(), prescribed_orientations, "device-input orientations"))
        return false;
    solver.GetOwnerVelocityToDevice(float3_output.data(), count, 0, tracker->GetOwnerID(), count);
    if (!compareVectors(float3_output.ToHost(), prescribed_velocities, "device-input velocities"))
        return false;
    solver.GetOwnerAngVelGlobalToDevice(float3_output.data(), count, 0, tracker->GetOwnerID(), count);
    if (!compareVectors(float3_output.ToHost(), prescribed_global_angular_velocities,
                        "device-input global angular velocities"))
        return false;

    // The local-frame flavor has the same physical interpretation as the original host SetOwnerAngVel API.
    const std::vector<float3> prescribed_local_angular_velocities = {
        make_float3(1.f, 2.f, 3.f), make_float3(-4.f, 5.f, -6.f), make_float3(0.25f, 0.5f, 0.75f)};
    float3_input.FromHost(prescribed_local_angular_velocities);
    tracker->SetAngularVelocitiesFromDevice(float3_input.data(), 0, false);
    solver.GetOwnerAngVelLocalToDevice(float3_output.data(), count, 0, tracker->GetOwnerID(), count);
    if (!compareVectors(float3_output.ToHost(), prescribed_local_angular_velocities,
                        "device-input local angular velocities"))
        return false;

    // Host and device orientation setters both normalize valid quaternions before storing them.
    solver.SetOwnerOriQ(tracker->GetOwnerID(), unnormalized_orientations);
    if (!compareVectors(tracker->OrientationQuaternions(), prescribed_orientations, "host-normalized orientations"))
        return false;

    // The common one-owner call uses the default count while retaining explicit pointer-device information.
    float3_input.FromHost({prescribed_velocities.front()});
    solver.SetOwnerVelocityFromDevice(tracker->GetOwnerID(), float3_input.data(), 0);
    solver.GetOwnerVelocityToDevice(float3_output.data(), count, 0, tracker->GetOwnerID());
    if (!compareVectors(float3_output.ToHost(1), {prescribed_velocities.front()}, "default one-owner exchange"))
        return false;

    tracker->PositionsToDevice(float3_output.data(), count, 0, false);
    if (!compareVectors(float3_output.ToHost(), tracker->Positions(), "positions"))
        return false;
    tracker->VelocitiesToDevice(float3_output.data(), count, 0, false);
    if (!compareVectors(float3_output.ToHost(), tracker->Velocities(), "velocities"))
        return false;
    tracker->AngularVelocitiesLocalToDevice(float3_output.data(), count, 0, false);
    if (!compareVectors(float3_output.ToHost(), tracker->AngularVelocitiesLocal(), "local angular velocities"))
        return false;
    tracker->AngularVelocitiesGlobalToDevice(float3_output.data(), count, 0, false);
    if (!compareVectors(float3_output.ToHost(), tracker->AngularVelocitiesGlobal(), "global angular velocities"))
        return false;
    tracker->OrientationQuaternionsToDevice(float4_output.data(), count, 0, false);
    if (!compareVectors(float4_output.ToHost(), tracker->OrientationQuaternions(), "orientations"))
        return false;
    tracker->ContactAccelerationsToDevice(float3_output.data(), count, 0, false);
    if (!compareVectors(float3_output.ToHost(), tracker->ContactAccelerations(), "contact accelerations"))
        return false;
    tracker->ContactAngularAccelerationsLocalToDevice(float3_output.data(), count, 0, false);
    if (!compareVectors(float3_output.ToHost(), tracker->ContactAngularAccelerationsLocal(),
                        "local contact angular accelerations"))
        return false;
    tracker->ContactAngularAccelerationsGlobalToDevice(float3_output.data(), count, 0, false);
    if (!compareVectors(float3_output.ToHost(), tracker->ContactAngularAccelerationsGlobal(),
                        "global contact angular accelerations"))
        return false;
    tracker->FamiliesToDevice(uint_output.data(), count, 0, false);
    if (!compareVectors(uint_output.ToHost(), tracker->GetFamilies(), "families"))
        return false;
    tracker->MassesToDevice(float_output.data(), count, 0, false);
    if (!compareVectors(float_output.ToHost(), tracker->Masses(), "masses"))
        return false;
    tracker->MOIsToDevice(float3_output.data(), count, 0, false);
    if (!compareVectors(float3_output.ToHost(), tracker->MOIs(), "moments of inertia"))
        return false;
    tracker->OwnerWildcardValuesToDevice("retrieval_tag", float_output.data(), count, 0, false);
    if (!compareVectors(float_output.ToHost(), tracker->GetOwnerWildcardValues("retrieval_tag"), "owner wildcards"))
        return false;

    TestDeviceBuffer<float3> range_output(2, 0);
    solver.GetOwnerVelocityToDevice(range_output.data(), 2, 0, tracker->GetOwnerID(1), 2);
    const auto host_velocities = tracker->Velocities();
    if (!compareVectors(range_output.ToHost(), std::vector<float3>(host_velocities.begin() + 1, host_velocities.end()),
                        "nonzero owner range"))
        return false;

    // Repeated output exercises reusable scratch/staging ownership and ensures no query leaves a claimed temp buffer.
    for (int iteration = 0; iteration < 3; iteration++) {
        tracker->PositionsToDevice(float3_output.data(), count, 0);
    }

    // Disabling validation trusts caller-provided metadata. The allocation really has `count` elements, so declaring
    // zero capacity here safely verifies that the optional capacity check is bypassed.
    tracker->PositionsToDevice(float3_output.data(), 0, 0, false);
    if (!compareVectors(float3_output.ToHost(), tracker->Positions(), "unchecked positions"))
        return false;

    // These calls intentionally throw. Suppress their expected logger output so a passing test does not look broken.
    solver.SetVerbosity("QUIET");
    bool capacity_rejected = false;
    try {
        tracker->PositionsToDevice(float3_output.data(), count - 1, 0);
    } catch (const SolverException&) {
        capacity_rejected = true;
    }
    if (!capacity_rejected) {
        std::cerr << "FAIL: insufficient fixed-output capacity was not rejected." << std::endl;
        return false;
    }

    bool host_pointer_rejected = false;
    std::vector<float3> host_only_output(count);
    try {
        tracker->PositionsToDevice(host_only_output.data(), count, 0);
    } catch (const SolverException&) {
        host_pointer_rejected = true;
    }
    if (!host_pointer_rejected) {
        std::cerr << "FAIL: ordinary host memory was accepted as CUDA device output." << std::endl;
        return false;
    }

    bool null_input_rejected = false;
    try {
        solver.SetOwnerPositionFromDevice(tracker->GetOwnerID(), nullptr, 0);
    } catch (const SolverException&) {
        null_input_rejected = true;
    }
    if (!null_input_rejected) {
        std::cerr << "FAIL: null CUDA owner-state input was not rejected." << std::endl;
        return false;
    }

    bool zero_orientation_rejected = false;
    try {
        float4_input.FromHost({make_float4(0.f)});
        solver.SetOwnerOriQFromDevice(tracker->GetOwnerID(), float4_input.data(), 0);
    } catch (const SolverException&) {
        zero_orientation_rejected = true;
    }
    if (!zero_orientation_rejected) {
        std::cerr << "FAIL: zero-length CUDA owner orientation was not rejected." << std::endl;
        return false;
    }

    bool input_range_rejected = false;
    try {
        solver.SetOwnerVelocityFromDevice(tracker->GetOwnerID(2), float3_input.data(), 0, 2);
    } catch (const SolverException&) {
        input_range_rejected = true;
    }
    if (!input_range_rejected) {
        std::cerr << "FAIL: out-of-range CUDA owner-state input was not rejected." << std::endl;
        return false;
    }

    if (visible_devices >= 2) {
        TestDeviceBuffer<float3> peer_output(count, 1);
        tracker->PositionsToDevice(peer_output.data(), count, 1);
        if (!compareVectors(peer_output.ToHost(), tracker->Positions(), "cross-device positions"))
            return false;

        bool wrong_device_rejected = false;
        try {
            tracker->PositionsToDevice(peer_output.data(), count, 0);
        } catch (const SolverException&) {
            wrong_device_rejected = true;
        }
        if (!wrong_device_rejected) {
            std::cerr << "FAIL: mismatched pointer device declaration was not rejected." << std::endl;
            return false;
        }

        const std::vector<float3> remote_velocities = {make_float3(3, -2, 1), make_float3(-4, 5, -6),
                                                       make_float3(7, 8, -9)};
        TestDeviceBuffer<float3> peer_input(count, 1);
        peer_input.FromHost(remote_velocities);
        solver.SetOwnerVelocityFromDevice(tracker->GetOwnerID(), peer_input.data(), 1, count);
        solver.GetOwnerVelocityToDevice(float3_output.data(), count, 0, tracker->GetOwnerID(), count);
        if (!compareVectors(float3_output.ToHost(), remote_velocities, "cross-device velocity input"))
            return false;

        bool wrong_source_device_rejected = false;
        try {
            solver.SetOwnerVelocityFromDevice(tracker->GetOwnerID(), peer_input.data(), 0, count);
        } catch (const SolverException&) {
            wrong_source_device_rejected = true;
        }
        if (!wrong_source_device_rejected) {
            std::cerr << "FAIL: mismatched CUDA owner-state source-device declaration was not rejected." << std::endl;
            return false;
        }
    }
    solver.SetVerbosity("ERROR");

    // Refresh contacts and forces only after all prescribed state is in place. Besides producing a coherent wrench
    // snapshot, this lets the fixed-field exchange checks remain independent of contact-detection setup.
    solver.RequestContactUpdate();
    solver.DoDynamicsThenSync(1e-5);
    const size_t contact_capacity = solver.GetNumContacts();
    if (contact_capacity == 0) {
        std::cerr << "FAIL: contact-force retrieval scenario produced no contacts." << std::endl;
        return false;
    }
    TestDeviceBuffer<float3> contact_points(contact_capacity, 0);
    TestDeviceBuffer<float3> contact_forces(contact_capacity, 0);
    TestDeviceBuffer<float3> contact_torques(contact_capacity, 0);

    std::vector<float3> host_points;
    std::vector<float3> host_forces;
    std::vector<float3> host_torques;
    const size_t host_count = tracker->GetContactForcesAndGlobalTorqueForAll(host_points, host_forces, host_torques);
    const size_t device_count = tracker->GetContactForcesAndGlobalTorqueForAllToDevice(
        contact_points.data(), contact_forces.data(), contact_torques.data(), contact_capacity, 0);
    if (device_count != host_count ||
        !compareVectors(contact_points.ToHost(device_count), host_points, "contact points") ||
        !compareVectors(contact_forces.ToHost(device_count), host_forces, "contact forces") ||
        !compareVectors(contact_torques.ToHost(device_count), host_torques, "global contact torques")) {
        return false;
    }

    const size_t force_only_count =
        tracker->GetContactForcesForAllToDevice(contact_points.data(), contact_forces.data(), contact_capacity, 0);
    if (force_only_count != host_count ||
        !compareVectors(contact_forces.ToHost(force_only_count), host_forces, "force-only contact retrieval")) {
        return false;
    }

    const size_t local_count = tracker->GetContactForcesAndLocalTorqueForAll(host_points, host_forces, host_torques);
    const size_t local_device_count = tracker->GetContactForcesAndLocalTorqueForAllToDevice(
        contact_points.data(), contact_forces.data(), contact_torques.data(), contact_capacity, 0);
    if (local_device_count != local_count ||
        !compareVectors(contact_torques.ToHost(local_device_count), host_torques, "local contact torques")) {
        return false;
    }

    // Independently reconstruct each owner's resultant from the existing per-contact API.
    TestDeviceBuffer<float3> wrench_forces(count, 0);
    TestDeviceBuffer<float3> wrench_torques(count, 0);
    tracker->ContactWrenchesToDevice(wrench_forces.data(), wrench_torques.data(), count, 0);
    std::vector<float3> reference_forces(count, make_float3(0));
    std::vector<float3> reference_torques(count, make_float3(0));
    bool observed_extra_torque = false;
    const auto owner_positions = tracker->Positions();
    for (size_t owner = 0; owner < count; owner++) {
        std::vector<float3> owner_points;
        std::vector<float3> owner_forces;
        std::vector<float3> owner_extra_torques;
        tracker->GetContactForcesAndGlobalTorque(owner_points, owner_forces, owner_extra_torques, owner);
        for (size_t contact = 0; contact < owner_forces.size(); contact++) {
            observed_extra_torque = observed_extra_torque || length(owner_extra_torques[contact]) > kTolerance;
            reference_forces[owner] += owner_forces[contact];
            reference_torques[owner] += cross(owner_points[contact] - owner_positions[owner], owner_forces[contact]) +
                                        owner_extra_torques[contact];
        }
    }
    if (!observed_extra_torque) {
        std::cerr << "FAIL: wrench reference scenario did not exercise force-model extra torque." << std::endl;
        return false;
    }
    if (!compareWrenchVectors(wrench_forces.ToHost(), reference_forces, "reduced owner contact forces") ||
        !compareWrenchVectors(wrench_torques.ToHost(), reference_torques, "reduced owner contact torques")) {
        return false;
    }
    std::vector<float3> host_wrench_forces;
    std::vector<float3> host_wrench_torques;
    tracker->ContactWrenches(host_wrench_forces, host_wrench_torques);
    if (!compareWrenchVectors(host_wrench_forces, wrench_forces.ToHost(), "host/device owner contact forces") ||
        !compareWrenchVectors(host_wrench_torques, wrench_torques.ToHost(), "host/device owner contact torques")) {
        return false;
    }
    std::vector<float3> solver_wrench_forces;
    std::vector<float3> solver_wrench_torques;
    solver.GetOwnerContactWrench(solver_wrench_forces, solver_wrench_torques, tracker->GetOwnerID(), count);
    if (!compareWrenchVectors(solver_wrench_forces, host_wrench_forces, "solver/tracker owner contact forces") ||
        !compareWrenchVectors(solver_wrench_torques, host_wrench_torques, "solver/tracker owner contact torques")) {
        return false;
    }

    if (visible_devices >= 2) {
        TestDeviceBuffer<float3> peer_wrench_forces(count, 1);
        TestDeviceBuffer<float3> peer_wrench_torques(count, 1);
        tracker->ContactWrenchesToDevice(peer_wrench_forces.data(), peer_wrench_torques.data(), count, 1);
        if (!compareWrenchVectors(peer_wrench_forces.ToHost(), reference_forces,
                                  "cross-device reduced owner contact forces") ||
            !compareWrenchVectors(peer_wrench_torques.ToHost(), reference_torques,
                                  "cross-device reduced owner contact torques")) {
            return false;
        }
    }

    bool wrench_capacity_rejected = false;
    solver.SetVerbosity("QUIET");
    try {
        tracker->ContactWrenchesToDevice(wrench_forces.data(), wrench_torques.data(), count - 1, 0);
    } catch (const SolverException&) {
        wrench_capacity_rejected = true;
    }
    solver.SetVerbosity("ERROR");
    if (!wrench_capacity_rejected) {
        std::cerr << "FAIL: insufficient owner-wrench capacity was not rejected." << std::endl;
        return false;
    }

    return true;
}

bool testNextStepAccelerationInput() {
    DEMSolver solver(1);
    solver.SetVerbosity("ERROR");
    solver.InstructBoxDomainDimension(8, 8, 8);
    solver.SetGravitationalAcceleration(make_float3(0));
    solver.SetCDUpdateFreq(1);
    solver.SetInitBinNumTarget(10);

    auto material = solver.LoadMaterial({{"E", 1e7}, {"nu", 0.3}, {"CoR", 0.2}, {"mu", 0.4}, {"Crr", 0.0}});
    auto sphere = solver.LoadSphereType(2.f, 0.25f, material);
    sphere->SetMOI(make_float3(0.3f, 0.4f, 0.5f));
    auto batch = solver.AddClumps(sphere, std::vector<float3>{make_float3(-2, 0, 0), make_float3(2, 0, 0)});
    auto tracker = solver.Track(batch);

    constexpr float step_size = 1e-4f;
    constexpr size_t count = 2;
    solver.SetTimeStepSize(step_size);
    solver.Initialize();

    TestDeviceBuffer<float3> input(count, 0);
    TestDeviceBuffer<float3> output(count, 0);
    const std::vector<float3> first_linear = {make_float3(1, 2, 3), make_float3(-1, -2, -3)};
    const std::vector<float3> queued_linear = {make_float3(4, -5, 6), make_float3(-7, 8, -9)};
    const std::vector<float3> first_angular = {make_float3(0.5f, 1.f, 1.5f), make_float3(-0.5f, -1.f, -1.5f)};
    const std::vector<float3> queued_angular = {make_float3(2, 3, 4), make_float3(-4, -3, -2)};

    // Exercise both solver and tracker entry points. A later submission replaces the already queued contribution,
    // matching the established host methods despite their historical "Add" name.
    input.FromHost(first_linear);
    solver.AddOwnerNextStepAccFromDevice(tracker->GetOwnerID(), input.data(), 0, count);
    input.FromHost(queued_linear);
    tracker->AddAccFromDevice(input.data(), 0, false);
    solver.GetOwnerAccToDevice(output.data(), count, 0, tracker->GetOwnerID(), count);
    if (!compareVectors(output.ToHost(), queued_linear, "queued device linear acceleration"))
        return false;

    input.FromHost(first_angular);
    tracker->AddAngAccFromDevice(input.data(), 0);
    input.FromHost(queued_angular);
    solver.AddOwnerNextStepAngAccFromDevice(tracker->GetOwnerID(), input.data(), 0, count, false);
    solver.GetOwnerAngAccLocalToDevice(output.data(), count, 0, tracker->GetOwnerID(), count);
    if (!compareVectors(output.ToHost(), queued_angular, "queued device local angular acceleration"))
        return false;

    // Zero-count calls must not inspect the pointer or alter the already queued values.
    solver.AddOwnerNextStepAccFromDevice(tracker->GetOwnerID(), nullptr, 0, 0);

    solver.SetVerbosity("QUIET");
    bool range_rejected = false;
    try {
        solver.AddOwnerNextStepAccFromDevice(tracker->GetOwnerID(1), input.data(), 0, count);
    } catch (const SolverException&) {
        range_rejected = true;
    }
    solver.SetVerbosity("ERROR");
    if (!range_rejected) {
        std::cerr << "FAIL: out-of-range CUDA next-step acceleration input was not rejected." << std::endl;
        return false;
    }

    solver.DoDynamicsThenSync(step_size);
    std::vector<float3> expected_velocity(count);
    std::vector<float3> expected_angular_velocity(count);
    for (size_t i = 0; i < count; i++) {
        expected_velocity[i] = queued_linear[i] * step_size;
        expected_angular_velocity[i] = queued_angular[i] * step_size;
    }
    if (!compareVectors(tracker->Velocities(), expected_velocity, "next-step linear velocity response") ||
        !compareVectors(tracker->AngularVelocitiesLocal(), expected_angular_velocity,
                        "next-step angular velocity response")) {
        return false;
    }

    // The preservation flags are consumed by the first force preparation, so a second step must not apply the same
    // acceleration again when there is no contact, gravity, or family acceleration prescription.
    solver.DoDynamicsThenSync(step_size);
    return compareVectors(tracker->Velocities(), expected_velocity, "one-shot linear acceleration") &&
           compareVectors(tracker->AngularVelocitiesLocal(), expected_angular_velocity,
                          "one-shot angular acceleration");
}

bool testNonJitifiedMassData() {
    DEMSolver solver(1);
    solver.SetVerbosity("ERROR");
    solver.DisableJitifyMassProperties();
    solver.InstructBoxDomainDimension(4, 4, 4);
    auto material = solver.LoadMaterial({{"E", 1e7}, {"nu", 0.3}, {"CoR", 0.2}, {"mu", 0.4}, {"Crr", 0.0}});
    auto sphere = solver.LoadSphereType(3.25f, 0.4f, material);
    auto batch = solver.AddClumps(sphere, std::vector<float3>{make_float3(1, 1, 1), make_float3(3, 1, 1)});
    auto tracker = solver.Track(batch);
    solver.SetTimeStepSize(1e-5);
    solver.Initialize();

    TestDeviceBuffer<float> masses(2, 0);
    TestDeviceBuffer<float3> moments(2, 0);
    tracker->MassesToDevice(masses.data(), 2, 0);
    tracker->MOIsToDevice(moments.data(), 2, 0);
    return compareVectors(masses.ToHost(), tracker->Masses(), "non-jitified masses") &&
           compareVectors(moments.ToHost(), tracker->MOIs(), "non-jitified moments of inertia");
}

bool testExternallyMovedFixedMeshes() {
    DEMSolver solver(1);
    solver.SetVerbosity("ERROR");
    // This scenario uses positive coordinates through x = 6. Use explicit bounds instead of the centered [-4, 4]
    // domain produced by the dimension-only overload, since positions outside the domain cannot be voxel-encoded.
    solver.InstructBoxDomainDimension({0.f, 8.f}, {0.f, 8.f}, {0.f, 8.f});
    solver.SetGravitationalAcceleration(make_float3(0));
    solver.SetMeshUniversalContact(true);
    solver.SetCDUpdateFreq(1);
    solver.SetInitBinNumTarget(10);
    solver.SetErrorOutAvgContacts(10000);
    solver.SetErrorOutVelocity(1e6f);

    auto material = solver.LoadMaterial({{"E", 1e7}, {"nu", 0.3}, {"CoR", 0.2}, {"mu", 0.4}, {"Crr", 0.0}});
    std::vector<std::shared_ptr<DEMMesh>> meshes;
    const auto cube_path = (GET_DATA_PATH() / "mesh/cube.obj").string();
    for (unsigned int i = 0; i < 3; i++) {
        auto mesh = solver.AddWavefrontMeshObject(cube_path, material);
        mesh->SetMass(1.f);
        mesh->SetMOI(make_float3(1.f));
        mesh->SetInitPos(make_float3(2.f + 2.f * i, 2.f, 2.f));
        mesh->SetFamily(20 + i);
        solver.SetFamilyFixed(20 + i);
        meshes.push_back(mesh);
    }
    auto first_tracker = solver.Track(meshes.front());

    constexpr size_t owner_count = 3;
    solver.SetTimeStepSize(1e-5);
    solver.Initialize();
    const bodyID_t first_owner = first_tracker->GetOwnerID();
    TestDeviceBuffer<float3> device_input(owner_count, 0);
    TestDeviceBuffer<float3> device_positions(owner_count, 0);
    TestDeviceBuffer<float3> device_velocities(owner_count, 0);
    TestDeviceBuffer<float3> device_forces(owner_count, 0);
    TestDeviceBuffer<float3> device_torques(owner_count, 0);
    std::vector<float3> prescribed_positions(owner_count);
    const std::vector<float> moving_owner_x = {3.02f, 3.01f, 3.00f, 2.99f};
    for (float x : moving_owner_x) {
        prescribed_positions = {make_float3(2.f, 2.f, 2.f), make_float3(x, 2.f, 2.f), make_float3(6.f, 2.f, 2.f)};
        device_input.FromHost(prescribed_positions);
        solver.SetOwnerPositionFromDevice(first_owner, device_input.data(), 0, owner_count);
        const std::vector<float3> prescribed_velocities = {make_float3(0), make_float3(-1000.f, 0, 0), make_float3(0)};
        device_velocities.FromHost(prescribed_velocities);
        solver.SetOwnerVelocityFromDevice(first_owner, device_velocities.data(), 0, owner_count);
        // Owner setters match their host counterparts and do not implicitly alter dT/kT scheduling policy.
        solver.RequestContactUpdate();
        solver.DoDynamicsThenSync(1e-5);

        // A fixed family must retain the just-prescribed pose rather than dynamically integrating its supplied speed.
        solver.GetOwnerPositionToDevice(device_positions.data(), owner_count, 0, first_owner, owner_count);
        if (!comparePositionVectors(device_positions.ToHost(), prescribed_positions,
                                    "fixed-mesh prescribed positions")) {
            return false;
        }
        solver.GetOwnerContactWrenchToDevice(device_forces.data(), device_torques.data(), owner_count, 0, first_owner,
                                             owner_count);
        const auto step_forces = device_forces.ToHost();
        if (x > 3.005f && (length(step_forces[0]) > kTolerance || length(step_forces[1]) > kTolerance)) {
            std::cerr << "FAIL: separated externally moved meshes produced a contact reaction." << std::endl;
            return false;
        }
    }

    solver.GetOwnerContactWrenchToDevice(device_forces.data(), device_torques.data(), owner_count, 0, first_owner,
                                         owner_count);
    const auto forces = device_forces.ToHost();
    const auto torques = device_torques.ToHost();
    for (size_t i = 0; i < owner_count; i++) {
        if (!std::isfinite(forces[i].x) || !std::isfinite(forces[i].y) || !std::isfinite(forces[i].z) ||
            !std::isfinite(torques[i].x) || !std::isfinite(torques[i].y) || !std::isfinite(torques[i].z)) {
            std::cerr << "FAIL: fixed-mesh contact wrench contains a non-finite value." << std::endl;
            return false;
        }
    }
    if (length(forces[0]) <= kTolerance || length(forces[1]) <= kTolerance || dot(forces[0], forces[1]) >= 0.f) {
        std::cerr << "FAIL: externally overlapped fixed meshes did not produce opposing contact reactions."
                  << std::endl;
        return false;
    }
    if (length(forces[2]) > kTolerance || length(torques[2]) > kTolerance) {
        std::cerr << "FAIL: fixed mesh without contact did not receive a zero wrench." << std::endl;
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int visible_devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&visible_devices);
    if (status != cudaSuccess) {
        std::cerr << "SKIP: CUDA runtime unavailable: " << cudaGetErrorString(status) << std::endl;
        return 0;
    }
    if (visible_devices == 0) {
        std::cout << "SKIP: DEMTest_DeviceDataRetrieval requires a visible CUDA device." << std::endl;
        return 0;
    }

    if (!testFixedOwnerData(visible_devices)) {
        return 1;
    }
    if (!testNextStepAccelerationInput()) {
        return 1;
    }
    if (!testNonJitifiedMassData()) {
        return 1;
    }
    if (!testExternallyMovedFixedMeshes()) {
        return 1;
    }
    std::cout << "PASS: GPU owner-state exchange and contact-wrench retrieval match reference behavior." << std::endl;
    return 0;
}
