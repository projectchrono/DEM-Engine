// Copyright (c) 2021, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

#include "DEM/API.h"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace deme;

namespace {

struct ContactSample {
    std::vector<float> position;
    std::vector<float> velocity;
    std::vector<float> angular_velocity;
    float3 force;
};

// Start separated, then enter contact at the third force evaluation. Zero extra margin makes a stale
// candidate list observable. Tangential motion exercises history migration after contact onset.
std::vector<ContactSample> runContactSequence(bool sync_each_step, bool legacy_setters, bool switch_mode) {
    constexpr double step = 1.e-4;
    DEMSolver solver;
    solver.SetVerbosity("ERROR");
    solver.InstructBoxDomainDimension(2., 2., 2.);
    solver.SetInitBinNumTarget(8);
    solver.SetTimeStepSize(step);
    solver.SetGravitationalAcceleration(make_float3(0.f));
    if (legacy_setters || switch_mode) {
        solver.SetCDUpdateFreq(switch_mode ? 4 : 0);
        solver.DisableAdaptiveUpdateFreq();
    } else {
        solver.SetSynchronizedMode();
    }
    auto material = solver.LoadMaterial({{"E", 1.e7f}, {"nu", 0.3f}, {"CoR", 0.8f}, {"mu", 0.5f}, {"Crr", 0.f}});
    auto sphere = solver.LoadSphereType(1.f, 0.05f, material);
    auto fixed = solver.AddClumps(sphere, make_float3(0.f));
    fixed->SetFamily(1);
    solver.SetFamilyFixed(1);
    auto moving = solver.AddClumps(sphere, make_float3(0.10015f, 0.f, 0.f));
    moving->SetVel(make_float3(-1.f, 0.1f, 0.f));
    auto tracker = solver.Track(moving);
    solver.Initialize();
    if (switch_mode) {
        // Change policy only from a synchronized stance; also exercise a zero-duration call after the change.
        solver.DoDynamicsThenSync(0.0);
        solver.SetSynchronizedMode();
        solver.UpdateSimParams();
        solver.DoDynamics(0.0);
    }
    solver.ClearThreadCollaborationStats();
    std::vector<ContactSample> samples;
    int previous_step = 0;
    for (int target_step : {1, 2, 3, 10, 25, 40}) {
        const double before = solver.GetSimTime();
        const int count = target_step - previous_step;
        if (sync_each_step) {
            for (int i = 0; i < count; ++i) {
                solver.DoDynamicsThenSync(0.5 * step);
            }
        } else {
            // Request an unambiguous number of steps despite the float device timestep.
            solver.DoDynamics((count - 0.5) * step);
        }
        if (std::abs(solver.GetSimTime() - before - count * step) > 1.e-8) {
            std::cerr << "FAIL: unexpected step count in contact sequence\n";
            std::exit(EXIT_FAILURE);
        }
        std::vector<float3> points, forces;
        tracker->GetContactForces(points, forces);
        float3 total = make_float3(0.f);
        for (const auto& force : forces) {
            total.x += force.x;
            total.y += force.y;
            total.z += force.z;
        }
        samples.push_back({tracker->GetPos(), tracker->GetVel(), tracker->GetAngVelGlobal(), total});
        previous_step = target_step;
    }
    solver.ShowThreadCollaborationStats();
    if (switch_mode) {
        // Verify the reverse transition and another restart with an outstanding asynchronous order.
        solver.DoDynamicsThenSync(0.0);
        solver.SetCDUpdateFreq(4);
        solver.UpdateSimParams();
        solver.DoDynamics(0.5 * step);
        solver.DoDynamicsThenSync(0.0);
        solver.SetSynchronizedMode();
        solver.UpdateSimParams();
        solver.DoDynamics(0.5 * step);
    }
    return samples;
}

// Compare contact onset and the subsequent frictional trajectory with the per-step synchronization reference.
bool checkSynchronizedContacts() {
    const auto reference = runContactSequence(true, false, false);
    if (reference[0].force.x != 0.f || reference[1].force.x != 0.f || reference[2].force.x <= 0.f) {
        std::cerr << "FAIL: reference contact must first generate force on step 3\n";
        return false;
    }
    for (int mode = 0; mode < 3; ++mode) {
        const auto actual = runContactSequence(false, mode == 1, mode == 2);
        for (size_t i = 0; i < reference.size(); ++i) {
            auto close = [](float a, float b) {
                return std::isfinite(a) && std::abs(a - b) <= 1.e-5f * (1.f + std::abs(b));
            };
            const auto& a = actual[i];
            const auto& b = reference[i];
            bool matched = close(a.force.x, b.force.x) && close(a.force.y, b.force.y) && close(a.force.z, b.force.z);
            for (size_t axis = 0; axis < 3; ++axis) {
                matched = matched && close(a.position[axis], b.position[axis]) &&
                          close(a.velocity[axis], b.velocity[axis]) &&
                          close(a.angular_velocity[axis], b.angular_velocity[axis]);
            }
            if (!matched) {
                std::cerr << "FAIL: synchronized contact mismatch, mode " << mode << ", sample " << i << ", force x "
                          << a.force.x << " versus " << b.force.x << "\n";
                return false;
            }
        }
    }
    std::cout << "PASS: synchronized contact onset, history, and mode switching\n";
    return true;
}

}  // namespace

// Exercise real worker handoffs with short calls, forced drift waits, sync/restart, and destruction.
// This is a scheduling stress regression, not a deterministic reproduction of every lost-wakeup interleaving.
int main() {
    std::mutex completion_mutex;
    std::condition_variable completion_cv;
    bool completed = false;
    // Fail rather than hang indefinitely if either worker loses a notification, including during destruction.
    // Include enough startup time for runtime kernel compilation with a cold JIT cache.
    std::thread watchdog([&]() {
        std::unique_lock<std::mutex> lock(completion_mutex);
        if (!completion_cv.wait_for(lock, std::chrono::seconds(600), [&]() { return completed; })) {
            std::cerr << "FAIL: worker handoff timed out\n" << std::flush;
            std::_Exit(EXIT_FAILURE);
        }
    });

    bool passed = checkSynchronizedContacts();
    for (int drift : {0, 1, 4}) {
        std::cout << "Initializing handoff test with update frequency " << drift << std::endl;
        DEMSolver solver;
        solver.SetVerbosity("ERROR");
        solver.InstructBoxDomainDimension(2.0, 2.0, 2.0);
        solver.SetInitBinNumTarget(8);
        solver.SetTimeStepSize(1.e-4);
        solver.SetGravitationalAcceleration(make_float3(0.f, 0.f, 0.f));
        if (drift == 0) {
            solver.SetSynchronizedMode();
        } else {
            solver.SetCDUpdateFreq(drift);
            solver.DisableAdaptiveUpdateFreq();
        }
        auto material = solver.LoadMaterial({{"E", 1.e7f}, {"nu", 0.3f}, {"CoR", 0.2f}, {"mu", 0.4f}, {"Crr", 0.f}});
        auto sphere = solver.LoadSphereType(1.f, 0.05f, material);
        solver.AddClumps(sphere, make_float3(0.f, 0.f, 0.f));
        solver.Initialize();
        std::cout << "Exercising handoffs" << std::endl;

        // Half a timestep requests exactly one step without sensitivity to float timestep rounding.
        for (int call = 0; call < 200; ++call) {
            const double before = solver.GetSimTime();
            solver.DoDynamics(5.e-5);
            const double advance = solver.GetSimTime() - before;
            passed = passed && std::isfinite(advance) && std::abs(advance - 1.e-4) < 1.e-9;
            if (call % 7 == 0) {
                solver.DoDynamicsThenSync(0.0);
            }
            if (call % 3 == 0) {
                std::this_thread::yield();
            }
        }
        // Leave the final call asynchronous to exercise destruction with a potentially outstanding kT order.
    }

    {
        std::lock_guard<std::mutex> lock(completion_mutex);
        completed = true;
    }
    completion_cv.notify_one();
    watchdog.join();
    std::cout << (passed ? "PASS" : "FAIL") << ": worker handoff and restart\n";
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
