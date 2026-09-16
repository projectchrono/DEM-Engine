//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Setup-time adaptive timestep smoke test.
//
// The hertz_const mode computes one fixed timestep during Initialize() from
// material stiffness, minimum clump mass, and minimum radius. This test keeps the
// system minimal and checks that the initialized timestep matches the documented
// Hertz estimate rather than the manually supplied initial value.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/Defines.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using namespace deme;

namespace {

int fail(const std::string& message) {
    std::cerr << "DEMTest_AdaptiveTimestep failed: " << message << std::endl;
    return 1;
}

double expectedHertzConstTimestep(double mass, double radius, double youngs_modulus, double poisson_ratio) {
    const double effective_E = 1.0 / (2.0 * (1.0 - poisson_ratio * poisson_ratio) / youngs_modulus);
    const double effective_radius = 0.5 * radius;
    const double effective_mass = 0.5 * mass;
    const double hertz_stiffness = FOUR_OVER_THREE * std::sqrt(0.1) * effective_E * effective_radius;
    return (PI / (2.0 * N_DT)) * std::sqrt(effective_mass / hertz_stiffness);
}

bool nearlyEqual(double a, double b, double rel_tol = 1e-10) {
    const double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return std::abs(a - b) <= rel_tol * scale;
}

}  // namespace

int main() {
    constexpr double manual_dt = 1.0e-5;
    constexpr double mass = 0.2;
    constexpr double radius = 0.05;
    constexpr double youngs_modulus = 1.0e8;
    constexpr double poisson_ratio = 0.25;

    DEMSolver DEMSim;
    DEMSim.SetVerbosity("ERROR");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.InstructBoxDomainDimension(1.0, 1.0, 1.0);
    DEMSim.SetGravitationalAcceleration(make_float3(0.f, 0.f, 0.f));
    DEMSim.SetInitBinNumTarget(8);
    DEMSim.SetTimeStepSize(manual_dt);

    auto mat = DEMSim.LoadMaterial({{"E", static_cast<float>(youngs_modulus)},
                                    {"nu", static_cast<float>(poisson_ratio)},
                                    {"CoR", 0.2f},
                                    {"mu", 0.4f},
                                    {"Crr", 0.0f}});
    auto sphere = DEMSim.LoadSphereType(static_cast<float>(mass), static_cast<float>(radius), mat);
    DEMSim.AddClumps(sphere, make_float3(0.f, 0.f, 0.f));

    DEMSim.UseHertzConstTimeStep();
    DEMSim.Initialize();

    const double actual_dt = DEMSim.GetTimeStepSize();
    const double expected_dt = expectedHertzConstTimestep(mass, radius, youngs_modulus, poisson_ratio);

    if (!std::isfinite(actual_dt) || actual_dt <= 0.0) {
        return fail("initialized timestep is not finite and positive");
    }
    if (nearlyEqual(actual_dt, manual_dt)) {
        return fail("hertz_const did not replace the manually supplied timestep");
    }
    if (!nearlyEqual(actual_dt, expected_dt)) {
        std::cerr << "Expected dt: " << expected_dt << ", actual dt: " << actual_dt << std::endl;
        return fail("hertz_const timestep does not match the Hertz estimate");
    }

    std::cout << "DEMTest_AdaptiveTimestep passed. "
              << "manual_dt=" << manual_dt << " hertz_dt=" << actual_dt << std::endl;
    return 0;
}
