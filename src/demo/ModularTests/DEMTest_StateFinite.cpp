// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

#include "DEM/API.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace deme;

// Run one isolated no-contact case. Invalid cases intentionally trap on the GPU; the Python driver checks
// their diagnostics in separate processes because a device trap invalidates the CUDA context.
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: DEMTest_StateFinite <healthy|linear|angular> <auto|fixed|no-angular> <nan|inf>\n";
        return 2;
    }
    const std::string kind = argv[1];
    const std::string margin = argv[2];
    const float invalid = std::string(argv[3]) == "inf" ? std::numeric_limits<float>::infinity()
                                                        : std::numeric_limits<float>::quiet_NaN();
    DEMSolver solver;
    solver.SetVerbosity("ERROR");
    solver.InstructBoxDomainDimension(2., 2., 2.);
    solver.SetInitBinNumTarget(8);
    solver.SetTimeStepSize(1.e-4);
    solver.SetGravitationalAcceleration(make_float3(0.f));
    if (margin == "fixed")
        solver.SetExpandFactor(0.01f);
    if (margin == "no-angular")
        solver.SetUseAngularVelocityMargin(false);
    auto material = solver.LoadMaterial({{"E", 1.e7f}, {"nu", 0.3f}, {"CoR", 0.2f}, {"mu", 0.4f}, {"Crr", 0.f}});
    auto sphere = solver.LoadSphereType(1.f, 0.05f, material);
    auto batch = solver.AddClumps(sphere, make_float3(0.f));
    auto tracker = solver.Track(batch);
    solver.Initialize();
    // Inject after initialization to exercise the runtime dT handoff, not setup validation.
    tracker->SetVel(make_float3(kind == "linear" ? invalid : 0.1f, 0.f, 0.f));
    tracker->SetAngVel(make_float3(0.f, kind == "angular" ? invalid : 0.2f, 0.f));
    solver.DoDynamicsThenSync(1.e-3);
    if (kind != "healthy") {
        std::cerr << "FAIL: invalid velocity escaped detection\n";
        return 1;
    }
    const auto pos = tracker->Pos();
    if (!std::isfinite(pos.x) || std::abs(pos.x - 0.1 * solver.GetSimTime()) > 1.e-6) {
        std::cerr << "FAIL: healthy free motion changed: x=" << pos.x << ", t=" << solver.GetSimTime() << "\n";
        return 1;
    }
    std::cout << "PASS: healthy no-contact state\n";
    return 0;
}
