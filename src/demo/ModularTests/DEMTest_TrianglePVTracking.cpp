//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Triangle P/V/P*V tracking and mesh-wear API smoke test.
//
// Two cube meshes are placed in shallow face contact. The lower cube is fixed
// and the upper cube is given a tangential velocity, so at least one tracked
// triangle on the upper cube should accumulate contact pressure and P*V.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace deme;

namespace {

int fail(const std::string& message) {
    std::cerr << "DEMTest_TrianglePVTracking failed: " << message << std::endl;
    return 1;
}

bool allFiniteAndNonnegative(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float val) { return std::isfinite(val) && val >= 0.f; });
}

float maxValue(const std::vector<float>& values) {
    return values.empty() ? 0.f : *std::max_element(values.begin(), values.end());
}

}  // namespace

int main() {
    constexpr float cube_side = 0.2f;
    constexpr float overlap = 0.004f;
    constexpr float density = 2500.f;
    constexpr float mass = density * cube_side * cube_side * cube_side;
    constexpr float moi = mass * cube_side * cube_side / 6.f;
    constexpr float step_size = 1e-5f;
    constexpr float run_time = 2e-4f;

    DEMSolver DEMSim;
    DEMSim.SetVerbosity("ERROR");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.InstructBoxDomainDimension(2.f, 2.f, 2.f);
    DEMSim.SetGravitationalAcceleration(make_float3(0.f, 0.f, 0.f));
    DEMSim.SetMeshUniversalContact(true);
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.SetCDUpdateFreq(1);
    DEMSim.SetInitBinNumTarget(8);
    DEMSim.SetErrorOutAvgContacts(10000);

    auto mat = DEMSim.LoadMaterial({{"E", 1e8f}, {"nu", 0.3f}, {"CoR", 0.1f}, {"mu", 0.5f}, {"Crr", 0.f}});

    auto lower_cube = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cube.obj").string(), mat);
    lower_cube->Scale(cube_side);
    lower_cube->SetMass(mass);
    lower_cube->SetMOI(make_float3(moi, moi, moi));
    lower_cube->SetInitPos(make_float3(0.f, 0.f, 0.f));
    lower_cube->SetFamily(100);
    DEMSim.SetFamilyFixed(100);

    auto upper_cube = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cube.obj").string(), mat);
    upper_cube->Scale(cube_side);
    upper_cube->SetMass(mass);
    upper_cube->SetMOI(make_float3(moi, moi, moi));
    upper_cube->SetInitPos(make_float3(0.f, 0.f, cube_side - overlap));
    upper_cube->SetFamily(1);
    auto upper_tracker = DEMSim.Track(upper_cube);

    DEMSim.SetTimeStepSize(step_size);
    DEMSim.Initialize();
    DEMSim.SetTriTriPenetration(overlap * 2.f);

    const bodyID_t upper_owner = upper_tracker->GetOwnerID();
    const size_t upper_triangles = upper_cube->GetNumTriangles();
    DEMSim.SetTrianglePVTrackingOwners({upper_owner});

    upper_tracker->SetVel(make_float3(0.05f, 0.f, 0.f));
    DEMSim.DoDynamicsThenSync(run_time);

    std::vector<float> avgP;
    std::vector<float> avgV;
    std::vector<float> avgPV;
    if (!DEMSim.GetTrackedOwnerTrianglePV(upper_owner, avgP, avgV, avgPV, false)) {
        return fail("tracked owner query returned false");
    }
    if (avgP.size() != upper_triangles || avgV.size() != upper_triangles || avgPV.size() != upper_triangles) {
        return fail("tracked vector sizes do not match mesh triangle count");
    }
    if (!allFiniteAndNonnegative(avgP) || !allFiniteAndNonnegative(avgV) || !allFiniteAndNonnegative(avgPV)) {
        return fail("tracked vectors contain non-finite or negative values");
    }
    if (maxValue(avgP) <= 0.f) {
        return fail("tracked pressure remained zero");
    }
    if (maxValue(avgPV) <= 0.f) {
        return fail("tracked P*V remained zero");
    }

    // Smoke-test the wear-control API using zero wear rate. This exercises the
    // enable/flush/disable path without intentionally deforming the mesh.
    DEMSim.EnableMeshWearModel(upper_owner, 0.0, step_size);
    DEMSim.DoDynamicsThenSync(step_size);
    DEMSim.FlushMeshWearModels();
    DEMSim.DisableMeshWearModel(upper_owner);
    DEMSim.DisableAllMeshWearModels();
    DEMSim.DisableTrianglePVTracking();

    std::cout << "DEMTest_TrianglePVTracking passed. "
              << "triangles=" << upper_triangles << " maxP=" << maxValue(avgP) << " maxV=" << maxValue(avgV)
              << " maxPV=" << maxValue(avgPV) << std::endl;
    return 0;
}
