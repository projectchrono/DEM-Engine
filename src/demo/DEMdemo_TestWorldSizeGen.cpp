//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(DEBUG);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);

    // E, nu, CoR, mu, Crr...
    auto mat_type = DEMSim.LoadMaterial({{"E", 1e10}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.0}, {"Crr", 0.0}});

    auto projectile = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/sphere.obj").string(), mat_type);
    std::cout << "Total num of triangles: " << projectile->GetNumTriangles() << std::endl;

    float step_size = 1e-5;
    double x_size = 3;
    double y_size = 4;
    double z_size = 5;

    auto pairs = DEMSim.ReadContactPairsFromCsv("example_cnt_pairs.csv");
    auto wcs = DEMSim.ReadContactWildcardsFromCsv("example_cnt_pairs.csv");
    // for (int i = 0; i < pairs.size(); i++) {
    //     std::cout << "Body pair: " << pairs[i].first << ", " << pairs[i].second << std::endl;
    //     std::cout << "delta_time: " << wcs.at("delta_time")[i] << std::endl;
    // }

    float rad = 0.01;
    auto template_terrain = DEMSim.LoadSphereType(rad * rad * rad * 2.6e3 * 4 / 3 * 3.14, rad, mat_type);

    float3 sample_center = make_float3(0, 0, 0);
    auto input_xyz = DEMBoxHCPSampler(sample_center, make_float3(1, 1, 1), 2.01 * rad);
    auto terrain_particles = DEMSim.AddClumps(template_terrain, input_xyz);
    auto terrain_tracker = DEMSim.Track(terrain_particles);

    terrain_particles->SetExistingContacts(pairs);
    terrain_particles->SetExistingContactWildcards(wcs);

    DEMSim.InstructBoxDomainDimension(x_size, y_size, z_size, SPATIAL_DIR::X);
    DEMSim.SetCoordSysOrigin("center");
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEMSim.SetCDUpdateFreq(10);
    // DEMSim.SetExpandFactor(1e-3);
    DEMSim.SetMaxVelocity(2.);
    DEMSim.SetExpandSafetyParam(1.1);
    DEMSim.Initialize();

    std::cout << "DEMdemo_TestWorldSizeGen exiting..." << std::endl;
    return 0;
}
