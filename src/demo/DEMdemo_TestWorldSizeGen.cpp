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
    auto mat_type_ball = DEMSim.LoadMaterial({{"E", 1e10}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.0}, {"Crr", 0.0}});

    auto projectile = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/sphere.obj").string(), mat_type_ball);
    std::cout << "Total num of triangles: " << projectile->GetNumTriangles() << std::endl;

    float step_size = 1e-5;
    double x_size = 15.;
    double y_size = 2;
    double z_size = 2.2;

    auto pairs = DEMSim.ReadContactPairsFromCsv("example_cnt_pairs.csv");
    auto wcs = DEMSim.ReadContactWildcardsFromCsv("example_cnt_pairs.csv");
    for (int i = 0; i < pairs.size(); i++) {
        std::cout << "Body pair: " << pairs[i].first << ", " << pairs[i].second << std::endl;
        std::cout << "delta_time: " << wcs.at("delta_time")[i] << std::endl;
    }

    // DEMSim.InstructBoxDomainNumVoxel(21, 21, 22, world_size / std::pow(2, 16) / std::pow(2, 21));
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
