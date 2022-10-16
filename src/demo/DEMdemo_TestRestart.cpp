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
    auto mat_type = DEMSim.LoadMaterial({{"E", 1e10}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.3}, {"Crr", 0.01}});

    // Bounding box...
    DEMSim.InstructBoxDomainBoundingBC("all", mat_type);

    float step_size = 1e-4;
    double x_size = 4;
    double y_size = 4;
    double z_size = 20;

    auto pairs = DEMSim.ReadContactPairsFromCsv((GET_SOURCE_DATA_PATH() / "sim_data/example_cnt_pairs.csv").string());
    auto wcs = DEMSim.ReadContactWildcardsFromCsv((GET_SOURCE_DATA_PATH() / "sim_data/example_cnt_pairs.csv").string());

    float rad = 2;
    auto template_sphere = DEMSim.LoadSphereType(rad * rad * rad * 2.6e3 * 4 / 3 * 3.14, rad, mat_type);

    std::vector<float3> input_xyz;
    input_xyz.push_back(make_float3(0, 0, -8));
    input_xyz.push_back(make_float3(0, 0, -4));

    // First 2 spheres...
    DEMClumpBatch base_batch(2);
    base_batch.SetTypes(template_sphere);
    base_batch.SetPos(input_xyz);
    base_batch.SetExistingContacts(pairs);
    base_batch.SetExistingContactWildcards(wcs);
    DEMSim.AddClumps(base_batch);

    // Add another 2 batches
    DEMClumpBatch batch_2 = base_batch;
    std::for_each(batch_2.xyz.begin(), batch_2.xyz.end(), [](float3& xyz) { xyz.z += 5; });
    DEMSim.AddClumps(batch_2);

    path out_dir = current_path();
    out_dir += "/DemoOutput_TestRestart";
    create_directory(out_dir);

    DEMSim.InstructBoxDomainDimension(x_size, y_size, z_size, SPATIAL_DIR::X);
    DEMSim.SetCoordSysOrigin("center");
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEMSim.SetCDUpdateFreq(10);
    // DEMSim.SetExpandFactor(1e-3);
    DEMSim.SetMaxVelocity(5.);
    DEMSim.SetExpandSafetyParam(1.1);
    DEMSim.Initialize();

    {
        char filename[100];
        sprintf(filename, "%s/DEMdemo_output_0.csv", out_dir.c_str());
        DEMSim.WriteSphereFile(std::string(filename));

        char cnt_filename[100];
        sprintf(cnt_filename, "%s/Contact_pairs_0.csv", out_dir.c_str());
        DEMSim.WriteContactFile(std::string(cnt_filename));
    }

    DEMSim.DoDynamicsThenSync(0.5);

    {
        char filename[100];
        sprintf(filename, "%s/DEMdemo_output_1.csv", out_dir.c_str());
        DEMSim.WriteSphereFile(std::string(filename));

        char cnt_filename[100];
        sprintf(cnt_filename, "%s/Contact_pairs_1.csv", out_dir.c_str());
        DEMSim.WriteContactFile(std::string(cnt_filename));
    }

    std::cout << "DEMdemo_TestRestart exiting..." << std::endl;
    return 0;
}
