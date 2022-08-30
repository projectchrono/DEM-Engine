//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <filesystem>
#include <cstdio>
#include <time.h>
#include <filesystem>

using namespace sgps;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;
    DEM_sim.SetVerbosity(DEBUG);
    DEM_sim.SetOutputFormat(DEM_OUTPUT_FORMAT::CSV);

    // srand(time(NULL));
    srand(4150);

    auto mat_type_1 = DEM_sim.LoadMaterialType(1e9, 0.3, 0.8);
    auto sph_type_1 = DEM_sim.LoadClumpSimpleSphere(11728., 1., mat_type_1);

    std::vector<float3> input_xyz1, input_xyz2;
    std::vector<float3> input_vel1, input_vel2;
    std::vector<std::shared_ptr<DEMClumpTemplate>> input_clump_type(1, sph_type_1);
    // std::vector<unsigned int> input_clump_type(1, sph_type_1);

    // Inputs are just 2 sphere
    float sphPos = 1.2f;
    input_xyz1.push_back(make_float3(-sphPos, 0, 0));
    input_xyz2.push_back(make_float3(sphPos, 0, 0));
    input_vel1.push_back(make_float3(1.f, 0, 0));
    input_vel2.push_back(make_float3(-1.f, 0, 0));

    auto particles1 = DEM_sim.AddClumps(input_clump_type, input_xyz1);
    particles1->SetVel(input_vel1);
    particles1->SetFamily(0);
    auto tracker1 = DEM_sim.Track(particles1);

    DEM_sim.DisableContactBetweenFamilies(0, 1);

    DEM_sim.AddBCPlane(make_float3(0, 0, -1.25), make_float3(0, 0, 1), mat_type_1);

    // Create a inspector to find out the highest point of this granular pile
    auto max_z_finder = DEM_sim.CreateInspector("max_z", DEM_INSPECT_ENTITY_TYPE::SPHERE);
    float max_z;

    DEM_sim.InstructBoxDomainNumVoxel(22, 21, 21, 3e-11);

    DEM_sim.SetCoordSysOrigin("center");
    DEM_sim.SetInitTimeStep(2e-5);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // Velocity not to exceed 3.0
    DEM_sim.SetCDUpdateFreq(10);
    DEM_sim.SetMaxVelocity(3.);
    DEM_sim.SetExpandSafetyParam(2.);

    DEM_sim.Initialize();

    // You can add more clumps to simulation after initialization, like this...
    DEM_sim.ClearCache();
    auto particles2 = DEM_sim.AddClumps(input_clump_type, input_xyz2);
    particles2->SetVel(input_vel2);
    particles2->SetFamily(1);
    auto tracker2 = DEM_sim.Track(particles2);
    DEM_sim.UpdateClumps();

    // Ready simulation
    path out_dir = current_path();
    out_dir += "/DEMdemo_SingleSphereCollide";
    create_directory(out_dir);
    bool changed_family = false;
    for (int i = 0; i < 100; i++) {
        std::cout << "Frame: " << i << std::endl;

        if ((!changed_family) && i >= 10) {
            // DEM_sim.ChangeFamily(1, 0);
            DEM_sim.EnableContactBetweenFamilies(0, 1);
            changed_family = true;
        }

        char filename[100];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), i);
        DEM_sim.WriteSphereFile(std::string(filename));

        DEM_sim.DoDynamicsThenSync(1e-2);
        max_z = max_z_finder->GetValue();
        std::cout << "Max Z coord is " << max_z << std::endl;
        std::cout << "Particle 1 X coord is " << tracker1->Pos().x << std::endl;
        std::cout << "Particle 2 X coord is " << tracker2->Pos().x << std::endl;
    }

    std::cout << "DEMdemo_SingleSphereCollide exiting..." << std::endl;
    return 0;
}
