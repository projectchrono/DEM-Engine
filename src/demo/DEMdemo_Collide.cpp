//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <granular/ApiSystem.h>
#include <granular/HostSideHelpers.cpp>

#include <cstdio>
#include <time.h>

using namespace sgps;

int main() {
    DEMSolver DEM_sim;

    srand(time(NULL));

    // total number of random clump templates to generate
    int num_template = 10;

    int min_sphere = 1;
    int max_sphere = 1;

    float min_rad = 0.08;
    float max_rad = 0.15;

    float min_relpos = -0.1;
    float max_relpos = 0.1;

    auto mat_type_1 = DEM_sim.LoadMaterialType(1, 10);

    for (int i = 0; i < num_template; i++) {
        // first decide the number of spheres that live in this clump
        int num_sphere = rand() % (max_sphere - min_sphere + 1) + 1;

        // then allocate the clump template definition arrays
        float mass = (float)rand() / RAND_MAX;
        float3 MOI = make_float3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
        std::vector<float> radii;
        std::vector<float3> relPos;
        std::vector<unsigned int> mat;

        // randomly generate clump template configurations
        // the relPos of a sphere is always seeded from one of the already-generated sphere
        float3 seed_pos = make_float3(0);
        for (int j = 0; j < num_sphere; j++) {
            radii.push_back(((float)rand() / RAND_MAX) * (max_rad - min_rad) + min_rad);
            float3 tmp;
            if (j == 0) {
                tmp.x = 0;
                tmp.y = 0;
                tmp.z = 0;
            } else {
                tmp.x = ((float)rand() / RAND_MAX) * (max_relpos - min_relpos) + min_relpos;
                tmp.y = ((float)rand() / RAND_MAX) * (max_relpos - min_relpos) + min_relpos;
                tmp.z = ((float)rand() / RAND_MAX) * (max_relpos - min_relpos) + min_relpos;
            }
            tmp += seed_pos;
            relPos.push_back(tmp);
            mat.push_back(mat_type_1);

            // seed relPos from one of the previously generated spheres
            int choose_from = rand() % (j + 1);
            seed_pos = relPos.at(choose_from);
        }

        // it returns the numbering of this clump template (although here we don't care)
        auto template_num = DEM_sim.LoadClumpType(mass, MOI, radii, relPos, mat);
    }

    // generate initial clumps (in this case just polydisperse spheres)
    auto input_xyz = DEMBoxGridSampler(make_float3(0), make_float3(1, 1, 1), 0.4);
    unsigned int num_clumps = input_xyz.size();
    std::vector<unsigned int> input_template_num;
    std::vector<float3> input_vel;
    for (unsigned int i = 0; i < num_clumps; i++) {
        input_template_num.push_back(i % num_template);
        float3 vel;
        vel.x = (float)rand() / RAND_MAX - 0.5;
        vel.y = (float)rand() / RAND_MAX - 0.5;
        vel.z = (float)rand() / RAND_MAX - 0.5;
        vel = normalize(vel) * 50.0;
        input_vel.push_back(vel);
    }

    DEM_sim.SetClumps(input_template_num, input_xyz);
    DEM_sim.SetClumpVels(input_vel);

    DEM_sim.InstructBoxDomainNumVoxel(22, 21, 21, 3e-10);

    DEM_sim.CenterCoordSys();
    DEM_sim.SetTimeStepSize(1e-4);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, 0));
    DEM_sim.SetCDUpdateFreq(1);

    DEM_sim.Initialize();

    for (int i = 0; i < 20; i++) {
        std::cout << "Iteration: " << i << std::endl;

        char filename[100];
        sprintf(filename, "./DEMdemo_collide_output_%04d.csv", i);
        DEM_sim.WriteFileAsSpheres(std::string(filename));

        DEM_sim.LaunchThreads(5e-4);
    }

    std::cout << "DEMdemo_Collide exiting..." << std::endl;
    return 0;
}
