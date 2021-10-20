//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <granular/ApiSystem.h>

#include <cstdio>
#include <time.h>

using namespace sgps;

int main() {
    DEMSolver aa(1.f);

    srand(time(NULL));

    // total number of random clump templates to generate
    int num_template = 27;

    int min_sphere = 1;
    int max_sphere = 5;

    float min_rad = 0.4;
    float max_rad = 1.0;

    float min_relpos = -0.5;
    float max_relpos = 0.5;

    /*
    std::vector<float> radii_a_vec(3, .4);
    std::vector<float> radii_b_vec(3, .6);
    std::vector<float> radii_c_vec(3, .8);

    std::vector<float3> pos_a_vec(3, make_float3(.1, .12, .08));
    std::vector<float3> pos_b_vec(3, make_float3(.1, .05, .06));
    std::vector<float3> pos_c_vec(3, make_float3(.12, .1, .14));
    std::vector<float3> pos_d_vec;
    pos_d_vec.push_back(2. * make_float3(.2, .3, .17));
    pos_d_vec.push_back(2. * make_float3(.12, .4, .37));
    pos_d_vec.push_back(2. * make_float3(.22, .19, .45));

    std::vector<unsigned int> mat_vec(3, 0);
    */

    aa.LoadMaterialType(1, 10);

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
            tmp.x = ((float)rand() / RAND_MAX) * (max_relpos - min_relpos) + min_relpos;
            tmp.y = ((float)rand() / RAND_MAX) * (max_relpos - min_relpos) + min_relpos;
            tmp.z = ((float)rand() / RAND_MAX) * (max_relpos - min_relpos) + min_relpos;
            tmp += seed_pos;
            relPos.push_back(tmp);
            mat.push_back(0);

            // seed relPos from one of the previously generated spheres
            int choose_from = rand() % (j + 1);
            seed_pos = relPos.at(choose_from);
        }

        // it returns the numbering of this clump template (although here we don't care)
        auto template_num = aa.LoadClumpType(mass, MOI, radii, relPos, mat);
    }
    // auto num = aa.LoadClumpSimpleSphere(3, 1., 0);

    std::vector<unsigned int> input_template_num;
    std::vector<float3> input_xyz;

    // show one for each template configuration
    for (int i = 0; i < num_template; i++) {
        input_template_num.push_back(i);

        float grid_size = 5.0;
        int ticks = 3;
        int ix = i % ticks;
        int iy = (i % (ticks * ticks)) / ticks;
        int iz = i / (ticks * ticks);
        input_xyz.push_back(grid_size * make_float3(ix, iy, iz));
    }
    aa.SetClumps(input_template_num, input_xyz);

    aa.InstructBoxDomainNumVoxel(22, 21, 21, 1e-10);
    aa.SetTimeStepSize(1e-4);
    aa.SetGravitationalAcceleration(make_float3(0, 0, -9.8));

    aa.Initialize();

    aa.UpdateSimParams();  // Not needed; just testing if this function works...

    aa.LaunchThreads();

    std::cout << aa.GetClumpVoxelID(0) << std::endl;

    char filename[100];
    sprintf(filename, "./test_gran_output.csv");
    aa.WriteFileAsSpheres(std::string(filename));

    std::cout << "Demo exiting..." << std::endl;
    return 0;
}
