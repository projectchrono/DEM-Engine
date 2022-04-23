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

    srand(759);

    // total number of random clump templates to generate
    int num_template = 6;

    int min_sphere = 1;
    int max_sphere = 5;

    float min_rad = 0.01;
    float max_rad = 0.02;

    float min_relpos = -0.01;
    float max_relpos = 0.01;

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

    auto mat_type_1 = DEM_sim.LoadMaterialType(1e8, 0.3, 0.2);

    // First create clump type 0 for representing the ground
    float ground_sp_r = 0.02;
    auto template_ground = DEM_sim.LoadClumpSimpleSphere(0.5, ground_sp_r, mat_type_1);

    // Then randomly create some clumps for piling up
    for (int i = 0; i < num_template; i++) {
        // first decide the number of spheres that live in this clump
        int num_sphere = rand() % (max_sphere - min_sphere + 1) + 1;

        // then allocate the clump template definition arrays (all in SI)
        float mass = 0.1 * (float)num_sphere;
        float3 MOI =
            make_float3(2e-5 * (float)num_sphere, 1.5e-5 * (float)num_sphere, 1.8e-5 * (float)num_sphere) * 10.;
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

    // generate ground clumps
    std::vector<unsigned int> input_template_num;
    std::vector<unsigned int> family_code;
    auto input_xyz = DEMBoxGridSampler(make_float3(0, 0, -3.8), make_float3(5.2, 5.2, 0.001), ground_sp_r * 1.3);
    // // generate domain bottom
    // auto domain_bottom = DEMBoxGridSampler(make_float3(0, 0, -10.0), make_float3(5.2, 5.2, 0.001), ground_sp_r
    // * 1.3); input_xyz.insert(input_xyz.end(), domain_bottom.begin(), domain_bottom.end()); Mark family 1 as fixed
    family_code.insert(family_code.end(), input_xyz.size(), 1);
    input_template_num.insert(input_template_num.end(), input_xyz.size(), template_ground);

    // generate initial clumps for piling
    float3 sample_center = make_float3(0, 0, -1);
    float sample_halfheight = 2;
    float sample_halfwidth = 0.7;
    auto pile =
        DEMBoxGridSampler(sample_center, make_float3(sample_halfwidth, sample_halfwidth, sample_halfheight), 0.05);
    input_xyz.insert(input_xyz.end(), pile.begin(), pile.end());
    unsigned int num_clumps = pile.size();
    for (unsigned int i = 0; i < num_clumps; i++) {
        input_template_num.push_back(i % (num_template) + 1);
        family_code.push_back(0);
    }
    DEM_sim.AddClumps(input_template_num, input_xyz);
    DEM_sim.SetClumpFamily(family_code);

    DEM_sim.InstructBoxDomainNumVoxel(21, 21, 22, 7.5e-11);
    // DEM_sim.InstructBoxDomainNumVoxel(11, 11, 10, 1e-10);
    DEM_sim.AddBCPlane(make_float3(0, 9, 0), make_float3(0, -1, 0), mat_type_1);
    DEM_sim.AddBCPlane(make_float3(0, -9, 0), make_float3(0, 1, 0), mat_type_1);
    DEM_sim.AddBCPlane(make_float3(9, 0, 0), make_float3(-1, 0, 0), mat_type_1);
    std::shared_ptr<DEMExternObj> plane_a = DEM_sim.AddExternalObject();
    plane_a->AddPlane(make_float3(-9, 0, 0), make_float3(1, 0, 0), mat_type_1);

    DEM_sim.CenterCoordSys();
    DEM_sim.SetTimeStepSize(5e-6);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEM_sim.SetCDUpdateFreq(5);
    // DEM_sim.SetExpandFactor(1e-3);
    DEM_sim.SuggestExpandFactor(10.);
    DEM_sim.SuggestExpandSafetyParam(2.);
    DEM_sim.Initialize();

    DEM_sim.UpdateSimParams();  // Not needed; just testing if this function works...

    for (int i = 0; i < 200; i++) {
        char filename[100];
        sprintf(filename, "./DEMdemo_collide_output_%04d.csv", i);
        DEM_sim.WriteFileAsSpheres(std::string(filename));
        std::cout << "Frame: " << i << std::endl;
        // float KE = DEM_sim.GetTotalKineticEnergy();
        // std::cout << "Total kinetic energy: " << KE << std::endl;
        DEM_sim.LaunchThreads(3e-2);
    }

    std::cout << "DEMdemo_Pile exiting..." << std::endl;
    // TODO: add end-game report APIs
    return 0;
}
