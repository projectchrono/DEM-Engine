//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <DEM/ApiSystem.h>
#include <DEM/HostSideHelpers.cpp>

#include <cstdio>
#include <time.h>

using namespace sgps;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;

    srand(42);

    // total number of random clump templates to generate
    int num_template = 6;

    int min_sphere = 1;
    int max_sphere = 5;

    float min_rad = 0.01;
    float max_rad = 0.02;

    float min_relpos = -0.01;
    float max_relpos = 0.01;

    auto mat_type_sand = DEM_sim.LoadMaterialType(1e8, 0.3, 0.8);
    auto mat_type_drum = DEM_sim.LoadMaterialType(1e9, 0.3, 0.9);

    // First create a clump type for representing the drum
    float drum_sp_r = 0.02;
    auto template_drum = DEM_sim.LoadClumpSimpleSphere(0.5, drum_sp_r, mat_type_drum);

    // Then randomly create some clumps for piling up
    for (int i = 0; i < num_template; i++) {
        // first decide the number of spheres that live in this clump
        int num_sphere = rand() % (max_sphere - min_sphere + 1) + 1;

        // then allocate the clump template definition arrays (all in SI)
        float mass = 0.1 * (float)num_sphere;
        float3 MOI =
            make_float3(2e-5 * (float)num_sphere, 1.5e-5 * (float)num_sphere, 1.8e-5 * (float)num_sphere) * 50.;
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
            mat.push_back(mat_type_sand);

            // seed relPos from one of the previously generated spheres
            int choose_from = rand() % (j + 1);
            seed_pos = relPos.at(choose_from);
        }

        // it returns the numbering of this clump template (although here we don't care)
        auto template_num = DEM_sim.LoadClumpType(mass, MOI, radii, relPos, mat);
    }

    std::vector<unsigned int> input_template_num;
    std::vector<unsigned int> family_code;

    // generate drum clumps
    float3 CylCenter = make_float3(0, 0, 0);
    float3 CylAxis = make_float3(1, 0, 0);
    float CylRad = 0.2;
    float CylHeight = 0.1;
    float SideIncr = 0.03;
    unsigned int NumRows = 300;
    auto Drum = DEMCylSurfSampler(CylCenter, CylAxis, CylRad, CylHeight, SideIncr, NumRows);
    // Drum is family 1
    family_code.insert(family_code.end(), Drum.size(), 1);
    // TODO: finish it!!
    DEM_sim.SetFamilyPrescribedLinVel(1, "0", "0", "0");
    input_template_num.insert(input_template_num.end(), Drum.size(), template_drum);

    DEM_sim.CenterCoordSys();
    DEM_sim.SetTimeStepSize(5e-6);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEM_sim.SetCDUpdateFreq(5);
    // DEM_sim.SetExpandFactor(1e-3);
    DEM_sim.SuggestExpandFactor(10.);
    DEM_sim.SuggestExpandSafetyParam(2.);
    DEM_sim.Initialize();

    path out_dir = current_path();
    out_dir += "/DEMdemo_Drum";
    create_directory(out_dir);
    for (int i = 0; i < 200; i++) {
        char filename[100];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), i);
        std::cout << "Frame: " << i << std::endl;
        DEM_sim.LaunchThreads(3e-2);
    }

    std::cout << "DEMdemo_Drum exiting..." << std::endl;
    // TODO: add end-game report APIs
    return 0;
}
