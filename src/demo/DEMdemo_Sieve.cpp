//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <DEM/ApiSystem.h>
#include <DEM/HostSideHelpers.cpp>

#include <cstdio>
#include <chrono>

using namespace sgps;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;

    srand(time(NULL));
    // srand(4150);

    // total number of random clump templates to generate
    int num_template = 10;

    int min_sphere = 1;
    int max_sphere = 5;

    float min_rad = 0.010;
    float max_rad = 0.024;

    float min_relpos = -0.015;
    float max_relpos = 0.015;

    auto mat_type_1 = DEM_sim.LoadMaterialType(1e8, 0.3, 0.2);
    auto mat_type_2 = DEM_sim.LoadMaterialType(1e8, 0.3, 0.3);

    // First create clump type 0 for representing the sieve
    float sieve_sp_r = 0.05;
    auto template_sieve = DEM_sim.LoadClumpSimpleSphere(5.0, sieve_sp_r, mat_type_1);

    for (int i = 0; i < num_template; i++) {
        // first decide the number of spheres that live in this clump
        int num_sphere = rand() % (max_sphere - min_sphere + 1) + 1;

        // then allocate the clump template definition arrays (all in SI)
        float mass = 0.4 * (float)num_sphere;
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
            mat.push_back(mat_type_1);

            // seed relPos from one of the previously generated spheres
            int choose_from = rand() % (j + 1);
            seed_pos = relPos.at(choose_from);
        }

        // it returns the numbering of this clump template (although here we don't care)
        auto template_num = DEM_sim.LoadClumpType(mass, MOI, radii, relPos, mat);
    }

    std::vector<unsigned int> input_template_num;
    std::vector<unsigned int> family_code;

    // generate sieve clumps
    auto input_xyz = DEMBoxGridSampler(make_float3(0, 0, 0), make_float3(5.0, 5.0, 0.001), sieve_sp_r * 2.0);
    // The sieve is family 1
    family_code.insert(family_code.end(), input_xyz.size(), 1);
    DEM_sim.SetFamilyPrescribedLinVel(1, "0", "0", "(t > 1.0) ? sin(2.0 * SGPS_PI * (t - 1.0)) : 0");
    // No contact within family 1
    DEM_sim.SetFamilyNoContact(1, 1);
    input_template_num.insert(input_template_num.end(), input_xyz.size(), template_sieve);

    // float sample_halfheight = 1.;
    float sample_halfheight = 0.15;
    float sample_halfwidth = 2.;
    // generate initial clumps for piling
    float3 sample_center = make_float3(0, 0, sample_halfheight + sieve_sp_r + 0.07);
    auto pile =
        DEMBoxGridSampler(sample_center, make_float3(sample_halfwidth, sample_halfwidth, sample_halfheight), 0.07);
    input_xyz.insert(input_xyz.end(), pile.begin(), pile.end());
    unsigned int num_clumps = pile.size();
    for (unsigned int i = 0; i < num_clumps; i++) {
        input_template_num.push_back(i % (num_template) + 1);
        family_code.push_back(0);
    }

    std::shared_ptr<DEMExternObj> BCs = DEM_sim.AddExternalObject();
    BCs->SetFamily(2);
    BCs->AddPlane(make_float3(0, 4.8, 0), make_float3(0, -1, 0), mat_type_2);
    BCs->AddPlane(make_float3(0, -4.8, 0), make_float3(0, 1, 0), mat_type_2);
    BCs->AddPlane(make_float3(4.8, 0, 0), make_float3(-1, 0, 0), mat_type_2);
    BCs->AddPlane(make_float3(-4.8, 0, 0), make_float3(1, 0, 0), mat_type_2);
    BCs->AddPlane(make_float3(0, 0, -10.0), make_float3(0, 0, 1), mat_type_2);
    BCs->AddPlane(make_float3(0, 0, 10.0), make_float3(0, 0, -1), mat_type_2);
    DEM_sim.SetFamilyFixed(2);  // BCs are fixed!
    // BC family does not interact with the sieve
    DEM_sim.SetFamilyNoContact(1, 2);

    DEM_sim.AddClumps(input_template_num, input_xyz);
    DEM_sim.SetClumpFamily(family_code);
    DEM_sim.InstructBoxDomainNumVoxel(21, 21, 22, 7.5e-11);

    DEM_sim.CenterCoordSys();
    DEM_sim.SetTimeStepSize(5e-6);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    DEM_sim.SetCDUpdateFreq(10);
    DEM_sim.SuggestExpandFactor(10.0, 5e-6 * 10);
    DEM_sim.SuggestExpandSafetyParam(2.);

    DEM_sim.Initialize();

    path out_dir = current_path();
    out_dir += "/DEMdemo_Sieve";
    create_directory(out_dir);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 200; i++) {
        char filename[100];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), i);
        DEM_sim.WriteFileAsSpheres(std::string(filename));
        std::cout << "Frame: " << i << std::endl;

        DEM_sim.LaunchThreads(1.0 / 20.);
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds" << std::endl;

    std::cout << "DEMdemo_Sieve exiting..." << std::endl;
    // TODO: add end-game report APIs
    return 0;
}
