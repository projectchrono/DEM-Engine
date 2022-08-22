//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>

using namespace sgps;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;
    // DEM_sim.SetVerbosity(INFO);
    DEM_sim.SetVerbosity(STEP_METRIC);

    srand(759);

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

    // An array to store these generated clump templates
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_types;
    for (int i = 0; i < num_template; i++) {
        // first decide the number of spheres that live in this clump
        int num_sphere = rand() % (max_sphere - min_sphere + 1) + 1;

        // then allocate the clump template definition arrays (all in SI)
        float mass = 0.4 * (float)num_sphere;
        float3 MOI =
            make_float3(2e-5 * (float)num_sphere, 1.5e-5 * (float)num_sphere, 1.8e-5 * (float)num_sphere) * 50.;
        std::vector<float> radii;
        std::vector<float3> relPos;
        std::vector<std::shared_ptr<DEMMaterial>> mat;

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

        // LoadClumpType returns the pointer to this clump template we just loaded
        clump_types.push_back(DEM_sim.LoadClumpType(mass, MOI, radii, relPos, mat));
    }

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_template_type;
    std::vector<unsigned int> family_code;

    // generate sieve clumps
    auto input_xyz = DEMBoxGridSampler(make_float3(0, 0, 0), make_float3(5.0, 5.0, 0.001), sieve_sp_r * 2.0);
    // The sieve is family 1
    family_code.insert(family_code.end(), input_xyz.size(), 1);
    DEM_sim.SetFamilyPrescribedLinVel(1, "0", "0", "(t > 1.0) ? 2.0 * sin(5.0 * sgps::PI * (t - 1.0)) : 0");
    // No contact within family 1
    DEM_sim.DisableContactBetweenFamilies(1, 1);
    input_template_type.insert(input_template_type.end(), input_xyz.size(), template_sieve);
    auto sieve = DEM_sim.AddClumps(input_template_type, input_xyz);
    sieve->SetFamilies(family_code);

    // Clear arrays and add the grains being sieved
    std::vector<std::shared_ptr<DEMClumpTemplate>>().swap(input_template_type);
    std::vector<unsigned int>().swap(family_code);
    float sample_halfheight = 0.15;
    float sample_halfwidth = 2.;
    // generate initial clumps for piling
    float3 sample_center = make_float3(0, 0, sample_halfheight + sieve_sp_r + 0.07);
    input_xyz =
        DEMBoxGridSampler(sample_center, make_float3(sample_halfwidth, sample_halfwidth, sample_halfheight), 0.07);
    unsigned int num_clumps = input_xyz.size();
    // Casually select from generated clump types
    for (unsigned int i = 0; i < num_clumps; i++) {
        input_template_type.push_back(clump_types.at(i % num_template));
        family_code.push_back(0);
    }
    auto grains = DEM_sim.AddClumps(input_template_type, input_xyz);
    grains->SetFamilies(family_code);

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
    DEM_sim.DisableContactBetweenFamilies(1, 2);

    float step_size = 5e-6;
    DEM_sim.InstructBoxDomainNumVoxel(21, 21, 22, 7.5e-11);
    DEM_sim.SetCoordSysOrigin("center");
    DEM_sim.SetInitTimeStep(step_size);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    DEM_sim.SetCDUpdateFreq(30);
    DEM_sim.SetExpandFactor(6.0 * 30 * step_size);
    DEM_sim.SetExpandSafetyParam(1.0);

    DEM_sim.Initialize();

    path out_dir = current_path();
    out_dir += "/DEMdemo_Sieve";
    create_directory(out_dir);

    float time_end = 20.0;
    unsigned int fps = 20;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (double t = 0; t < (double)time_end; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            std::cout << "Frame: " << currframe << std::endl;
            char filename[100];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEM_sim.WriteSphereFile(std::string(filename));
        }

        DEM_sim.DoDynamics(step_size);
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds" << std::endl;

    DEM_sim.ShowThreadCollaborationStats();
    DEM_sim.ShowTimingStats();
    std::cout << "DEMdemo_Sieve exiting..." << std::endl;
    // TODO: add end-game report APIs
    return 0;
}
