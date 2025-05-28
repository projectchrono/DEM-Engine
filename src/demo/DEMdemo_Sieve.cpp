//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A clump-represented sieve with a presecibed back-and-forth motion is in this
// demo. It lets particles that are small enough fall through it.
// =============================================================================

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
    // I generally use this demo to inspect if I have "lost contact pairs", so the verbosity is set to STEP_METRIC...
    DEMSim.SetVerbosity(STEP_METRIC);

    // If you don't need individual force information, then this option makes the solver run a bit faster.
    DEMSim.SetNoForceRecord();

    srand(759);

    // total number of random clump templates to generate
    int num_template = 10;

    int min_sphere = 1;
    int max_sphere = 5;

    float min_rad = 0.010;
    float max_rad = 0.024;

    float min_relpos = -0.015;
    float max_relpos = 0.015;

    auto mat_type_1 = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.5}});

    // First create clump type 0 for representing the sieve
    float sieve_sp_r = 0.05;
    auto template_sieve = DEMSim.LoadSphereType(5.0, sieve_sp_r, mat_type_1);

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
        // the relPos of a sphere is always seeded from one of some previously-generated spheres
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

            // seed relPos from one of the previously-generated spheres
            int choose_from = rand() % (j + 1);
            seed_pos = relPos.at(choose_from);
        }

        // LoadClumpType returns the pointer to this clump template we just loaded
        clump_types.push_back(DEMSim.LoadClumpType(mass, MOI, radii, relPos, mat));
    }

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_template_type;
    std::vector<unsigned int> family_code;

    // generate sieve clumps
    auto input_xyz = DEMBoxGridSampler(make_float3(0, 0, 0), make_float3(5.0, 5.0, 0.001), sieve_sp_r * 2.0);
    // The sieve is family 1
    family_code.insert(family_code.end(), input_xyz.size(), 1);
    DEMSim.SetFamilyPrescribedLinVel(1, "0", "0", "(t > 1.0) ? 2.0 * sin(5.0 * deme::PI * (t - 1.0)) : 0");
    // No contact within family 1
    DEMSim.DisableContactBetweenFamilies(1, 1);
    input_template_type.insert(input_template_type.end(), input_xyz.size(), template_sieve);
    auto sieve = DEMSim.AddClumps(input_template_type, input_xyz);
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
    auto grains = DEMSim.AddClumps(input_template_type, input_xyz);
    grains->SetFamilies(family_code);

    std::shared_ptr<DEMExternObj> BCs = DEMSim.AddExternalObject();
    BCs->SetFamily(2);
    BCs->AddPlane(make_float3(0, 4.8, 0), make_float3(0, -1, 0), mat_type_1);
    BCs->AddPlane(make_float3(0, -4.8, 0), make_float3(0, 1, 0), mat_type_1);
    BCs->AddPlane(make_float3(4.8, 0, 0), make_float3(-1, 0, 0), mat_type_1);
    BCs->AddPlane(make_float3(-4.8, 0, 0), make_float3(1, 0, 0), mat_type_1);
    BCs->AddPlane(make_float3(0, 0, -10.0), make_float3(0, 0, 1), mat_type_1);
    BCs->AddPlane(make_float3(0, 0, 10.0), make_float3(0, 0, -1), mat_type_1);
    DEMSim.SetFamilyFixed(2);  // BCs are fixed!
    // BC family does not interact with the sieve
    DEMSim.DisableContactBetweenFamilies(1, 2);

    // Keep tab of the max velocity in simulation
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");
    float max_v;

    float step_size = 1e-5;
    DEMSim.InstructBoxDomainDimension(12, 12, 25);
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    DEMSim.SetCDUpdateFreq(30);
    DEMSim.SetExpandSafetyMultiplier(1.0);
    DEMSim.SetExpandSafetyAdder(1.0);
    // You usually don't have to worry about initial bin size. In very rare cases, init bin size is so bad that auto bin
    // size adaption is effectless, and you should notice in that case kT runs extremely slow. Then in that case setting
    // init bin size may save the simulation.
    // DEMSim.SetInitBinSize(0.1);
    // DEMSim.DisableAdaptiveBinSize();

    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir /= "DemoOutput_Sieve";
    create_directory(out_dir);

    float time_end = 40.0;
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
            sprintf(filename, "DEMdemo_output_%04d.csv", currframe++);
            DEMSim.WriteSphereFile(out_dir / filename);
            max_v = max_v_finder->GetValue();
            std::cout << "Max velocity of any point in simulation is " << max_v << std::endl;
        }

        DEMSim.DoDynamics(step_size);
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;

    DEMSim.ShowThreadCollaborationStats();
    DEMSim.ShowTimingStats();

    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "DEMdemo_Sieve exiting..." << std::endl;
    return 0;
}
