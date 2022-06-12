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
    DEM_sim.SetVerbosity(INFO);

    srand(42);

    // total number of random clump templates to generate
    int num_template = 6;

    int min_sphere = 1;
    int max_sphere = 5;

    float min_rad = 0.005;
    float max_rad = 0.01;

    float min_relpos = -0.005;
    float max_relpos = 0.005;

    auto mat_type_sand = DEM_sim.LoadMaterialType(1e9, 0.3, 0.8);
    auto mat_type_drum = DEM_sim.LoadMaterialType(2e9, 0.3, 0.9);

    // Standard bin size
    DEM_sim.InstructBinSize(min_rad * 2.0);

    // Create some random clump templates for the filling materials
    // An array to store these generated clump templates
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_types;
    // Then randomly create some clump templates for filling the drum
    for (int i = 0; i < num_template; i++) {
        // first decide the number of spheres that live in this clump
        int num_sphere = rand() % (max_sphere - min_sphere + 1) + 1;

        // then allocate the clump template definition arrays (all in SI)
        float mass = 0.1 * (float)num_sphere;
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
            mat.push_back(mat_type_sand);

            // seed relPos from one of the previously generated spheres
            int choose_from = rand() % (j + 1);
            seed_pos = relPos.at(choose_from);
        }

        // it returns the numbering of this clump template (although here we don't care)
        clump_types.push_back(DEM_sim.LoadClumpType(mass, MOI, radii, relPos, mat));
    }

    // Drum is a `big clump', we now generate its template
    float3 CylCenter = make_float3(0, 0, 0);
    float3 CylAxis = make_float3(1, 0, 0);
    float CylRad = 2.0;
    float CylHeight = 1.0;
    float CylMass = 1.0;
    float CylParticleRad = 0.05;
    float IXX = CylMass * CylRad * CylRad;
    float IYY = (CylMass / 12) * (3 * CylRad * CylRad + CylHeight * CylHeight);
    auto Drum_particles = DEMCylSurfSampler(CylCenter, CylAxis, CylRad, CylHeight, CylParticleRad);
    auto Drum_template =
        DEM_sim.LoadClumpType(CylMass, make_float3(IXX, IYY, IYY),
                              std::vector<float>(Drum_particles.size(), CylParticleRad), Drum_particles, mat_type_drum);
    std::cout << Drum_particles.size() << " spheres make up the rotating drum" << std::endl;

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_template_type;
    std::vector<float3> input_xyz;
    std::vector<unsigned int> family_code;

    // Add drum
    auto Drum = DEM_sim.AddClumpTracked(Drum_template, make_float3(0));
    // Drum is family 1
    family_code.push_back(1);
    // The drum rotates (facing X direction)
    DEM_sim.SetFamilyPrescribedAngVel(1, "6.0", "0", "0");

    // Then add top and bottom planes to `close up' the drum
    float safe_delta = 0.03;
    DEM_sim.AddBCPlane(make_float3(CylHeight / 2. - safe_delta, 0, 0), make_float3(-1, 0, 0), mat_type_drum);
    DEM_sim.AddBCPlane(make_float3(-CylHeight / 2. + safe_delta, 0, 0), make_float3(1, 0, 0), mat_type_drum);

    // Then sample some particles inside the drum
    float3 sample_center = make_float3(0, 0, 0);
    float sample_halfheight = CylHeight / 2.0 - 3.0 * safe_delta;
    float sample_halfwidth = CylRad / 1.45;
    auto input_material_xyz =
        DEMBoxGridSampler(sample_center, make_float3(sample_halfheight, sample_halfwidth, sample_halfwidth), 0.025);
    input_xyz.insert(input_xyz.end(), input_material_xyz.begin(), input_material_xyz.end());
    unsigned int num_clumps = input_material_xyz.size();
    family_code.insert(family_code.end(), num_clumps, 0);
    // Casually select from generated clump types
    for (unsigned int i = 0; i < num_clumps; i++) {
        input_template_type.push_back(clump_types.at(i % num_template));
    }

    // Finally, input to system
    DEM_sim.AddClumps(input_template_type, input_xyz);
    DEM_sim.SetClumpFamilies(family_code);
    DEM_sim.InstructBoxDomainNumVoxel(21, 21, 22, 4e-11);

    float step_size = 5e-6;
    DEM_sim.CenterCoordSys();
    DEM_sim.SetTimeStepSize(step_size);
    DEM_sim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEM_sim.SetCDUpdateFreq(20);
    // DEM_sim.SetExpandFactor(1e-3);
    DEM_sim.SuggestExpandFactor(12.);
    DEM_sim.SuggestExpandSafetyParam(1.1);
    DEM_sim.Initialize();

    path out_dir = current_path();
    out_dir += "/DEMdemo_Drum";
    create_directory(out_dir);

    float time_end = 10.0;
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
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
            DEM_sim.WriteFileAsSpheres(std::string(filename));
            currframe++;
        }

        DEM_sim.DoStepDynamics(step_size);
        // We can quarry info out of this drum, since it is tracked
        float3 drum_pos = Drum->Pos();
        float3 drum_angVel = Drum->AngVel();
        // std::cout << "Position of the drum: " << drum_pos.x << ", " << drum_pos.y << ", " << drum_pos.z
        //           << std::endl;
        // std::cout << "Angular velocity of the drum: " << drum_angVel.x << ", " << drum_angVel.y << ", "
        //           << drum_angVel.z << std::endl;
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds" << std::endl;
    DEM_sim.ShowThreadCollaborationStats();
    DEM_sim.ResetWorkerThreads();
    DEM_sim.ClearThreadCollaborationStats();

    std::cout << "DEMdemo_Drum exiting..." << std::endl;
    // TODO: add end-game report APIs
    return 0;
}
