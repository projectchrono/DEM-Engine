//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <DEM/ApiSystem.h>
#include <DEM/HostSideHelpers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>

using namespace sgps;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;
    DEM_sim.SetVerbosity(INFO);
    DEM_sim.SetOutputFormat(DEM_OUTPUT_FORMAT::CSV);
    DEM_sim.SetOutputContent(DEM_OUTPUT_CONTENT::ABSV);
    std::cout << "Note: This is a relatively large demo and should take hours/days to run!!" << std::endl;

    srand(42);

    // A general template for ellipsoid with b = c = 1 and a = 2, where Z is the long axis
    std::vector<float> radii = {1.0, 0.88, 0.64, 0.88, 0.64};
    std::vector<float3> relPos = {make_float3(0, 0, 0), make_float3(0, 0, 0.86), make_float3(0, 0, 1.44),
                                  make_float3(0, 0, -0.86), make_float3(0, 0, -1.44)};
    // Then calculate mass and MOI
    float mass = 2.6e3 * 4. / 3. * SGPS_PI * 2 * 1 * 1;
    float3 MOI = make_float3(1. / 5. * mass * (1 * 1 + 2 * 2), 1. / 5. * mass * (1 * 1 + 2 * 2),
                             1. / 5. * mass * (1 * 1 + 1 * 1));
    // We can scale this general template to make it smaller, like a DEM particle that you would actually use
    float scaling = 0.01;

    auto mat_type_sand = DEM_sim.LoadMaterialType(1e9, 0.3, 0.3);
    auto mat_type_drum = DEM_sim.LoadMaterialType(2e9, 0.3, 0.4);

    // Bin size needs to make sure no too-many-sphere-per-bin situation happens
    DEM_sim.InstructBinSize(scaling);

    // Create some random clump templates for the filling materials
    // An array to store these generated clump templates
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_types;
    // Allocate the clump template definition arrays (all in SI)
    float this_mass = scaling * scaling * scaling * mass;
    float3 this_MOI = scaling * scaling * scaling * scaling * scaling * MOI;
    std::vector<float> this_radii(radii);
    std::vector<float3> this_relPos(relPos);
    std::transform(radii.begin(), radii.end(), this_radii.begin(), [scaling](float& r) { return r * scaling; });
    std::transform(relPos.begin(), relPos.end(), this_relPos.begin(), [scaling](float3& r) { return r * scaling; });

    // Load particle template(s)
    clump_types.push_back(DEM_sim.LoadClumpType(this_mass, this_MOI, this_radii, this_relPos, mat_type_sand));
    // clump_types.push_back(DEM_sim.LoadClumpSimpleSphere(this_mass, std::cbrt(2.0) * scaling, mat_type_sand));

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
    // Drum is family 10
    unsigned int drum_family = 100;
    family_code.push_back(drum_family);
    // The drum rotates (facing X direction)
    DEM_sim.SetFamilyPrescribedAngVel(drum_family, "0.1", "0", "0");
    // Disable contacts within drum components
    DEM_sim.DisableContactBetweenFamilies(drum_family, drum_family);

    // Then add top and bottom planes to `close up' the drum
    float safe_delta = 0.03;
    auto top_bot_planes = DEM_sim.AddExternalObject();
    top_bot_planes->AddPlane(make_float3(CylHeight / 2. - safe_delta, 0, 0), make_float3(-1, 0, 0), mat_type_drum);
    top_bot_planes->AddPlane(make_float3(-CylHeight / 2. + safe_delta, 0, 0), make_float3(1, 0, 0), mat_type_drum);
    // top_bot_planes->SetFamily(drum_family);
    auto planes_tracker = DEM_sim.Track(top_bot_planes);

    // Then sample some particles inside the drum
    float3 sample_center = make_float3(0, 0, 0);
    float sample_halfheight = CylHeight / 2.0 - 3.0 * safe_delta;
    float sample_halfwidth = CylRad / 1.5;
    // Ensure the spacing is acceptable for both spherical and ellipsoidal fillers
    auto input_material_xyz =
        DEMBoxGridSampler(sample_center, make_float3(sample_halfheight, sample_halfwidth, sample_halfwidth),
                          scaling * std::cbrt(2.0) * 2.1, scaling * std::cbrt(2.0) * 2.1, scaling * 2 * 2.1);
    input_xyz.insert(input_xyz.end(), input_material_xyz.begin(), input_material_xyz.end());
    unsigned int num_clumps = input_material_xyz.size();
    // Casually select from generated clump types
    for (unsigned int i = 0; i < num_clumps; i++) {
        input_template_type.push_back(clump_types.at(i % clump_types.size()));
        // Every clump type that has a unique mass, gets a unique family number
        family_code.push_back((i % clump_types.size()) / 2);
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
    DEM_sim.SetCDUpdateFreq(40);
    // DEM_sim.SetExpandFactor(1e-3);
    DEM_sim.SuggestExpandFactor(3.);
    DEM_sim.SuggestExpandSafetyParam(1.1);
    DEM_sim.Initialize();

    path out_dir = current_path();
    out_dir += "/DEMdemo_RotatingDrum";
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
            DEM_sim.ShowThreadCollaborationStats();
            char filename[100];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
            DEM_sim.WriteClumpFile(std::string(filename));
            currframe++;
            // float3 plane_vel = planes_tracker->Vel();
            // float4 plane_quat = planes_tracker->OriQ();
            // std::cout << "Vel of the planes: " << plane_vel.x << ", " << plane_vel.y << ", " << plane_vel.z
            //           << std::endl;
            // std::cout << "Quaternion of the planes: " << plane_quat.x << ", " << plane_quat.y << ", " << plane_quat.z
            //           << ", " << plane_quat.w << std::endl;
        }

        DEM_sim.DoStepDynamics(step_size);
        // We can quarry info out of this drum, since it is tracked
        // float3 drum_pos = Drum->Pos();
        // float3 drum_angVel = Drum->AngVel();
        // std::cout << "Position of the drum: " << drum_pos.x << ", " << drum_pos.y << ", " << drum_pos.z
        //           << std::endl;
        // std::cout << "Angular velocity of the drum: " << drum_angVel.x << ", " << drum_angVel.y << ", "
        //           << drum_angVel.z << std::endl;
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << (time_sec.count()) / time_end * 10.0 << " seconds (wall time) to finish 10 seconds' simulation"
              << std::endl;
    DEM_sim.ShowThreadCollaborationStats();
    DEM_sim.ResetWorkerThreads();
    DEM_sim.ClearThreadCollaborationStats();

    std::cout << "DEMdemo_RotatingDrum exiting..." << std::endl;
    // TODO: add end-game report APIs
    return 0;
}
