//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <map>
#include <random>
#include <cmath>

// =============================================================================
// In GRCPrep demo series, we try to prepare a sample of the GRC simulant, which
// are supposed to be used for extraterrestrial rover mobility simulations. It is
// made of particles of various sizes and shapes following a certain distribution.
// In Part1, it creates several batches of clumps and let them settle at the bottom
// of the domain.
// =============================================================================

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    // DEMSim.SetOutputContent(OUTPUT_CONTENT::FAMILY);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ);

    srand(759);

    // Define materials
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.4}});
    auto mat_type_wheel = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.4}});

    // Define the simulation world
    double world_y_size = 0.99;
    DEMSim.InstructBoxDomainDimension(world_y_size, world_y_size, world_y_size);
    // Add 5 bounding planes around the simulation world, and leave the top open
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);
    float bottom = -0.5;
    DEMSim.AddBCPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);

    // Define the terrain particle templates
    // Calculate its mass and MOI
    float terrain_density = 2.6e3;
    float volume1 = 4.2520508;
    float mass1 = terrain_density * volume1;
    float3 MOI1 = make_float3(1.6850426, 1.6375114, 2.1187753) * terrain_density;
    // Scale the template we just created
    std::vector<double> scales = {0.007};
    // Then load it to system
    std::shared_ptr<DEMClumpTemplate> my_template1 =
        DEMSim.LoadClumpType(mass1, MOI1, GetDEMEDataFile("clumps/triangular_flat.csv"), mat_type_terrain);
    std::vector<std::shared_ptr<DEMClumpTemplate>> ground_particle_templates = {my_template1};
    // Now scale those templates
    for (int i = 0; i < scales.size(); i++) {
        std::shared_ptr<DEMClumpTemplate>& my_template = ground_particle_templates.at(i);
        // Note the mass and MOI are also scaled in the process, automatically. But if you are not happy with this, you
        // can always manually change mass and MOI afterwards.
        my_template->Scale(scales.at(i));
        // Give these templates names, 0000, 0001 etc.
        char t_name[20];
        sprintf(t_name, "%04d", i);
        my_template->AssignName(std::string(t_name));
    }

    // Instatiate particles with a probability that is in line with their weight distribution.
    std::vector<double> weight_perc = {1.};
    std::vector<double> grain_perc;
    for (int i = 0; i < scales.size(); i++) {
        grain_perc.push_back(weight_perc.at(i) / std::pow(scales.at(i), 3));
    }
    {
        double tmp = vector_sum(grain_perc);
        std::for_each(grain_perc.begin(), grain_perc.end(), [tmp](double& p) { p /= tmp; });
        std::cout << "Percentage of grains add up to " << vector_sum(grain_perc) << std::endl;
    }

    // Sampler to use
    HCPSampler sampler(scales.at(0) * 2.5);

    // Make ready for simulation
    float step_size = 1e-6;
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    // wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    // happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(15.);
    // Error out vel is used to force the simulation to abort when something goes wrong.
    DEMSim.SetErrorOutVelocity(15.);
    DEMSim.SetExpandSafetyMultiplier(1.);
    DEMSim.Initialize();

    float time_end = 10.0;
    unsigned int fps = 10;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    path out_dir = current_path();
    out_dir += "/CheapGRCPrep_Part1";
    create_directory(out_dir);
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    float sample_halfheight = 0.18;
    float sample_halfwidth_x = (world_y_size * 0.96) / 2;
    float sample_halfwidth_y = (world_y_size * 0.96) / 2;
    float offset_z = bottom + sample_halfheight + 0.15;
    float settle_frame_time = 0.2;
    float settle_batch_time = 2.0;
    {
        DEMSim.ClearCache();
        float3 sample_center = make_float3(0, 0, offset_z);
        std::vector<std::shared_ptr<DEMClumpTemplate>> heap_template_in_use;
        std::vector<unsigned int> heap_family;
        // Sample and add heap particles
        auto heap_particles_xyz =
            sampler.SampleBox(sample_center, make_float3(sample_halfwidth_x, sample_halfwidth_y, sample_halfheight));
        for (unsigned int i = 0; i < heap_particles_xyz.size(); i++) {
            heap_template_in_use.push_back(ground_particle_templates.at(0));
            heap_family.push_back(0);
        }
        auto heap_particles = DEMSim.AddClumps(heap_template_in_use, heap_particles_xyz);
        // Give ground particles a small initial velocity so they `collapse' at the start of the simulation
        heap_particles->SetVel(make_float3(0.00, 0, -0.05));
        heap_particles->SetFamilies(heap_family);
        DEMSim.UpdateClumps();
        std::cout << "Current number of clumps: " << DEMSim.GetNumClumps() << std::endl;

        // Allow for some settling
        // Must DoDynamicsThenSync (not DoDynamics), as adding entities to the simulation is only allowed at a sync-ed
        // point of time.
        for (float t = 0; t < settle_batch_time; t += settle_frame_time) {
            std::cout << "Frame: " << currframe << std::endl;
            char filename[200];
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe++);
            DEMSim.WriteSphereFile(std::string(filename));
            DEMSim.DoDynamicsThenSync(settle_frame_time);
        }

        DEMSim.ShowThreadCollaborationStats();
    }

    // Settle for some time more
    DEMSim.DoDynamicsThenSync(1.0);

    char cp_filename[200];
    sprintf(cp_filename, "%s/GRC_3e5.csv", out_dir.c_str());
    DEMSim.WriteClumpFile(std::string(cp_filename));

    DEMSim.ShowThreadCollaborationStats();
    DEMSim.ClearThreadCollaborationStats();

    char cnt_filename[200];
    sprintf(cnt_filename, "%s/Contact_pairs_3e5.csv", out_dir.c_str());
    DEMSim.WriteContactFile(std::string(cnt_filename));

    std::cout << "DEMdemo_GRCPrep_Part1 exiting..." << std::endl;
    return 0;
}
