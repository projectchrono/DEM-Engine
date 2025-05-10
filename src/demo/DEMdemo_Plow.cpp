//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A bowl plowing in a pile of granular material.
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

const double math_PI = 3.1415927;

void AdvanceSimulation(DEMSolver& DEMSim,
                       double time,
                       float step_size,
                       unsigned int out_steps,
                       const path& out_dir,
                       unsigned int& curr_step,
                       unsigned int& currframe) {
    for (double t = 0; t < time; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            char filename[100], meshfile[100];
            sprintf(filename, "DEMdemo_output_%04d.csv", currframe);
            sprintf(meshfile, "DEMdemo_excavator_%04d.vtk", currframe);
            DEMSim.WriteSphereFile(out_dir / filename);
            DEMSim.WriteMeshFile(out_dir / meshfile);
            std::cout << "Frame: " << currframe << std::endl;
            currframe++;
            DEMSim.ShowThreadCollaborationStats();
        }

        DEMSim.DoStepDynamics();
    }
}

int main() {
    DEMSolver DEMSim;
    DEMSim.UseFrictionalHertzianModel();
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);

    // Scale-defining numbers of this simulation.
    float world_halfsize = 5.;
    float bowl_bottom = -world_halfsize;

    auto mat_type_walls = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.5}});
    auto mat_type_particles = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.7}, {"mu", 0.5}});
    // If you don't have this line, then CoR between wall material and granular material will be 0.5 (average of the
    // two).
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_walls, mat_type_particles, 0.3);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_walls, mat_type_particles, 0.5);

    // Define the terrain particle templates
    // Calculate its mass and MOI
    float mass = 2.6e3 * 4. / 3. * math_PI * 2 * 1 * 1;
    float3 MOI = make_float3(1. / 5. * mass * (1 * 1 + 2 * 2), 1. / 5. * mass * (1 * 1 + 2 * 2),
                             1. / 5. * mass * (1 * 1 + 1 * 1));
    // We can scale this general template to make it smaller, like a DEM particle that you would actually use
    float scaling = 0.03;
    std::shared_ptr<DEMClumpTemplate> my_template =
        DEMSim.LoadClumpType(mass, MOI, GetDEMEDataFile("clumps/ellipsoid_2_1_1.csv"), mat_type_particles);
    my_template->Scale(scaling);

    // Load the excavator mesh...
    auto excavator = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/excavator.obj"), mat_type_walls);
    // Upon-loading move command is used to move the mesh's (0,0,0) to its MOI, this is needed for DEME simulations. But
    // the mesh is already created so.
    excavator->Move(make_float3(0, 0, 0), make_float4(0, 0, 0, 1));
    // Scale so it fits the size of our simulation world.
    excavator->Scale(1. / 20.);
    // Initial position (mid-air), and initial quaternion (up-side-down).
    float3 init_pos = make_float3(0, 0.6 * world_halfsize, 0.2 * world_halfsize);
    float4 init_Q = make_float4(0.7071, 0, 0, 0.7071);  // 90 deg about x
    excavator->SetInitPos(init_pos);
    excavator->SetInitQuat(init_Q);
    excavator->SetFamily(10);
    // No contact in the settling phase
    DEMSim.DisableContactBetweenFamilies(0, 10);
    DEMSim.SetFamilyFixed(10);

    // Generate initial clumps for piling
    float spacing = 2. * scaling;
    float fill_halfwidth = world_halfsize - 4. * scaling;
    float fill_height = world_halfsize * 1.5;
    float fill_bottom = bowl_bottom + 3. * scaling;
    PDSampler sampler(spacing);
    // Use a PDSampler-based clump generation process. For PD sampler it is better to do it layer by layer.
    std::vector<float3> input_pile_xyz;
    float layer_z = 0;
    while (layer_z < fill_height) {
        float3 sample_center = make_float3(0, 0, fill_bottom + layer_z);
        auto layer_xyz = sampler.SampleBox(sample_center, make_float3(fill_halfwidth, fill_halfwidth, 0));
        input_pile_xyz.insert(input_pile_xyz.end(), layer_xyz.begin(), layer_xyz.end());
        layer_z += 4.5 * scaling;
    }
    // Note: AddClumps can be called multiple times before initialization to add more clumps to the system.
    auto the_pile = DEMSim.AddClumps(my_template, input_pile_xyz);
    the_pile->SetFamily(0);

    // Two sets of prescribed motions. We are just using families, since families are for bulk control of know
    // prescribed motions. You can use trackers too, but remember trackers are usually for fine-grain explicit motion
    // control.
    // Family 1 rotated about Point (0, 0, 0.4 * world_halfsize), and this is a fixed point in space not on the bowl. So
    // what we have to prescribe is the bowl's linear velocity. And its angular velocity is just a constant spin about
    // its own x.
    DEMSim.SetFamilyPrescribedLinVel(1, "0",
                                     "-" + to_string_with_precision(0.3 * world_halfsize) + " * sin(3.14 / 4. * t)",
                                     "-" + to_string_with_precision(0.3 * world_halfsize) + " * cos(3.14 / 4. * t)");
    DEMSim.SetFamilyPrescribedAngVel(1, "-3.14 / 4", "0", "0");
    // For family 2, it represents the bowl rotating about itself to pour out the granular material it holds. So it is a
    // rotation about its own frame.
    DEMSim.SetFamilyPrescribedLinVel(2, "0", "0", "0");
    DEMSim.SetFamilyPrescribedAngVel(2, "3.14 / 4", "0", "0");

    float step_size = 5e-6;
    DEMSim.InstructBoxDomainDimension({-world_halfsize, world_halfsize}, {-world_halfsize, world_halfsize},
                                      {-world_halfsize, world_halfsize});
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_walls);
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir /= "DemoOutput_Plow";
    create_directory(out_dir);

    unsigned int fps = 20;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;
    unsigned int curr_step = 0;

    // Settle
    DEMSim.DoDynamicsThenSync(1.5);
    DEMSim.ChangeFamily(10, 1);
    // We'll later use family 10 for fixing the excavator, so let's re-enable 10--0 contact.
    DEMSim.EnableContactBetweenFamilies(0, 10);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Use 3 seconds to plow
    AdvanceSimulation(DEMSim, 3, step_size, out_steps, out_dir, curr_step, currframe);
    // Stay for 2 seconds
    DEMSim.ChangeFamily(1, 10);
    AdvanceSimulation(DEMSim, 2, step_size, out_steps, out_dir, curr_step, currframe);
    // Then 2 seconds to pour
    DEMSim.ChangeFamily(10, 2);
    AdvanceSimulation(DEMSim, 2, step_size, out_steps, out_dir, curr_step, currframe);
    // Then rest in-place for a while
    DEMSim.ChangeFamily(2, 10);
    AdvanceSimulation(DEMSim, 3, step_size, out_steps, out_dir, curr_step, currframe);

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds (wall time) to finish the 10-second plowing simulation." << std::endl;

    DEMSim.ShowTimingStats();
    DEMSim.ClearTimingStats();

    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "DEMdemo_Plow exiting..." << std::endl;
    return 0;
}
