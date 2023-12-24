//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Fracture
// =============================================================================

#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace deme;

const double math_PI = 3.1415927;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetContactOutputContent(OWNER | CNT_WILDCARD | DEME_POINT);

    DEMSim.SetErrorOutAvgContacts(150);
    // E, nu, CoR, mu, Crr...
    auto mat_type_container =
        DEMSim.LoadMaterial({{"E", 100e9}, {"nu", 0.3}, {"CoR", 0.7}, {"mu", 1.0}, {"Crr", 0.10}});
    auto mat_type_particle = DEMSim.LoadMaterial({{"E", 6e9}, {"nu", 0.33}, {"CoR", 0.5}, {"mu", 0.50}, {"Crr", 0.50}});
    // If you don't have this line, then values will take average between 2 materials, when they are in contact
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_container, mat_type_particle, 0.7);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_container, mat_type_particle, 0.6);
    // We can specify the force model using a file.
    auto my_force_model = DEMSim.ReadContactForceModel("ForceModelWithFractureModel.cu");

    // Those following lines are needed. We must let the solver know that those var names are history variable etc.
    my_force_model->SetMustHaveMatProp({"E", "nu", "CoR", "mu", "Crr"});
    my_force_model->SetMustPairwiseMatProp({"CoR", "mu", "Crr"});
    // Pay attention to the extra per-contact wildcard `unbroken' here.
    my_force_model->SetPerContactWildcards(
        {"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z", "unbroken", "initialLength","damage"});

    float world_size = 5.;
    float container_diameter = 0.50;
    float step_size = 2e-7;
    DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size);
    // No need to add simulation `world' boundaries, b/c we'll add a cylinderical container manually
    DEMSim.InstructBoxDomainBoundingBC("all", mat_type_container);
    // Now add a cylinderical boundary along with a bottom plane
    double bottom = -0.52/2;
    float beamHeight = 0.30;
    float beamWidth = 0.40;
    float top = 0.2;
    float sphere_rad = 0.01;

    auto walls = DEMSim.AddExternalObject();
    walls->AddPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_container);
    walls->SetFamily(10);
    auto fixed = DEMSim.AddWavefrontMeshObject("../data/granularFlow/drum.obj", mat_type_container);
    fixed->Scale(0.20 * 1.0);
    fixed->SetFamily(10);
    fixed->Move(make_float3(-0.150, +0.10, -0.11), make_float4(0.7071,0.7071, 0, 0));

    auto fixed_2 = DEMSim.AddWavefrontMeshObject("../data/granularFlow/drum.obj", mat_type_container);
    fixed_2->Scale(0.20 * 1.0);
    fixed_2->SetFamily(10);
    fixed_2->Move(make_float3(0.150, +0.10, -0.11), make_float4(0.7071,0.7071, 0, 0));

    DEMSim.SetFamilyFixed(10);

    auto mobile = DEMSim.AddWavefrontMeshObject("../data/granularFlow/drum.obj", mat_type_container);
    mobile->Scale(0.20 * 1.0);
    mobile->SetFamily(10);
    mobile->Move(make_float3(0.0, +0.10, beamHeight+2.5*sphere_rad + 0.20/2.0), make_float4(0.7071,0.7071, 0, 0));
    mobile->SetFamily(20);
    DEMSim.SetFamilyFixed(20);
    DEMSim.SetFamilyPrescribedLinVel(21, "0", "0", to_string_with_precision(-0.01));
    // Define the terrain particle templates
    // Calculate its mass and MOI
    float terrain_density = 2.6e3;
    
    float sphere_vol = 4. / 3. * math_PI * sphere_rad * sphere_rad * sphere_rad;
    float mass = terrain_density * sphere_vol;
    // Then load it to system
    std::shared_ptr<DEMClumpTemplate> my_template = DEMSim.LoadSphereType(mass, sphere_rad, mat_type_particle);

    // Sampler to sample
    HCPSampler sampler(sphere_rad * 2.0);
    
    float3 fill_center = make_float3(0, beamWidth / 2.0 , beamHeight/2.0 +sphere_rad);
    float3 fill_size = make_float3(0.40, beamWidth / 2.0, beamHeight / 2.0);
    auto input_xyz = sampler.SampleBox(fill_center, fill_size);
    auto particles = DEMSim.AddClumps(my_template, input_xyz);
    particles->SetFamily(1);
    std::cout << "Total num of particles: " << particles->GetNumClumps() << std::endl;

    std::filesystem::path out_dir = std::filesystem::current_path();
    out_dir += "/DemoOutput_FractureBeam";
    remove_all(out_dir);
    create_directory(out_dir);

    // Some inspectors
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");

    DEMSim.SetFamilyExtraMargin(1, 1.40 * sphere_rad); // compared to 0

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0.0,  1* -9.81));
    DEMSim.Initialize();
    //DEMSim.DisableContactBetweenFamilies(20, 1);
    std::cout << "Initial number of contacts: " << DEMSim.GetNumContacts() << std::endl;

    float sim_end = 4.0;
    unsigned int fps = 100;
    float frame_time = 1.0 / fps;
    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));
    unsigned int frame_count = 0;
    unsigned int step_count = 0;

    bool status_1 = true;
    bool status_2 = true;

    DEMSim.SetFamilyContactWildcardValueAll(1, "initialLength", 0.0);
     DEMSim.SetFamilyContactWildcardValueAll(1, "damage", 0.0);
    DEMSim.SetFamilyContactWildcardValueAll(1, "unbroken", 2.0);
    // Simulation loop
    for (float t = 0; t < sim_end; t += frame_time) {
        char filename[200];
        char meshname[200];
        char cnt_filename[200];
        std::cout << "Outputting frame: " << frame_count << std::endl;
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), frame_count);
        sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), frame_count);
        DEMSim.WriteSphereFile(std::string(filename));
        DEMSim.WriteMeshFile(std::string(meshname));
        sprintf(cnt_filename, "%s/DEMdemo_contact_%04d.csv", out_dir.c_str(), frame_count);
        DEMSim.WriteContactFile(std::string(cnt_filename));
        frame_count++;
        DEMSim.ShowThreadCollaborationStats();
        std::cout << "Initial number of contacts: " << DEMSim.GetNumContacts() << std::endl;
        DEMSim.DoDynamics(frame_time);

        if (t > 0.01 && status_1) {
            status_1 = false;
            DEMSim.DoDynamicsThenSync(0);
            //DEMSim.DisableContactBetweenFamilies(10, 1);
            DEMSim.ChangeFamily(20, 21);

            std::cout << "Establishing inner forces: " << frame_count << std::endl;
        }

        if (t > 6.0 && status_2) {
            status_2 = false;
            DEMSim.DoDynamicsThenSync(0);
            // DEMSim.SetFamilyContactWildcardValueAll(1, "unbroken", -1.0);

            std::cout << "freeing the material: " << frame_count << std::endl;
        }
    }

    DEMSim.ShowTimingStats();
    std::cout << "Fracture demo exiting..." << std::endl;
    return 0;
}
