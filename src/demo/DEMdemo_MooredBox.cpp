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
#include <iostream>
#include <fstream>

using namespace deme;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::VEL);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetContactOutputContent(XYZ | OWNER | FORCE | CNT_WILDCARD);

    DEMSim.SetErrorOutAvgContacts(200);
    // DEMSim.SetForceCalcThreadsPerBlock(256);
    //  E, nu, CoR, mu, Crr...
    auto mat_type_container =
        DEMSim.LoadMaterial({{"E", 10e9}, {"nu", 0.3}, {"CoR", 0.7}, {"mu", 0.10}, {"Crr", 0.10}});
    auto mat_type_particle = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.20}, {"CoR", 0.5}, {"mu", 0.10}, {"Crr", 0.05}});
    // If you don't have this line, then values will take average between 2 materials, when they are in contact
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_container, mat_type_particle, 0.2);
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_container, mat_type_particle, 0.5);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_container, mat_type_particle, 0.5);
    // We can specify the force model using a file.
    auto my_force_model = DEMSim.ReadContactForceModel("ForceModelMooring.cu");

    // Those following lines are needed. We must let the solver know that those var names are history variable etc.
    my_force_model->SetMustHaveMatProp({"E", "nu", "CoR", "mu", "Crr"});
    my_force_model->SetMustPairwiseMatProp({"CoR", "mu", "Crr"});
    // Pay attention to the extra per-contact wildcard `unbroken' here.
    my_force_model->SetPerContactWildcards(
        {"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z", "innerInteraction", "initialLength"});

    float world_size = 30;
    float container_diameter = 0.06;
    float terrain_density = 7.80e3;
    float sphere_rad = 0.04;

    float step_size = 1e-6;
    float fact_radius = 1.0;

    DEMSim.InstructBoxDomainDimension({-30, 30}, {-10, 10}, {-12, 10});
    // No need to add simulation `world' boundaries, b/c we'll add a cylinderical container manually
    DEMSim.InstructBoxDomainBoundingBC("all", mat_type_container);
    // DEMSim.SetInitBinSize(sphere_rad * 5);
    //  Now add a cylinderical boundary along with a bottom plane
    double bottom = -0;
    double top = 0.10;

    // Creating the two clump templates we need, which are just spheres
    std::vector<std::shared_ptr<DEMClumpTemplate>> templates_terrain;

    templates_terrain.push_back(DEMSim.LoadSphereType(sphere_rad * sphere_rad * sphere_rad * 4 / 3 * 1.0e3 * PI,
                                                      sphere_rad, mat_type_particle));

    auto data_xyz = DEMSim.ReadClumpXyzFromCsv("../data/clumps/line_1.csv");
    std::vector<float3> input_xyz;

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;

    std::cout << data_xyz.size() << " Data points are loaded from the external list." << std::endl;

    for (unsigned int i = 0; i < (data_xyz.size()); i++) {
        char t_name[20];
        sprintf(t_name, "%d", i);

        auto this_type_xyz = data_xyz[std::string(t_name)];
        input_xyz.insert(input_xyz.end(), this_type_xyz.begin(), this_type_xyz.end());

        input_pile_template_type.push_back(templates_terrain[0]);
    }

    auto allParticles = DEMSim.AddClumps(input_pile_template_type, input_xyz);
    allParticles->SetFamily(1);

    auto top_plane = DEMSim.AddWavefrontMeshObject("../data/granularFlow/cylinder.obj", mat_type_container);
    top_plane->SetInitPos(make_float3(0, 0, 0.25));
    top_plane->SetMass(1.);
    top_plane->Scale(make_float3(1, 1, 0.01));
    top_plane->SetFamily(10);
    DEMSim.SetFamilyFixed(10);

    auto bottom_plane = DEMSim.AddWavefrontMeshObject("../data/granularFlow/cylinder.obj", mat_type_container);
    bottom_plane->SetInitPos(make_float3(0, 0, -5.06));
    bottom_plane->SetMass(1.);
    bottom_plane->Scale(make_float3(8, 8, 0.01));
    bottom_plane->SetFamily(20);
    DEMSim.SetFamilyFixed(20);

    std::cout << "Total num of particles: " << allParticles->GetNumClumps() << std::endl;

    std::filesystem::path out_dir = std::filesystem::current_path();
    std::string nameOutFolder = "R" + std::to_string(sphere_rad) + "_Int" + std::to_string(fact_radius) + "";
    out_dir += "/DemoOutput_MooringLine";
    remove_all(out_dir);
    create_directory(out_dir);

    // Some inspectors

    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    auto min_z_finder = DEMSim.CreateInspector("clump_min_z");

    DEMSim.SetFamilyExtraMargin(1, fact_radius * sphere_rad);

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0.00, 1 * -9.81));
    DEMSim.Initialize();
    // DEMSim.DisableContactBetweenFamilies(20, 1);
    std::cout << "Initial number of contacts: " << DEMSim.GetNumContacts() << std::endl;

    float sim_end = 50;

    unsigned int fps = 100;
    unsigned int datafps = 25;
    unsigned int modfpsGeo = datafps / fps;
    float frame_time = 1.0 / datafps;
    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int out_steps = (unsigned int)(1.0 / (datafps * step_size));
    unsigned int frame_count = 0;
    unsigned int step_count = 0;

    bool status_1 = true;
    bool status_2 = true;

    // DEMSim.DisableContactBetweenFamilies(10, 1);

    double L0 = 0.0;
    double stress = 0.0;
    std::string nameOutFile = "data_R" + std::to_string(sphere_rad) + "_Int" + std::to_string(fact_radius) + ".csv";
    std::ofstream csvFile(nameOutFile);

    DEMSim.SetFamilyContactWildcardValueAll(1, "initialLength", 0.0);
    // DEMSim.SetFamilyContactWildcardValueAll(1, "damage", 0.0);
    DEMSim.SetFamilyContactWildcardValueAll(1, "innerInteraction", 0.0);

    // Simulation loop
    for (float t = 0; t < sim_end; t += frame_time) {
        // DEMSim.ShowThreadCollaborationStats();

        std::cout << "Contacts now: " << DEMSim.GetNumContacts() << std::endl;

        if (t >= 0.0 && status_1) {
            status_1 = false;
            DEMSim.DoDynamicsThenSync(0);
            DEMSim.SetFamilyContactWildcardValueAll(1, "innerInteraction", 2.0);
            DEMSim.SetFamilyContactWildcardValue(1, 10, "innerInteraction", 2.0);
            DEMSim.SetFamilyContactWildcardValue(1, 20, "innerInteraction", 2.0);
            std::cout << "Establishing inner forces: " << frame_count << std::endl;
        }

        if (frame_count % 1 == 0) {
            char filename[200];
            char meshname[200];
            char cnt_filename[200];

            std::cout << "Outputting frame: " << frame_count << std::endl;
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), frame_count);
            sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), frame_count);
            sprintf(cnt_filename, "%s/DEMdemo_contact_%04d.csv", out_dir.c_str(), frame_count);

            DEMSim.WriteSphereFile(std::string(filename));
            DEMSim.WriteMeshFile(std::string(meshname));
            DEMSim.WriteContactFile(std::string(cnt_filename));
        }

        DEMSim.DoDynamics(frame_time);
        frame_count++;
    }
    csvFile.close();
    DEMSim.ShowTimingStats();
    std::cout << "Fracture demo exiting..." << std::endl;
    return 0;
}
