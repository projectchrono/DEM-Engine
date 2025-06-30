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
    DEMSim.SetOutputContent(OUTPUT_CONTENT::VEL | FAMILY);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetContactOutputContent(CNT_POINT | OWNER | FORCE | CNT_WILDCARD);

    DEMSim.SetErrorOutAvgContacts(200);
    // DEMSim.SetForceCalcThreadsPerBlock(256);
    //  E, nu, CoR, mu, Crr...
    auto mat_type_container =
        DEMSim.LoadMaterial({{"E", 10e9}, {"nu", 0.3}, {"CoR", 0.7}, {"mu", 0.10}, {"Crr", 0.10}});
    auto mat_type_particle = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.20}, {"CoR", 0.5}, {"mu", 0.10}, {"Crr", 0.05}});
    // If you don't have this line, then values will take average between 2 materials, when they are in contact
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_container, mat_type_particle, 0.2);
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_container, mat_type_particle, 0.5);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_container, mat_type_particle, 0.5);
    // We can specify the force model using a file.
    auto my_force_model = DEMSim.ReadContactForceModel("ForceModelMooringPosition.cu");

    // Those following lines are needed. We must let the solver know that those var names are history variable etc.
    my_force_model->SetMustHaveMatProp({"E", "nu", "CoR", "mu", "Crr"});
    my_force_model->SetMustPairwiseMatProp({"CoR", "mu", "Crr"});
    // Pay attention to the extra per-contact wildcard `unbroken' here.
    my_force_model->SetPerContactWildcards(
        {"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z", "innerInteraction", "initialLength", "restLength"});

    float world_size = 100;
    float container_diameter = 0.06;
    float terrain_density = 7.85e3;
    float sphere_rad = 0.045;

    float step_size = 2e-6;
    float fact_radius = 1.0;

    DEMSim.InstructBoxDomainDimension({-30, 30}, {-10, 10}, {-12, 10});
    // No need to add simulation `world' boundaries, b/c we'll add a cylinderical container manually
    DEMSim.InstructBoxDomainBoundingBC("all", mat_type_container);
    // DEMSim.SetInitBinSize(sphere_rad * 5);
    //  Now add a cylinderical boundary along with a bottom plane
    double bottom = -0;
    double top = 0.10;

    // Creating the two clump templates we need, which are just spheres
    std::shared_ptr<DEMClumpTemplate> templates_terrain;

    templates_terrain =
        DEMSim.LoadSphereType(sphere_rad * sphere_rad * sphere_rad * 4 / 3 * 1.0e3 * PI, sphere_rad, mat_type_particle);

    auto data_xyz = DEMSim.ReadClumpXyzFromCsv("../data/my/line_1.csv");
    std::vector<float3> input_xyz;

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;
    std::cout << data_xyz.size() << " Data points are loaded from the external list." << std::endl;

    for (unsigned int i = 0; i < (data_xyz.size()); i++) {
        char t_name[20];
        sprintf(t_name, "%d", i);

        auto this_type_xyz = data_xyz[std::string(t_name)];
        input_xyz.insert(input_xyz.end(), this_type_xyz.begin(), this_type_xyz.end());

        input_pile_template_type.push_back(templates_terrain);
    }
    auto allParticles = DEMSim.AddClumps(input_pile_template_type, input_xyz);
    allParticles->SetFamily(1);

    auto data_xyz_anchor = DEMSim.ReadClumpXyzFromCsv("../data/my/line_anchor.csv");
    std::vector<float3> input_xyz_2;

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type_2;
    std::cout << data_xyz_anchor.size() << " Data points are loaded from the external list." << std::endl;

    for (unsigned int i = 0; i < (data_xyz_anchor.size()); i++) {
        char t_name[20];
        sprintf(t_name, "%d", i);

        auto this_type_xyz = data_xyz_anchor[std::string(t_name)];
        input_xyz_2.insert(input_xyz_2.end(), this_type_xyz.begin(), this_type_xyz.end());

        input_pile_template_type_2.push_back(templates_terrain);
    }

    auto allParticles_2 = DEMSim.AddClumps(input_pile_template_type_2, input_xyz_2);
    allParticles_2->SetFamily(2);
    DEMSim.SetFamilyFixed(2);

    auto data_xyz_fairlead = DEMSim.ReadClumpXyzFromCsv("../data/my/line_fairlead.csv");
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_cylinder;
    // Then load it to system
    {  // initialize cylinder clump
        std::vector<float3> relPos;
        std::vector<float> radii;
        std::vector<std::shared_ptr<DEMMaterial>> mat;
        for (int i = 0; i < data_xyz_fairlead.size(); i++) {
            char t_name[20];
            sprintf(t_name, "%d", i);

            auto tmp = data_xyz_fairlead[std::string(t_name)];
            relPos.insert(relPos.end(), tmp.begin(), tmp.end());

            mat.push_back(mat_type_particle);
            radii.push_back(sphere_rad);
        }

        float mass = 100;
        float Ixx = 1.f / 2.f * mass;
        float Iyy = Ixx;
        float3 MOI = make_float3(Ixx, Iyy, Iyy);

        auto clump_ptr = DEMSim.LoadClumpType(mass, MOI, radii, relPos, mat_type_particle);
        // clump_ptr->AssignName("fsfs");
        clump_cylinder.push_back(clump_ptr);
    }
    std::cout << "Total num of clumps: " << clump_cylinder.size() << std::endl;
    std::vector<float3> input_pile_xyz;
    input_pile_xyz.insert(input_pile_xyz.end(), make_float3(0.0, 0, 0.0));

    auto the_pile = DEMSim.AddClumps(clump_cylinder, input_pile_xyz);
    the_pile->SetFamily(3);
    // DEMSim.SetFamilyFixed(3);
    auto anchoring_track = DEMSim.Track(the_pile);

    DEMSim.AddFamilyPrescribedAcc(3, "0", "0", to_string_with_precision(9.81 + 5.00));

    std::cout << "Total num of particles: " << the_pile->GetNumClumps() << std::endl;
    std::cout << "Total num of spheres: " << the_pile->GetNumSpheres() << std::endl;

    auto top_plane = DEMSim.AddWavefrontMeshObject("../data/my/cylinder.obj", mat_type_container);
    top_plane->SetInitPos(make_float3(0, 0, 0.26));
    top_plane->SetMass(100.);
    top_plane->Scale(make_float3(1, 1, 0.01));
    top_plane->SetFamily(10);
    DEMSim.SetFamilyFixed(10);
    auto phantom_track = DEMSim.Track(top_plane);

    auto bottom_plane = DEMSim.AddWavefrontMeshObject("../data/my/cylinder.obj", mat_type_container);
    bottom_plane->SetInitPos(make_float3(0, 0, -5.06));
    bottom_plane->SetMass(1.);
    bottom_plane->Scale(make_float3(8, 8, 0.01));
    bottom_plane->SetFamily(20);
    DEMSim.SetFamilyFixed(20);

    std::cout << "Total num of particles: " << allParticles->GetNumClumps() << std::endl;

    std::filesystem::path out_dir = std::filesystem::current_path();
    std::string nameOutFolder = "R" + std::to_string(sphere_rad) + "_Int" + std::to_string(fact_radius) + "";
    out_dir /= "DemoOutput_MooringLine";
    remove_all(out_dir);
    create_directory(out_dir);

    // Some inspectors

    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    auto min_z_finder = DEMSim.CreateInspector("clump_min_z");

    DEMSim.SetFamilyExtraMargin(1, fact_radius * sphere_rad);
    DEMSim.SetFamilyExtraMargin(2, fact_radius * sphere_rad);

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0.00, 1 * -9.81));
    DEMSim.Initialize();
    // DEMSim.DisableContactBetweenFamilies(20, 1);
    std::cout << "Initial number of contacts: " << DEMSim.GetNumContacts() << std::endl;

    float sim_end = 50;

    unsigned int fps = 20;
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

    DEMSim.SetFamilyContactWildcardValueBoth(1, "initialLength", 0.0);
    DEMSim.SetFamilyContactWildcardValueBoth(1, "innerInteraction", 0.0);
    DEMSim.SetFamilyContactWildcardValueBoth(2, "initialLength", 0.0);
    DEMSim.SetFamilyContactWildcardValueBoth(2, "innerInteraction", 0.0);
    DEMSim.SetFamilyContactWildcardValueBoth(3, "initialLength", 0.0);
    DEMSim.SetFamilyContactWildcardValueBoth(3, "innerInteraction", 0.0);
    std::cout << "Contacts now: " << DEMSim.GetNumContacts() << std::endl;
    DEMSim.DoDynamicsThenSync(0);
    DEMSim.SetFamilyContactWildcardValueBoth(1, "innerInteraction", 2.0);
    DEMSim.SetFamilyContactWildcardValue(1, 2, "innerInteraction", 2.0);
    DEMSim.SetFamilyContactWildcardValue(1, 3, "innerInteraction", 2.0);

    std::cout << "Contacts now: " << DEMSim.GetNumContacts() << std::endl;
    DEMSim.MarkFamilyPersistentContactEither(1);
    DEMSim.MarkFamilyPersistentContact(1, 2);
    DEMSim.MarkFamilyPersistentContact(1, 3);
    DEMSim.DoDynamicsThenSync(0);

    DEMSim.DisableContactBetweenFamilies(1, 10);
    DEMSim.DisableContactBetweenFamilies(3, 10);
    std::cout << "Establishing inner forces: " << frame_count << std::endl;

    float3 position = anchoring_track->Pos();
    float3 phantom_position = phantom_track->Pos();
    // Simulation loop
    for (float t = 0; t < sim_end; t += frame_time) {
        // DEMSim.ShowThreadCollaborationStats();

        std::cout << "Contacts now: " << DEMSim.GetNumContacts() << std::endl;

        if (t >= 0.0 && status_1) {
            status_1 = false;
        }

        // anchoring_track->SetPos(position + make_float3(0.1 * 0, 0, 0.1 * 0));
        phantom_track->SetPos(phantom_position + make_float3(0.1 * 0, 0, 0.1 * 0));
        auto temp = anchoring_track->ContactAcc();
        DEMSim.DoDynamicsThenSync(0);

        if (frame_count % 1 == 0) {
            char filename[200];
            char meshname[200];
            char cnt_filename[200];

            std::cout << "Outputting frame: " << frame_count << std::endl;
            std::cout << "Force: " << temp.z * 100.0 << std::endl;
            sprintf(filename, "DEMdemo_output_%04d.csv", frame_count);
            sprintf(meshname, "DEMdemo_mesh_%04d.vtk", frame_count);
            sprintf(cnt_filename, "DEMdemo_contact_%04d.csv", frame_count);

            DEMSim.WriteSphereFile(out_dir / filename);
            DEMSim.WriteMeshFile(out_dir / meshname);
            DEMSim.WriteContactFile(out_dir / cnt_filename);
        }

        DEMSim.DoDynamics(frame_time);
        frame_count++;
    }
    csvFile.close();
    DEMSim.ShowTimingStats();
    std::cout << "Fracture demo exiting..." << std::endl;
    return 0;
}