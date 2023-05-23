//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// This demo lets a rod with some electric charges stick into a pile of granular
// material that is also charged. The electrostatic force shows its effect. The
// electric charges are even moving during the simulation. This is done through
// `Geometry Wildcard', where the amount of charges is associated with each
// sphere component (of clumps) and triangles (of meshes). Then a custom force
// model is used to derive the electrostatic force in addition to contact forces.
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
    DEMSim.SetContactOutputContent(OWNER | FORCE | POINT);

    // E, nu, CoR, mu, Crr...
    auto mat_type_rod = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.7}, {"Crr", 0.00}});
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}, {"Crr", 0.00}});
    // If you don't have this line, then values will take average between 2 materials, when they are in contact
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_rod, mat_type_terrain, 0.8);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_rod, mat_type_terrain, 0.7);

    float cone_speed = 0.1;
    float step_size = 5e-6;
    double world_size = 2;
    double soil_bin_diameter = 0.584;
    double rod_length = 0.5;
    double rod_surf_area = 323e-6;
    double rod_diameter = std::sqrt(rod_surf_area / math_PI) * 2;
    DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size);
    // No need to add simulation `world' boundaries, b/c we'll add a cylinderical container manually
    DEMSim.InstructBoxDomainBoundingBC("none", mat_type_terrain);
    // Now add a cylinderical boundary along with a bottom plane
    double bottom = -0.5;
    auto walls = DEMSim.AddExternalObject();
    walls->AddCylinder(make_float3(0), make_float3(0, 0, 1), soil_bin_diameter / 2., mat_type_terrain, 0);
    walls->AddPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);

    // Define the terrain particle templates
    // Calculate its mass and MOI
    float terrain_density = 2.6e3;
    double clump_vol = 4. / 3. * math_PI;
    float mass = terrain_density * clump_vol;
    float3 MOI = make_float3(2. / 5.) * mass;
    // Then load it to system
    std::shared_ptr<DEMClumpTemplate> my_template =
        DEMSim.LoadClumpType(mass, MOI, GetDEMEDataFile("clumps/spiky_sphere.csv"), mat_type_terrain);
    my_template->SetVolume(clump_vol);
    // Decide the scalings of the templates we just created (so that they are... like particles, not rocks)
    double scale = 0.01;
    my_template->Scale(scale);

    // Sampler to sample
    GridSampler sampler(scale * 2.4);
    float fill_height = 1.;
    float3 fill_center = make_float3(0, 0, bottom + fill_height / 2);
    const float fill_radius = soil_bin_diameter / 2. - scale * 3.;
    auto input_xyz = sampler.SampleCylinderZ(fill_center, fill_radius, fill_height / 2 - scale * 2.);
    DEMSim.AddClumps(my_template, input_xyz);
    std::cout << "Total num of particles: " << input_xyz.size() << std::endl;

    // Load in the cone used for this penetration test
    auto rod_body = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/cyl_r1_h2.obj"), mat_type_rod);
    std::cout << "Total num of triangles: " << rod_body->GetNumTriangles() << std::endl;

    // The define the properties of the rod
    float body_mass = 7.8e3 * math_PI;
    rod_body->SetMass(body_mass);
    rod_body->SetMOI(make_float3(body_mass * 7 / 12, body_mass * 7 / 12, body_mass / 2));
    // This cyl mesh (h = 2m, r = 1m) has its center at the origin. So the following call actually has no effect...
    rod_body->InformCentroidPrincipal(make_float3(0, 0, 0), make_float4(0, 0, 0, 1));
    rod_body->Scale(make_float3(rod_diameter / 2., rod_diameter / 2., rod_length / 2.));
    rod_body->SetFamily(1);
    // Just fix it: We will manually impose its motion later.
    DEMSim.SetFamilyFixed(1);

    // Track the rod
    auto rod_tracker = DEMSim.Track(rod_body);

    // Some inspectors
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.Initialize();

    std::filesystem::path out_dir = std::filesystem::current_path();
    out_dir += "/DemoOutput_Electrostatic";
    std::filesystem::create_directory(out_dir);

    // Settle
    DEMSim.DisableContactBetweenFamilies(0, 1);
    DEMSim.DoDynamicsThenSync(1.);

    float sim_end = 9.0;
    unsigned int fps = 20;
    float frame_time = 1.0 / fps;
    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    // Put the cone in place
    float terrain_max_z = max_z_finder->GetValue();
    double current_height = terrain_max_z + 0.03;
    // Its initial position should be right above the granular material (but accounting for the fact that the coordinate
    // system center of the rod is in its middle)
    rod_tracker->SetPos(make_float3(0, 0, rod_length / 2. + current_height));

    unsigned int frame_count = 0;
    unsigned int step_count = 0;
    DEMSim.EnableContactBetweenFamilies(0, 1);
    for (float t = 0; t < sim_end; t += step_size, step_count++) {
        if (step_count % out_steps == 0) {
            char filename[200], meshname[200];
            std::cout << "Outputting frame: " << frame_count << std::endl;
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), frame_count);
            sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), frame_count++);
            DEMSim.WriteSphereFile(std::string(filename));
            DEMSim.WriteMeshFile(std::string(meshname));
            DEMSim.ShowThreadCollaborationStats();
        }

        // Means advance simulation by one time step
        DEMSim.DoStepDynamics();
        // The rod first goes down into the material, then goes up, then stay in place to let us watch how the particles
        // affected by the electrostatic force move.
        if (t < 1. / 3. * sim_end) {
            current_height -= cone_speed * step_size;
        } else if (t < 2. / 3. * sim_end) {
            current_height += cone_speed * step_size;
        }  // else the rod does not move
        rod_tracker->SetPos(make_float3(0, 0, rod_length / 2. + current_height));
    }

    std::cout << "Electrostatic demo exiting..." << std::endl;
    return 0;
}
