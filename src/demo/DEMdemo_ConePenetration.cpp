//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// This demo presents a cone penetrameter test with a soil sample made of clumped
// particles of various sizes. Before the test starts, when compress the terrain
// first, and note that the compressor used in this process has its position
// explicitly controlled step-by-step.
// =============================================================================

#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <random>

using namespace deme;

const double math_PI = 3.14159;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetContactOutputContent({"OWNER", "FORCE", "POINT"});

    // E, nu, CoR, mu, Crr...
    auto mat_type_cone = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.8}, {"mu", 0.7}, {"Crr", 0.00}});
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.8}, {"mu", 0.4}, {"Crr", 0.00}});
    // If you don't have this line, then values will take average between 2 materials, when they are in contact
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_cone, mat_type_terrain, 0.8);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_cone, mat_type_terrain, 0.7);

    float cone_speed = 0.03;
    float step_size = 5e-6;
    double world_size = 2;
    double soil_bin_diameter = 0.584;
    double cone_surf_area = 323e-6;
    double cone_diameter = std::sqrt(cone_surf_area / math_PI) * 2;
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
    double clump_vol = 5.5886717;
    float mass = terrain_density * clump_vol;
    float3 MOI = make_float3(2.928, 2.6029, 3.9908) * terrain_density;
    // Then load it to system
    std::shared_ptr<DEMClumpTemplate> my_template =
        DEMSim.LoadClumpType(mass, MOI, GetDEMEDataFile("clumps/3_clump.csv"), mat_type_terrain);
    my_template->SetVolume(clump_vol);
    // Decide the scalings of the templates we just created (so that they are... like particles, not rocks)
    double scale = 0.0044;
    my_template->Scale(scale);

    // Sampler to sample
    HCPSampler sampler(scale * 3.);
    float fill_height = 0.5;
    float3 fill_center = make_float3(0, 0, bottom + fill_height / 2);
    const float fill_radius = soil_bin_diameter / 2. - scale * 3.;
    auto input_xyz = sampler.SampleCylinderZ(fill_center, fill_radius, fill_height / 2 - scale * 2.);
    DEMSim.AddClumps(my_template, input_xyz);
    std::cout << "Total num of particles: " << input_xyz.size() << std::endl;

    // Load in the cone used for this penetration test
    auto cone_tip = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/cone.obj"), mat_type_cone);
    auto cone_body = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/cyl_r1_h2.obj"), mat_type_cone);
    std::cout << "Total num of triangles: " << cone_tip->GetNumTriangles() + cone_body->GetNumTriangles() << std::endl;

    // The initial cone mesh has base radius 1, and height 1. Let's stretch it a bit so it has a 60deg tip, instead of
    // 90deg.
    float tip_height = std::sqrt(3.);
    cone_tip->Scale(make_float3(1, 1, tip_height));
    // Then set mass properties
    float cone_mass = 7.8e3 * tip_height / 3 * math_PI;
    cone_tip->SetMass(cone_mass);
    // You can checkout https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    cone_tip->SetMOI(make_float3(cone_mass * (3. / 20. + 3. / 80. * tip_height * tip_height),
                                 cone_mass * (3. / 20. + 3. / 80. * tip_height * tip_height), 3 * cone_mass / 10));
    // This cone mesh has its tip at the origin. And, float4 quaternion pattern is (x, y, z, w).
    cone_tip->InformCentroidPrincipal(make_float3(0, 0, 3. / 4. * tip_height), make_float4(0, 0, 0, 1));
    // Note the scale method will scale mass and MOI automatically. But this only goes for the case you scale xyz all
    // together; otherwise, the MOI scaling will not be accurate and you should manually reset them.
    cone_tip->Scale(cone_diameter / 2);
    cone_tip->SetFamily(2);

    // The define the body that is connected to the tip
    float body_mass = 7.8e3 * math_PI;
    cone_body->SetMass(body_mass);
    cone_body->SetMOI(make_float3(body_mass * 7 / 12, body_mass * 7 / 12, body_mass / 2));
    // This cyl mesh (h = 2m, r = 1m) has its center at the origin. So the following call actually has no effect...
    cone_body->InformCentroidPrincipal(make_float3(0, 0, 0), make_float4(0, 0, 0, 1));
    cone_body->Scale(make_float3(cone_diameter / 2, cone_diameter / 2, 0.5));
    cone_body->SetFamily(2);

    // Track the cone_tip
    auto tip_tracker = DEMSim.Track(cone_tip);
    auto body_tracker = DEMSim.Track(cone_body);

    // Because the cone's motion is completely pre-determined, we can just prescribe family 1
    DEMSim.SetFamilyPrescribedLinVel(1, "0", "0", "-" + to_string_with_precision(cone_speed));
    // Cone is initially in family 2, sleeping...
    DEMSim.SetFamilyFixed(2);
    DEMSim.DisableContactBetweenFamilies(0, 2);

    // Now add a plane to compress the sample
    auto compressor = DEMSim.AddExternalObject();
    compressor->AddPlane(make_float3(0, 0, 0), make_float3(0, 0, -1), mat_type_terrain);
    compressor->SetFamily(10);
    DEMSim.SetFamilyFixed(10);
    auto compressor_tracker = DEMSim.Track(compressor);

    // Some inspectors
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    // auto total_volume_finder = DEMSim.CreateInspector("clump_volume", "return (X * X + Y * Y <= 0.25 * 0.25) && (Z <=
    // -0.3);");
    auto total_mass_finder = DEMSim.CreateInspector("clump_mass");

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // CD freq will be auto-adapted so it does not matter much here.
    DEMSim.SetCDUpdateFreq(20);
    // Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    // wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    // happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(10.);

    DEMSim.Initialize();

    std::filesystem::path out_dir = std::filesystem::current_path();
    out_dir /= "DemoOutput_ConePenetration";
    std::filesystem::create_directory(out_dir);

    // Settle
    DEMSim.DoDynamicsThenSync(0.8);

    // Compress until dense enough
    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    unsigned int fps = 20;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));
    double compressor_vel = 0.05;
    float terrain_max_z = max_z_finder->GetValue();
    double init_max_z = terrain_max_z;
    float bulk_density = -10000.;
    while (bulk_density < 1500.) {
        float matter_mass = total_mass_finder->GetValue();
        float total_volume = math_PI * (soil_bin_diameter * soil_bin_diameter / 4) * (terrain_max_z - bottom);
        bulk_density = matter_mass / total_volume;
        if (curr_step % out_steps == 0) {
            char filename[100], meshname[100];
            sprintf(filename, "DEMdemo_output_%04d.csv", currframe);
            sprintf(meshname, "DEMdemo_mesh_%04d.vtk", currframe);
            DEMSim.WriteSphereFile(out_dir / filename);
            // DEMSim.WriteMeshFile(out_dir / meshname);
            std::cout << "Compression bulk density: " << bulk_density << std::endl;
            currframe++;
        }

        terrain_max_z -= compressor_vel * step_size;
        compressor_tracker->SetPos(make_float3(0, 0, terrain_max_z));
        DEMSim.DoDynamics(step_size);
        curr_step++;
    }
    // Then gradually remove the compressor
    while (terrain_max_z < init_max_z) {
        if (curr_step % out_steps == 0) {
            char filename[100], meshname[100];
            sprintf(filename, "DEMdemo_output_%04d.csv", currframe);
            sprintf(meshname, "DEMdemo_mesh_%04d.vtk", currframe);
            DEMSim.WriteSphereFile(out_dir / filename);
            // DEMSim.WriteMeshFile(out_dir / meshname);
            float matter_mass = total_mass_finder->GetValue();
            float total_volume =
                math_PI * (soil_bin_diameter * soil_bin_diameter / 4) * (max_z_finder->GetValue() - bottom);
            bulk_density = matter_mass / total_volume;
            std::cout << "Compression bulk density: " << bulk_density << std::endl;
            currframe++;
        }

        terrain_max_z += compressor_vel * step_size;
        compressor_tracker->SetPos(make_float3(0, 0, terrain_max_z));
        DEMSim.DoDynamics(step_size);
        curr_step++;
    }

    // Remove compressor
    DEMSim.DoDynamicsThenSync(0.);
    DEMSim.DisableContactBetweenFamilies(0, 10);
    DEMSim.DoDynamicsThenSync(0.2);
    terrain_max_z = max_z_finder->GetValue();

    float sim_end = 7.0;
    fps = 2500;
    float frame_time = 1.0 / fps;
    std::cout << "Output at " << fps << " FPS" << std::endl;

    // Put the cone in place
    double starting_height = terrain_max_z + 0.03;
    // Its initial position should be right above the cone tip...
    body_tracker->SetPos(make_float3(0, 0, 0.5 + (cone_diameter / 2 / 4 * tip_height) + starting_height));
    // Note that position of objects is always the location of their centroid
    tip_tracker->SetPos(make_float3(0, 0, starting_height));
    // The tip location, used to measure penetration length
    double tip_z = -cone_diameter / 2 * 3 / 4 * tip_height + starting_height;

    // Enable cone
    DEMSim.ChangeFamily(2, 1);
    float matter_mass = total_mass_finder->GetValue();
    float total_volume = math_PI * (soil_bin_diameter * soil_bin_diameter / 4) * (terrain_max_z - bottom);
    bulk_density = matter_mass / total_volume;
    std::cout << "Bulk density: " << bulk_density << std::endl;

    double tip_z_when_first_hit;
    bool hit_terrain = false;
    unsigned int frame_count = 0;

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (float t = 0; t < sim_end; t += frame_time) {
        // float terrain_max_z = max_z_finder->GetValue();
        float3 forces = tip_tracker->ContactAcc();
        // Note cone_mass is not the true mass, b/c we scaled the the cone tip! So we use true mass by using
        // cone_tip->mass.
        forces *= cone_tip->mass;
        float pressure = std::abs(forces.z) / cone_surf_area;
        if (pressure > 1e-4 && !hit_terrain) {
            hit_terrain = true;
            tip_z_when_first_hit = tip_z;
        }
        float penetration = (hit_terrain) ? tip_z_when_first_hit - tip_z : 0;
        std::cout << "Time: " << t << std::endl;
        std::cout << "Z coord of tip: " << tip_z << std::endl;
        std::cout << "Penetration: " << penetration << std::endl;
        std::cout << "Force on cone: " << forces.x << ", " << forces.y << ", " << forces.z << std::endl;
        std::cout << "Pressure: " << pressure << std::endl;

        if (frame_count % 500 == 0) {
            char filename[100], meshname[100];
            std::cout << "Outputting frame: " << currframe << std::endl;
            sprintf(filename, "DEMdemo_output_%04d.csv", currframe);
            sprintf(meshname, "DEMdemo_mesh_%04d.vtk", currframe++);
            DEMSim.WriteSphereFile(out_dir / filename);
            DEMSim.WriteMeshFile(out_dir / meshname);
            DEMSim.ShowThreadCollaborationStats();
        }

        DEMSim.DoDynamicsThenSync(frame_time);
        tip_z -= cone_speed * frame_time;

        frame_count++;
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;

    std::cout << "ConePenetration demo exiting..." << std::endl;
    return 0;
}