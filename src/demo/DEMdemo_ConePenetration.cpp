//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

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
const float kg_g_conv = 1.;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetContactOutputContent(OWNER | FORCE | POINT);

    // E, nu, CoR, mu, Crr...
    auto mat_type_cone = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.8}});
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.8}});
    // If you don't have this line, then values will take average between 2 materials, when they are in contact
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_cone, mat_type_terrain, 0.8);
    // DEMSim.SetMaterialPropertyPair("mu", mat_type_cone, mat_type_terrain, 0.7);

    std::filesystem::path out_dir = std::filesystem::current_path();
    // out_dir += "/Cone_Penetration_HighDensity_CoR0.8";
    // out_dir += "/Cone_Penetration_LowDensity_CoR0.8";
    out_dir += "/Cone_Penetration_1650Density";
    std::filesystem::create_directory(out_dir);
    float settle_mu = 0.5;
    float sim_mu = 0.5;
    float cone_mu = 0.5;
    float target_density = 1650.;

    // A custom force model can be read in through a file and used by the simulation. Magic, right?
    auto my_force_model = DEMSim.ReadContactForceModel("SampleCustomForceModel.cu");
    // This custom force model still uses contact history arrays, so let's define it
    my_force_model->SetPerContactWildcards({"delta_tan_x", "delta_tan_y", "delta_tan_z"});
    // Owner wildcards. In this demo, we define a changable friction coefficient mu_custom.
    my_force_model->SetPerOwnerWildcards({"mu_custom"});

    float shake_amp = 0.1;
    float shake_speed = 2;  // Num of periods per second
    float cone_speed = 0.03;
    float step_size = 2e-6;
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
    walls->SetFamily(5);  // Fixed wall bc
    // Family 6 shakes, family 5 is fixed
    DEMSim.SetFamilyFixed(5);
    DEMSim.SetFamilyPrescribedLinVel(6, "0", "0",
                                     to_string_with_precision(shake_amp) + " * sin(" +
                                         to_string_with_precision(shake_speed) + " * 2 * deme::PI * t)");

    // Then the ground particle template
    DEMClumpTemplate shape_template1, shape_template2;
    shape_template1.ReadComponentFromFile((GET_DATA_PATH() / "clumps/triangular_flat.csv").string());
    shape_template2.ReadComponentFromFile((GET_DATA_PATH() / "clumps/triangular_flat_6comp.csv").string());
    std::vector<DEMClumpTemplate> shape_template = {shape_template2, shape_template2, shape_template1,
                                                    shape_template1, shape_template1, shape_template1,
                                                    shape_template1};
    // Calculate its mass and MOI
    float mass1 = 2.6e3 * 4.2520508;  
    float3 MOI1 = make_float3(1.6850426, 1.6375114, 2.1187753) * 2.6e3;
    float mass2 = 2.6e3 * 2.1670011;  
    float3 MOI2 = make_float3(0.57402126, 0.60616378, 0.92890173) * 2.6e3;
    std::vector<float> mass = {mass2, mass2, mass1, mass1, mass1, mass1, mass1};
    std::vector<float3> MOI = {MOI2, MOI2, MOI1, MOI1, MOI1, MOI1, MOI1};
    // Scale the template we just created
    std::vector<std::shared_ptr<DEMClumpTemplate>> ground_particle_templates;
    std::vector<double> volume = {2.1670011, 2.1670011, 4.2520508, 4.2520508, 4.2520508, 4.2520508, 4.2520508};
    std::vector<double> scales = {0.0014, 0.00075833, 0.00044, 0.0003, 0.0002, 0.00018333, 0.00017};
    std::for_each(scales.begin(), scales.end(), [](double& r) { r *= 10.; });
    unsigned int t_num = 0;
    for (double scaling : scales) {
        auto this_template = shape_template[t_num];
        this_template.mass = (double)mass[t_num] * scaling * scaling * scaling;
        this_template.MOI.x = (double)MOI[t_num].x * (double)(scaling * scaling * scaling * scaling * scaling);
        this_template.MOI.y = (double)MOI[t_num].y * (double)(scaling * scaling * scaling * scaling * scaling);
        this_template.MOI.z = (double)MOI[t_num].z * (double)(scaling * scaling * scaling * scaling * scaling);
        std::cout << "Mass: " << this_template.mass << std::endl;
        std::cout << "MOIX: " << this_template.MOI.x << std::endl;
        std::cout << "MOIY: " << this_template.MOI.y << std::endl;
        std::cout << "MOIZ: " << this_template.MOI.z << std::endl;
        std::cout << "=====================" << std::endl;
        std::for_each(this_template.radii.begin(), this_template.radii.end(),
                        [scaling](float& r) { r *= scaling; });
        std::for_each(this_template.relPos.begin(), this_template.relPos.end(),
                        [scaling](float3& r) { r *= scaling; });
        this_template.materials = std::vector<std::shared_ptr<DEMMaterial>>(this_template.nComp, mat_type_terrain);

        // Give these templates names, 0000, 0001 etc.
        char t_name[20];
        sprintf(t_name, "%04d", t_num);
        this_template.AssignName(std::string(t_name));
        ground_particle_templates.push_back(DEMSim.LoadClumpType(this_template));
        t_num++;
    }

    // Now we load clump locations from a checkpointed file
    {
        std::cout << "Making terrain..." << std::endl;
        auto clump_xyz = DEMSim.ReadClumpXyzFromCsv("./GRC_3e6.csv");
        auto clump_quaternion = DEMSim.ReadClumpQuatFromCsv("./GRC_3e6.csv");
        std::vector<float3> in_xyz;
        std::vector<float4> in_quat;
        std::vector<std::shared_ptr<DEMClumpTemplate>> in_types;
        unsigned int t_num = 0;
        for (int i = 0; i < scales.size(); i++) {
            char t_name[20];
            sprintf(t_name, "%04d", t_num);

            auto this_type_xyz = clump_xyz[std::string(t_name)];
            auto this_type_quat = clump_quaternion[std::string(t_name)];

            size_t n_clump_this_type = this_type_xyz.size();
            std::cout << "Loading clump " << std::string(t_name) << " which has particle num: " << n_clump_this_type
                      << std::endl;
            // Prepare clump type identification vector for loading into the system (don't forget type 0 in
            // ground_particle_templates is the template for rover wheel)
            std::vector<std::shared_ptr<DEMClumpTemplate>> this_type(n_clump_this_type,
                                                                     ground_particle_templates.at(t_num));

            // Add them to the big long vector
            in_xyz.insert(in_xyz.end(), this_type_xyz.begin(), this_type_xyz.end());
            in_quat.insert(in_quat.end(), this_type_quat.begin(), this_type_quat.end());
            in_types.insert(in_types.end(), this_type.begin(), this_type.end());
            std::cout << "Added clump type " << t_num << std::endl;
            // Our template names are 0000, 0001 etc.
            t_num++;
        }
        // Now, we don't need all particles loaded... we just need a cylinderical portion out of it, to fill the soil
        // bin Remove the particles that are outside a cylinderical region
        std::vector<notStupidBool_t> elem_to_remove(in_xyz.size(), 0);
        for (size_t i = 0; i < in_xyz.size(); i++) {
            if (std::pow(in_xyz.at(i).x, 2) + std::pow(in_xyz.at(i).y, 2) >= std::pow(soil_bin_diameter / 2. - 0.02, 2))
                elem_to_remove.at(i) = 1;
        }
        in_xyz.erase(std::remove_if(
                         in_xyz.begin(), in_xyz.end(),
                         [&elem_to_remove, &in_xyz](const float3& i) { return elem_to_remove.at(&i - in_xyz.data()); }),
                     in_xyz.end());
        in_quat.erase(std::remove_if(in_quat.begin(), in_quat.end(),
                                     [&elem_to_remove, &in_quat](const float4& i) {
                                         return elem_to_remove.at(&i - in_quat.data());
                                     }),
                      in_quat.end());
        in_types.erase(std::remove_if(in_types.begin(), in_types.end(),
                                      [&elem_to_remove, &in_types](const auto& i) {
                                          return elem_to_remove.at(&i - in_types.data());
                                      }),
                       in_types.end());
        DEMClumpBatch base_batch(in_xyz.size());
        base_batch.SetTypes(in_types);
        base_batch.SetPos(in_xyz);
        base_batch.SetOriQ(in_quat);

        DEMSim.AddClumps(base_batch);

        // This batch is about 10cm thick... let's add another 2 batches, so we have something like 30cm
        float shift_dist = 0.15;
        for (int i = 0; i < 2; i++) {
            std::for_each(in_xyz.begin(), in_xyz.end(), [shift_dist](float3& xyz) { xyz.z += shift_dist; });
            DEMClumpBatch another_batch = base_batch;
            another_batch.SetPos(in_xyz);
            DEMSim.AddClumps(another_batch);
        }
    }

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

    // In fact, because the cone's motion is completely pre-determined, we can just prescribe family 1
    DEMSim.SetFamilyPrescribedLinVel(1, "0", "0", "-" + to_string_with_precision(cone_speed));
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
    DEMSim.SetCDUpdateFreq(30);
    DEMSim.SetMaxVelocity(10.);
    DEMSim.SetInitBinSize(2 * scales.at(2));
    DEMSim.Initialize();

    // Settle
    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    {
        float tot_mass = total_mass_finder->GetValue();
        std::cout << "Total granular mass is " << tot_mass << std::endl;
    }
    DEMSim.SetOwnerWildcardValue("mu_custom", settle_mu);
    {
        char filename[200], meshname[200];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
        sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), currframe++);
        DEMSim.WriteSphereFile(std::string(filename));
        DEMSim.WriteMeshFile(std::string(meshname));
        DEMSim.DoDynamicsThenSync(1.2);
    }

    unsigned int fps = 60;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));
    double compressor_vel = 0.2;
    float terrain_max_z = max_z_finder->GetValue();
    std::cout << "Max Z after settling: " << terrain_max_z << std::endl;
    double init_max_z = terrain_max_z;
    // Shake--compress several times
    for (int shake_times = 0; shake_times < 1; shake_times++) {
        // Shake
        std::cout << "Shake round " << shake_times << "..." << std::endl;
        DEMSim.ChangeFamily(5, 6);
        DEMSim.DisableContactBetweenFamilies(0, 10);
        DEMSim.DoDynamicsThenSync(1.);
        DEMSim.ChangeFamily(6, 5);
        DEMSim.DoDynamicsThenSync(0.5);
        // Compress until dense
        DEMSim.EnableContactBetweenFamilies(0, 10);
        terrain_max_z = max_z_finder->GetValue();
        std::cout << "Max Z after settling: " << terrain_max_z << std::endl;
        init_max_z = terrain_max_z;
        float bulk_density;
        {
            float matter_mass = total_mass_finder->GetValue();
            float total_volume =
                math_PI * (soil_bin_diameter * soil_bin_diameter / 4) * (max_z_finder->GetValue() - bottom);
            bulk_density = matter_mass / total_volume;
            std::cout << "Compression bulk density: " << bulk_density << std::endl;
        }

        while (bulk_density < target_density) {
            if (curr_step % out_steps == 0) {
                char filename[200], meshname[200];
                sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
                sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), currframe++);
                DEMSim.WriteSphereFile(std::string(filename));
                DEMSim.WriteMeshFile(std::string(meshname));
                float matter_mass = total_mass_finder->GetValue();
                float total_volume =
                    math_PI * (soil_bin_diameter * soil_bin_diameter / 4) * (max_z_finder->GetValue() - bottom);
                bulk_density = matter_mass / total_volume;
                std::cout << "Compression bulk density: " << bulk_density << std::endl;
            }

            terrain_max_z -= compressor_vel * step_size;
            compressor_tracker->SetPos(make_float3(0, 0, terrain_max_z));
            DEMSim.DoDynamics(step_size);
            curr_step++;
        }
        while (terrain_max_z < init_max_z && terrain_max_z < -0.15) {
            if (curr_step % out_steps == 0) {
                char filename[200], meshname[200];
                sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
                sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), currframe++);
                DEMSim.WriteSphereFile(std::string(filename));
                DEMSim.WriteMeshFile(std::string(meshname));
                float matter_mass = total_mass_finder->GetValue();
                float total_volume =
                    math_PI * (soil_bin_diameter * soil_bin_diameter / 4) * (max_z_finder->GetValue() - bottom);
                bulk_density = matter_mass / total_volume;
                std::cout << "Compression bulk density: " << bulk_density << std::endl;
            }

            terrain_max_z += compressor_vel * step_size;
            compressor_tracker->SetPos(make_float3(0, 0, terrain_max_z));
            DEMSim.DoDynamics(step_size);
            curr_step++;
        }
    }

    // Remove compressor
    DEMSim.DoDynamicsThenSync(0);
    DEMSim.DisableContactBetweenFamilies(0, 10);
    DEMSim.SetFamilyOwnerWildcardValue(0, "mu_custom", sim_mu);
    DEMSim.SetFamilyOwnerWildcardValue(2, "mu_custom", cone_mu);  // For cone
    DEMSim.DoDynamicsThenSync(0.5);
    terrain_max_z = max_z_finder->GetValue();

    double starting_height = terrain_max_z + 0.02;
    // Its initial position should be right above the cone tip...
    body_tracker->SetPos(make_float3(0, 0, 0.5 + (cone_diameter / 2 / 4 * tip_height) + starting_height));
    // Note that position of objects is always the location of their centroid
    tip_tracker->SetPos(make_float3(0, 0, starting_height));
    // The tip location, used to measure penetration length
    double tip_z = -cone_diameter / 2 * 3 / 4 * tip_height + starting_height;

    float sim_end = 8.;
    fps = 2500;
    float frame_time = 1.0 / fps;
    // Re-enable cone
    DEMSim.ChangeFamily(2, 1);

    int step_size_marker = 0;
    double tip_z_when_first_hit;
    bool hit_terrain = false;
    unsigned int frame_count = 0;
    for (float t = 0; t < sim_end; t += frame_time) {
        float matter_mass = total_mass_finder->GetValue();
        float total_volume = math_PI * (soil_bin_diameter * soil_bin_diameter / 4) * (terrain_max_z - bottom);
        float bulk_density = matter_mass / total_volume;
        // float terrain_max_z = max_z_finder->GetValue();
        float3 forces = tip_tracker->ContactAcc();
        // Note cone_mass is not the true mass, b/c we scaled the the cone tip!
        forces *= cone_tip->mass;
        float pressure = std::abs(forces.z) / cone_surf_area;
        if (pressure > 1e-8 && !hit_terrain) {
            hit_terrain = true;
            tip_z_when_first_hit = tip_z;
        }
        float penetration = (hit_terrain) ? tip_z_when_first_hit - tip_z : 0;
        std::cout << "Time: " << t << std::endl;
        std::cout << "Bulk density: " << bulk_density << std::endl;
        std::cout << "Penetration: " << penetration << std::endl;
        std::cout << "Force on cone: " << forces.x << ", " << forces.y << ", " << forces.z << std::endl;
        std::cout << "Pressure: " << pressure << std::endl;

        if (frame_count % 1000 == 0) {
            char filename[200], meshname[200];
            std::cout << "Outputting frame: " << currframe << std::endl;
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
            sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), currframe++);
            DEMSim.WriteSphereFile(std::string(filename));
            DEMSim.WriteMeshFile(std::string(meshname));
            DEMSim.ShowThreadCollaborationStats();
        }

        DEMSim.DoDynamicsThenSync(frame_time);
        tip_z -= cone_speed * frame_time;

        frame_count++;
    }

    std::cout << "ConeDrop demo exiting..." << std::endl;
    return 0;
}