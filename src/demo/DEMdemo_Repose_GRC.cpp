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

using namespace deme;
using namespace std::filesystem;

const float kg_g_conv = 1.;

int main() {
    DEMSolver DEMSim;
    DEMSim.UseFrictionalHertzianModel();
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);

    float mu = 0.7;
    auto mat_type_walls = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", mu}});
    auto mat_type_particles = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", mu}});
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_walls, mat_type_particles, 0.2);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_walls, mat_type_particles, 2.);

    float scaling = 10.;
    // Loaded meshes are by-default fixed
    auto funnel = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/funnel.obj"), mat_type_walls);
    funnel->Scale(0.0005 * scaling);
    float funnel_bottom = 0.f;
    std::cout << "Total num of triangles: " << funnel->GetNumTriangles() << std::endl;

    float gap = 0.0015 * scaling;
    float fill_width = 0.05f * scaling;
    float fill_height = 0.15f;
    float fill_bottom = funnel_bottom + fill_width + gap;

    // Calculate its mass and MOI
    float mass1 = 2.6e3 * 5.5886717 * kg_g_conv;  // in kg or g
    float3 MOI1 = make_float3(1.8327927, 2.1580013, 0.77010059) * 2.6e3 * kg_g_conv;
    float mass2 = 2.6e3 * 2.7564385 * kg_g_conv;  // in kg or g
    float3 MOI2 = make_float3(1.0352626, 0.9616627, 1.6978352) * 2.6e3 * kg_g_conv;
    float mass3 = 2.6e3 * 8.1812 * kg_g_conv;  // in kg or g
    float3 MOI3 = make_float3(5.11336, 5.1133, 5.1133) * 2.6e3 * kg_g_conv;
    std::vector<float> mass = {mass2, mass2, mass1, mass1, mass3, mass3, mass3};
    std::vector<float3> MOI = {MOI2, MOI2, MOI1, MOI1, MOI3, MOI3, MOI3};
    // Then the ground particle template
    auto shape_template1 =
        DEMSim.LoadClumpType(mass1, MOI1, (GET_DATA_PATH() / "clumps/triangular_flat.csv").string(), mat_type_particles);
    auto shape_template2 = DEMSim.LoadClumpType(
        mass2, MOI2, (GET_DATA_PATH() / "clumps/triangular_flat_6comp.csv").string(), mat_type_particles);
    auto shape_template3 = DEMSim.LoadSphereType(mass3, 1.25, mat_type_particles);
    std::vector<std::shared_ptr<DEMClumpTemplate>> ground_particle_templates = {
        shape_template2, DEMSim.Duplicate(shape_template2), shape_template1, DEMSim.Duplicate(shape_template1),
        shape_template3, DEMSim.Duplicate(shape_template3), DEMSim.Duplicate(shape_template3)};
    // Scale the template we just created
    std::vector<double> volume = {2.7564385, 2.7564385, 5.5886717, 5.5886717, 8.1812, 8.1812, 8.1812};
    std::vector<double> scales = {0.0014, 0.00075833, 0.00044, 0.0003, 0.00016667, 0.00014667, 0.00012};
    std::for_each(scales.begin(), scales.end(), [](double& r) { r *= 10.; });
    unsigned int t_num = 0;
    for (double scaling : scales) {
        auto& this_template = ground_particle_templates[t_num];
        // this_template->mass = (double)mass[t_num] * scaling * scaling * scaling;
        // this_template->MOI.x = (double)MOI[t_num].x * (double)(scaling * scaling * scaling * scaling * scaling);
        // this_template->MOI.y = (double)MOI[t_num].y * (double)(scaling * scaling * scaling * scaling * scaling);
        // this_template->MOI.z = (double)MOI[t_num].z * (double)(scaling * scaling * scaling * scaling * scaling);
        // std::for_each(this_template->radii.begin(), this_template->radii.end(), [scaling](float& r) { r *= scaling;
        // }); std::for_each(this_template->relPos.begin(), this_template->relPos.end(), [scaling](float3& r) { r *=
        // scaling; });
        this_template->Scale(scaling);
        std::cout << "Mass: " << this_template->mass << std::endl;
        std::cout << "MOIX: " << this_template->MOI.x << std::endl;
        std::cout << "MOIY: " << this_template->MOI.y << std::endl;
        std::cout << "MOIZ: " << this_template->MOI.z << std::endl;
        std::cout << "=====================" << std::endl;

        // Give these templates names, 0000, 0001 etc.
        char t_name[20];
        sprintf(t_name, "%04d", t_num);
        this_template->AssignName(std::string(t_name));
        t_num++;
    }

    // Now we load clump locations from a checkpointed file
    {
        std::cout << "Making terrain..." << std::endl;
        auto clump_xyz = DEMSim.ReadClumpXyzFromCsv("./GRC_20e6.csv");
        auto clump_quaternion = DEMSim.ReadClumpQuatFromCsv("./GRC_20e6.csv");
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

        // Now, we don't need all particles loaded...
        std::vector<notStupidBool_t> elem_to_remove(in_xyz.size(), 0);
        for (size_t i = 0; i < in_xyz.size(); i++) {
            if (std::pow(in_xyz.at(i).x, 2) + std::pow(in_xyz.at(i).y, 2) >= std::pow(fill_width / 2. - 0.02, 2))
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

        // Move clumps from -0.5 to fill_bottom
        std::for_each(in_xyz.begin(), in_xyz.end(), [fill_bottom](float3& xyz) { xyz.z += (fill_bottom + 0.25); });

        DEMClumpBatch base_batch(in_xyz.size());
        base_batch.SetTypes(in_types);
        base_batch.SetPos(in_xyz);
        base_batch.SetOriQ(in_quat);

        DEMSim.AddClumps(base_batch);

        // Based on the `base_batch', we can create more batches
        std::vector<float> y_shift_dist = {fill_height, fill_height * 2, fill_height * 3};
        // Add some patches of such graular bed
        for (float y_shift : y_shift_dist) {
            DEMClumpBatch another_batch = base_batch;
            std::vector<float3> my_xyz = in_xyz;
            std::for_each(my_xyz.begin(), my_xyz.end(), [y_shift](float3& xyz) { xyz.z += y_shift; });
            another_batch.SetPos(my_xyz);
            DEMSim.AddClumps(another_batch);
        }
    }

    DEMSim.InstructBoxDomainDimension({-0.1 * scaling, 0.1 * scaling}, {-0.1 * scaling, 0.1 * scaling}, {funnel_bottom - 0.05f * scaling, funnel_bottom + 0.2f * scaling});
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_walls);

    DEMSim.SetInitTimeStep(1e-6);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
    DEMSim.SetCDUpdateFreq(100);
    DEMSim.SetExpandSafetyAdder(0.5);
    DEMSim.SetInitBinSizeAsMultipleOfSmallestSphere(2.5);
    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir += "/DemoOutput_Repose_GRC_HighGroundFric";
    create_directory(out_dir);

    for (int i = 0; i < 300; i++) {
        char filename[200], meshfile[200];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), i);
        sprintf(meshfile, "%s/DEMdemo_funnel_%04d.vtk", out_dir.c_str(), i);
        DEMSim.WriteSphereFile(std::string(filename));
        DEMSim.WriteMeshFile(std::string(meshfile));
        std::cout << "Frame: " << i << std::endl;
        DEMSim.DoDynamics(1e-1);
        DEMSim.ShowThreadCollaborationStats();
    }
    DEMSim.ShowTimingStats();
    DEMSim.ClearTimingStats();

    std::cout << "DEMdemo_Repose exiting..." << std::endl;
    return 0;
}
