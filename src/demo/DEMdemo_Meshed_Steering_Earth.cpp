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
#include <cstdlib>

using namespace deme;

const double math_PI = 3.1415927;

int main(int argc, char* argv[]) {
    int cur_test = atoi(argv[1]);

    std::filesystem::path out_dir = std::filesystem::current_path();
    out_dir += "/DEMdemo_Wheel_Steer_2";
    std::filesystem::create_directory(out_dir);

    // `World'
    float G_mag = 9.81;
    float step_size = 5e-6;
    double world_size_y = .75;
    double world_size_x = 2.;
    double world_size_z = 4.0;

    // Define the wheel geometry
    float wheel_rad = atof(argv[2]);
    float wheel_width = 0.2;
    float wheel_mass = 5.;               // 8.7;
    float total_pressure = atof(argv[3]) * 9.81;  // 22.
    float added_pressure = (total_pressure - wheel_mass * G_mag);
    float wheel_IYY = wheel_mass * wheel_rad * wheel_rad / 2;
    float wheel_IXX = (wheel_mass / 12) * (3 * wheel_rad * wheel_rad + wheel_width * wheel_width);
    float grouser_height = atof(argv[4]);

    // Wheel movements
    float w_r = 0.8;
    float forward_slip = 0.1;
    double sim_end = 5.;
    float tilt = 5. / 180. * math_PI;
    float forward_v = w_r * wheel_rad * std::cos(tilt) * (1. - forward_slip);
    float forward_ref_v = w_r * wheel_rad * std::cos(tilt);

    float Slopes_deg[] = {0};

    for (float Slope_deg : Slopes_deg) {
        DEMSolver DEMSim;
        DEMSim.SetVerbosity(ERROR);
        DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
        DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
        DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
        DEMSim.SetIntegrator(TIME_INTEGRATOR::FORWARD_EULER);
        DEMSim.SetCollectAccRightAfterForceCalc(true);

        // E, nu, CoR, mu, Crr...
        float mu = 0.4;
        float mu_wheel = 0.8;
        float mu_wall = 1.;
        auto mat_type_wall =
            DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.4}, {"mu", mu_wall}, {"Crr", 0.00}});
        auto mat_type_wheel =
            DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.4}, {"mu", mu_wheel}, {"Crr", 0.00}});
        auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.4}, {"mu", mu}, {"Crr", 0.00}});
        DEMSim.SetMaterialPropertyPair("mu", mat_type_wheel, mat_type_terrain, mu_wheel);
        DEMSim.SetMaterialPropertyPair("mu", mat_type_wall, mat_type_terrain, mu_wall);

        DEMSim.InstructBoxDomainDimension(world_size_x, world_size_y, world_size_z);
        DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_wall);

        float bottom = -0.5;
        auto bot_wall = DEMSim.AddBCPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_wall);
        auto bot_wall_tracker = DEMSim.Track(bot_wall);

        auto wheel = DEMSim.AddWavefrontMeshObject("./wheel.obj", mat_type_wheel);
        wheel->SetMass(wheel_mass);
        wheel->SetMOI(make_float3(wheel_IXX, wheel_IYY, wheel_IXX));
        // Give the wheel a family number so we can potentially add prescription
        wheel->SetFamily(11);
        float4 initMeshQ = make_float4(0, 0, 1 * std::sin(tilt / 2.), std::cos(tilt / 2.));
        wheel->SetInitQuat(initMeshQ);
        DEMSim.SetFamilyFixed(11);
        DEMSim.DisableContactBetweenFamilies(11, 0);
        // Track it
        auto wheel_tracker = DEMSim.Track(wheel);

        // Define the terrain particle templates
        // Calculate its mass and MOI
        float terrain_density = 2.6e3;
        float volume1 = 4.2520508;
        float mass1 = terrain_density * volume1;
        float3 MOI1 = make_float3(1.6850426, 1.6375114, 2.1187753) * terrain_density;
        // Scale the template we just created
        std::vector<double> scales = {0.008};
        // Then load it to system
        std::shared_ptr<DEMClumpTemplate> my_template1 =
            DEMSim.LoadClumpType(mass1, MOI1, GetDEMEDataFile("clumps/triangular_flat.csv"), mat_type_terrain);
        std::vector<std::shared_ptr<DEMClumpTemplate>> ground_particle_templates = {my_template1,
                                                                                    DEMSim.Duplicate(my_template1)};
        // Now scale those templates
        for (int i = 0; i < scales.size(); i++) {
            std::shared_ptr<DEMClumpTemplate>& my_template = ground_particle_templates.at(i);
            // Note the mass and MOI are also scaled in the process, automatically. But if you are not happy with this,
            // you can always manually change mass and MOI afterwards.
            my_template->Scale(scales.at(i));
            // Give these templates names, 0000, 0001 etc.
            char t_name[20];
            sprintf(t_name, "%04d", i);
            my_template->AssignName(std::string(t_name));
        }

        // Now we load clump locations from a checkpointed file
        {
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
                // Prepare clump type identification vector for loading into the system (don't forget type 0 in
                // ground_particle_templates is the template for rover wheel)
                std::vector<std::shared_ptr<DEMClumpTemplate>> this_type(n_clump_this_type,
                                                                         ground_particle_templates.at(t_num));

                // Add them to the big long vector
                in_xyz.insert(in_xyz.end(), this_type_xyz.begin(), this_type_xyz.end());
                in_quat.insert(in_quat.end(), this_type_quat.begin(), this_type_quat.end());
                in_types.insert(in_types.end(), this_type.begin(), this_type.end());
                // Our template names are 0000, 0001 etc.
                t_num++;
            }

            // Now, we don't need all particles loaded...
            std::vector<notStupidBool_t> elem_to_remove(in_xyz.size(), 0);
            for (size_t i = 0; i < in_xyz.size(); i++) {
                if (std::abs(in_xyz.at(i).y) > (world_size_y - 0.05) / 2 ||
                    std::abs(in_xyz.at(i).x) > (world_size_x - 0.05) / 2)
                    elem_to_remove.at(i) = 1;
            }
            in_xyz.erase(std::remove_if(in_xyz.begin(), in_xyz.end(),
                                        [&elem_to_remove, &in_xyz](const float3& i) {
                                            return elem_to_remove.at(&i - in_xyz.data());
                                        }),
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
        }

        // Now add a plane to compress the sample
        // auto compressor = DEMSim.AddExternalObject();
        // compressor->AddPlane(make_float3(0, 0, 0), make_float3(0, 0, -1), mat_type_terrain);
        // compressor->SetFamily(2);
        // auto compressor_tracker = DEMSim.Track(compressor);

        // Families' prescribed motions (Earth)
        float v_ref = w_r * wheel_rad;
        double G_ang = Slope_deg * math_PI / 180.;

        // Note: this wheel is not `dictated' by our prescrption of motion because it can still fall onto the ground
        // (move freely linearly)
        DEMSim.SetFamilyPrescribedAngVel(1, "0", "0", "0", false);
        // DEMSim.SetFamilyPrescribedLinVel(2, to_string_with_precision(forward_v), "0", "none", false);
        DEMSim.SetFamilyPrescribedLinVel(2, "none", "0", "none", false);
        DEMSim.SetFamilyPrescribedAngVel(2, "0", to_string_with_precision(w_r), "0", false);
        DEMSim.AddFamilyPrescribedAcc(2, to_string_with_precision(-added_pressure * std::sin(G_ang) / wheel_mass),
                                      "none", to_string_with_precision(-added_pressure * std::cos(G_ang) / wheel_mass));
        DEMSim.SetFamilyFixed(10);
        DEMSim.DisableContactBetweenFamilies(10, 10);
        DEMSim.DisableContactBetweenFamilies(10, 255);

        // Some inspectors
        auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
        // auto min_z_finder = DEMSim.CreateInspector("clump_min_z");
        // auto total_mass_finder = DEMSim.CreateInspector("clump_mass");
        // auto partial_mass_finder = DEMSim.CreateInspector("clump_mass", "return (Z <= -0.41);");
        // auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");

        float3 this_G = make_float3(-G_mag * std::sin(G_ang), 0, -G_mag * std::cos(G_ang));
        DEMSim.SetGravitationalAcceleration(this_G);

        DEMSim.SetInitTimeStep(step_size);
        DEMSim.SetCDUpdateFreq(10);
        DEMSim.SetExpandSafetyAdder(.1);
        DEMSim.SetCDNumStepsMaxDriftMultipleOfAvg(1);
        DEMSim.SetCDNumStepsMaxDriftAheadOfAvg(5);
        DEMSim.SetErrorOutVelocity(150.);
        DEMSim.Initialize();

        // Put the wheel in place, then let the wheel sink in initially
        float corr = 1.0;  // 0
        float init_x = -1. + corr;
        if (Slope_deg < 21) {
            init_x = -1.6 + corr;
        }

        float settle_time = 0.2;
        { DEMSim.DoDynamicsThenSync(settle_time); }

        // Put the wheel in place, then let the wheel sink in initially
        float max_z = max_z_finder->GetValue();
        wheel_tracker->SetPos(make_float3(init_x, 0, max_z + 0.05 + grouser_height + wheel_rad));

        int report_ps = 1000;
        float report_time = report_ps * step_size;
        unsigned int report_steps = (unsigned int)(1.0 / report_time);
        unsigned int cur_step = 0;
        double xforce = 0., yforce = 0.;
        unsigned int report_num = 0;

        DEMSim.ChangeFamily(11, 1);

        {
            {
                char filename[200], meshname[200];
                sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), cur_test);
                sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), cur_test);
                DEMSim.WriteSphereFile(std::string(filename));
                DEMSim.WriteMeshFile(std::string(meshname));
            }

            std::cout << "Test num: " << cur_test << std::endl;
            DEMSim.DoDynamicsThenSync(0.5);
            DEMSim.ChangeFamily(1, 2);
            DEMSim.DoDynamicsThenSync(1.);

            float x1 = wheel_tracker->Pos().x;

            float t;
            for (t = 0; t < sim_end; t += report_time) {
                float3 force = wheel_tracker->ContactAcc();
                xforce += (double)force.x * wheel_mass;
                yforce += (double)force.y * wheel_mass;
                report_num++;

                DEMSim.DoDynamics(report_time);
            }

            float adv = wheel_tracker->Pos().x - x1;
            std::cout << "Slip: " << 1. - adv / (forward_ref_v * t) << std::endl;
            std::cout << "Force X: " << xforce / report_num << std::endl;
            std::cout << "Force Y: " << yforce / report_num << std::endl;

            {
                // Overwrite previous...
                char filename[200], meshname[200];
                sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), cur_test);
                sprintf(meshname, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), cur_test);
                DEMSim.WriteSphereFile(std::string(filename));
                DEMSim.WriteMeshFile(std::string(meshname));
                std::cout << "=================================" << std::endl;
            }
        }

        // DEMSim.ShowTimingStats();
        // DEMSim.ShowAnomalies();
    }

    return 0;
}