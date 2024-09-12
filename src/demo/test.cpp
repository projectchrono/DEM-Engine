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
#include <fstream>

// =============================================================================
// In GRCPrep demo series, we try to prepare a sample of the GRC simulant, which
// are supposed to be used for extraterrestrial rover mobility simulations. It is
// made of particles of various sizes and shapes following a certain distribution.
// In Part1, it creates several batches of clumps and let them settle at the bottom
// of the domain.
// =============================================================================

using namespace deme;
using namespace std::filesystem;
using namespace std;
// =============================================================================
// PARAMETER DEFINITION
// =============================================================================

float E_terrain = 1e8, nu_terrain = 0.3, CoR_terrain = 0.3, mu_terrain = 0.60, Crr_terrain = 0.14,
      Cohesion = 10;  // 0.01 to 0.1
float E_ball = 1e9, nu_ball = 0.3, CoR_ball = 0.3, mu_ball = 0.5, Crr_ball = Crr_terrain,
      Cohesion_Ball = 0;  // 0.01 to 0.1

int srandValue = 759;

float world_x_size = 0.510, world_y_size = 0.510, world_z_size = 0.99;  // chrono simulation dimensions

float sampleheight = 0.15;     // for the generation of the random material
float bottom = -sampleheight;  // location of bottom boundary plane
// Define the terrain particle templates
// Default values for GPR1 example
// Calculate its mass and MOI
float terrain_density = 2.60e3;  // 2.6e3;
float ballDrop_simH = 0.45;

// Coarse sand
// Define radius
// std::vector<double> radius = {0.004,0.0025,0.002,0.0016,0.001,0.00071,0.0005};
//  Instatiate particles with a probability that is in line with their weight distribution.
// std::vector<double> weight_perc = {0.01732, 0.03013, 0.05466, 0.16078, 0.14146, 0.15697, 0.43868};
// Fine gravel
// std::vector<double> radius      = {0.008,0.0063,0.0056,0.00475,0.004,0.002,0.001};
std::vector<double> radius = {0.008 * 0.5, 0.0063 * 0.5, 0.0056 * 0.5, 0.00475 * 0.5,
                              0.004 * 0.5, 0.002 * 0.5,  0.001 * 0.5};
std::vector<double> weight_perc = {0.0042, 0.0423, 0.13191, 0.20089, 0.54679, 0.04379, 0.02889};
std::vector<int> particlePerClump = {2, 2, 2, 2, 2, 1, 1};

// Make ready for simulation
float step_size = 5e-7;
float sim_time = 1.20;

float paddingFactor = 1.7;  // distance between particles
float toleranceFactor = 0.96;
float settletoleranceFactor = 0.10;  // m
float gravity = -9.81;

std::string nameOutFile =
    "Test_width_" + std::to_string(world_y_size) + "Crr_" + std::to_string(Crr_terrain) + "" + ".out";
std::ofstream csvFile(nameOutFile);

int main() {
    float Cohesion_values[] = {0, 5, 10, 20, 40, 80};

    for (float Cohesion0 : Cohesion_values) {
        DEMSolver DEMSim;
        DEMSim.SetVerbosity(INFO);
        DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
        // DEMSim.SetOutputContent(OUTPUT_CONTENT::FAMILY);
        DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ);
        DEMSim.SetOutputContent(OUTPUT_CONTENT::VEL);

        csvFile << "Rolling friction: " << Crr_terrain << std::endl;
        csvFile << "Cohesion: " << Cohesion0 << std::endl;
        csvFile << "___________" << std::endl;

        srand(srandValue);
        // Define ball
        float ball_density = 3561.50;
        float ball_radius = 0.0722 / 2;
        float drop_height = 0.50;
        // Define materials
        auto mat_type_terrain = DEMSim.LoadMaterial({{"E", E_terrain},
                                                     {"nu", nu_terrain},
                                                     {"CoR", CoR_terrain},
                                                     {"mu", mu_terrain},
                                                     {"Crr", Crr_terrain},
                                                     {"Cohesion", Cohesion0}});

        // Define ball
        auto mat_type_ball = DEMSim.LoadMaterial({{"E", E_terrain},
                                                  {"nu", nu_terrain},
                                                  {"CoR", CoR_terrain},
                                                  {"mu", mu_terrain},
                                                  {"Crr", Crr_terrain},
                                                  {"Cohesion", Cohesion_Ball}});

        DEMSim.SetMaterialPropertyPair("Cohesion", mat_type_terrain, mat_type_ball, 0.);

        // Define the simulation world
        DEMSim.InstructBoxDomainDimension({-world_x_size / 1.5, world_x_size / 1.5},
                                          {-world_y_size / 1.5, world_y_size / 1.5}, {2.0 * bottom, world_z_size});
        // Add 5 bounding planes around the simulation world, and leave the top open
        // DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);
        auto top_plane = DEMSim.AddWavefrontMeshObject("../data/mesh/box.obj", mat_type_ball);
        top_plane->SetInitPos(make_float3(0, 0, bottom - 0.010));
        top_plane->SetMass(1.);
        top_plane->Scale(make_float3(world_x_size / 2.0, world_x_size / 2.0, 0.50));
        top_plane->SetFamily(10);
        DEMSim.SetFamilyFixed(10);

        // Then the ground particle template
        std::vector<std::shared_ptr<DEMClumpTemplate>> ground_particle_templates;
        unsigned int t_num = 0;

        for (int i = 0; i < radius.size(); i++) {
            std::vector<float> radii;
            std::vector<float3> relPos;
            std::vector<std::shared_ptr<DEMMaterial>> mat;

            double radiusMax = radius[i];
            double radiusMin = 8.0 / 8.0 * radiusMax;
            double eccentricity = 3.0 / 8.0 * radiusMax;
            /// particle one of the two-particle clump
            float3 tmp = make_float3(0, 0, 0);
            double a = 0, b = 0, c = 0;

            if (particlePerClump[i] == int(1)) {
                relPos.push_back(tmp);
                mat.push_back(mat_type_terrain);
                radii.push_back(radiusMax);

                a = radiusMax;
                b = radiusMax;
                c = radiusMax;
                std::cout << "A clump with only one particle" << std::endl;
            } else if (particlePerClump[i] == int(2)) {
                tmp.x = -1 * eccentricity / 2.0;
                tmp.y = 0;
                tmp.z = 0;
                relPos.push_back(tmp);
                mat.push_back(mat_type_terrain);
                radii.push_back(radiusMin);

                double x = eccentricity / 2.0;
                double y = 0;
                double z = 0;
                tmp.x = x;
                tmp.y = y;
                tmp.z = z;
                relPos.push_back(tmp);
                mat.push_back(mat_type_terrain);
                radii.push_back(radiusMin);

                c = radiusMin;  // smaller dim of the ellipse
                b = radiusMin;
                a = radiusMax;
                std::cout << "A clump with two particles" << std::endl;
            }

            float mass = 4.0 / 3.0 * deme::PI * a * b * c * terrain_density;
            float3 MOI = make_float3(1.f / 5.f * mass * (b * b + c * c), 1.f / 5.f * mass * (a * a + c * c),
                                     1.f / 5.f * mass * (b * b + a * a));

            auto clump_ptr = DEMSim.LoadClumpType(mass, MOI, radii, relPos, mat_type_terrain);
            // clump_ptr->AssignName("fsfs");

            char t_name[20];
            sprintf(t_name, "%04d", i);
            clump_ptr->AssignName(std::string(t_name));
            ground_particle_templates.push_back(clump_ptr);
        }

        // std::cout << "Name " << ground_particle_templates.at(0)->m_name << std::endl;

        // Now we load patch clump locations from an output file
        auto part2_clump_xyz = DEMSim.ReadClumpXyzFromCsv("./data/my/GRC_cons0.510000_Final.csv");
        auto part2_clump_quaternion = DEMSim.ReadClumpQuatFromCsv("./data/my/GRC_cons0.510000_Final.csv");

        std::vector<float3> in_xyz;
        std::vector<float4> in_quat;
        std::vector<std::shared_ptr<DEMClumpTemplate>> in_types;
        t_num = 0;

        for (int i = 0; i < radius.size(); i++) {
            char t_name[20];
            sprintf(t_name, "%04d", t_num);

            auto this_type_xyz = part2_clump_xyz[std::string(t_name)];
            auto this_type_quat = part2_clump_quaternion[std::string(t_name)];

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
        std::cout << "Point " << in_xyz.data() << std::endl;

        // Finally, load the info into this batch
        DEMClumpBatch base_batch(in_xyz.size());
        base_batch.SetTypes(in_types);
        base_batch.SetPos(in_xyz);
        base_batch.SetOriQ(in_quat);

        DEMSim.AddClumps(base_batch);

        // Include ball for drop ball test
        auto projectile = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/sphere.obj").string(), mat_type_ball);
        projectile->Scale(ball_radius);
        std::cout << "Total num of triangles: " << projectile->GetNumTriangles() << std::endl;

        projectile->SetInitPos(make_float3(0, 0, 1.0));

        // float ball_mass = ball_density * 4. / 3. * PI * ball_radius * ball_radius * ball_radius;
        float ball_mass = 0.70185;
        projectile->SetMass(ball_mass);
        projectile->SetMOI(make_float3(ball_mass * 2 / 5 * ball_radius * ball_radius,
                                       ball_mass * 2 / 5 * ball_radius * ball_radius,
                                       ball_mass * 2 / 5 * ball_radius * ball_radius));
        projectile->SetFamily(2);
        DEMSim.SetFamilyFixed(2);
        DEMSim.DisableContactBetweenFamilies(0, 2);
        // Track the projectile
        auto proj_tracker = DEMSim.Track(projectile);

        // A custom force model can be read in through a file and used by the simulation. Magic, right?
        auto my_force_model = DEMSim.ReadContactForceModel("ForceModelWithCohesion.cu");
        // This custom force model still uses contact history arrays, so let's define it
        my_force_model->SetPerContactWildcards({"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z"});
        my_force_model->SetMustPairwiseMatProp({"CoR", "mu", "Crr", "Cohesion"});

        // Make ready for simulation
        DEMSim.SetInitTimeStep(step_size);
        DEMSim.SetGravitationalAcceleration(make_float3(0, 0, gravity));
        // Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
        // wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
        // happen anyway and if it does, something already went wrong.
        DEMSim.SetMaxVelocity(30.);
        // Error out vel is used to force the simulation to abort when something goes wrong.
        DEMSim.SetErrorOutVelocity(15.);
        DEMSim.SetExpandSafetyMultiplier(1.2);
        DEMSim.SetInitBinNumTarget(1e7);
        DEMSim.Initialize();

        unsigned int fps = 300;
        unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

        path out_dir = current_path();
        char drop_height_buf[100];
        char Crr0_buf[100];
        char Cohesion_buf[100];
        string folderName;
        snprintf(drop_height_buf, sizeof drop_height_buf, "%.2f", drop_height);
        snprintf(Crr0_buf, sizeof Crr0_buf, "%.2f", Crr_terrain);
        snprintf(Cohesion_buf, sizeof Cohesion_buf, "%.2f", Cohesion0);
        folderName = "_H_";
        folderName += drop_height_buf;
        folderName += "_Crr_";
        folderName += Crr0_buf;
        folderName += "_Cohesion_";
        folderName += Cohesion_buf;

        out_dir += "/Output" + folderName;
        create_directory(out_dir);
        unsigned int currframe = 0;
        unsigned int curr_step = 0;

        // Print initial time step
        char filename[200];
        sprintf(filename, "%s/DEMdemo_output_ini.csv", out_dir.c_str());
        DEMSim.WriteSphereFile(std::string(filename));
        // Settle a bit
        // DEMSim.DoDynamicsThenSync(0.3);
        DEMSim.SetInitTimeStep(step_size);
        //DEMSim.UpdateStepSize();

        float frame_time = 1.0 / fps;

        auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
        float terrain_max_z = max_z_finder->GetValue();

        DEMSim.ChangeFamily(2, 0);
        //proj_tracker->SetPos(make_float3(0, 0, terrain_max_z + drop_height - ballDrop_simH));
        //proj_tracker->SetVel(make_float3(0, 0, -sqrt(2 * ballDrop_simH * abs(gravity))));

        sprintf(filename, "%s/BallDrop_mesh_ini.vtk", out_dir.c_str());
        DEMSim.WriteMeshFile(std::string(filename));

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        for (float t = 0; t < sim_time; t += frame_time) {
            std::cout << "Frame: " << currframe << std::endl;
            char filename[200], meshfilename[200], cnt_filename[200];
            sprintf(filename, "%s/BallDrop_output_%04d.csv", out_dir.c_str(), currframe);
            sprintf(meshfilename, "%s/BallDrop_mesh_%04d.vtk", out_dir.c_str(), currframe);
            // sprintf(cnt_filename, "%s/Contact_pairs_%04d.csv", out_dir.c_str(), currframe);
            DEMSim.WriteSphereFile(std::string(filename));
            DEMSim.WriteMeshFile(std::string(meshfilename));
            // DEMSim.WriteContactFile(std::string(cnt_filename));
            currframe++;
            float3 force = proj_tracker->ContactAcc();
            std::cout << "pos z " << proj_tracker->Pos().z << "; vel z " << std::abs(proj_tracker->Vel().z)
                      << std::endl;
            std::cout << "Penetration: " << terrain_max_z - (proj_tracker->Pos().z - ball_radius) << std::endl;
            csvFile << "Time: " << t << "; Pos.z: " << proj_tracker->Pos().z << "; force Z: " << force.z
                    << "; Penetration: " << terrain_max_z - (proj_tracker->Pos().z - ball_radius) << std::endl;
            DEMSim.DoDynamics(frame_time);
            // DEMSim.ShowThreadCollaborationStats();

            if (std::abs(proj_tracker->Vel().z) < 1e-5) {
                break;
            }
        }

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;

        DEMSim.ShowTimingStats();

        float3 final_pos = proj_tracker->Pos();
        std::cout << "Ball density: " << ball_density << std::endl;
        std::cout << "Ball rad: " << ball_radius << std::endl;
        std::cout << "Drop height: " << drop_height << std::endl;
        std::cout << "Penetration: " << terrain_max_z - (final_pos.z - ball_radius) << std::endl;

        csvFile << "Drop height: " << drop_height << std::endl;
        csvFile << "Penetration: " << terrain_max_z - (final_pos.z - ball_radius) << std::endl;

        std::cout << "==============================================================" << std::endl;
    }
    std::cout << "DEMdemo_BallDrop exiting..." << std::endl;
    return 0;
}