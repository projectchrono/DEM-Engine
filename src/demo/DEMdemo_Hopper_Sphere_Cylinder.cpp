//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A repose angle test. Particles flow through a mesh-represented funnel and form
// a pile that has an apparent angle.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <random>

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.UseFrictionalHertzianModel();
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ | OUTPUT_CONTENT::VEL | OUTPUT_CONTENT::FAMILY);
    DEMSim.EnsureKernelErrMsgLineNum();

    srand(7001);
    DEMSim.SetCollectAccRightAfterForceCalc(true);
    DEMSim.SetErrorOutAvgContacts(80);

    // DEMSim.SetExpandSafetyAdder(0.5);
    int totalCyl = 10500;
    int totalSph = 7000 + totalCyl;

    int num_template = 1;

    // total number of random clump templates to generate
    // data for the cylinders
    double radiusCyl = 0.0040 / 2.0;
    double length = 0.0080;
    int n_sphere = 5;  // spheres that are required to form a cylinder
    double densityCyl = 1128;

    // data for the spheres
    double radiusSph = 0.0060 / 2.0;
    double densitySph = 1592;

    float plane_bottom = 0.02f;
    float funnel_bottom = 0.02f;

    double gateOpen = 0.30;
    double gateSpeed = -3.5;
    double hopperW = 0.04;
    double gateWidth = 0.1295;

    path out_dir = current_path();
    out_dir /= "Test_Plastic_Sphere_Cylinder";
    out_dir /= "Hopper";

    auto mat_type_bottom = DEMSim.LoadMaterial({{"E", 10e9}, {"nu", 0.3}, {"CoR", 0.60}});
    auto mat_type_flume = DEMSim.LoadMaterial({{"E", 10e9}, {"nu", 0.3}, {"CoR", 0.60}});
    auto mat_type_walls = DEMSim.LoadMaterial({{"E", 10e9}, {"nu", 0.3}, {"CoR", 0.60}});

    // auto mat_spheres = DEMSim.LoadMaterial({{"E", 1.0e7}, {"nu", 0.35}, {"CoR", 0.85}, {"mu", 0.50}, {"Crr", 0.03}});

    // auto mat_cylinders = DEMSim.LoadMaterial({{"E", 1.0e7}, {"nu", 0.35}, {"CoR", 0.85}, {"mu", 0.60}, {"Crr",
    // 0.05}});

    auto mat_spheres = DEMSim.LoadMaterial({{"E", 1.0e7}, {"nu", 0.35}, {"CoR", 0.85}, {"mu", 0.40}, {"Crr", 0.04}});

    auto mat_cylinders = DEMSim.LoadMaterial({{"E", 1.0e7}, {"nu", 0.35}, {"CoR", 0.85}, {"mu", 0.30}, {"Crr", 0.03}});

    DEMSim.SetMaterialPropertyPair("CoR", mat_type_walls, mat_cylinders, 0.7);
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_walls, mat_cylinders, 0.05);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_walls, mat_spheres, 0.30);

    DEMSim.SetMaterialPropertyPair("CoR", mat_type_walls, mat_spheres, 0.7);
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_walls, mat_spheres, 0.05);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_walls, mat_spheres, 0.30);

    DEMSim.SetMaterialPropertyPair("CoR", mat_type_flume, mat_spheres, 0.70);
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_flume, mat_spheres, 0.05);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_flume, mat_spheres, 0.30);

    DEMSim.SetMaterialPropertyPair("CoR", mat_type_flume, mat_cylinders, 0.70);  // it is supposed to be
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_flume, mat_cylinders, 0.05);  // plexiglass
    DEMSim.SetMaterialPropertyPair("mu", mat_type_flume, mat_cylinders, 0.30);

    // Make ready for simulation
    float step_size = 5.0e-6;
    DEMSim.InstructBoxDomainDimension({-0.10, 0.10}, {-0.02, 0.02}, {-0.50, 1.0});
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_walls);
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    // wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    // happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(25.);
    DEMSim.SetInitBinSize(radiusCyl * 2);

    // Loaded meshes are by-default fixed

    auto fixed_left = DEMSim.AddWavefrontMeshObject("../data/mesh/funnel_left.obj", mat_type_flume);
    float3 move = make_float3(-hopperW / 2.0, 0, -0.01);
    float4 rot = make_float4(0.7071, 0, 0, 0.7071);
    fixed_left->Move(move, rot);

    auto fixed_right = DEMSim.AddWavefrontMeshObject("../data/mesh/funnel_left.obj", mat_type_flume);
    move = make_float3(gateWidth + hopperW / 2.0, 0, -0.01);
    fixed_right->Move(move, rot);

    auto gate = DEMSim.AddWavefrontMeshObject("../data/mesh/funnel_left.obj", mat_type_flume);
    gate->Move(make_float3(gateWidth / 2, 0, -0.011), rot);

    fixed_left->SetFamily(10);
    fixed_right->SetFamily(10);
    gate->SetFamily(3);

    std::string shake_pattern_xz = " 0.0 * sin( 300 * 2 * deme::PI * t)";
    std::string shake_pattern_y = " 0.0 * sin( 30 * 2 * deme::PI * t)";

    DEMSim.SetFamilyFixed(1);
    DEMSim.SetFamilyFixed(3);
    DEMSim.SetFamilyPrescribedLinVel(4, "0", "0", to_string_with_precision(gateSpeed));
    DEMSim.SetFamilyPrescribedLinVel(10, shake_pattern_xz, shake_pattern_y, shake_pattern_xz);

    // Make an array to store these generated clump templates
    float z = 0;
    int frame = 0;
    float timeTotal = 0.0;

    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    float settle_frame_time = 0.005;

    remove_all(out_dir);
    create_directories(out_dir);
    char filename[100], meshfile[100];

    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    auto min_z_finder = DEMSim.CreateInspector("clump_min_z");
    auto total_mass_finder = DEMSim.CreateInspector("clump_mass");
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");

    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_cylinder;
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_sphere;

    {  // initialize cylinder clump
        for (int i = 0; i < num_template; i++) {
            std::vector<float> radii;
            std::vector<float3> relPos;
            std::vector<std::shared_ptr<DEMMaterial>> mat;

            double radiusMed = radiusCyl;

            float init = -1.0 * length / 2.0 + radiusMed;
            float trail = length / 2.0 - radiusMed;
            float incrR = (trail - init) / (n_sphere - 1);
            float3 tmp;

            for (int j = 0; j < n_sphere; j++) {
                radii.push_back(radiusMed);

                tmp.x = init + (j * incrR);
                tmp.y = 0;
                tmp.z = 0;
                relPos.push_back(tmp);
                mat.push_back(mat_cylinders);
            }

            float mass = PI * radiusMed * radiusMed * length * densityCyl;
            float Ixx = 1.f / 2.f * mass * radiusMed * radiusMed;
            float Iyy = Ixx / 2.0 + 1.0 / 12.0 * mass * length * length;
            float3 MOI = make_float3(Ixx, Iyy, Iyy);
            std::cout << mass << " chosen moi ..." << radiusMed / radiusCyl << std::endl;

            auto clump_ptr = DEMSim.LoadClumpType(mass, MOI, radii, relPos, mat_cylinders);
            // clump_ptr->AssignName("fsfs");
            clump_cylinder.push_back(clump_ptr);
        }
    }

    {  // initialize spheres
        for (int i = 0; i < num_template; i++) {
            std::vector<float> radii;
            std::vector<float3> relPos;
            std::vector<std::shared_ptr<DEMMaterial>> mat;
            auto tmp = make_float3(0, 0, 0);
            // double radiusMax = distribution(generator);
            double radiusMax = radiusSph;

            relPos.push_back(tmp);
            mat.push_back(mat_spheres);
            radii.push_back(radiusMax);

            double c = radiusMax;
            double b = radiusMax;
            double a = radiusMax;

            float mass = 4.0 / 3.0 * PI * a * b * c * densitySph;
            float3 MOI = make_float3(1.f / 5.f * mass * (b * b + c * c), 1.f / 5.f * mass * (a * a + c * c),
                                     1.f / 5.f * mass * (b * b + a * a));
            std::cout << a << " chosen moi ..." << a / radiusSph << std::endl;

            auto clump_ptr = DEMSim.LoadClumpType(mass, MOI, radii, relPos, mat_spheres);
            // clump_ptr->AssignName("fsfs");
            clump_sphere.push_back(clump_ptr);
        }
    }

    DEMSim.Initialize();

    //
    // make cylinders
    //
    {
        float shift_xyz = 1.0 * (length)*1.1;
        float x = 0;
        float y = 0;

        z = shift_xyz / 2;  // by default we create beads at 0
        double emitterZ = 0.60;
        unsigned int actualTotalSpheres = 0;

        bool generate = true;
        bool initialization = true;
        double consolidation = true;

        sprintf(meshfile, "DEMdemo_funnel_%04d.vtk", frame);
        DEMSim.WriteMeshFile(out_dir / meshfile);

        while (initialization) {
            std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;
            std::vector<float3> input_pile_xyz;
            PDSampler sampler(shift_xyz);

            bool generate = (plane_bottom + shift_xyz / 2 > emitterZ) ? false : true;

            if (generate) {
                float sizeZ = (frame == 0) ? 0.20 : 0;
                float sizeX = 0.20;
                float z = plane_bottom + shift_xyz + sizeZ / 2.0;

                float3 center_xyz = make_float3(0, 0, z);
                float3 size_xyz = make_float3((sizeX - shift_xyz) / 2.0, (0.04 - shift_xyz) / 2.0, sizeZ / 2.0);

                std::cout << "level of particles position ... " << center_xyz.z << std::endl;

                auto heap_particles_xyz = sampler.SampleBox(center_xyz, size_xyz);
                unsigned int num_clumps = heap_particles_xyz.size();
                std::cout << "number of particles at this level ... " << num_clumps << std::endl;

                for (unsigned int i = actualTotalSpheres; i < actualTotalSpheres + num_clumps; i++) {
                    input_pile_template_type.push_back(clump_cylinder.at(i % num_template));
                }

                input_pile_xyz.insert(input_pile_xyz.end(), heap_particles_xyz.begin(), heap_particles_xyz.end());

                auto the_pile = DEMSim.AddClumps(input_pile_template_type, input_pile_xyz);
                the_pile->SetVel(make_float3(-0.00, 0.0, -0.80));
                the_pile->SetFamily(100);

                DEMSim.UpdateClumps();

                std::cout << "Total num of particles: " << (int)DEMSim.GetNumClumps() << std::endl;
                actualTotalSpheres = (int)DEMSim.GetNumClumps();
                // Generate initial clumps for piling
            }

            timeTotal += settle_frame_time;
            std::cout << "Total runtime: " << timeTotal << "s; settling for: " << settle_frame_time << std::endl;
            std::cout << "maxZ is: " << max_z_finder->GetValue() << std::endl;

            initialization = (actualTotalSpheres < totalCyl) ? true : false;

            if (generate) {
                std::cout << "frame : " << frame << std::endl;
                sprintf(filename, "DEMdemo_settling_%04d.csv", frame);
                DEMSim.WriteSphereFile(out_dir / filename);
                // DEMSim.ShowThreadCollaborationStats();
                frame++;
            }

            DEMSim.DoDynamicsThenSync(settle_frame_time);

            plane_bottom = max_z_finder->GetValue();
            /// here the settling phase starts
            if (!initialization) {
                for (int i = 0; i < (int)(0.3 / settle_frame_time); i++) {
                    DEMSim.DoDynamics(settle_frame_time);
                    sprintf(filename, "DEMdemo_settling_%04d.csv", i);
                    DEMSim.WriteSphereFile(out_dir / filename);
                    std::cout << "consolidating for " << i * settle_frame_time << "s " << std::endl;
                }
            }
        }
    }

    DEMSim.DoDynamicsThenSync(0.0);

    //
    // Sphere generation
    //

    {
        float shift_xyz = 1.0 * (radiusSph)*2.0;
        float x = 0;
        float y = 0;

        double emitterZ = 0.60;
        unsigned int actualTotalSpheres = 0;

        bool generate = true;
        bool initialization = true;
        double consolidation = true;

        sprintf(meshfile, "DEMdemo_funnel_%04d.vtk", frame);
        DEMSim.WriteMeshFile(out_dir / meshfile);

        while (initialization) {
            std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;
            std::vector<float3> input_pile_xyz;
            PDSampler sampler(shift_xyz);

            bool generate = (plane_bottom + shift_xyz > emitterZ) ? false : true;

            if (generate) {
                float sizeZ = (frame == 0) ? 0.20 : 0;
                float sizeX = 0.20;
                float z = plane_bottom + shift_xyz + sizeZ / 2.0;

                float3 center_xyz = make_float3(0, 0, z);
                float3 size_xyz = make_float3((sizeX - shift_xyz) / 2.0, (0.04 - shift_xyz) / 2.0, sizeZ / 2.0);

                std::cout << "level of particles position ... " << center_xyz.z << std::endl;

                auto heap_particles_xyz = sampler.SampleBox(center_xyz, size_xyz);
                unsigned int num_clumps = heap_particles_xyz.size();
                std::cout << "number of particles at this level ... " << num_clumps << std::endl;

                for (unsigned int i = actualTotalSpheres; i < actualTotalSpheres + num_clumps; i++) {
                    input_pile_template_type.push_back(clump_sphere.at(i % num_template));
                }

                input_pile_xyz.insert(input_pile_xyz.end(), heap_particles_xyz.begin(), heap_particles_xyz.end());

                auto the_pile = DEMSim.AddClumps(input_pile_template_type, input_pile_xyz);
                the_pile->SetVel(make_float3(-0.00, 0.0, -0.80));
                the_pile->SetFamily(99);

                DEMSim.UpdateClumps();

                std::cout << "Total num of particles: " << (int)DEMSim.GetNumClumps() << std::endl;
                actualTotalSpheres = (int)DEMSim.GetNumClumps();
                // Generate initial clumps for piling
            }
            timeTotal += settle_frame_time;
            std::cout << "Total runtime: " << timeTotal << "s; settling for: " << settle_frame_time << std::endl;
            std::cout << "maxZ is: " << max_z_finder->GetValue() << std::endl;

            initialization = (actualTotalSpheres < totalSph) ? true : false;

            if (generate) {
                std::cout << "frame : " << frame << std::endl;
                sprintf(filename, "DEMdemo_settling_%04d.csv", frame);
                DEMSim.WriteSphereFile(out_dir / filename);
                // DEMSim.ShowThreadCollaborationStats();
                frame++;
            }

            DEMSim.DoDynamicsThenSync(settle_frame_time);

            plane_bottom = max_z_finder->GetValue();
            /// here the settling phase starts
            if (!initialization) {
                for (int i = 0; i < (int)(0.4 / settle_frame_time); i++) {
                    DEMSim.DoDynamics(settle_frame_time);
                    sprintf(filename, "DEMdemo_settling_%04d.csv", frame++);
                    DEMSim.WriteSphereFile(out_dir / filename);
                    std::cout << "consolidating for " << i * settle_frame_time << "s " << std::endl;
                }
            }
        }
    }

    DEMSim.DoDynamicsThenSync(0.0);

    float timeStep = step_size * 500.0;
    int numStep = 7.5 / timeStep;
    int timeOut = 0.01 / timeStep;
    int gateMotion = (gateOpen / gateSpeed) / timeStep;
    std::cout << "Frame: " << timeOut << std::endl;
    frame = 0;

    DEMSim.WriteMeshFile(out_dir / meshfile);
    char cnt_filename[100];
    // sprintf(cnt_filename, "Contact_pairs_1_.csv");
    sprintf(meshfile, "DEMdemo_funnel_%04d.vtk", frame);

    bool status = true;
    bool stopGate = true;

    float totalRunTime = 0.0f;

    for (int i = 0; i < numStep; i++) {
        DEMSim.DoDynamics(timeStep);
        totalRunTime += timeStep;

        if (!(i % timeOut) || i == 0) {
            sprintf(filename, "DEMdemo_output_%04d.csv", frame);
            sprintf(meshfile, "DEMdemo_mesh_%04d.vtk", frame);

            DEMSim.WriteMeshFile(out_dir / meshfile);
            DEMSim.WriteSphereFile(out_dir / filename);

            std::cout << "Frame: " << frame << std::endl;
            std::cout << "Elapsed time: " << totalRunTime << std::endl;
            // DEMSim.ShowThreadCollaborationStats();

            frame++;
        }

        if ((i > (timeOut * 2)) && status) {
            DEMSim.DoDynamicsThenSync(0);
            std::cout << "gate is in motion from: " << timeStep * i << " s" << std::endl;
            DEMSim.ChangeFamily(10, 1);
            DEMSim.ChangeFamily(3, 4);
            status = false;
        }

        if ((i >= (timeOut * (2) + gateMotion - 1)) && stopGate) {
            DEMSim.DoDynamicsThenSync(0);
            std::cout << "gate has stopped at: " << timeStep * i << " s" << std::endl;
            DEMSim.ChangeFamily(4, 3);
            stopGate = false;
        }
    }
    std::cout << "The simulated time is: " << totalRunTime << " s" << std::endl;
    DEMSim.ShowTimingStats();
    DEMSim.ClearTimingStats();

    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "DEMdemo_Hopper exiting..." << std::endl;
    return 0;
}