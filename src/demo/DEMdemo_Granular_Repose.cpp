//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A repose angle test. Particles flow through a mesh-represented funnel and form
// a pile that has an apparent angle.
// =============================================================================
// Created by btagliafierro 07/04/2023

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
    DEMSim.EnsureKernelErrMsgLineNum();

    srand(7001);
    DEMSim.SetCollectAccRightAfterForceCalc(true);
    //DEMSim.SetExpandSafetyAdder(0.1);
    // Scale factor
    float scaling = 1.f;

    // total number of random clump templates to generate

    double radius = 0.003300 * scaling / 2.0;
    double density = 1410.0;
  

    int num_template = 10000;

    float plane_bottom = -0.20f * scaling;
    float funnel_bottom =0.050 * scaling;
    float funnel_outlet = 0.020f * scaling;
    float funnel_slope = 1.0 / 4.0 * 3.14;

    double tilt = 3.141592 / 6.0;

    auto mat_type_funnel = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.60}});

    auto mat_type_walls = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.90}, {"mu", 0.04}, {"Crr", 0.04}});
    auto mat_type_particles =
        DEMSim.LoadMaterial({{"E", 2.7e9}, {"nu", 0.35}, {"CoR", 0.83}, {"mu", 0.50}, {"Crr", 0.08}});

    DEMSim.SetMaterialPropertyPair("CoR", mat_type_walls, mat_type_particles, 0.50);
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_walls, mat_type_particles, 0.50);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_walls, mat_type_particles, 0.80);

    DEMSim.SetMaterialPropertyPair("CoR", mat_type_funnel, mat_type_particles, 0.7);    // it is supposed to be
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_funnel, mat_type_particles, 0.01);  // bakelite
    DEMSim.SetMaterialPropertyPair("mu", mat_type_funnel, mat_type_particles, 0.01);
    

    /*
    // First create clump type 0 for representing the ground
    float ground_sp_r = 0.02;
    auto template_ground = DEMSim.LoadSphereType(0.5, ground_sp_r, mat_type_walls);
    */
    // Make ready for simulation
    float step_size = 1e-6;
    
    DEMSim.InstructBoxDomainDimension({-0.35, 0.35}, {-0.35, 0.35}, {plane_bottom, 1.00});
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_walls);
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    // wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    // happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(25.);
    DEMSim.SetInitBinSize(0.9*radius* 6);

    // Loaded meshes are by-default fixed
    auto funnel = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/funnel.obj"), mat_type_funnel);
    funnel->Scale(0.003 * scaling);


    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    auto min_z_finder = DEMSim.CreateInspector("clump_min_z");
    auto total_mass_finder = DEMSim.CreateInspector("clump_mass");
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");



// Make an array to store these generated clump templates
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_types;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(radius, radius * 0.050);
    float maxRadius = 0.0;

    for (int i = 0; i < num_template; i++) {

        std::vector<float> radii;
        std::vector<float3> relPos;
        std::vector<std::shared_ptr<DEMMaterial>> mat;


        double radiusMax = distribution(generator);

        double radiusMin = 3.0 / 4.0 * radiusMax;
        double eccentricity = 1.0 / 4.0 * radiusMax;

        radii.push_back(radiusMin);
        float3 tmp;
        tmp.x = 0;
        tmp.y = 0;
        tmp.z = 0;
        relPos.push_back(tmp);
        mat.push_back(mat_type_particles);

        double x = 1.0 * eccentricity;
        double y = 0.0;
        double z = 0.0;
        tmp.x = x;
        tmp.y = y;
        tmp.z = z;
        relPos.push_back(tmp);
        mat.push_back(mat_type_particles);

        radii.push_back(radiusMin);

        double c = radiusMin;  // smaller dim of the ellipse
        double b = radiusMin;
        double a = radiusMin + 0.50*eccentricity;

        float mass = 4.0 / 3.0 * 3.141592 * a * b * c * density;
        float3 MOI = make_float3(   1.f / 5.f * mass * (b * b + c * c),
                                    1.f / 5.f * mass * (a * a + c * c),
                                    1.f / 5.f * mass * (b * b + a * a)
                                );
        std::cout << x << " chosen moi ..." << a / radius << std::endl;

        maxRadius = (radiusMax > maxRadius) ? radiusMax : maxRadius;
        
        auto clump_ptr = DEMSim.LoadClumpType(mass, MOI, radii, relPos, mat_type_particles);
        // clump_ptr->AssignName("fsfs");
        clump_types.push_back(clump_ptr);
    }

   unsigned int currframe = 0;
    unsigned int curr_step = 0;
    float settle_frame_time = 0.20;
    // Track the projectile

    float shift_xyz = 1.00 * maxRadius * 2;
    float x = 0;
    float y = 0;
    float z = plane_bottom + shift_xyz;  // by default we create beads at 0

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;
    std::vector<float3> input_pile_xyz;
    PDSampler sampler(shift_xyz);

    while (z < funnel_bottom) {
        float3 center_xyz = make_float3(x, y, z);

        std::cout << "level of particles position ... " << center_xyz.z << std::endl;
        std::cout << "pile radius  ... " << funnel_outlet << std::endl;
        auto heap_particles_xyz = sampler.SampleCylinderZ(center_xyz, funnel_outlet, 0);
        unsigned int num_clumps = heap_particles_xyz.size();
        std::cout << "number of particles at this level ... " << num_clumps << std::endl;

        for (unsigned int i = 0; i < num_clumps; i++) {
            input_pile_template_type.push_back(clump_types.at(i % num_template));
        }

        input_pile_xyz.insert(input_pile_xyz.end(), heap_particles_xyz.begin(), heap_particles_xyz.end());
        z += shift_xyz;
    }

    for (unsigned int i = 0; i < 45; i++) {
        float3 center_xyz = make_float3(x, y, z);

        std::cout << "level of particles position ... " << center_xyz.z << std::endl;
        std::cout << "pile radius  ... " << funnel_outlet + (z - funnel_bottom) * std::tan(funnel_slope) << std::endl;
        auto heap_particles_xyz =
            sampler.SampleCylinderZ(center_xyz, funnel_outlet + (z - funnel_bottom) * std::tan(funnel_slope), 0);
        unsigned int num_clumps = heap_particles_xyz.size();
        std::cout << "number of particles at this level ... " << num_clumps << std::endl;

        for (unsigned int i = 0; i < num_clumps; i++) {
            input_pile_template_type.push_back(clump_types.at(i % num_template));
        }
        input_pile_xyz.insert(input_pile_xyz.end(), heap_particles_xyz.begin(), heap_particles_xyz.end());

        z += shift_xyz;
    }

    auto the_pile = DEMSim.AddClumps(input_pile_template_type, input_pile_xyz);

    std::cout << "Total num of particles: " << (int)DEMSim.GetNumClumps() << std::endl;
    // Generate initial clumps for piling

    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir += "/DemoOutput_Granular_Repose";

    remove_all(out_dir);
    create_directory(out_dir);

    double timeStep = 5e-5;
    int numStep = 7. / timeStep;
    int timeOut = 0.01 / timeStep;

    int frame = 0;
    char filename[200], meshfile[200];
    sprintf(meshfile, "%s/DEMdemo_funnel_%04d.vtk", out_dir.c_str(), frame);
    DEMSim.WriteMeshFile(std::string(meshfile));
    char cnt_filename[200];
    sprintf(cnt_filename, "%s/Contact_pairs_1_.csv", out_dir.c_str());

    for (int i = 0; i < numStep; i++) {
        

        if (!(i % timeOut) || i == 0) {
            sprintf(filename, "%s/DEMdemo_output_1_%04d.csv", out_dir.c_str(), frame);
            // sprintf(cnt_filename, "%s/Contact_pairs_1_%04d.csv", out_dir.c_str(), frame);

            DEMSim.WriteSphereFile(std::string(filename));
            DEMSim.WriteContactFile(std::string(cnt_filename));
            std::cout << "Frame: " << frame << std::endl;
            std::cout << "Elapsed time: " << timeStep * i << std::endl;
            // DEMSim.ShowThreadCollaborationStats();
            frame++;
        }
        DEMSim.DoDynamics(timeStep);
        // if (!(i % ((int)timeOut / 10)) && i>0 ) {
        //     std::cout << "Elapsed time: " << timeStep * i << std::endl;
        //  DEMSim.ShowThreadCollaborationStats();
        //}
        //        if (!(i % 4*timeOut) && i>0) {
        //            DEMSim.ClearCache();
        //           auto the_pile = DEMSim.AddClumps(input_pile_template_type, input_pile_xyz);
        //           DEMSim.UpdateClumps();
        //       }
    }

    DEMSim.ShowTimingStats();
    DEMSim.ClearTimingStats();

    std::cout << "DEMdemo_Repose exiting..." << std::endl;
    return 0;
}
