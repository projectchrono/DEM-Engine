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
    DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ | OUTPUT_CONTENT::VEL | OUTPUT_CONTENT::ANG_VEL);
    DEMSim.EnsureKernelErrMsgLineNum();

    srand(7001);
    DEMSim.SetCollectAccRightAfterForceCalc(true);
    DEMSim.SetErrorOutAvgContacts(80);

    // DEMSim.SetExpandSafetyAdder(0.5);

    // Scale factor
    float scaling = 1.f;

    // total number of random clump templates to generate

    double radius = 0.0040 / 2.0;
    double length = 0.00850;
    int n_sphere = 5;

    double density = 1000;


    int num_template = 5;

    float plane_bottom = 0.02f * scaling;
    float funnel_bottom = 0.02f * scaling;

    double gateOpen = 0.30;
    double gateSpeed = -3.5;
    double hopperW = 0.10;

    path out_dir = current_path();
    out_dir += "/DemoOutput_FreeFalling/";
    out_dir += "";

    auto mat_type_bottom = DEMSim.LoadMaterial({{"E", 10e9}, {"nu", 0.3}, {"CoR", 0.60}});
    auto mat_type_flume = DEMSim.LoadMaterial({{"E", 10e9}, {"nu", 0.3}, {"CoR", 0.60}});
    auto mat_type_walls = DEMSim.LoadMaterial({{"E", 10e9}, {"nu", 0.3}, {"CoR", 0.60}});

    auto mat_type_particles =
        DEMSim.LoadMaterial({{"E", 1.0e7}, {"nu", 0.35}, {"CoR", 0.50}, {"mu", 0.70}, {"Crr", 0.01}});

    DEMSim.SetMaterialPropertyPair("CoR", mat_type_walls, mat_type_particles, 0.5);
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_walls, mat_type_particles, 0.02);

    DEMSim.SetMaterialPropertyPair("CoR", mat_type_flume, mat_type_particles, 0.7);   // it is supposed to be
    DEMSim.SetMaterialPropertyPair("Crr", mat_type_flume, mat_type_particles, 0.05);  // plexiglass
    DEMSim.SetMaterialPropertyPair("mu", mat_type_flume, mat_type_particles, 0.30);

    // Make ready for simulation
    float step_size = 5.0e-6;
    DEMSim.InstructBoxDomainDimension({-0.30, 0.30}, {-0.10, 0.10}, {-4.00, 1.0});
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_walls);
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    // Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    // wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    // happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(25.);
    DEMSim.SetInitBinSize(radius * 5);



    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
    auto min_z_finder = DEMSim.CreateInspector("clump_min_z");
    auto total_mass_finder = DEMSim.CreateInspector("clump_mass");
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");

    // Make an array to store these generated clump templates
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_types;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(radius, radius * 0.00);
    double maxRadius = 0;

    for (int i = 0; i < num_template; i++) {
        std::vector<float> radii;
        std::vector<float3> relPos;
        std::vector<std::shared_ptr<DEMMaterial>> mat;

        double radiusMed = distribution(generator);
        radiusMed = radius;
        double eccentricity = 0.0 / 8.0 * radiusMed;

        float init = -length / 2.0 + radiusMed;
        float trail = length / 2.0 - radiusMed;
        float incrR = (trail - init) / (n_sphere - 1);
        float3 tmp;

        for (int j = 0; j < n_sphere; j++) {
            radii.push_back(radiusMed);

            tmp.x = init + (j * incrR) + eccentricity;
            tmp.y = 0;
            tmp.z = 0;
            relPos.push_back(tmp);
            mat.push_back(mat_type_particles);
        }

        float mass = PI * radiusMed * radiusMed * length * density;
        float Ixx = 1.f / 2.f * mass * radiusMed * radiusMed;
        float Iyy = Ixx / 2.0 + 1.0 / 12.0 * mass * length * length;
        float3 MOI = make_float3(Ixx, Iyy, Iyy);
        std::cout << mass << " chosen moi ..." << radiusMed / radius << std::endl;

        maxRadius = (radiusMed > maxRadius) ? radiusMed : maxRadius;
        auto clump_ptr = DEMSim.LoadClumpType(mass, MOI, radii, relPos, mat_type_particles);
        // clump_ptr->AssignName("fsfs");
        clump_types.push_back(clump_ptr);
    }

    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    float settle_frame_time = 0.005;

    remove_all(out_dir);
    create_directories(out_dir);

    char filename[200], meshfile[200];

    float shift_xyz = 1.0 * (length)*1.1;
    float x = 0;
    float y = 0;

    float z = shift_xyz / 2;  // by default we create beads at 0
    double emitterZ = 0.60;
    unsigned int actualTotalSpheres = 0;

    DEMSim.Initialize();

    int frame = 0;
    bool generate = true;
    bool initialization = true;
    float timeTotal = 0.0;
    double consolidation = true;

    sprintf(meshfile, "%s/DEMdemo_funnel_%04d.vtk", out_dir.c_str(), frame);
    DEMSim.WriteMeshFile(std::string(meshfile));

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;
    std::vector<float3> input_pile_xyz;
    PDSampler sampler(shift_xyz);

    float sizeZ = 0.95;
    float sizeX = 0.20;
    

    float3 center_xyz = make_float3(0, 0, z);
    float3 size_xyz = make_float3((sizeX - shift_xyz) / 2.0, (0.04 - shift_xyz) / 2.0, sizeZ / 2.0);

    std::cout << "level of particles position ... " << center_xyz.z << std::endl;

    auto heap_particles_xyz = sampler.SampleBox(center_xyz, size_xyz);
    unsigned int num_clumps = heap_particles_xyz.size();
    std::cout << "number of particles at this level ... " << num_clumps << std::endl;

    for (unsigned int i = actualTotalSpheres; i < actualTotalSpheres + num_clumps; i++) {
        input_pile_template_type.push_back(clump_types.at(i % num_template));
    }

    input_pile_xyz.insert(input_pile_xyz.end(), heap_particles_xyz.begin(), heap_particles_xyz.end());

    auto the_pile = DEMSim.AddClumps(input_pile_template_type, input_pile_xyz);
    the_pile->SetVel(make_float3(-0.00, 0.0, -0.00));
    the_pile->SetFamily(100);

    DEMSim.UpdateClumps();

    std::cout << "Total num of particles: " << (int)DEMSim.GetNumClumps() << std::endl;
    actualTotalSpheres = (int)DEMSim.GetNumClumps();
    // Generate initial clumps for piling

    DEMSim.DoDynamicsThenSync(0.0);


    float timeStep = step_size * 500.0;
    int numStep = 7.0 / timeStep;
    int timeOut = 0.01 / timeStep;
    int gateMotion = (gateOpen / gateSpeed) / timeStep;
    std::cout << "Frame: " << timeOut << std::endl;
    frame = 0;

    DEMSim.WriteMeshFile(std::string(meshfile));
    char cnt_filename[200];
    // sprintf(cnt_filename, "%s/Contact_pairs_1_.csv", out_dir.c_str());
    sprintf(meshfile, "%s/DEMdemo_funnel_%04d.vtk", out_dir.c_str(), frame);

    bool status = true;
    bool stopGate = true;

    float totalRunTime = 0.0f;

    for (int i = 0; i < numStep; i++) {
        DEMSim.DoDynamics(timeStep);
        totalRunTime += timeStep;

        if (!(i % timeOut) || i == 0) {
            sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), frame);
            sprintf(meshfile, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), frame);

            DEMSim.WriteMeshFile(std::string(meshfile));
            DEMSim.WriteSphereFile(std::string(filename));

            std::cout << "Frame: " << frame << std::endl;
            std::cout << "Elapsed time: " << totalRunTime << std::endl;
            // DEMSim.ShowThreadCollaborationStats();

            frame++;
        }
       
    }
    std::cout << "The simulated time is: " << totalRunTime << " s" << std::endl;
    DEMSim.ShowTimingStats();
    DEMSim.ClearTimingStats();

    std::cout << "DEMdemo_Repose exiting..." << std::endl;
    return 0;
}
