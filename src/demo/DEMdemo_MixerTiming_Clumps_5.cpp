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
#include <cmath>
#include <chrono>
#include <filesystem>

using namespace deme;
using namespace std::filesystem;

int main() {
    float granular_rad = 0.005;
    unsigned int num_particles = 0;
    double world_size = 1;
    double CDFreq = 25.1;
    double pi = 3.14159;
    float step_size = 5e-7;
    size_t n_steps = 5e5;
    int test_num = 0;

    while (num_particles < 2e8) {
        DEMSolver DEMSim;
        DEMSim.SetVerbosity(ERROR);
        DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
        DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
        DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
        DEMSim.SetCollectAccRightAfterForceCalc(true);
        DEMSim.SetNoForceRecord();

        // E, nu, CoR, mu, Crr...
        auto mat_type_mixer = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.2}, {"mu", 0.5}, {"Crr", 0.0}});
        auto mat_type_granular =
            DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.2}, {"mu", 0.5}, {"Crr", 0.0}});

        const float chamber_height = world_size / 3.;
        const float fill_height = chamber_height;
        const float chamber_bottom = -world_size / 2.;
        const float fill_bottom = chamber_bottom + chamber_height;

        DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size);
        DEMSim.InstructBoxDomainBoundingBC("all", mat_type_granular);

        // Now add a cylinderical boundary
        auto walls = DEMSim.AddExternalObject();
        walls->AddCylinder(make_float3(0), make_float3(0, 0, 1), world_size / 2., mat_type_mixer, 0);

        auto mixer =
            DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/internal_mixer.obj").string(), mat_type_mixer);
        std::cout << "Total num of triangles: " << mixer->GetNumTriangles() << std::endl;
        mixer->Scale(make_float3(world_size / 2, world_size / 2, chamber_height));
        mixer->SetFamily(10);
        // Define the prescribed motion of mixer
        DEMSim.SetFamilyPrescribedAngVel(10, "0", "0", "1 * " + to_string_with_precision(pi / world_size));

        DEMClumpTemplate shape_template;
        shape_template.ReadComponentFromFile((GET_DATA_PATH() / "clumps/spiky_sphere.csv").string());
        // Calculate its mass and MOI
        shape_template.mass = 2.6e3 * 5.5886717;  // in kg or g
        shape_template.MOI = make_float3(1.8327927, 2.1580013, 0.77010059) * 2.6e3;
        shape_template.materials = std::vector<std::shared_ptr<DEMMaterial>>(shape_template.nComp, mat_type_granular);
        shape_template.Scale(granular_rad);
        auto template_granular = DEMSim.LoadClumpType(shape_template);

        // Track the mixer
        auto mixer_tracker = DEMSim.Track(mixer);

        // Sampler to use
        HCPSampler sampler(3.f * granular_rad);
        float3 fill_center = make_float3(0, 0, fill_bottom + fill_height / 2);
        const float fill_radius = world_size / 2. - 2. * granular_rad;
        auto input_xyz = sampler.SampleCylinderZ(fill_center, fill_radius, fill_height / 2);
        DEMSim.AddClumps(template_granular, input_xyz);
        num_particles = input_xyz.size();
        std::cout << "Particle size: " << granular_rad << std::endl;
        std::cout << "Time step size: " << step_size << std::endl;
        std::cout << "Total num of particles: " << num_particles << std::endl;
        std::cout << "Total num of spheres: " << num_particles * shape_template.nComp << std::endl;
        std::cout << "World size: " << world_size << std::endl;

        DEMSim.SetInitTimeStep(step_size);
        DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
        // If you want to use a large UpdateFreq then you have to expand spheres to ensure safety
        DEMSim.SetCDUpdateFreq((unsigned int)CDFreq);
        DEMSim.SetInitBinSize(4 * granular_rad);
        DEMSim.SetCDNumStepsMaxDriftMultipleOfAvg(1.1);
        DEMSim.SetCDNumStepsMaxDriftAheadOfAvg(4);
        DEMSim.SetErrorOutAvgContacts(100);
        DEMSim.SetErrorOutVelocity(15.);
        DEMSim.Initialize();

        path out_dir = current_path();
        out_dir += "/DemoOutput_Mixer_5clump";
        create_directory(out_dir);

        float sim_end = step_size * n_steps;
        unsigned int fps = 20;

        mixer_tracker->SetPos(make_float3(0, 0, chamber_bottom + chamber_height / 2.0));
        DEMSim.DoDynamicsThenSync(.75);
        DEMSim.ClearThreadCollaborationStats();
        DEMSim.ClearTimingStats();
        char filename[200], meshfilename[200];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), test_num);
        sprintf(meshfilename, "%s/DEMdemo_mesh_%04d.vtk", out_dir.c_str(), test_num);
        DEMSim.WriteSphereFile(std::string(filename));
        DEMSim.WriteMeshFile(std::string(meshfilename));

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

        DEMSim.DoDynamicsThenSync(sim_end);

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

        DEMSim.ShowThreadCollaborationStats();
        DEMSim.ShowTimingStats();

        std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << (time_sec.count()) / sim_end / (1e-5 / step_size)
                  << " seconds (wall time) to finish 1 second's simulation" << std::endl;
        // Compensate for smaller ts

        granular_rad *= std::pow(0.75, 1. / 3.);
        // step_size *= std::pow(0.75, 1. / 3.);
        // world_size *= std::pow(2., 1. / 3.);
        // CDFreq *= std::pow(0.95, 1. / 3.);

        test_num++;
    }

    return 0;
}
