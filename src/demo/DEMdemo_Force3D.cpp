//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A meshed ball hitting a granular bed under gravity. A collection of different
// ball densities and drop heights are tested against loosely packed material.
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

void print(int id, const std::vector<int>& container) {
    std::cout << id << ". ";
    for (const int x : container)
        std::cout << x << ' ';
    std::cout << '\n';
}

void runDEME(std::string dir_output, float friction, float massMultiplier);

int main(int argc, char* argv[]) {
    if (argc != 6) {
        printf("You have entered %d arguments, which is wrong!\n", argc);
        return 0;
    }

    int case_Folder = atoi(argv[1]);  // takes the test ID
    int case_ID = atoi(argv[2]);      // takes the test ID
    std::string tag = argv[3];
    float conctact_friction = atof(argv[4]);  // takes the value
    float massMultiplier = atof(argv[5]);     // takes the value

    std::string out_dir = "/DemoOutput_Force3D_" + tag + "/";
    out_dir += "Test_" + std::to_string(case_Folder) + "/" + std::to_string(case_ID) + "/";

    std::cout << "Running case with friction: " << conctact_friction << ", and Mass multiplier: " << massMultiplier
              << std::endl;
    std::cout << "Dir out is " << out_dir << std::endl;

    runDEME(out_dir, conctact_friction, massMultiplier);

    return 0;
}

void runDEME(std::string dir_output, float frictionMaterial, float massMultiplier) {
    double terrain_rad = 0.02 / 2.;

    DEMSolver DEMSim;
    DEMSim.UseFrictionalHertzianModel();
    DEMSim.SetVerbosity("ERROR");
    DEMSim.SetOutputFormat("CSV");
    DEMSim.SetOutputContent({"ABSV"});
    DEMSim.SetMeshOutputFormat("VTK");
    DEMSim.SetContactOutputContent(DEME_POINT | OWNER | FORCE | CNT_WILDCARD);

    path out_dir = current_path();
    out_dir += dir_output;
    remove_all(out_dir);
    create_directories(out_dir);

    // E, nu, CoR, mu, Crr...
    auto mat_type_terrain =
        DEMSim.LoadMaterial({{"E", 1e7}, {"nu", 0.33}, {"CoR", 0.5}, {"mu", frictionMaterial}, {"Crr", 0.0}});

    float gravityMagnitude = 9.81;
    float step_size = 5 * 1.0e-5 * sqrt(terrain_rad * gravityMagnitude);
    double world_sizeX = 122.0 * terrain_rad;
    double world_sizeZ = 26.7 * terrain_rad;

    DEMSim.InstructBoxDomainDimension({-world_sizeX / 2., world_sizeX / 2.}, {-5 * terrain_rad, 5 * terrain_rad},
                                      {-1 * world_sizeZ, 8 * terrain_rad});
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);

    // Force model to use
    // auto model2D = DEMSim.ReadContactForceModel("ForceModel2D.cu");
    // model2D->SetMustHaveMatProp({"E", "nu", "CoR", "mu", "Crr"});
    // model2D->SetMustPairwiseMatProp({"CoR", "mu", "Crr"});
    // model2D->SetPerContactWildcards({"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z"});

    // creating the two clump templates we need, which are just spheres
    std::vector<std::shared_ptr<DEMClumpTemplate>> templates_terrain;

    templates_terrain.push_back(DEMSim.LoadSphereType(terrain_rad * terrain_rad * terrain_rad * 4 / 3 * 1.0e3 * PI,
                                                      terrain_rad, mat_type_terrain));

    templates_terrain.push_back(DEMSim.LoadSphereType(terrain_rad * terrain_rad * terrain_rad * 4 / 3 * 1.0e3 * PI,
                                                      terrain_rad, mat_type_terrain));

    unsigned int num_particle = 0;

    auto data_xyz = DEMSim.ReadClumpXyzFromCsv("../data/clumps/xyz.csv");
    std::vector<float3> input_xyz;

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_pile_template_type;

    std::cout << data_xyz.size() << std::endl;
    for (unsigned int i = 0; i < data_xyz.size(); i++) {
        char t_name[20];
        sprintf(t_name, "%d", i);

        auto this_type_xyz = data_xyz[std::string(t_name)];
        input_xyz.insert(input_xyz.end(), this_type_xyz.begin(), this_type_xyz.end());

        input_pile_template_type.push_back(templates_terrain[0]);
    }

    std::cout << input_xyz.size() << std::endl;
    std::cout << input_pile_template_type.size() << std::endl;

    auto allParticles = DEMSim.AddClumps(input_pile_template_type, input_xyz);
    allParticles->SetFamily(1);

    auto zeroParticle = DEMSim.AddClumps(templates_terrain[1], make_float3(0, 0, -terrain_rad));
    zeroParticle->SetFamily(3);
    auto driver = DEMSim.Track(zeroParticle);

    float Aext = -gravityMagnitude * (massMultiplier);
    float timeApplication = abs(Aext)>abs(gravityMagnitude)? 2*sqrt(terrain_rad*abs(Aext)): 2*sqrt(terrain_rad*abs(gravityMagnitude));
    std::string Aext_pattern = to_string_with_precision(Aext) + "*erf(t/sqrt(" + to_string_with_precision(timeApplication) + "))";
    std::cout << "applying this force law" << timeApplication << std::endl;
    DEMSim.AddFamilyPrescribedAcc(2, "none", "none", Aext_pattern);

    num_particle += input_xyz.size();

    std::cout << "Total num of particles: " << (int)DEMSim.GetNumClumps() << std::endl;

    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetMaxVelocity(30.);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -gravityMagnitude));

    DEMSim.Initialize();

    float sim_time = 20.0;
    float time_settling=5.0;
    unsigned int fps = 5;
    float frame_time = 1.0 / fps;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;
    double terrain_max_z;

    bool status = true;

    for (float t = 0; t < sim_time; t += frame_time) {
        std::cout << "Output file: " << currframe << std::endl;
        char filename[200], meshfilename[200], cnt_filename[200];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), currframe);
        sprintf(cnt_filename, "%s/Contact_pairs_%04d.csv", out_dir.c_str(), currframe);

        DEMSim.WriteSphereFile(std::string(filename));
        DEMSim.WriteContactFile(std::string(cnt_filename));
        currframe++;

        DEMSim.DoDynamicsThenSync(frame_time);

        if (t > time_settling && status) {
            DEMSim.DoDynamicsThenSync(0);
            DEMSim.ChangeFamily(3, 2);
            std::cout << "Extra mass applied" << std::endl;
            status = false;
            DEMSim.DoDynamicsThenSync(timeApplication);
        }
    }

    DEMSim.ShowTimingStats();

    std::cout << "==============================================================" << std::endl;

    std::cout << "DEMdemo_2DForce exiting..." << std::endl;
}
