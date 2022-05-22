//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

// =============================================================================
// This demo reproduces the Game of Life in SGPS DEM simulator, to showcase the flexibility of its APIs.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <DEM/ApiSystem.h>
#include <DEM/HostSideHelpers.cpp>

#include <cstdio>
#include <chrono>

using namespace sgps;
using namespace std::filesystem;

int main() {
    DEMSolver DEM_sim;
    DEM_sim.SetVerbosity(INFO);
    DEM_sim.UseFrictionlessModel(true);

    srand(777);

    // Formality
    auto mat_type_1 = DEM_sim.LoadMaterialType(1e8, 0.3, 0.2);

    float grid_size = 1.0;
    float r = (1.45 * grid_size) / 2.0;
    float world_size = 100.0;

    auto template_sphere = DEM_sim.LoadClumpSimpleSphere(1.0, r, mat_type_1);

    std::vector<unsigned int> input_template_num;
    std::vector<unsigned int> family_code;

    // Generate a flat layer of particles in this periodic domain
    auto input_xyz =
        DEMBoxGridSampler(make_float3(0, 0, 0), make_float3(world_size / 2.0, world_size / 2.0, 0.001), grid_size);
    unsigned int num_cells = input_xyz.size();

    // Use code DEM_RESERVED_FAMILY_NUM to represent dead cells
    family_code.insert(family_code.end(), input_xyz.size(), DEM_RESERVED_FAMILY_NUM);
    // But mark some of them to be alive (family 0). 8 cells in a row seems to be a good start.
    unsigned int offset = num_cells / 2;
    for (int i = 0; i < 8; i++) {
        family_code.at(offset + i) = 0;
    }

    input_template_num.insert(input_template_num.end(), input_xyz.size(), template_sphere);

    // All objects in the game are fixed
    DEM_sim.SetFamilyFixed(0);
    DEM_sim.SetFamilyFixed(DEM_RESERVED_FAMILY_NUM);

    // The force model just serves as a contact number register here. So let's say if 2 spheres are in contact, the
    // force is constant 1, so you have 4+ neighbours if you feel 4+ force, and 2- if you got 2- neighbours.
    DEM_sim.DefineContactForceModel(
        "if (AOwnerFamily == 0) force = make_float3(0, 0, 1); else force = make_float3(0, 0, -1);");
    // To avoid force canceling out, we let alive cells always get positive force, and dead cells negative
    DEM_sim.DisableContactBetweenFamilies(0, 0);
    DEM_sim.DisableContactBetweenFamilies(DEM_RESERVED_FAMILY_NUM, DEM_RESERVED_FAMILY_NUM);

    // The rule for changing family numbers is simple: you die, move to DEM_RESERVED_FAMILY_NUM; live, move to 0. And
    // then, family DEM_RESERVED_FAMILY_NUM will not be outputted. Dead to alive: if you have 3 alive neighbours
    DEM_sim.ChangeFamilyWhen(
        DEM_RESERVED_FAMILY_NUM, 0,
        "float my_neighbours = length(acc * mass); return (my_neighbours > 2.9) && (my_neighbours < 3.1);");
    // Alive to dead, if less than 2 alive neighbours, or more than 3 alive neighbours (more than 6 dead neighbors, or
    // less than 5 dead neighbors)
    DEM_sim.ChangeFamilyWhen(
        0, DEM_RESERVED_FAMILY_NUM,
        "float my_neighbours = length(acc * mass); return (my_neighbours < 4.9) || (my_neighbours > 6.1);");

    DEM_sim.AddClumps(input_template_num, input_xyz);
    DEM_sim.SetClumpFamily(family_code);
    DEM_sim.InstructBoxDomainNumVoxel(22, 22, 20, (world_size + grid_size) / std::pow(2, 16) / std::pow(2, 22));

    DEM_sim.CenterCoordSys();
    DEM_sim.SetTimeStepSize(1.);
    DEM_sim.SetCDUpdateFreq(0);

    DEM_sim.Initialize();

    path out_dir = current_path();
    out_dir += "/DEMdemo_GameOfLife";
    create_directory(out_dir);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        char filename[100];
        sprintf(filename, "%s/DEMdemo_output_%04d.csv", out_dir.c_str(), i);
        DEM_sim.WriteFileAsSpheres(std::string(filename));
        std::cout << "Frame: " << i << std::endl;

        DEM_sim.LaunchThreads(1.);
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds" << std::endl;

    std::cout << "DEMdemo_GameOfLife exiting..." << std::endl;
    // TODO: add end-game report APIs
    return 0;
}
