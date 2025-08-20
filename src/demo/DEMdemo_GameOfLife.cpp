//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// This demo reproduces the Game of Life in DEME simulator, to showcase the
// flexibility of its APIs.
// =============================================================================

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

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    // Output as CSV so no post-processing is needed
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ);
    DEMSim.EnsureKernelErrMsgLineNum();

    float grid_size = 1.0;
    float r = (1.45 * grid_size) / 2.0;
    float world_size = 2500.0;
    unsigned int n_init = 10000;

    // Material is formaility... you can opt not to set it at all, it works the same
    auto mat_type_1 = DEMSim.LoadMaterial({{"junk", 1.0}});

    auto template_sphere = DEMSim.LoadSphereType(1.0, r, mat_type_1);

    std::vector<std::shared_ptr<DEMClumpTemplate>> input_template_num;
    std::vector<unsigned int> family_code;

    // Generate a flat layer of particles in this domain
    auto input_xyz =
        DEMBoxGridSampler(make_float3(0, 0, 0), make_float3(world_size / 2.0, world_size / 2.0, 0.001), grid_size);
    unsigned int num_cells = input_xyz.size();
    // Number cells per row
    unsigned int num_row = std::round(std::sqrt(num_cells));

    // Use code 10 to represent dead cells, and don't output dead cells
    family_code.insert(family_code.end(), input_xyz.size(), 10);
    DEMSim.DisableFamilyOutput(10);

    // Init patterns adding...
    {
        for (unsigned int i = 0; i < n_init; i++) {
            unsigned int offset = 0;

            while (offset < 5 * num_row || offset > (num_row - 5) * num_row) {
                offset = rand() % num_cells;
            }
            // Mark some of them to be alive (family 0). 8 cells in a row seems to be a good start.
            for (int j = 0; j < 8; j++) {
                family_code.at(offset + j) = 0;
            }
        }

        for (unsigned int i = 0; i < n_init; i++) {
            unsigned int offset = 0;

            while (offset < 5 * num_row || offset > (num_row - 5) * num_row) {
                offset = rand() % num_cells;
            }
            // Another interesting one is the spaceship pattern
            family_code.at(offset) = 0;
            family_code.at(offset + 1) = 0;
            family_code.at(offset + 2) = 0;
            family_code.at(offset + 3) = 0;
            family_code.at(offset + num_row - 1) = 0;
            family_code.at(offset + num_row + 3) = 0;
            family_code.at(offset + 2 * num_row + 3) = 0;
            family_code.at(offset + 3 * num_row - 1) = 0;
            family_code.at(offset + 3 * num_row + 2) = 0;
        }
    }

    input_template_num.insert(input_template_num.end(), input_xyz.size(), template_sphere);

    // All objects in the game are fixed
    DEMSim.SetFamilyFixed(0);
    DEMSim.SetFamilyFixed(10);

    // The force model just serves as a contact number register here. So let's say if 2 spheres are in contact, the
    // force is constant 1, so you have 4+ neighbours if you feel 4+ force, and 2- if you got 2- neighbours.
    // To avoid force cancelling out, we let alive cells always get positive force, and dead cells negative.
    DEMSim.DefineContactForceModel(
        "if (AOwnerFamily == 0) force = make_float3(0, 0, 1); else force = make_float3(0, 0, -1);");
    DEMSim.DisableContactBetweenFamilies(0, 0);
    DEMSim.DisableContactBetweenFamilies(10, 10);

    // The rule for changing family numbers is simple: you die, move to 10; live, move to 0. And then, family 10 will
    // not be outputted.
    // Dead to alive: if you have 3 alive neighbours
    DEMSim.ChangeFamilyWhen(
        10, 0, "float my_neighbours = length(acc * mass); return (my_neighbours > 2.9) && (my_neighbours < 3.1);");
    // Alive to dead, if less than 2 alive neighbours, or more than 3 alive neighbours (more than 6 dead neighbors, or
    // less than 5 dead neighbors)
    DEMSim.ChangeFamilyWhen(
        0, 10, "float my_neighbours = length(acc * mass); return (my_neighbours < 4.9) || (my_neighbours > 6.1);");

    auto particles = DEMSim.AddClumps(input_template_num, input_xyz);
    particles->SetFamilies(family_code);
    // The game board is somewhat large so we have to define it, because the solver defaults the world size to be 1000.
    DEMSim.InstructBoxDomainDimension(world_size * 1.1, world_size * 1.1, world_size * 1.1);
    // You usually don't have to worry about initial bin size. In very rare cases, init bin size is so bad that auto bin
    // size adaption is effectless, and you should notice in that case kT runs extremely slow. Then in that case setting
    // init bin size may save the simulation.
    // DEMSim.SetInitBinSize(world_size / 100.);

    DEMSim.SetInitTimeStep(1.);
    DEMSim.SetCDUpdateFreq(0);
    // Must disable this if you want to run dT and kT synchronizely, or the solver will automatically find a non-zero
    // update frequency that it sees fit to run it in an async fashion.
    DEMSim.DisableAdaptiveUpdateFreq();

    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir /= "DemoOutput_GameOfLife";
    create_directory(out_dir);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 3000; i++) {
        char filename[100];
        sprintf(filename, "DEMdemo_output_%04d.csv", i);
        DEMSim.WriteSphereFile(out_dir / filename);
        std::cout << "Frame: " << i << std::endl;
        std::cout << "Average contacts each sphere has: " << DEMSim.GetAvgSphContacts() << std::endl;

        DEMSim.DoDynamicsThenSync(1.);
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds" << std::endl;

    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "DEMdemo_GameOfLife exiting..." << std::endl;
    return 0;
}
