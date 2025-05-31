#  Copyright (c) 2021, SBEL GPU Development Team
#  Copyright (c) 2021, University of Wisconsin - Madison
#
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# This demo reproduces the Game of Life in DEME simulator, to showcase the
# flexibility of its APIs.
# =============================================================================

import DEME

import numpy as np
import os
import random

if __name__ == "__main__":
    out_dir = "DemoOutput_GameOfLife/"
    out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    DEMSim = DEME.DEMSolver()
    DEMSim.SetVerbosity("INFO")
    DEMSim.SetOutputFormat("CSV")
    DEMSim.SetOutputContent(["XYZ"])
    DEMSim.SetMeshOutputFormat("VTK")
    DEMSim.EnsureKernelErrMsgLineNum(True)

    grid_size = 1.0
    r = (1.45 * grid_size) / 2.0
    world_size = 2500.0
    n_init = 10000

    # Material is formaility... you can opt not to set it at all, it works the same
    mat_type_1 = DEMSim.LoadMaterial({"junk": 1.0})

    template_sphere = DEMSim.LoadSphereType(1.0, r, mat_type_1)

    input_template_num = []
    family_code = []

    # Generate a flat layer of particles in this domain
    input_xyz = DEME.DEMBoxGridSampler(
        [0, 0, 0], [world_size / 2.0, world_size / 2.0, 0.001], grid_size)
    num_cells = len(input_xyz)
    # Number cells per row
    num_row = round(np.sqrt(num_cells))

    # Use code 10 to represent dead cells, and don't output dead cells
    family_code += [10] * len(input_xyz)
    DEMSim.DisableFamilyOutput(10)

    # Init patterns adding...
    for i in range(n_init):
        offset = 0

        while (offset < 5 * num_row) or (offset > (num_row - 5) * num_row):
            offset = random.randint(0, num_cells - 1)
        # Mark some of them to be alive (family 0). 8 cells in a row seems to be a good start.
        for j in range(8):
            family_code[offset + j] = 0

    for i in range(n_init):
        offset = 0

        while (offset < 5 * num_row) or (offset > (num_row - 5) * num_row):
            offset = random.randint(0, num_cells-1)
        # Another interesting one is the spaceship pattern
        family_code[offset] = 0
        family_code[offset + 1] = 0
        family_code[offset + 2] = 0
        family_code[offset + 3] = 0
        family_code[offset + num_row - 1] = 0
        family_code[offset + num_row + 3] = 0
        family_code[offset + 2 * num_row + 3] = 0
        family_code[offset + 3 * num_row - 1] = 0
        family_code[offset + 3 * num_row + 2] = 0

    input_template_num += [template_sphere] * len(input_xyz)

    # All objects in the game are fixed
    DEMSim.SetFamilyFixed(0)
    DEMSim.SetFamilyFixed(10)

    # The force model just serves as a contact number register here. So let's say if 2 spheres are in contact, the
    # force is constant 1, so you have 4+ neighbours if you feel 4+ force, and 2- if you got 2- neighbours.
    # To avoid force cancelling out, we let alive cells always get positive force, and dead cells negative.
    model = DEMSim.DefineContactForceModel(
        "if (AOwnerFamily == 0) force = make_float3(0, 0, 1); else force = make_float3(0, 0, -1);")
    DEMSim.DisableContactBetweenFamilies(0, 0)
    DEMSim.DisableContactBetweenFamilies(10, 10)

    # The rule for changing family numbers is simple: you die, move to 10; live, move to 0. And then, family 10 will
    # not be outputted.
    # Dead to alive: if you have 3 alive neighbours
    DEMSim.ChangeFamilyWhen(
        10, 0, "float my_neighbours = length(acc * mass); return (my_neighbours > 2.9) && (my_neighbours < 3.1);")
    # Alive to dead, if less than 2 alive neighbours, or more than 3 alive neighbours (more than 6 dead neighbors, or
    # less than 5 dead neighbors)
    DEMSim.ChangeFamilyWhen(
        0, 10, "float my_neighbours = length(acc * mass); return (my_neighbours < 4.9) || (my_neighbours > 6.1);")

    particles = DEMSim.AddClumps(input_template_num, input_xyz)
    particles.SetFamilies(family_code)
    # The game board is somewhat large so we have to define it, because the solver defaults the world size to be 1000.
    DEMSim.InstructBoxDomainDimension(
        world_size * 1.1, world_size * 1.1, world_size * 1.1)
    DEMSim.SetInitBinSize(world_size / 100.)

    DEMSim.SetInitTimeStep(1.)
    DEMSim.SetCDUpdateFreq(0)
    # Must disable this if you want to run dT and kT synchronizely, or the solver will automatically find a non-zero
    # update frequency that it sees fit to run it in an async fashion.
    DEMSim.DisableAdaptiveUpdateFreq()
    DEMSim.Initialize()

    for i in range(3000):
        print(f"Outputting frame: {i}", flush=True)
        filename = os.path.join(out_dir, f"DEMdemo_output_{i:04d}.csv")
        DEMSim.WriteSphereFile(filename)
        print(
            f"Average contacts each sphere has: {DEMSim.GetAvgSphContacts()}", flush=True)

        DEMSim.DoDynamicsThenSync(1.)

    DEMSim.ShowMemStats()

    print("DEMdemo_GameOfLife exiting...")
