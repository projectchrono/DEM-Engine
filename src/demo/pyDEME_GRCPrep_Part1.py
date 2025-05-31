#  Copyright (c) 2021, SBEL GPU Development Team
#  Copyright (c) 2021, University of Wisconsin - Madison
#
# SPDX-License-Identifier: BSD-3-Clause

import DEME
from DEME import HCPSampler

import numpy as np
import os

# =============================================================================
# In GRCPrep demo series, we try to prepare a sample of the GRC simulant, which
# are supposed to be used for extraterrestrial rover mobility simulations. It is
# made of particles of various sizes and shapes following a certain distribution.
# In Part1, it creates several batches of clumps and let them settle at the bottom
# of the domain.
# =============================================================================

if __name__ == "__main__":
    out_dir = "DemoOutput_GRCPrep_Part1/"
    out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    DEMSim = DEME.DEMSolver()
    DEMSim.SetVerbosity("INFO")
    DEMSim.SetOutputFormat("CSV")
    # XYZ is default so this doesn't do anything
    DEMSim.SetOutputContent(["XYZ"])

    # Define materials
    mat_type_terrain = DEMSim.LoadMaterial(
        {"E": 1e9, "nu": 0.3, "CoR": 0.3, "mu": 0.5})
    mat_type_wheel = DEMSim.LoadMaterial(
        {"E": 1e9, "nu": 0.3, "CoR": 0.3, "mu": 0.5})

    # Define the simulation world
    world_y_size = 0.99
    DEMSim.InstructBoxDomainDimension(world_y_size, world_y_size, world_y_size)
    # Add 5 bounding planes around the simulation world, and leave the top open
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain)
    bottom = -0.5
    DEMSim.AddBCPlane([0, 0, bottom], [0, 0, 1], mat_type_terrain)

    # Define the terrain particle templates
    # Calculate its mass and MOI
    terrain_density = 2.6e3
    volume1 = 4.2520508
    mass1 = terrain_density * volume1
    MOI1 = np.array([1.6850426, 1.6375114, 2.1187753]) * terrain_density
    volume2 = 2.1670011
    mass2 = terrain_density * volume2
    MOI2 = np.array([0.57402126, 0.60616378, 0.92890173]) * terrain_density
    # Scale the template we just created
    scales = [0.014, 0.0075833, 0.0044, 0.003, 0.002, 0.0018333, 0.0017]
    # Then load it to system
    my_template2 = DEMSim.LoadClumpType(mass2, MOI2.tolist(), DEME.GetDEMEDataFile(
        "clumps/triangular_flat_6comp.csv"), mat_type_terrain)
    my_template1 = DEMSim.LoadClumpType(mass1, MOI1.tolist(), DEME.GetDEMEDataFile(
        "clumps/triangular_flat.csv"), mat_type_terrain)
    ground_particle_templates = [my_template2, DEMSim.Duplicate(my_template2),
                                 my_template1,
                                 DEMSim.Duplicate(my_template1),
                                 DEMSim.Duplicate(my_template1),
                                 DEMSim.Duplicate(my_template1),
                                 DEMSim.Duplicate(my_template1)]
    # Now scale those templates
    for i in range(len(scales)):
        my_template = ground_particle_templates[i]
        # Note the mass and MOI are also scaled in the process, automatically. But if you are not happy with this, you
        # can always manually change mass and MOI afterwards.
        my_template.Scale(scales[i])
        # Give these templates names, 0000, 0001 etc.
        t_name = f"{i:04d}"
        my_template.AssignName(t_name)

    # Instatiate particles with a probability that is in line with their weight distribution.
    weight_perc = [0.1700, 0.2100, 0.1400, 0.1900, 0.1600, 0.0500, 0.0800]
    grain_perc = []
    for i in range(len(scales)):
        grain_perc.append(weight_perc[i] / (scales[i]**3))

    tmp = sum(grain_perc)
    grain_perc = [x / tmp for x in grain_perc]

    # Sampler to use
    sampler = HCPSampler(scales[0] * 2.2)

    # Make ready for simulation
    step_size = 1e-6
    DEMSim.SetInitTimeStep(step_size)
    DEMSim.SetGravitationalAcceleration([0, 0, -9.81])
    # Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    # wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    # happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(15.)
    # Error out vel is used to force the simulation to abort when something goes wrong.
    DEMSim.SetErrorOutVelocity(15.)
    DEMSim.SetExpandSafetyMultiplier(1.2)
    DEMSim.SetInitBinSize(scales[2])
    DEMSim.Initialize()

    time_end = 10.0

    currframe = 0
    curr_step = 0

    sample_halfheight = 0.4
    sample_halfwidth_x = (world_y_size * 0.96) / 2
    sample_halfwidth_y = (world_y_size * 0.96) / 2
    offset_z = bottom + sample_halfheight + 0.15
    settle_frame_time = 0.2
    settle_batch_time = 2.0

    while (DEMSim.GetNumClumps() < 0.25e6):
        # DEMSim.ClearCache() # Clearing cache is no longer needed
        sample_center = [0, 0, offset_z]

        # Sample and add heap particles
        heap_particles_xyz = sampler.SampleBox(
            sample_center, [sample_halfwidth_x, sample_halfwidth_y, sample_halfheight])

        # Pick from possible grain templates with a pre-defined possibility
        heap_family = np.random.choice(
            range(len(grain_perc)), len(heap_particles_xyz), p=grain_perc)
        heap_template_in_use = [ground_particle_templates[i]
                                for i in heap_family]

        heap_particles = DEMSim.AddClumps(
            heap_template_in_use, heap_particles_xyz)
        # Give ground particles a small initial velocity so they `collapse' at the start of the simulation
        heap_particles.SetVel([0.00, 0, -0.05])
        heap_particles.SetFamilies(heap_family)
        DEMSim.UpdateClumps()
        print(f"Current number of clumps: {DEMSim.GetNumClumps()}")

        # Allow for some settling
        # Must DoDynamicsThenSync (not DoDynamics), as adding entities to the simulation is only allowed at a sync-ed
        # point of time.
        t = 0.
        while (t < settle_batch_time):
            print(f"Frame: {currframe}", flush=True)
            filename = os.path.join(
                out_dir, f"DEMdemo_output_{currframe:04d}.csv")
            # meshname = os.path.join(out_dir, f"DEMdemo_mesh_{currframe:04d}.vtk")
            DEMSim.WriteSphereFile(filename)
            # DEMSim.WriteMeshFile(meshname)
            currframe += 1

            DEMSim.DoDynamicsThenSync(settle_frame_time)
            t += settle_frame_time

        DEMSim.ShowThreadCollaborationStats()

    # Settle for some time more
    DEMSim.DoDynamicsThenSync(1.0)

    DEMSim.ShowThreadCollaborationStats()
    DEMSim.ClearThreadCollaborationStats()
    DEMSim.ShowMemStats()

    cp_filename = os.path.join(out_dir, f"GRC_3e5.csv")
    cnt_filename = os.path.join(out_dir, f"Contact_pairs_3e5.csv")
    DEMSim.WriteClumpFile(cp_filename)
    DEMSim.WriteMeshFile(cnt_filename)

    print("DEMdemo_GRCPrep_Part1 exiting...")
