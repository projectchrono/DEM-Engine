# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
#
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# Put particles in a jar, the shake the jar in the hope to change the bulk density.
# =============================================================================

import DEME

import numpy as np
import os


if __name__ == "__main__":
    out_dir = "DemoOutput_Shake/"
    out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    DEMSim = DEME.DEMSolver()
    DEMSim.SetVerbosity("INFO")
    DEMSim.SetOutputFormat("CSV")
    DEMSim.SetOutputContent(["ABSV"])
    DEMSim.SetMeshOutputFormat("VTK")
    DEMSim.SetContactOutputContent(["OWNER", "FORCE", "POINT"])

    # If you don't need individual force information, then this option makes the solver run a bit faster.
    DEMSim.SetNoForceRecord()

    # E, nu, CoR, mu, Crr...
    mat_type_cone = DEMSim.LoadMaterial(
        {"E": 5e7, "nu": 0.3, "CoR": 0.5})
    mat_type_terrain = DEMSim.LoadMaterial(
        {"E": 5e7, "nu": 0.3, "CoR": 0.5})
    DEMSim.UseFrictionlessHertzianModel()

    
    shake_amp = 0.1
    shake_speed = 2  # Num of periods per second
    step_size = 1e-5
    world_size = 2.
    soil_bin_diameter = 0.584
    cone_surf_area = 323e-6
    cone_diameter = np.sqrt(cone_surf_area / np.pi) * 2
    DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size)
    # No need to add simulation `world' boundaries, b/c we'll add a cylinderical container manually
    DEMSim.InstructBoxDomainBoundingBC("none", mat_type_terrain)
    # Now add a cylinderical boundary along with a bottom plane
    bottom = -0.5
    walls = DEMSim.AddExternalObject()
    walls.AddCylinder([0, 0, 0], [0, 0, 1], soil_bin_diameter / 2., mat_type_terrain, 0)
    walls.AddPlane([0, 0, bottom], [0, 0, 1], mat_type_terrain)
    walls.SetFamily(1)

    # Define the terrain particle templates. Two types of clumps.
    terrain_density = 2.6e3
    clump_vol = 5.5886717
    mass = terrain_density * clump_vol
    MOI1 = np.array([2.928, 2.6029, 3.9908]) * terrain_density
    clump_vol2 = 2.1670011
    mass2 = terrain_density * clump_vol2
    MOI2 = np.array([0.57402126, 0.60616378, 0.92890173]) * terrain_density
    # Then load it to system
    template_1 = DEMSim.LoadClumpType(mass, MOI1.tolist(
        ), DEME.GetDEMEDataFile("clumps/triangular_flat.csv"), mat_type_terrain)
    template_2 = DEMSim.LoadClumpType(mass, MOI1.tolist(
        ), DEME.GetDEMEDataFile("clumps/triangular_flat_6comp.csv"), mat_type_terrain)
    # Decide the scalings of the templates we just created (so that they are... like particles, not rocks)
    scale1 = 0.015
    scale2 = 0.004
    template_1.Scale(scale1)
    template_2.Scale(scale2)

    # Sampler to sample
    sampler1 = DEME.GridSampler(scale1 * 3.)
    fill_height = 0.5
    fill_center = [0, 0, bottom + fill_height / 2]
    fill_radius = soil_bin_diameter / 2. - scale1 * 2.
    input_xyz = sampler1.SampleCylinderZ(
        fill_center, fill_radius, fill_height / 2 - scale1 * 2.)
    DEMSim.AddClumps(template_1, input_xyz)
    # Another batch...
    sampler2 = DEME.GridSampler(scale2 * 3.)
    fill_center += [0, 0, fill_height]
    fill_radius = soil_bin_diameter / 2. - scale2 * 2.
    input_xyz = sampler2.SampleCylinderZ(
        fill_center, fill_radius, fill_height / 2 - scale2 * 2.)
    DEMSim.AddClumps(template_2, input_xyz)

    # Now add a `cap' to the container when we shake it
    compressor = DEMSim.AddExternalObject()
    compressor.AddPlane([0, 0, 0], [0, 0, -1], mat_type_terrain)
    compressor.SetFamily(1)
    compressor_tracker = DEMSim.Track(compressor)

    # Family 1 shakes, family 2 is fixed
    shake_pattern_x = DEME.to_string_with_precision(shake_amp) + " * sin(" + DEME.to_string_with_precision(shake_speed) + " * 2 * deme::PI * t)"
    DEMSim.SetFamilyPrescribedMotion(1, shake_pattern_x, "0", shake_pattern_x)
    DEMSim.SetFamilyFixed(2)

    # Some inspectors
    max_z_finder = DEMSim.CreateInspector("clump_max_z")
    min_z_finder = DEMSim.CreateInspector("clump_min_z")
    total_mass_finder = DEMSim.CreateInspector("clump_mass")
    max_v_finder = DEMSim.CreateInspector("clump_max_absv")

    DEMSim.SetInitTimeStep(step_size)
    DEMSim.SetGravitationalAcceleration([0, 0, -9.81])

    # 256 or 512 are common choices. Note that in cases where the force model is modified, too many registers may be
    # used in the kernel, so we have to reduce this number to use 256. In other cases (and most cases), 512 is fine and
    # may make the code run a bit faster. Usually, the user do not have to call SetForceCalcThreadsPerBlock if they
    # don't know the implication.
    DEMSim.SetForceCalcThreadsPerBlock(512)
    DEMSim.Initialize()

    # Settle phase
    currframe = 0
    curr_step = 0
    fps = 10
    out_steps = (int)(1.0 / (fps * step_size))
    compressor_tracker.SetPos([0, 0, max_z_finder.GetValue()])
    for i in range (0, 5):
        filename = os.path.join(out_dir, f"DEMdemo_output_{currframe:04d}.csv")
        DEMSim.WriteSphereFile(filename)
        DEMSim.DoDynamicsThenSync(0.1)
        currframe += 1

    stop_time = 2.0 # Time before stopping shaking and measure bulk density
    stop_steps = (int)(stop_time * (1.0 / step_size))
    sim_end = 6.0
    print(f"Output at {fps} FPS")
    
    t = 0.
    while(t < sim_end):
        if (curr_step % out_steps == 0):
            print(f"Frame: {currframe}", flush=True)
            filename = os.path.join(
                out_dir, f"DEMdemo_output_{currframe:04d}.csv")
            DEMSim.WriteSphereFile(filename)
            print(f"Max system velocity: {max_z_finder.GetValue()}")
            currframe += 1
            DEMSim.ShowThreadCollaborationStats()
        if (curr_step % stop_steps == 0):
            # Measure
            max_z = max_z_finder.GetValue()
            min_z = min_z_finder.GetValue()
            matter_mass = total_mass_finder.GetValue()
            total_volume = np.pi * (soil_bin_diameter * soil_bin_diameter /4)
            bulk_density = matter_mass / total_volume
            print(f"Max z: {max_z}")
            print(f"Min z: {min_z}")
            print(f"Bulk density: {bulk_density}")
            # Put the cap to its new position and start shaking
            compressor_tracker.SetPos([0, 0, max_z])

        DEMSim.DoDynamics(step_size)
        t += step_size
        curr_step += 1

    # Output the final configuration of the clumps as a file. This is just a demonstration. This particular
    # configuration is not that useful as no other demos actually use it, unlike the GRC-1 soil.
    cp_filename = os.path.join(out_dir, f"GRC_rho{bulk_density}.csv")
    DEMSim.WriteSphereFile(cp_filename)

    print("Shake demo exiting...")
