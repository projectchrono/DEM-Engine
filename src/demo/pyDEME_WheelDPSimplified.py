#  Copyright (c) 2021, SBEL GPU Development Team
#  Copyright (c) 2021, University of Wisconsin - Madison
#
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# A simplified wheel drawbar-pull test, featuring Curiosity wheel geometry and
# terrain particles represented by irregular DEM elements. Unlike WheelDP, this
# demo does not have a prerequisite and can run on its own, and the terrain is
# simpler since there is only one type of particle (3-sphere clump). The velocity
# and angular velocity of the wheel are prescribed and the terrain force on the
# wheel is measured (we only test one slip case in this demo, to make it faster).
# =============================================================================

import DEME
from DEME import HCPSampler

import numpy as np
import os
# import ctypes

if __name__ == "__main__":
    out_dir = "DemoOutput_WheelDPSimplified/"
    out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    DEMSim = DEME.DEMSolver()
    DEMSim.SetVerbosity("INFO")
    DEMSim.SetOutputFormat("CSV")
    DEMSim.SetMeshOutputFormat("VTK")
    DEMSim.SetContactOutputContent(["OWNER", "FORCE", "POINT"])

    # If you don't need individual force information, then this option makes the solver run a bit faster.
    DEMSim.SetNoForceRecord(True)

    # E, nu, CoR, mu, Crr...
    mat_type_wheel = DEMSim.LoadMaterial(
        {"E": 1e9, "nu": 0.3, "CoR": 0.6, "mu": 0.5, "Crr": 0.01})
    mat_type_terrain = DEMSim.LoadMaterial(
        {"E": 1e9, "nu": 0.3, "CoR": 0.4, "mu": 0.5, "Crr": 0.01})

    # If you don't have this line, then mu between drum material and granular material will be the average of the two.
    DEMSim.SetMaterialPropertyPair("mu", mat_type_wheel, mat_type_terrain, 0.8)
    DEMSim.SetMaterialPropertyPair(
        "CoR", mat_type_wheel, mat_type_terrain, 0.6)

    # `World'
    G_mag = 9.81
    step_size = 5e-6
    world_size_y = 1.
    world_size_x = 2.
    world_size_z = 2.
    DEMSim.InstructBoxDomainDimension(
        world_size_x, world_size_y, world_size_z, "none")
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain)
    bottom = -0.5
    bot_wall = DEMSim.AddBCPlane([0, 0, bottom], [0, 0, 1], mat_type_terrain)

    # Define the wheel geometry
    wheel_rad = 0.25
    wheel_width = 0.2
    wheel_weight = 0000100.000
    wheel_mass = wheel_weight / G_mag
    total_pressure = 200.0
    added_pressure = total_pressure - wheel_weight
    wheel_IYY = wheel_mass * wheel_rad * wheel_rad / 2
    wheel_IXX = (wheel_mass / 12) * (3 * wheel_rad *
                                     wheel_rad + wheel_width * wheel_width)

    wheel = DEMSim.AddWavefrontMeshObject(DEME.GetDEMEDataFile(
        "mesh/rover_wheels/viper_wheel_right.obj"), mat_type_wheel, True, False)

    wheel.SetMass(wheel_mass)
    wheel.SetMOI([wheel_IXX, wheel_IYY, wheel_IXX])

    # Give the wheel a family number so we can potentially add prescription
    wheel.SetFamily(1)

    # Define the terrain particle templates
    # Calculate its mass and MOI
    terrain_density = 2.6e3
    volume1 = 4.2520508
    mass1 = terrain_density * volume1
    MOI1 = np.array([1.6850426, 1.6375114, 2.1187753]) * terrain_density

    # Scale the template we just created
    scale = 0.02
    # Then load it to system
    my_template = DEMSim.LoadClumpType(mass1, MOI1.tolist(), DEME.GetDEMEDataFile(
        "clumps/triangular_flat.csv"), mat_type_terrain)
    # Now scale the template
    # Note the mass and MOI are also scaled in the process, automatically. But if you are not happy with this, you
    # can always manually change mass and MOI afterwards.
    my_template.Scale(scale)
    # Give these templates names, 0000, 0001 etc.
    template_num = 0
    t_name = f"{template_num:04d}"
    my_template.AssignName(t_name)

    # Sampler to use
    sampler = HCPSampler(scale * 2.7)
    sample_halfheight = 0.25
    sample_halfwidth_x = (world_size_x * 0.95) / 2
    sample_halfwidth_y = (world_size_y * 0.95) / 2
    offset_z = bottom + sample_halfheight + 0.03
    # Sample initial particles
    sample_center = np.array([0, 0, offset_z])
    terrain_particles_xyz = sampler.SampleBox(sample_center.tolist(
    ), [sample_halfwidth_x, sample_halfwidth_y, sample_halfheight])
    terrain_template_in_use = [my_template] * len(terrain_particles_xyz)
    heap_family = [0] * len(terrain_particles_xyz)

    terrain_particles = DEMSim.AddClumps(
        terrain_template_in_use, terrain_particles_xyz)
    # Give ground particles a small initial velocity so they `collapse' at the start of the simulation
    terrain_particles.SetVel([0.00, 0, -0.05])
    terrain_particles.SetFamilies(heap_family)
    print(f"Current number of clumps: {len(terrain_particles_xyz)}")

    # Track it
    bot_wall_tracker = DEMSim.Track(bot_wall)
    wheel_tracker = DEMSim.Track(wheel)

    # Families' prescribed motions
    math_PI = 3.1415927
    w_r = math_PI / 4
    v_ref = w_r * wheel_rad

    sim_end = 6.
    # Note: this wheel is not `dictated' by our prescrption of motion because it can still fall onto the ground
    # (move freely linearly)
    DEMSim.SetFamilyPrescribedAngVel(1, "0", f"{w_r:09}", "0", False)
    # An extra force (acceleration) is addedd to simulate the load that the wheel carries
    DEMSim.AddFamilyPrescribedAcc(
        1, "none", "none", f"{(-added_pressure / wheel_mass):09}")
    # `Real sim' family number
    DEMSim.SetFamilyPrescribedAngVel(2, "0", f"{w_r:09}", "0", False)
    # Note: this wheel is not `dictated' by our prescrption of motion (hence the false argument), because it
    # can sink into the ground (move on Z dir); but its X and Y motions are explicitly controlled.
    # This one says when the experiment is going, the slip ratio is 0.5 (by our prescribing linear and angular vel)
    DEMSim.SetFamilyPrescribedLinVel(
        2, f"{(v_ref * 0.5):09}", "0", "none", False)
    # An extra force (acceleration) is addedd to simulate the load that the wheel carries
    DEMSim.AddFamilyPrescribedAcc(
        2, "none", "none", f"{(-added_pressure / wheel_mass):09}")

    # Some inspectors
    max_z_finder = DEMSim.CreateInspector("clump_max_z")
    min_z_finder = DEMSim.CreateInspector("clump_min_z")
    total_mass_finder = DEMSim.CreateInspector("clump_mass")
    max_v_finder = DEMSim.CreateInspector("clump_max_absv")

    # Make ready for simulation
    DEMSim.SetInitTimeStep(step_size)
    DEMSim.SetGravitationalAcceleration([0, 0, -G_mag])
    # Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    # wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    # happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(20.)
    # Error out vel is used to force the simulation to abort when something goes wrong.
    DEMSim.SetErrorOutVelocity(35.)
    DEMSim.SetExpandSafetyMultiplier(1.)
    DEMSim.Initialize()

    fps = 10
    out_steps = (int)(1.0 / (fps * step_size))
    curr_step = 0
    currframe = 0
    frame_time = 1.0 / fps
    report_ps = 100
    report_steps = (int)(1.0 / (report_ps * step_size))
    print(f"Output at {fps} FPS")

    # Put the wheel in place, then let the wheel sink in initially
    max_z = max_z_finder.GetValue()
    wheel_tracker.SetPos([-0.45, 0, max_z + 0.03 + wheel_rad], 0)

    t = 0.
    while t < 1.:
        print(f"Outputting frame: {currframe}", flush=True)
        filename = os.path.join(out_dir, f"DEMdemo_output_{currframe:04d}.csv")
        meshname = os.path.join(out_dir, f"DEMdemo_mesh_{currframe:04d}.vtk")
        DEMSim.WriteSphereFile(filename)
        DEMSim.WriteMeshFile(meshname)
        currframe += 1

        DEMSim.DoDynamics(frame_time)
        t += frame_time

    # Switch wheel from free fall into DP test
    DEMSim.DoDynamicsThenSync(0)
    DEMSim.ChangeFamily(1, 2)

    t = 0.
    while (t < sim_end):
        if (curr_step % out_steps == 0):
            print(f"Outputting frame: {currframe}", flush=True)
            filename = os.path.join(
                out_dir, f"DEMdemo_output_{currframe:04d}.csv")
            meshname = os.path.join(
                out_dir, f"DEMdemo_mesh_{currframe:04d}.vtk")
            DEMSim.WriteSphereFile(filename)
            DEMSim.WriteMeshFile(meshname)
            currframe += 1
            DEMSim.ShowThreadCollaborationStats()

        if (curr_step % report_steps == 0):
            force_list = wheel_tracker.ContactAcc()
            forces = np.array(force_list) * wheel_mass
            print(f"Time: {t}", flush=True)
            print(
                f"Force on wheel: {forces[0]}, {forces[1]}, {forces[2]}", flush=True)
            print(
                f"Drawbar pull coeff: {(forces[0] / total_pressure)}", flush=True)

        DEMSim.DoDynamics(step_size)
        curr_step += 1
        t += step_size

    DEMSim.ShowTimingStats()
    DEMSim.ShowMemStats()

    print("WheelDPSimpilified demo exiting...")
