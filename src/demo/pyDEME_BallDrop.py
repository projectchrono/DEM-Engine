# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
#
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# A meshed ball hitting a granular bed under gravity.
# ========= ====================================================================

import DEME

import numpy as np
import os

if __name__ == "__main__":
    out_dir = "DemoOutput_BallDrop/"
    out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    DEMSim = DEME.DEMSolver()
    DEMSim.SetVerbosity("STEP_METRIC")
    DEMSim.SetOutputFormat("CSV")
    DEMSim.SetOutputContent(["ABSV", "XYZ"])
    DEMSim.SetMeshOutputFormat("VTK")
    DEMSim.EnsureKernelErrMsgLineNum(True)

    # E, nu, CoR, mu, Crr... Material properties
    mat_type_ball = DEMSim.LoadMaterial(
        {"E": 1e10, "nu":  0.3, "CoR":  0.6, "mu":  0.3, "Crr": 0.01})
    mat_type_terrain = DEMSim.LoadMaterial(
        {"E":  5e9, "nu":  0.3, "CoR":  0.8, "mu":  0.3, "Crr":  0.01})
    # If you don't have this line, then CoR between mixer material and granular material will be 0.7 (average of the
    # two).
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_ball, mat_type_terrain, 0.6)
    # Should do the same for mu and Crr, but since they are the same across 2 materials, it won't have an effect...

    step_size = 1e-5
    world_size = 10

    DEMSim.InstructBoxDomainDimension(
        tuple([0, world_size]), tuple([0, world_size]), tuple([0, world_size]), "none")
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain)

    projectile = DEMSim.AddWavefrontMeshObject(
        DEME.GetDEMEDataFile("mesh/sphere.obj"), mat_type_ball, True, False)
    print(f"Total num of triangles: {projectile.GetNumTriangles()}")

    projectile.SetInitPos([world_size / 2, world_size / 2, world_size / 3 * 2])
    ball_mass = 7.8e3 * 4 / 3 * 3.1416
    projectile.SetMass(ball_mass)
    projectile.SetMOI(
        [ball_mass * 2 / 5, ball_mass * 2 / 5, ball_mass * 2 / 5])
    projectile.SetFamily(2)
    DEMSim.SetFamilyFixed(2)

    terrain_rad = 0.05
    template_terrain = DEMSim.LoadSphereType(
        terrain_rad * terrain_rad * terrain_rad * 2.6e3 * 4 / 3 * 3.14,  terrain_rad, mat_type_terrain)

    # Track the projectile
    proj_tracker = DEMSim.Track(projectile)

    sample_halfheight = world_size / 8
    sample_center = [world_size / 2, world_size / 2, sample_halfheight + 0.05]
    sample_halfwidth = world_size / 2 * 0.95

    input_xyz = DEME.DEMBoxHCPSampler(sample_center, [
                                      sample_halfwidth, sample_halfwidth, sample_halfheight], 2.01 * terrain_rad)
    DEMSim.AddClumps(template_terrain, input_xyz)
    print(f"Total num of particles: {len(input_xyz)}")

    DEMSim.SetInitTimeStep(step_size)
    DEMSim.SetGravitationalAcceleration([0, 0, -9.81])
    # Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    # wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    # happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(15.)
    # In general you don't have to worry about SetExpandSafetyAdder, unless if an entity has the property that a point
    # on it can move much faster than its CoM. In this demo, you are dealing with a meshed ball and you in fact don't
    # have this problem. In the Centrifuge demo though, this can be a problem since the centrifuge's CoM is not moving,
    # but its pointwise velocity can be high, so it needs to be accounted for using this method.
    DEMSim.SetExpandSafetyAdder(5.)
    DEMSim.SetInitBinSize(4 * terrain_rad)
    DEMSim.Initialize()

    sim_time = 6.0
    settle_time = 2.0
    fps = 20
    frame_time = (1.0 / fps)

    print(f"Output at {fps} FPS")
    currframe = 0

    # We can let it settle first
    t = 0.
    while (t < settle_time):
        print(f"Frame: {currframe}", flush=True)
        filename = os.path.join(out_dir, f"DEMdemo_output_{currframe:04d}.csv")
        meshname = os.path.join(out_dir, f"DEMdemo_mesh_{currframe:04d}.vtk")
        DEMSim.WriteSphereFile(filename)
        DEMSim.WriteMeshFile(meshname)
        currframe += 1

        DEMSim.DoDynamicsThenSync(frame_time)
        DEMSim.ShowThreadCollaborationStats()
        t += frame_time

    # Then drop the ball. I also wanted to test if changing step size method works fine here...
    step_size *= 0.5
    DEMSim.UpdateStepSize(step_size)
    DEMSim.ChangeFamily(2, 1)

    t = 0.
    while (t < sim_time):
        print(f"Frame: {currframe}", flush=True)
        filename = os.path.join(out_dir, f"DEMdemo_output_{currframe:04d}.csv")
        meshname = os.path.join(out_dir, f"DEMdemo_mesh_{currframe:04d}.vtk")
        DEMSim.WriteSphereFile(filename)
        DEMSim.WriteMeshFile(meshname)
        DEMSim.ShowThreadCollaborationStats()
        currframe += 1

        DEMSim.DoDynamicsThenSync(frame_time)
        DEMSim.ShowThreadCollaborationStats()
        t += frame_time

    DEMSim.ShowTimingStats()
    DEMSim.ShowMemStats()
    print("BallDrop demo exiting...")
