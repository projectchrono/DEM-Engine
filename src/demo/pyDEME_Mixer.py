# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
#
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# This demo features a mesh-represented bladed mixer interacting with clump-represented
# DEM particles.
# =============================================================================

import DEME
from DEME import HCPSampler

import numpy as np
import os
import time

if __name__ == "__main__":
    out_dir = "DemoOutput_Mixer/"
    out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    DEMSim = DEME.DEMSolver()
    DEMSim.SetVerbosity("STEP_METRIC")
    DEMSim.SetOutputFormat("CSV")
    DEMSim.SetOutputContent(["ABSV", "XYZ"])
    DEMSim.SetMeshOutputFormat("VTK")

    # If you don't need individual force information, then this option makes the solver run a bit faster.
    DEMSim.SetNoForceRecord(True)

    # E, nu, CoR, mu, Crr... Material properties
    mat_type_mixer = DEMSim.LoadMaterial(
        {"E": 1e8, "nu":  0.3, "CoR":  0.6, "mu":  0.5, "Crr": 0.0})
    mat_type_granular = DEMSim.LoadMaterial(
        {"E":  1e8, "nu":  0.3, "CoR":  0.8, "mu":  0.2, "Crr":  0.0})
    # If you don't have this line, then mu between mixer material and granular material will be 0.35 (average of the two).
    DEMSim.SetMaterialPropertyPair(
        "CoR", mat_type_mixer, mat_type_granular, 0.5)

    step_size = 5e-6
    world_size = 1
    chamber_height = world_size / 3.
    fill_height = chamber_height
    chamber_bottom = -world_size / 2.
    fill_bottom = chamber_bottom + chamber_height

    DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size)
    DEMSim.InstructBoxDomainBoundingBC("all", mat_type_granular)

    # Now add a cylinderical boundary
    walls = DEMSim.AddExternalObject()
    walls.AddCylinder([0, 0, 0], [0, 0, 1], world_size / 2., mat_type_mixer, 0)

    mixer = DEMSim.AddWavefrontMeshObject(
        DEME.GetDEMEDataFile("mesh/internal_mixer.obj"), mat_type_mixer)
    print(f"Total num of triangles: {mixer.GetNumTriangles()}")
    mixer.Scale([world_size / 2, world_size / 2, chamber_height])
    mixer.SetFamily(10)
    # Define the prescribed motion of mixer
    DEMSim.SetFamilyPrescribedAngVel(10, "0", "0", "3.14159")

    granular_rad = 0.005
    # Calculate its mass and MOI
    mass = 2.6e3 * 5.5886717
    MOI = np.array([2.928, 2.6029, 3.9908]) * 2.6e3
    template_granular = DEMSim.LoadClumpType(mass, MOI.tolist(
    ), DEME.GetDEMEDataFile("clumps/3_clump.csv"), mat_type_granular)
    template_granular.Scale(granular_rad)

    # Track the mixer
    mixer_tracker = DEMSim.Track(mixer)

    # Sampler uses hex close-packing
    sampler = HCPSampler(3.0 * granular_rad)
    fill_center = [0, 0, fill_bottom + fill_height / 2]
    fill_radius = world_size / 2. - 2. * granular_rad
    input_xyz = sampler.SampleCylinderZ(
        fill_center, fill_radius, fill_height / 2)
    DEMSim.AddClumps(template_granular, input_xyz)
    print(f"Total num of particles: {len(input_xyz)}")

    DEMSim.SetInitTimeStep(step_size)
    DEMSim.SetGravitationalAcceleration([0, 0, -9.81])
    DEMSim.SetCDUpdateFreq(40)
    # Mixer has a big angular velocity-contributed linear speed at its blades, this is something the solver do not
    # account for, for now. And that means it needs to be added as an estimated value.
    DEMSim.SetExpandSafetyAdder(2.0)
    # You usually don't have to worry about initial bin size. In very rare cases, init bin size is so bad that auto bin
    # size adaption is effectless, and you should notice in that case kT runs extremely slow. Then in that case setting
    # init bin size may save the simulation.
    # DEMSim.SetInitBinSize(25 * granular_rad);
    DEMSim.SetCDNumStepsMaxDriftMultipleOfAvg(1.2)
    DEMSim.SetCDNumStepsMaxDriftAheadOfAvg(6)
    DEMSim.SetSortContactPairs(True)
    # DEMSim.DisableAdaptiveBinSize();
    DEMSim.SetErrorOutVelocity(20.)
    # Force the solver to error out if something went crazy. A good practice to add them, but not necessary.
    DEMSim.SetErrorOutAvgContacts(50)
    DEMSim.Initialize()

    sim_end = 10.0
    fps = 20
    frame_time = 1.0 / fps

    # Keep tab of the max velocity in simulation
    max_v_finder = DEMSim.CreateInspector("clump_max_absv")

    print(f"Output at {fps} FPS")
    currframe = 0

    mixer_tracker.SetPos([0, 0, chamber_bottom + chamber_height / 2.0])

    t = 0.
    start = time.process_time()
    while (t < sim_end):
        print(f"Frame: {currframe}", flush=True)
        filename = os.path.join(out_dir, f"DEMdemo_output_{currframe:04d}.csv")
        meshname = os.path.join(out_dir, f"DEMdemo_mesh_{currframe:04d}.vtk")
        DEMSim.WriteSphereFile(filename)
        DEMSim.WriteMeshFile(meshname)
        currframe += 1

        max_v = max_v_finder.GetValue()
        print(
            f"Max velocity of any point in simulation is {max_v}", flush=True)
        print(
            f"Solver's current update frequency (auto-adapted): {DEMSim.GetUpdateFreq()}", flush=True)
        print(
            f"Average contacts each sphere has: {DEMSim.GetAvgSphContacts()}", flush=True)

        mixer_moi = np.array(mixer_tracker.MOI())
        mixer_acc = np.array(mixer_tracker.ContactAngAccLocal())
        mixer_torque = mixer_acc * mixer_moi
        print(
            f"Contact torque on the mixer is {mixer_torque[0]}, {mixer_torque[1]}, {mixer_torque[2]}", flush=True)

        DEMSim.DoDynamics(frame_time)
        DEMSim.ShowThreadCollaborationStats()

        t += frame_time

    elapsed_time = time.process_time() - start
    print(f"{elapsed_time} seconds (wall time) to finish this simulation")

    DEMSim.ShowTimingStats()
    DEMSim.ShowMemStats()
    print("Mixer demo exiting...")
