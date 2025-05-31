#  Copyright (c) 2021, SBEL GPU Development Team
#  Copyright (c) 2021, University of Wisconsin - Madison
#
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# This demo presents a cone penetrameter test with a soil sample made of clumped
# particles of various sizes. Before the test starts, when compress the terrain
# first, and note that the compressor used in this process has its position
# explicitly controlled step-by-step.
# =============================================================================

import DEME

import numpy as np
import os

if __name__ == "__main__":
    out_dir = "DemoOutput_ConePenetration/"
    out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    DEMSim = DEME.DEMSolver()
    DEMSim.SetVerbosity("INFO")
    DEMSim.SetOutputFormat("CSV")
    DEMSim.SetOutputContent(["ABSV"])
    DEMSim.SetMeshOutputFormat("VTK")
    DEMSim.SetContactOutputContent(["OWNER", "FORCE", "POINT"])

    # E, nu, CoR, mu, Crr... Material properties.
    mat_type_cone = DEMSim.LoadMaterial(
        {"E": 1e9, "nu": 0.3, "CoR": 0.8, "mu": 0.7, "Crr": 0.00})
    mat_type_terrain = DEMSim.LoadMaterial(
        {"E": 1e9, "nu": 0.3, "CoR": 0.8, "mu": 0.4, "Crr": 0.00})
    # If you don't have this line, then values will take average between 2 materials, when they are in contact
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_cone, mat_type_terrain, 0.8)
    DEMSim.SetMaterialPropertyPair("mu", mat_type_cone, mat_type_terrain, 0.7)

    math_PI = 3.1415926
    cone_speed = 0.03
    step_size = 5e-6
    world_size = 2
    soil_bin_diameter = 0.584
    cone_surf_area = 323e-6
    cone_diameter = np.sqrt(cone_surf_area / math_PI) * 2
    DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size)
    # No need to add simulation `world' boundaries, b/c we'll add a cylinderical container manually
    DEMSim.InstructBoxDomainBoundingBC("none", mat_type_terrain)
    # Now add a cylinderical boundary along with a bottom plane
    bottom = -0.5
    walls = DEMSim.AddExternalObject()
    walls.AddCylinder([0, 0, 0], [0, 0, 1],
                      soil_bin_diameter / 2., mat_type_terrain, 0)
    walls.AddPlane([0, 0, bottom], [0, 0, 1], mat_type_terrain)

    # Define the terrain particle templates
    # Calculate its mass and MOI
    terrain_density = 2.6e3
    clump_vol = 5.5886717
    mass = terrain_density * clump_vol
    MOI = np.array([2.928, 2.6029, 3.9908]) * terrain_density
    # Then load it to system
    my_template = DEMSim.LoadClumpType(mass, MOI.tolist(
    ), DEME.GetDEMEDataFile("clumps/3_clump.csv"), mat_type_terrain)
    my_template.SetVolume(clump_vol)
    # Decide the scalings of the templates we just created (so that they are... like particles, not rocks)
    scale = 0.0044
    my_template.Scale(scale)

    # Sampler to sample
    sampler = DEME.HCPSampler(scale * 3.)
    fill_height = 0.5
    fill_center = [0, 0, bottom + fill_height / 2]
    fill_radius = soil_bin_diameter / 2. - scale * 3.
    input_xyz = sampler.SampleCylinderZ(
        fill_center, fill_radius, fill_height / 2 - scale * 2.)
    DEMSim.AddClumps(my_template, input_xyz)
    print(f"Total num of particles: {len(input_xyz)}")

    # Load in the cone used for this penetration test
    cone_tip = DEMSim.AddWavefrontMeshObject(
        DEME.GetDEMEDataFile("mesh/cone.obj"), mat_type_cone)
    cone_body = DEMSim.AddWavefrontMeshObject(
        DEME.GetDEMEDataFile("mesh/cyl_r1_h2.obj"), mat_type_cone)
    print(
        f"Total num of triangles: {cone_tip.GetNumTriangles() + cone_body.GetNumTriangles()}")

    # The initial cone mesh has base radius 1, and height 1. Let's stretch it a bit so it has a 60deg tip, instead of
    # 90deg.
    tip_height = np.sqrt(3.)
    cone_tip.Scale([1, 1, tip_height])
    # Then set mass properties
    cone_mass = 7.8e3 * tip_height / 3 * math_PI
    cone_tip.SetMass(cone_mass)
    # You can checkout https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    cone_tip.SetMOI([cone_mass * (3. / 20. + 3. / 80. * tip_height * tip_height),
                    cone_mass * (3. / 20. + 3. / 80. * tip_height * tip_height), 3 * cone_mass / 10])
    # This cone mesh has its tip at the origin. And, float4 quaternion pattern is (x, y, z, w).
    cone_tip.InformCentroidPrincipal(
        [0, 0, 3. / 4. * tip_height], [0, 0, 0, 1])
    # Note the scale method will scale mass and MOI automatically. But this only goes for the case you scale xyz all
    # together; otherwise, the MOI scaling will not be accurate and you should manually reset them.
    cone_tip.Scale(cone_diameter / 2)
    cone_tip.SetFamily(2)

    # The define the body that is connected to the tip
    body_mass = 7.8e3 * math_PI
    cone_body.SetMass(body_mass)
    cone_body.SetMOI([body_mass * 7 / 12, body_mass * 7 / 12, body_mass / 2])
    # This cyl mesh (h = 2m, r = 1m) has its center at the origin. So the following call actually has no effect...
    cone_body.InformCentroidPrincipal([0, 0, 0], [0, 0, 0, 1])
    cone_body.Scale([cone_diameter / 2, cone_diameter / 2, 0.5])
    cone_body.SetFamily(2)

    # Track the cone_tip
    tip_tracker = DEMSim.Track(cone_tip)
    body_tracker = DEMSim.Track(cone_body)

    # Because the cone's motion is completely pre-determined, we can just prescribe family 1
    DEMSim.SetFamilyPrescribedLinVel(1, "0", "0", f"-{cone_speed}")
    # Cone is initially in family 2, sleeping...
    DEMSim.SetFamilyFixed(2)
    DEMSim.DisableContactBetweenFamilies(0, 2)

    # Now add a plane to compress the sample
    compressor = DEMSim.AddExternalObject()
    compressor.AddPlane([0, 0, 0], [0, 0, -1], mat_type_terrain)
    compressor.SetFamily(10)
    DEMSim.SetFamilyFixed(10)
    compressor_tracker = DEMSim.Track(compressor)

    # Some inspectors
    max_z_finder = DEMSim.CreateInspector("clump_max_z")
    total_mass_finder = DEMSim.CreateInspector("clump_mass")

    DEMSim.SetInitTimeStep(step_size)
    DEMSim.SetGravitationalAcceleration([0, 0, -9.81])
    # CD freq will be auto-adapted so it does not matter much here.
    DEMSim.SetCDUpdateFreq(20)
    # Max velocity info is generally just for the solver's reference and the user do not have to set it. The solver
    # wouldn't take into account a vel larger than this when doing async-ed contact detection: but this vel won't
    # happen anyway and if it does, something already went wrong.
    DEMSim.SetMaxVelocity(10.)
    DEMSim.Initialize()

    # Settle
    DEMSim.DoDynamicsThenSync(0.8)

    # Compress until dense enough
    currframe = 0
    curr_step = 0
    fps = 20
    out_steps = (int)(1.0 / (fps * step_size))
    compressor_vel = 0.05
    terrain_max_z = max_z_finder.GetValue()
    init_max_z = terrain_max_z
    bulk_density = -10000.
    while (bulk_density < 1500.):
        matter_mass = total_mass_finder.GetValue()
        total_volume = math_PI * \
            (soil_bin_diameter * soil_bin_diameter / 4) * (terrain_max_z - bottom)
        bulk_density = matter_mass / total_volume
        if (curr_step % out_steps == 0):
            print(f"Outputting frame: {currframe}", flush=True)
            filename = os.path.join(
                out_dir, f"DEMdemo_output_{currframe:04d}.csv")
            # meshname = os.path.join(out_dir, f"DEMdemo_mesh_{currframe:04d}.vtk")
            DEMSim.WriteSphereFile(filename)
            # DEMSim.WriteMeshFile(meshname)
            currframe += 1
            print(f"Compression bulk density: {bulk_density}", flush=True)

        terrain_max_z -= compressor_vel * step_size
        compressor_tracker.SetPos([0, 0, terrain_max_z])
        DEMSim.DoDynamics(step_size)
        curr_step += 1

    # Then gradually remove the compressor
    while (terrain_max_z < init_max_z):
        if (curr_step % out_steps == 0):
            print(f"Outputting frame: {currframe}", flush=True)
            filename = os.path.join(
                out_dir, f"DEMdemo_output_{currframe:04d}.csv")
            # meshname = os.path.join(out_dir, f"DEMdemo_mesh_{currframe:04d}.vtk")
            DEMSim.WriteSphereFile(filename)
            # DEMSim.WriteMeshFile(meshname)
            matter_mass = total_mass_finder.GetValue()
            total_volume = math_PI * \
                (soil_bin_diameter * soil_bin_diameter / 4) * \
                (max_z_finder.GetValue() - bottom)
            bulk_density = matter_mass / total_volume
            print(f"Compression bulk density: {bulk_density}", flush=True)
            currframe += 1

        terrain_max_z += compressor_vel * step_size
        compressor_tracker.SetPos([0, 0, terrain_max_z])
        DEMSim.DoDynamics(step_size)
        curr_step += 1

    # Remove compressor
    DEMSim.DoDynamicsThenSync(0.)
    DEMSim.DisableContactBetweenFamilies(0, 10)
    DEMSim.DoDynamicsThenSync(0.2)
    terrain_max_z = max_z_finder.GetValue()

    sim_end = 7.0
    fps = 2500
    frame_time = 1.0 / fps
    print(f"Output at {fps} FPS")

    # Put the cone in place
    starting_height = terrain_max_z + 0.03
    # Its initial position should be right above the cone tip...
    body_tracker.SetPos(
        [0, 0, 0.5 + (cone_diameter / 2 / 4 * tip_height) + starting_height])
    # Note that position of objects is always the location of their centroid
    tip_tracker.SetPos([0, 0, starting_height])
    # The tip location, used to measure penetration length
    tip_z = -cone_diameter / 2 * 3 / 4 * tip_height + starting_height

    # Enable cone
    DEMSim.ChangeFamily(2, 1)
    matter_mass = total_mass_finder.GetValue()
    total_volume = math_PI * (soil_bin_diameter *
                              soil_bin_diameter / 4) * (terrain_max_z - bottom)
    bulk_density = matter_mass / total_volume
    print(f"Bulk density: {bulk_density}", flush=True)

    tip_z_when_first_hit = 9999.
    hit_terrain = False
    frame_count = 0

    t = 0.
    while (t < sim_end):
        # terrain_max_z = max_z_finder.GetValue()
        forces = tip_tracker.ContactAcc()
        # Note cone_mass is not the true mass, b/c we scaled the the cone tip! So we use true mass by using cone_mass
        forces = np.array(forces) * cone_tip.Mass()
        # forces[2] is the z dir force
        pressure = np.abs(forces[2]) / cone_surf_area
        if ((pressure > 1e-4) and not (hit_terrain)):
            hit_terrain = True
            tip_z_when_first_hit = tip_z

        penetration = tip_z_when_first_hit - tip_z if hit_terrain else 0.
        print(f"Time: {t}", flush=True)
        print(f"Z coord of tip: {tip_z}", flush=True)
        print(f"Penetration: {penetration}", flush=True)
        print(
            f"Force on cone: {forces[0]}, {forces[1]}, {forces[2]}", flush=True)
        print(f"Pressure: {pressure}", flush=True)

        if (frame_count % 500 == 0):
            print(f"Outputting frame: {currframe}", flush=True)
            filename = os.path.join(
                out_dir, f"DEMdemo_output_{currframe:04d}.csv")
            meshname = os.path.join(
                out_dir, f"DEMdemo_mesh_{currframe:04d}.vtk")
            DEMSim.WriteSphereFile(filename)
            DEMSim.WriteMeshFile(meshname)
            DEMSim.ShowThreadCollaborationStats()
            currframe += 1

        DEMSim.DoDynamicsThenSync(frame_time)
        tip_z -= cone_speed * frame_time

        frame_count += 1
        t += frame_time

    DEMSim.ShowMemStats()
    print(f"ConePenetration demo exiting...")
