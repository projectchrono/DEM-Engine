#  Copyright (c) 2021, SBEL GPU Development Team
#  Copyright (c) 2021, University of Wisconsin - Madison
#
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# This demo shows how to control the deformation of a flexibile mesh and use
# it in a DEM simulation.
# We show how mesh node coordinates can be extracted and modified. This can
# potentially be used with an external solid mechanics solver to do co-simulations
# involving flexible bodies.
# =============================================================================

import DEME

import numpy as np
import os
import csv

math_PI = 3.1415927
force_csv_header = ["point_x", "point_y",
                    "point_z", "force_x", "force_y", "force_z"]

# Used to write pairwise force concerning the mesh to a file.


def writePointsForcesToCSV(force_csv_header,
                           points,
                           forces,
                           filename):
    combined_array = np.hstack((np.array(points), np.array(forces)))
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar=',')
        writer.writerow(force_csv_header)
        for row in combined_array:
            writer.writerow(row)


if __name__ == "__main__":
    out_dir = "DemoOutput_FlexibleMesh/"
    out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    DEMSim = DEME.DEMSolver()
    DEMSim.SetVerbosity("INFO")
    DEMSim.SetOutputFormat("CSV")
    DEMSim.SetOutputContent(["ABSV", "XYZ"])
    DEMSim.SetMeshOutputFormat("VTK")
    DEMSim.SetContactOutputContent(["OWNER", "FORCE", "POINT", "TORQUE"])

    # E, nu, CoR, mu, Crr...
    mat_type_mesh = DEMSim.LoadMaterial(
        {"E": 1e8, "nu": 0.3, "CoR": 0.5, "mu": 0.7, "Crr": 0.00})
    mat_type_particle = DEMSim.LoadMaterial(
        {"E": 1e8, "nu": 0.3, "CoR": 0.5, "mu": 0.4, "Crr": 0.00})
    # If you don't have this line, then values will take average between 2 materials, when they are in contact
    DEMSim.SetMaterialPropertyPair("mu", mat_type_mesh, mat_type_particle, 0.5)

    world_size = 2.
    # You have the choice to not specify the world size. In that case, the world is 20^3. And you might want to add
    # world boundaries yourself, like we did in this demo.
    DEMSim.InstructBoxDomainDimension([-world_size / 2, world_size / 2],
                                      [-world_size / 2, world_size / 2], [0, world_size])

    # No need to add simulation `world' boundaries, b/c we'll add a cylinderical container manually
    DEMSim.InstructBoxDomainBoundingBC("none", mat_type_mesh)
    # Now manually add boundaries (you can choose to add them automatically using InstructBoxDomainBoundingBC, too)
    walls = DEMSim.AddExternalObject()
    walls.AddPlane([0, 0, 0], [0, 0, 1], mat_type_mesh)
    walls.AddPlane([0, 0, world_size], [0, 0, -1], mat_type_mesh)
    walls.AddPlane([world_size / 2, 0, 0], [-1, 0, 0], mat_type_mesh)
    walls.AddPlane([-world_size / 2, 0, 0], [1, 0, 0], mat_type_mesh)
    walls.AddPlane([0, world_size / 2, 0], [0, -1, 0], mat_type_mesh)
    walls.AddPlane([0, -world_size / 2, 0], [0, 1, 0], mat_type_mesh)

    # Define the terrain particle templates
    # Calculate its mass and MOI
    terrain_density = 2.6e3
    clump_vol = 4. / 3. * math_PI
    mass = terrain_density * clump_vol
    MOI = np.array([2. / 5., 2. / 5., 2. / 5.]) * mass
    # Then load it to system
    my_template = DEMSim.LoadClumpType(mass, MOI.tolist(), DEME.GetDEMEDataFile(
        "clumps/spiky_sphere.csv"), mat_type_particle)
    my_template.SetVolume(clump_vol)
    # Decide the scalings of the templates we just created (so that they are... like particles, not rocks)
    scale = 0.05
    my_template.Scale(scale)

    # Sample 2 chunks of materials in this part
    sampler = DEME.HCPSampler(scale * 2.2)
    fill_height = 1.75
    fill_center1 = [-world_size / 4, 0, fill_height / 2 + 2 * scale]
    fill_center2 = [world_size / 4, 0, fill_height / 2 + 2 * scale]
    fill_halfsize = [world_size / 4 - 2 * scale,
                     world_size / 4 - 2 * scale, fill_height / 2]
    input_xyz1 = sampler.SampleBox(fill_center1, fill_halfsize)
    input_xyz2 = sampler.SampleBox(fill_center2, fill_halfsize)
    particles = DEMSim.AddClumps(my_template, input_xyz1 + input_xyz2)
    print(f"Total num of particles: {particles.GetNumClumps()}")
    print(f"Total num of spheres: {particles.GetNumSpheres()}")

    # Load in the mesh which is a 2x2 (yz) plate. Its thickness is 0.05 in the x direction.
    flex_mesh = DEMSim.AddWavefrontMeshObject(
        DEME.GetDEMEDataFile("mesh/thin_plate.obj"), mat_type_mesh, True, False)
    num_tri = flex_mesh.GetNumTriangles()
    print(f"Total num of triangles: {num_tri}")

    # The define the properties
    body_mass = 1.5e3 * 1. * 1. * 0.05
    flex_mesh.SetMass(body_mass)
    Iz = 1. / 12. * (0.05 * 0.05 + 1 * 1) * body_mass
    Ix = 1. / 12. * (1 * 1 + 1 * 1) * body_mass
    flex_mesh.SetMOI([Ix, Iz, Iz])
    # This mesh is created with CoM being 0,0,0, so no effect by InformCentroidPrincipal. But it is good that you know
    # this method exists. Also, [0,0,0,1] means unit quaternion instead of [1,0,0,0].
    flex_mesh.InformCentroidPrincipal([0, 0, 0], [0, 0, 0, 1])
    # Attach it to the ceiling
    flex_mesh.SetInitPos([0, 0, 1.2])
    flex_mesh.SetFamily(1)
    # If you call SetFamilyPrescribedPosition and SetFamilyPrescribedQuaternion without specifying what position it
    # actually take, then its position is kept `as is' during simulation, without being affected by physics. It's
    # similar to fixing it but allows you manually impose velocities (which may have implications on your force model),
    # even though the velocity won't change its location. If you prescribe position by do not prescribe velocities, it
    # may make the object accumulate `phantom' velocity and de-stabilize the simulation. Fixing both position and
    # velocity is equivalent to fixing the family.
    DEMSim.SetFamilyPrescribedPosition(1)
    DEMSim.SetFamilyPrescribedQuaternion(1)
    DEMSim.SetFamilyPrescribedLinVel(1)
    DEMSim.SetFamilyPrescribedAngVel(1)
    # DEMSim.SetFamilyFixed(1);

    # Track the mesh
    flex_mesh_tracker = DEMSim.Track(flex_mesh)

    # Some inspectors
    max_z_finder = DEMSim.CreateInspector("clump_max_z")

    step_size = 5e-6
    DEMSim.SetInitTimeStep(step_size)
    DEMSim.SetGravitationalAcceleration([0, 0, -9.81])
    # Mesh has user-enforced deformation that the solver won't expect, so it can be better to allow larger safety
    # adder.
    DEMSim.SetExpandSafetyAdder(1.0)
    DEMSim.SetErrorOutAvgContacts(50)
    DEMSim.Initialize()

    # After system initialization, you can still get an handle of the mesh components using trackers (GetMesh method).
    # But note that if you do changes using this handle, the changes are done to the mesh and it immediate affects the
    # simulation. So sometimes you want to copy the information you are after and keep for your record.
    mesh_handle = flex_mesh_tracker.GetMesh()
    # This is keeping a copy of the RELATIVE (to the CoM) locations of the mesh nodes. In our case, the Z coordinates
    # of these nodes range from -0.5 to 0.5. In Python, you get a n by 3 matrix, and here we create a copy of this matrix.

    node_resting_location = np.array(mesh_handle.GetCoordsVertices())

    sim_end = 9.0
    fps = 20
    frame_time = 1.0 / fps
    print(f"Output at {fps} FPS")
    out_steps = (int)(1.0 / (fps * step_size))
    frame_count = 0
    step_count = 0

    # Used to store forces and points of contact
    forces = np.array([])
    points = np.array([])
    num_force_pairs = 0

    # Settle
    t = 0.
    while (t < 0.5):
        print(f"Outputting frame: {frame_count}", flush=True)
        filename = os.path.join(
            out_dir, f"DEMdemo_output_{frame_count:04d}.csv")
        meshname = os.path.join(
            out_dir, f"DEMdemo_mesh_{frame_count:04d}.vtk")
        force_filename = os.path.join(
            out_dir, f"DEMdemo_forces_{frame_count:04d}.csv")
        DEMSim.WriteSphereFile(filename)
        DEMSim.WriteMeshFile(meshname)
        writePointsForcesToCSV(force_csv_header, points,
                               forces, force_filename)
        DEMSim.ShowThreadCollaborationStats()
        frame_count += 1

        DEMSim.DoDynamics(frame_time)
        t += frame_time

    # It's possible that you don't have to update the mesh every time step so you can set this number larger than 1.
    # However, you have to then ensure the simulation does not de-stabilize because the mesh--particles contacts are
    # running in a delayed fashion and large penetrations can occur. If the mesh is super soft, then it's probably OK.
    ts_per_mesh_update = 5
    # Some constants that are used to define the artificial mesh motion. You'll see in the main simulation loop.
    max_wave_magnitude = 0.3
    wave_period = 3.0

    # Main simulation loop starts...
    t = 0.
    while (t < sim_end):
        if (step_count % out_steps == 0):
            print(f"Outputting frame: {frame_count}", flush=True)
            filename = os.path.join(
                out_dir, f"DEMdemo_output_{frame_count:04d}.csv")
            meshname = os.path.join(
                out_dir, f"DEMdemo_mesh_{frame_count:04d}.vtk")
            force_filename = os.path.join(
                out_dir, f"DEMdemo_forces_{frame_count:04d}.csv")
            DEMSim.WriteSphereFile(filename)
            DEMSim.WriteMeshFile(meshname)
            frame_count += 1
            # We write force pairs that are related to the mesh to a file
            [points, forces] = flex_mesh_tracker.GetContactForces()
            writePointsForcesToCSV(
                force_csv_header, points, forces, force_filename)
            DEMSim.ShowThreadCollaborationStats()

        # We probably don't have to update the mesh every time step
        if (step_count % ts_per_mesh_update == 0):
            # For real use cases, you probably will use an external solver to solve the defomration of the mesh,
            # then feed it to DEME. Here, we create an artificial defomration pattern for the mesh based on mesh node
            # location and time. This is just for show.

            # First, get where the mesh nodes are currently. We copy so we can modify it later on.
            # Shlok: This GetMeshNodesGlobal method maps to cpp version GetMeshNodesGlobalAsVectorOfVector method.
            node_current_location = np.array(
                flex_mesh_tracker.GetMeshNodesGlobal())
            # If you need the current RELATIVE (to the CoM) locations of the mesh nodes instead of global coordinates,
            # you can get it like the following:
            # node_current_location = mesh_handle.GetCoordsVertices()

            # Now calculate how much each node should `wave' and update the node location array. Remember z = 1 is
            # where the highest (relative) mesh node is. Again, this is artificial and only for showcasing this
            # utility.
            for i in range(node_current_location.shape[0]):
                # Use resting locations to calculate the magnitude of waving for nodes...
                my_wave_distance = (
                    (1. - node_resting_location[i, 2]) / 2.)**2 * max_wave_magnitude * np.sin(t / wave_period * 2 * np.pi)
                # Then update the current location array...
                node_current_location[i,
                                      0] = node_resting_location[i, 0] + my_wave_distance

            # Now instruct the mesh to deform. Two things to pay attention to:

            # 1. We should respect the actual CoM location of the mesh. We get the global coords of mesh nodes using
            # GetMeshNodesGlobal, but UpdateMesh works with mesh's local or say relative coordinates, and that is why
            # we do FrameTransformGlobalToLocal first. And depending on your setup, the CoM and coord frame of
            # your mesh might be moving, and if it moves and rotates then you probably need to move and rotate the
            # points you got to offset the influence of CoM and local frame first. That said, if you use
            # mesh_handle.GetCoordsVertices() as I mentioned above to get the relative node positions of the mesh,
            # then no need to FrameTransformGlobalToLocal the CoM and rotate the frame.

            # 2. UpdateMesh will update the relative locations of mesh nodes to your specified locations. But if you
            # just have the information on the amount of mesh deformation, then you can use UpdateMeshByIncrement
            # instead, to incremenet mesh nodes' relative locations.

            mesh_CoM_pos = flex_mesh_tracker.Pos()
            mesh_frame_oriQ = flex_mesh_tracker.OriQ()
            for i in range(node_current_location.shape[0]):
                node_current_location[i, :] = DEME.FrameTransformGlobalToLocal(
                    node_current_location[i, :], mesh_CoM_pos, mesh_frame_oriQ)
            flex_mesh_tracker.UpdateMesh(node_current_location)

            # Forces need to be extracted, if you want to use an external solver to solve the mesh's deformation. You
            # can do it like shown below. In this example, we did not use it other than writing it to a file; however
            # you may want to feed the array directly to your soild mechanics solver.
            [points, forces] = flex_mesh_tracker.GetContactForces()

        # Means advance simulation by one time step
        DEMSim.DoStepDynamics()
        t += step_size
        step_count += 1

    DEMSim.ShowTimingStats()
    DEMSim.ShowMemStats()
    print("FlexibleMesh demo exiting...")
