#  Copyright (c) 2021, SBEL GPU Development Team
#  Copyright (c) 2021, University of Wisconsin - Madison
#
#  SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# This demo features an analytical boundary-represented fast rotating container
# with particles of various shapes pulled into it. Different types of particles
# are marked with different family numbers (identification numbers) for easier
# visualizations.
# =============================================================================

import DEME

import numpy as np
import os

if __name__ == "__main__":
    out_dir = "DemoOutput_Centrifuge/"
    out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    DEMSim = DEME.DEMSolver()
    DEMSim.SetOutputFormat("CSV")
    # Output family numbers (used to identify the centrifuging effect)
    DEMSim.SetOutputContent(["ABSV", "FAMILY"])
    # DEMSim.SetVerbosity("STEP_METRIC");

    # If you don't need individual force information, then this option makes the solver run a bit faster.
    DEMSim.SetNoForceRecord(True)

    mat_type_sand = DEMSim.LoadMaterial(
        {"E": 1e9, "nu": 0.3, "CoR": 0.6, "mu": 0.5, "Crr": 0.01})
    mat_type_drum = DEMSim.LoadMaterial(
        {"E": 2e9, "nu": 0.3, "CoR": 0.6, "mu": 0.5, "Crr": 0.01})
    # Since two types of materials have the same mu, this following call does not change the default mu for their
    # interaction, it's still 0.5.
    DEMSim.SetMaterialPropertyPair("mu", mat_type_sand, mat_type_drum, 0.5)

    # We can scale this general template to make it smaller, like a DEM particle that you would actually use
    scaling = 0.01
    # Calculate its mass and MOI (scaled according to the size)
    mass = 2.6e3 * 4. / 3. * DEME.PI * 2 * 1 * 1
    MOI = np.array([1. / 5. * mass * (1 * 1 + 2 * 2), 1. / 5. *
                   mass * (1 * 1 + 2 * 2), 1. / 5. * mass * (1 * 1 + 1 * 1)])
    # What will be loaded from the file, is a template for ellipsoid with b = c = 1 and a = 2, where Z is the long axis
    ellipsoid = DEMSim.LoadClumpType(mass, MOI.tolist(), DEME.GetDEMEDataFile(
        "clumps/ellipsoid_2_1_1.csv"), mat_type_sand)
    # Scale the template we just created
    ellipsoid.Scale(scaling)
    mass *= scaling * scaling * scaling
    MOI *= scaling * scaling * scaling * scaling * scaling
    ellipsoid.SetMass(mass)
    ellipsoid.SetMOI(MOI.tolist())

    # Define material type for the particles (on a per-sphere-component basis)
    ellipsoid.SetMaterial(mat_type_sand)

    # Create some random clump templates for the filling materials
    # An array to store these generated clump templates
    clump_types = []
    # Then randomly create some clump templates for filling the drum
    for i in range(3):
        # A multiplier is added to the masses of different clumps, so that centrifuging separate those types. Consider
        # it separating materials with different densities.
        mult = 1.5**i
        # Then make a new copy of the template then do the scaling of mass
        ellipsoid_template = DEMSim.Duplicate(ellipsoid)
        ellipsoid_template.SetMass(mass*mult)
        ellipsoid_template.SetMOI((MOI*mult).tolist())

        # Load a (ellipsoid-shaped) clump and a sphere
        clump_types.append(ellipsoid_template)
        clump_types.append(DEMSim.LoadSphereType(
            mass*mult, np.cbrt(2.0) * scaling, mat_type_sand))

    # Add the centrifuge
    CylCenter = [0, 0, 0]
    CylAxis = [0, 0, 1]
    CylRad = 2.0
    CylHeight = 1.0
    CylMass = 1.0
    safe_delta = 0.03
    IZZ = CylMass * CylRad * CylRad / 2
    IYY = (CylMass / 12) * (3 * CylRad * CylRad + CylHeight * CylHeight)
    Drum = DEMSim.AddExternalObject()
    Drum.AddCylinder(CylCenter, CylAxis, CylRad, mat_type_drum, 0)
    Drum.SetMass(CylMass)
    Drum.SetMOI([IYY, IYY, IZZ])
    Drum_tracker = DEMSim.Track(Drum)
    # Drum is family 100
    drum_family = 100
    Drum.SetFamily(drum_family)
    # The drum rotates (facing Z direction)
    DEMSim.SetFamilyPrescribedAngVel(drum_family, "0", "0", "6.0")
    # Then add planes to `close up' the drum. We add it as another object b/c we want to track the force on it
    # separately.
    top_bot_planes = DEMSim.AddExternalObject()
    top_bot_planes.AddPlane(
        [0, 0, CylHeight / 2. - safe_delta], [0, 0, -1], mat_type_drum)
    top_bot_planes.AddPlane(
        [0, 0, -CylHeight / 2. + safe_delta], [0, 0, 1], mat_type_drum)
    # Planes should rotate together with the drum wall.
    top_bot_planes.SetFamily(drum_family)
    planes_tracker = DEMSim.Track(top_bot_planes)

    # Then sample some particles inside the drum
    input_template_type = []
    family_code = []
    sample_center = [0, 0, 0]
    sample_halfheight = CylHeight / 2.0 - 3.0 * safe_delta
    sample_halfwidth = CylRad / 1.5
    input_xyz = DEME.DEMBoxGridSampler(sample_center, [sample_halfwidth, sample_halfwidth, sample_halfheight],
                                       scaling * np.cbrt(2.0) * 2.1, scaling * np.cbrt(2.0) * 2.1, scaling * 2 * 2.1)
    num_clumps = len(input_xyz)
    # Casually select from generated clump types
    for i in range(num_clumps):
        input_template_type.append(clump_types[i % len(clump_types)])
        # Every clump type that has a unique mass, gets a unique family number
        family_code.append((i % len(clump_types)) // 2)

    particles = DEMSim.AddClumps(input_template_type, input_xyz)
    particles.SetFamilies(family_code)

    # Keep tab of the max velocity in simulation
    max_v_finder = DEMSim.CreateInspector("clump_max_absv")
    max_v = -1.

    # Make the domain large enough
    DEMSim.InstructBoxDomainDimension(5, 5, 5)
    step_size = 5e-6
    DEMSim.SetInitTimeStep(step_size)
    DEMSim.SetGravitationalAcceleration([0, 0, -9.81])
    DEMSim.SetExpandSafetyType("auto")
    # If there is a velocity that an analytical object (i.e. the drum) has that you'd like the solver to take into
    # account in consideration of adding contact margins, you have to specify it here, since the solver's automatic max
    # velocity derivation algorithm currently cannot take analytical object's angular velocity-induced velocity into
    # account.
    DEMSim.SetExpandSafetyAdder(6.0)
    DEMSim.Initialize()

    time_end = 20.0
    fps = 20
    out_steps = (int)(1.0 / (fps * step_size))

    print(f"Output at {fps} FPS")
    currframe = 0
    curr_step = 0
    t = 0.
    while (t < time_end):
        if (curr_step % out_steps == 0):
            print(f"Frame: {currframe}", flush=True)
            filename = os.path.join(
                out_dir, f"DEMdemo_output_{currframe:04d}.csv")
            DEMSim.WriteSphereFile(filename)
            currframe += 1

            DEMSim.ShowThreadCollaborationStats()
            max_v = max_v_finder.GetValue()
            print(
                f"Max velocity of any point in simulation is {max_v}", flush=True)

            # Torque on the side walls are?
            drum_moi = np.array(Drum_tracker.MOI())
            drum_pos = np.array(Drum_tracker.ContactAngAccLocal())
            drum_torque = np.multiply(drum_pos, drum_moi)
            print(
                f"Contact torque on the side walls is {drum_torque[0]}, {drum_torque[1]}, {drum_torque[2]}", flush=True)

            # The force on the bottom plane?
            force_on_BC = np.array(
                planes_tracker.ContactAcc()) * planes_tracker.Mass()
            print(
                f"Contact force on bottom plane is {force_on_BC[2]}", flush=True)

        DEMSim.DoDynamics(step_size)
        t += step_size
        curr_step += 1

    DEMSim.ShowThreadCollaborationStats()
    DEMSim.ClearThreadCollaborationStats()

    DEMSim.ShowTimingStats()
    DEMSim.ShowMemStats()

    print("DEMdemo_Centrifuge exiting...")
