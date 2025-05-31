# Copyright(c) 2021, SBEL GPU Development Team
# Copyright(c) 2021, University of Wisconsin - Madison
#
# SPDX - License - Identifier : BSD - 3 - Clause

# =============================================================
# A meshed ball hitting a granular bed under gravity.
# =============================================================

import DEME

import numpy as np
import os
import random

def force_model():
    model = """
        if (overlapDepth > 0.0) {
            // Material properties

            // DEM force calculation strategies for grain breakage
            float E_cnt, G_cnt, CoR_cnt, mu_cnt, Crr_cnt, E_A, E_B;
            {
                // E and nu are associated with each material, so obtain them this way
                E_A = E[bodyAMatType];
                float nu_A = nu[bodyAMatType];
                E_B = E[bodyBMatType];
                float nu_B = nu[bodyBMatType];
                matProxy2ContactParam<float>(E_cnt, G_cnt, E_A, nu_A, E_B, nu_B);
                // CoR, mu and Crr are pair-wise, so obtain them this way
                CoR_cnt = CoR[bodyAMatType][bodyBMatType];
                mu_cnt = mu[bodyAMatType][bodyBMatType];
                Crr_cnt = Crr[bodyAMatType][bodyBMatType];
            }
            float3 rotVelCPA, rotVelCPB;
            {
                // We also need the relative velocity between A and B in global frame to use in the damping terms
                // To get that, we need contact points' rotational velocity in GLOBAL frame
                // This is local rotational velocity (the portion of linear vel contributed by rotation)
                rotVelCPA = cross(ARotVel, locCPA);
                rotVelCPB = cross(BRotVel, locCPB);
                // This is mapping from local rotational velocity to global
                applyOriQToVector3<float, deme::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AOriQ.w, AOriQ.x, AOriQ.y,
                                                        AOriQ.z);
                applyOriQToVector3<float, deme::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BOriQ.w, BOriQ.x, BOriQ.y,
                                                        BOriQ.z);
            }
            float mass_eff, sqrt_Rd, beta;
            float3 vrel_tan;
            float3 delta_tan = make_float3(delta_tan_x, 0.0, delta_tan_z);

            // The (total) relative linear velocity of A relative to B
            const float3 velB2A = (ALinVel + rotVelCPA) - (BLinVel + rotVelCPB);
            const float projection = dot(velB2A, B2A);
            vrel_tan = velB2A - projection * B2A;
            vrel_tan.y = 0.0;

            const float3 v_rot = rotVelCPB - rotVelCPA;
            // This v_rot is only used for identifying resistance direction
            const float v_rot_mag = length(v_rot);
            mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);

            // Now we already have sufficient info to update contact history
            {
                delta_tan += ts * vrel_tan;
                const float disp_proj = dot(delta_tan, B2A);
                delta_tan -= disp_proj * B2A;
                delta_time += ts;
            }

            // Normal force part
            {
                sqrt_Rd = sqrt(overlapDepth * (ARadius * BRadius) / (ARadius + BRadius));
                const float Sn = 2. * E_cnt * sqrt_Rd;

                const float loge = (CoR_cnt < DEME_TINY_FLOAT) ? log(DEME_TINY_FLOAT) : log(CoR_cnt);
                beta = loge / sqrt(loge * loge + deme::PI_SQUARED);

                const float k_n = deme::TWO_OVER_THREE * Sn;
                const float gamma_n = deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

                force += (k_n * overlapDepth + gamma_n * projection) * B2A;
            }

            // Rolling resistance part
            if (Crr_cnt > 0.0) {
                // Figure out if we should apply rolling resistance force
                bool should_add_rolling_resistance = true;
                {
                    const float R_eff = sqrtf((ARadius * BRadius) / (ARadius + BRadius));
                    const float kn_simple = deme::FOUR_OVER_THREE * E_cnt * sqrtf(R_eff);
                    const float gn_simple = -2.f * sqrtf(deme::FIVE_OVER_THREE * mass_eff * E_cnt) * beta * powf(R_eff, 0.25f);

                    const float d_coeff = gn_simple / (2.f * sqrtf(kn_simple * mass_eff));

                    if (d_coeff < 1.0) {
                        float t_collision = deme::PI * sqrtf(mass_eff / (kn_simple * (1.f - d_coeff * d_coeff)));
                        if (delta_time <= t_collision) {
                            should_add_rolling_resistance = false;
                        }
                    }
                }
                // If should, then compute it (using Schwartz model)
                if (should_add_rolling_resistance) {
                    // Tangential velocity (only rolling contribution) of B relative to A, at contact point, in global
                    const float3 v_rot = make_float3(0.0, rotVelCPB.y - rotVelCPA.y, 0.0);
                    // This v_rot is only used for identifying resistance direction
                    const float v_rot_mag = length(v_rot);
                    if (v_rot_mag > DEME_TINY_FLOAT) {
                        // You should know that Crr_cnt * normal_force is the underlying formula, and in our model,
                        // it is a `force' that produces torque only, instead of also cancelling out friction.
                        // Its direction is that it `resists' rotation, see picture in
                        // https://en.wikipedia.org/wiki/Rolling_resistance.
                        torque_only_force = (v_rot / v_rot_mag) * (Crr_cnt * length(force));
                    }
                }
            }

            // Tangential force part
            if (mu_cnt > 0.0) {
                const float kt = 8. * G_cnt * sqrt_Rd;
                const float gt = -deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(mass_eff * kt);
                float3 tangent_force = -kt * delta_tan - gt * vrel_tan;
                const float ft = length(tangent_force);
                if (ft > DEME_TINY_FLOAT) {
                    // Reverse-engineer to get tangential displacement
                    const float ft_max = length(force) * mu_cnt;
                    if (ft > ft_max) {
                        tangent_force = (ft_max / ft) * tangent_force;
                        delta_tan = (tangent_force + gt * vrel_tan) / (-kt);
                    }
                } else {
                    tangent_force = make_float3(0, 0, 0);
                }
                // Use force to collect tangent_force
                force += tangent_force;
            }

            // Make sure we update those wildcards (in this case, contact history)
            delta_tan_x = delta_tan.x;
            delta_tan_y = 0.0;
            delta_tan_z = delta_tan.z;
            force.y = 0.0;
        }
    """

    return model

if __name__ == "__main__":
    ball_density = 6.2e3
    H = 0.1
    R = 0.0254 / 2.0

    terrain_rad = 0.006 / 2.0

    out_dir = "DemoOutput_BallDrop2D/"
    out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    DEMSim = DEME.DEMSolver()
    DEMSim.SetVerbosity("STEP_METRIC")
    DEMSim.SetOutputFormat("CSV")
    DEMSim.SetOutputContent(["ABSV", "XYZ"])
    DEMSim.SetMeshOutputFormat("VTK")

    # E, nu, CoR, mu, Crr...
    mat_type_ball = DEMSim.LoadMaterial(
        {"E": 7e7, "nu": 0.24, "CoR": 0.9, "mu": 0.3, "Crr": 0.0}
    )
    mat_type_terrain = DEMSim.LoadMaterial(
        {"E": 7e7, "nu": 0.24, "CoR": 0.9, "mu": 0.3, "Crr": 0.0}
    )
    mat_type_terrain_sim = DEMSim.LoadMaterial(
        {"E": 7e7, "nu": 0.24, "CoR": 0.9, "mu": 0.3, "Crr": 0.0}
    )

    step_size = 2e-6
    world_size = 0.2
    DEMSim.InstructBoxDomainDimension(
        tuple([-world_size / 2.0, world_size / 2.0]),
        tuple([-world_size / 2.0, world_size / 2.0]),
        tuple([0, 10 * world_size]),
    )
    DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain)

    projectile = DEMSim.AddWavefrontMeshObject(
        DEME.GetDEMEDataFile("mesh/sphere.obj"), mat_type_ball
    )
    projectile.Scale(R)
    print(f"Total num of triangles: {projectile.GetNumTriangles()}")

    projectile.SetInitPos([0, 0, 8 * world_size])
    ball_mass = ball_density * 4.0 / 3.0 * np.pi * R * R * R
    projectile.SetMass(ball_mass)
    projectile.SetMOI(
        [
            ball_mass * 2 / 5 * R * R,
            ball_mass * 2 / 5 * R * R,
            ball_mass * 2 / 5 * R * R,
        ]
    )
    projectile.SetFamily(2)
    DEMSim.SetFamilyFixed(2)
    DEMSim.DisableContactBetweenFamilies(0, 2)
    # Track the projectile
    proj_tracker = DEMSim.Track(projectile)

    # Force model to use
    # model2D = DEMSim.ReadContactForceModel("ForceModel2D.cu")
    model2D = DEMSim.DefineContactForceModel(force_model())
    model2D.SetMustHaveMatProp(set(["E", "nu", "CoR", "mu", "Crr"]))
    model2D.SetMustPairwiseMatProp(set(["CoR", "mu", "Crr"]))
    model2D.SetPerContactWildcards(
        set(["delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z"])
    )

    templates_terrain = []
    num_templates = 11
    for i in range(num_templates):
        templates_terrain.append(
            DEMSim.LoadSphereType(
                terrain_rad * terrain_rad * terrain_rad * 2.0e3 * 4 / 3 * np.pi,
                terrain_rad,
                mat_type_terrain,
            )
        )
        terrain_rad += 0.0001 / 2.0

    num_particle = 0
    sample_z = 1.5 * terrain_rad
    fullheight = world_size * 6.0
    sample_halfwidth = world_size / 2 - 2 * terrain_rad
    init_v = 0.01

    sampler = DEME.HCPSampler(2.05 * terrain_rad)

    sample_center = [0, 0, fullheight / 2 + 1 * terrain_rad]
    input_xyz = sampler.SampleBox(
        sample_center, [sample_halfwidth, 0.0, fullheight / 2.0]
    )
    template_to_use = []
    # Randomly select from pre - generated clump templates
    for i in range(len(input_xyz)):
        template_to_use.append(templates_terrain[random.randint(0, num_templates - 1)])

    DEMSim.AddClumps(template_to_use, input_xyz)
    num_particle += len(input_xyz)

    print(f"Total num of particles: {num_particle}")

    max_z_finder = DEMSim.CreateInspector("clump_max_z")
    total_mass_finder = DEMSim.CreateInspector("clump_mass")

    DEMSim.SetInitTimeStep(step_size)
    DEMSim.SetMaxVelocity(30.0)
    DEMSim.SetGravitationalAcceleration([0, 0, -9.81])

    DEMSim.Initialize()

    sim_time = 3.0
    settle_time = 1.0
    fps = 20
    frame_time = 1.0 / fps
    out_steps = (int)(1.0 / (fps * step_size))

    print(f"Output at {fps} FPS")
    currframe = 0

    # We can let it settle first
    t = 0.0
    while t < settle_time:
        print(f"Frame: {currframe}", flush=True)
        filename = os.path.join(out_dir, f"DEMdemo_output_{currframe:04d}.csv")
        meshname = os.path.join(out_dir, f"DEMdemo_mesh_{currframe:04d}.vtk")
        DEMSim.WriteSphereFile(filename)
        DEMSim.WriteMeshFile(meshname)
        currframe += 1

        DEMSim.DoDynamicsThenSync(frame_time)
        DEMSim.ShowThreadCollaborationStats()
        t += frame_time

    cp_filename = os.path.join(out_dir, f"bed.csv")
    DEMSim.WriteClumpFile(cp_filename)

    # This is to show that you can change the material for all the particles in a family... although here,
    # mat_type_terrain_sim and mat_type_terrain are the same material so there is no effect  you can define
    # them differently though.
    DEMSim.SetFamilyClumpMaterial(0, mat_type_terrain_sim)
    DEMSim.DoDynamicsThenSync(0.2)
    terrain_max_z = max_z_finder.GetValue()
    matter_mass = total_mass_finder.GetValue()
    total_volume = (world_size * world_size) * (terrain_max_z - 0.0)
    bulk_density = matter_mass / total_volume
    print(f"Original terrain height: {terrain_max_z}")
    print(f"Bulk density: {bulk_density}")

    # Then drop the ball
    DEMSim.ChangeFamily(2, 0)
    proj_tracker.SetPos([0, 0, terrain_max_z + R + H])

    t = 0.0
    while t < sim_time:
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

        if (abs(proj_tracker.Vel()[2]) < 1e-4) and (t > 0.1):
            break

        DEMSim.ShowTimingStats()

    DEMSim.ShowMemStats()

    final_pos = proj_tracker.Pos()
    print(f"Ball density: {ball_density}")
    print(f"Ball rad: {R}")
    print(f"Drop height: {H}")
    # final_pos[2] is the z coord
    print(f"Penetration: {terrain_max_z - (final_pos[2] - R)}")
    print(f"==============================================================")
    print(f"DEMdemo_BallDrop exiting...")
