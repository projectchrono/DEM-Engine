//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// This demo lets a rod with some electric charges stick into a pile of granular
// material that is also charged. The electrostatic force shows its effect. The
// electric charges are even moving during the simulation. This is done through
// `Geometry Wildcard', where the amount of charges is associated with each
// sphere component (of clumps) and triangles (of meshes). Then a custom force
// model is used to derive the electrostatic force in addition to contact forces.
//
// NOTE: If you want to create your own force model, it's probably a good idea
// to understand the default model in the file FullHertzianForceModel.cu
// (Hertzian--Mindlin), how the normal and tangential forces are calculated,
// then use it as a starting point to add your own force, like in this demo.
// =============================================================================

#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace deme;

const double math_PI = 3.1415927;

// For clarity, we attach the force model file at the end of this script.
// This model can be found in file ForceModelWithElectrostatic.cu, too.
std::string force_model();

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat("CSV");
    DEMSim.SetOutputContent({"ABSV"});
    DEMSim.SetMeshOutputFormat("VTK");
    DEMSim.SetContactOutputContent({"OWNER", "FORCE", "POINT"});

    // If you don't need individual force information, then this option makes the solver run a bit faster.
    DEMSim.SetNoForceRecord();

    // E, nu, CoR, mu, Crr...
    auto mat_type_rod = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.7}, {"Crr", 0.00}});
    auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}, {"Crr", 0.00}});
    // If you don't have this line, then values will take average between 2 materials, when they are in contact
    DEMSim.SetMaterialPropertyPair("CoR", mat_type_rod, mat_type_terrain, 0.8);
    DEMSim.SetMaterialPropertyPair("mu", mat_type_rod, mat_type_terrain, 0.7);
    // We can specify the force model using a string.
    // This force model is the standard Hertzian--Mindlin model, plus an electrostatic force, based on something
    // called a `geometry wildcard'. This is an extra property that we can associate with each geometry entity,
    // such as triangle and sphere. We use this value to derive the electrostatic force. But first, we need to
    // declare in the force model that a geometry wildcard is in use...
    auto my_force_model = DEMSim.DefineContactForceModel(force_model());
    // auto my_force_model = DEMSim.ReadContactForceModel("ForceModelWithElectrostatic.cu");

    // Those following lines are needed. We must let the solver know that those var names are history variable etc.
    my_force_model->SetMustHaveMatProp({"E", "nu", "CoR", "mu", "Crr"});
    my_force_model->SetMustPairwiseMatProp({"CoR", "mu", "Crr"});
    my_force_model->SetPerContactWildcards({"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z"});
    // Use variable name `Q' for the amount of electrc charge.
    // NOTE! If you call it Q here, then you can refer to this wildcard array using variable names Q_A amd Q_B in
    // your custom force model.
    my_force_model->SetPerGeometryWildcards({"Q"});

    float init_charge = 2e-8;  // Coulomb as the unit...
    float cone_speed = 0.1;
    float step_size = 5e-6;
    double world_size = 2;
    double soil_bin_diameter = 0.584;
    double rod_length = 0.5;
    double rod_surf_area = 323e-6;
    double rod_diameter = std::sqrt(rod_surf_area / math_PI) * 2;
    DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size);
    // No need to add simulation `world' boundaries, b/c we'll add a cylinderical container manually
    DEMSim.InstructBoxDomainBoundingBC("none", mat_type_terrain);
    // Now add a cylinderical boundary along with a bottom plane
    double bottom = -0.5;
    auto walls = DEMSim.AddExternalObject();
    walls->AddCylinder(make_float3(0), make_float3(0, 0, 1), soil_bin_diameter / 2., mat_type_terrain, 0);
    walls->AddPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);
    walls->AddPlane(make_float3(0, 0, world_size / 2. - world_size / 20.), make_float3(0, 0, -1), mat_type_terrain);

    // Define the terrain particle templates
    // Calculate its mass and MOI
    float terrain_density = 2.6e3;
    double clump_vol = 4. / 3. * math_PI;
    float mass = terrain_density * clump_vol;
    float3 MOI = make_float3(2. / 5.) * mass;
    // Then load it to system
    std::shared_ptr<DEMClumpTemplate> my_template =
        DEMSim.LoadClumpType(mass, MOI, GetDEMEDataFile("clumps/spiky_sphere.csv"), mat_type_terrain);
    my_template->SetVolume(clump_vol);
    // Decide the scalings of the templates we just created (so that they are... like particles, not rocks)
    double scale = 0.01;
    my_template->Scale(scale);

    // Sampler to sample
    GridSampler sampler(scale * 2.4);
    float fill_height = 1.;
    float3 fill_center = make_float3(0, 0, bottom + fill_height / 2);
    const float fill_radius = soil_bin_diameter / 2. - scale * 3.;
    auto input_xyz = sampler.SampleCylinderZ(fill_center, fill_radius, fill_height / 2 - scale * 2.);
    auto particles = DEMSim.AddClumps(my_template, input_xyz);
    std::cout << "Total num of particles: " << particles->GetNumClumps() << std::endl;
    std::cout << "Total num of spheres: " << particles->GetNumSpheres() << std::endl;
    // Add electric charge Q to each sphere. It is important to add it to each sphere, not each clump. This
    // is because the contact pairs are resolved between geometries, not clumps. Using clumps leads to
    // double-count or triple-count or...
    particles->AddGeometryWildcard("Q", std::vector<float>(particles->GetNumSpheres(), init_charge));

    // Load in the cone used for this penetration test
    auto rod_body = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/cyl_r1_h2.obj"), mat_type_rod);
    unsigned int num_tri = rod_body->GetNumTriangles();
    std::cout << "Total num of triangles: " << num_tri << std::endl;

    // The define the properties of the rod
    float body_mass = 7.8e3 * math_PI;
    rod_body->SetMass(body_mass);
    rod_body->SetMOI(make_float3(body_mass * 7 / 12, body_mass * 7 / 12, body_mass / 2));
    // This cyl mesh (h = 2m, r = 1m) has its center at the origin. So the following call actually has no effect...
    rod_body->InformCentroidPrincipal(make_float3(0, 0, 0), make_float4(0, 0, 0, 1));
    rod_body->Scale(make_float3(rod_diameter / 2., rod_diameter / 2., rod_length / 2.));
    rod_body->SetFamily(1);
    // Just fix it: We will manually impose its motion later.
    DEMSim.SetFamilyFixed(1);
    // We can set the geometry wildcard Q here. But we'll instead show how to modify that using a tracker.
    // rod_body->AddGeometryWildcard("Q", std::vector<float>(num_tri, init_charge));

    // Track the rod
    auto rod_tracker = DEMSim.Track(rod_body);

    // Some inspectors
    auto max_z_finder = DEMSim.CreateInspector("clump_max_z");

    // This ensures that the force model is in effect even for some `contacts' that are not physical contacts, since the
    // electrostatic force should work in a distance. We assume that this effect won't be longer than 7mm. This means
    // for any particle, it can affect another particle (to be considered in contact with) for up to 7mm away from it.
    DEMSim.SetFamilyExtraMargin(0, 0.007);
    DEMSim.SetFamilyExtraMargin(1, 0.007);
    // If you know your extra margin policy gonna make the average number of contacts per sphere huge, then set this so
    // that the solver does not error out when checking it.
    DEMSim.SetErrorOutAvgContacts(100);

    DEMSim.SetInitTimeStep(step_size);
    // In this test we don't want dT to drift too much ahead of kT, or the contact margin could be too large, resulting
    // in too many false positive contacts, making the solver slower.
    DEMSim.SetCDMaxUpdateFreq(60);

    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.Initialize();

    std::filesystem::path out_dir = std::filesystem::current_path();
    out_dir /= "DemoOutput_Electrostatic";
    std::filesystem::create_directory(out_dir);

    float sim_end = 9.0;
    unsigned int fps = 20;
    float frame_time = 1.0 / fps;
    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));
    unsigned int frame_count = 0;
    unsigned int step_count = 0;

    // Settle
    DEMSim.DisableContactBetweenFamilies(0, 1);
    for (float t = 0; t < 1.; t += frame_time) {
        char filename[100], meshname[100];
        std::cout << "Outputting frame: " << frame_count << std::endl;
        sprintf(filename, "DEMdemo_output_%04d.csv", frame_count);
        sprintf(meshname, "DEMdemo_mesh_%04d.vtk", frame_count++);
        DEMSim.WriteSphereFile(out_dir / filename);
        DEMSim.WriteMeshFile(out_dir / meshname);
        DEMSim.ShowThreadCollaborationStats();

        DEMSim.DoDynamics(frame_time);
    }
    DEMSim.DoDynamicsThenSync(0);

    // Put the cone in place
    float terrain_max_z = max_z_finder->GetValue();
    double current_height = terrain_max_z + 0.03;
    // Its initial position should be right above the granular material (but accounting for the fact that the coordinate
    // system center of the rod is in its middle)
    rod_tracker->SetPos(make_float3(0, 0, rod_length / 2. + current_height));

    DEMSim.EnableContactBetweenFamilies(0, 1);

    // We demonstrate using trackers to set a geometry wildcard. Q is now set for each triangle facet, and it's
    // the opposite charge to the particles. So the rod should attract the particles.
    rod_tracker->SetGeometryWildcardValues("Q", std::vector<float>(num_tri, -10. * init_charge));

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (float t = 0; t < sim_end; t += step_size, step_count++) {
        if (step_count % out_steps == 0) {
            char filename[100], meshname[100];
            std::cout << "Outputting frame: " << frame_count << std::endl;
            sprintf(filename, "DEMdemo_output_%04d.csv", frame_count);
            sprintf(meshname, "DEMdemo_mesh_%04d.vtk", frame_count++);
            DEMSim.WriteSphereFile(out_dir / filename);
            DEMSim.WriteMeshFile(out_dir / meshname);
            DEMSim.ShowThreadCollaborationStats();
        }

        // Means advance simulation by one time step
        DEMSim.DoStepDynamics();
        // The rod first goes down into the material, then goes up, then stay in place to let us watch how the particles
        // affected by the electrostatic force move.
        if (t < 1. / 3. * sim_end) {
            current_height -= cone_speed * step_size;
        } else if (t < 2. / 3. * sim_end) {
            current_height += cone_speed * step_size;
        }  // else the rod does not move
        rod_tracker->SetPos(make_float3(0, 0, rod_length / 2. + current_height));
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << time_sec.count() << " seconds (wall time) to finish the simulation" << std::endl;

    DEMSim.ShowTimingStats();

    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Electrostatic demo exiting..." << std::endl;
    return 0;
}

std::string force_model() {
    std::string model = R"V0G0N(
        /////////////////////////////////////////////////////////////
        // The first part is just the standard full Hertzian--Mindlin
        /////////////////////////////////////////////////////////////

        // No need to do any contact force calculation if no contact. And it can happen,
        // since we added extra contact margin for adding electrostatic forces before 
        // physical contacts emerge.
        if (overlapDepth > 0) {
            // Material properties
            float E_cnt, G_cnt, CoR_cnt, mu_cnt, Crr_cnt;
            {
                // E and nu are associated with each material, so obtain them this way
                float E_A = E[bodyAMatType];
                float nu_A = nu[bodyAMatType];
                float E_B = E[bodyBMatType];
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
                applyOriQToVector3<float, deme::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AOriQ.w, AOriQ.x, AOriQ.y, AOriQ.z);
                applyOriQToVector3<float, deme::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BOriQ.w, BOriQ.x, BOriQ.y, BOriQ.z);
            }

            // A few re-usables
            float mass_eff, sqrt_Rd, beta;
            float3 vrel_tan;
            float3 delta_tan = make_float3(delta_tan_x, delta_tan_y, delta_tan_z);

            // Normal force part
            {
                // The (total) relative linear velocity of A relative to B
                const float3 velB2A = (ALinVel + rotVelCPA) - (BLinVel + rotVelCPB);
                const float projection = dot(velB2A, B2A);
                vrel_tan = velB2A - projection * B2A;

                // Now we already have sufficient info to update contact history
                {
                    delta_tan += ts * vrel_tan;
                    const float disp_proj = dot(delta_tan, B2A);
                    delta_tan -= disp_proj * B2A;
                    delta_time += ts;
                }

                mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);
                sqrt_Rd = sqrt(overlapDepth * (ARadius * BRadius) / (ARadius + BRadius));
                const float Sn = 2. * E_cnt * sqrt_Rd;

                const float loge = (CoR_cnt < DEME_TINY_FLOAT) ? log(DEME_TINY_FLOAT) : log(CoR_cnt);
                beta = loge / sqrt(loge * loge + deme::PI_SQUARED);

                const float k_n = deme::TWO_OVER_THREE * Sn;
                const float gamma_n = deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

                force += (k_n * overlapDepth + gamma_n * projection) * B2A;
                // printf("normal force: %f, %f, %f\n", force.x, force.y, force.z);
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
                    const float3 v_rot = rotVelCPB - rotVelCPA;
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
                // printf("tangent force: %f, %f, %f\n", tangent_force.x, tangent_force.y, tangent_force.z);
            }

            // Finally, make sure we update those wildcards (in this case, contact history)
            delta_tan_x = delta_tan.x;
            delta_tan_y = delta_tan.y;
            delta_tan_z = delta_tan.z;
        } else { 
            // This is to be more rigorous. If in fact no physical contact, then contact wildcards (such as contact history)
            // should be cleared (they will also be automatically cleared if the contact is no longer detected).
            delta_time = 0;
            delta_tan_x = 0;
            delta_tan_y = 0;
            delta_tan_z = 0;
        }

        /////////////////////////////////////////////////
        // Now we add an extra electrostatic force
        ////////////////////////////////////////////////
        {
            const float k = 8.99e9;
            const double ABdist2 = dot(bodyAPos - bodyBPos, bodyAPos - bodyBPos);
            // If Q_A and Q_B are the same sign, then the force pushes A away from B, so B2A is the direction.
            force += k * Q_A[AGeo] * Q_B[BGeo] / ABdist2 * (B2A);
            // Fun part: we can modify the electric charge on the fly. But we have to use atomic, since multiple contacts
            // can modify the same Q.
            // But this is not recommend unless you understand what you are doing, and there are a lot of details related to it.
            // For example, although the charges transfer between geometries, the geometries within one clump cannot
            // re-distribute elec charges among them, since no contact among geometries in one clump. Still, you could write
            // your own subroutine to further modify those geometry and/or own wildcards in your script, or within the force
            // model. 
            // On the other hand, if you do not need to modify the wildcards, you just need to use them for calculating
            // the force, then that is probably easier and with less strings attached to it. I can see this being more
            // useful.
            if (overlapDepth > 0) {  // Exchange the elec charge only in physical contact
                float avg_Q = (Q_A[AGeo] + Q_B[BGeo]) / 2.;
                float A_change_dir = (abs(avg_Q - Q_A[AGeo]) > 1e-11) ? (avg_Q - Q_A[AGeo]) / abs(avg_Q - Q_A[AGeo]) : 0.;
                // Modify the charge they carry... the rate is 1e-8 per second
                atomicAdd(Q_A + AGeo, A_change_dir * 1e-8 * ts);
                atomicAdd(Q_B + BGeo, -A_change_dir * 1e-8 * ts);
            }
        }
    )V0G0N";
    return model;
}
