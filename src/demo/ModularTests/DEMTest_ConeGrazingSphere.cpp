//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Cone-tip grazing sphere test: a cone mesh is driven in a circular orbit so
// that its tip grazes the surface of a fixed sphere mesh (no gravity).
// Uses cone.obj and sphere.obj from the data folder.
// The sphere is fixed at the origin; the cone orbits around it at a prescribed
// angular velocity, keeping the tip in contact with the sphere surface.
// We inspect the time evolution of the contact force magnitude and direction
// to verify stability of the mesh-mesh contact during circular grazing.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/HostSideHelpers.hpp>

#include <cmath>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace std::filesystem;
using namespace deme;

int main() {
    // =========================================================================
    // Parameters
    // =========================================================================
    // Material properties (same for cone and sphere)
    const float mu = 0.3f;   // friction coefficient
    const float CoR = 0.3f;  // coefficient of restitution
    const float E = 1e8f;    // Young's modulus (Pa)
    const float nu = 0.3f;   // Poisson's ratio

    // Sphere geometry (sphere.obj: unit sphere centered at origin, radius=1)
    const float sphere_scale = 0.25f;  // sphere radius after scaling (m)

    // Cone geometry (cone.obj: tip at origin, base at z=1, base radius=1)
    const float cone_scale = 0.2f;         // scale factor
    const float cone_height = cone_scale;  // 0.2 m after scaling
    const float cone_radius = cone_scale;  // 0.2 m base radius after scaling
    const float cone_density = 2600.0f;    // kg/m^3
    const float cone_volume = (1.0f / 3.0f) * PI * cone_radius * cone_radius * cone_height;
    const float cone_mass = cone_density * cone_volume;
    // MOI of solid cone about centroid (at 3/4 height from tip = 3/4 * cone_height)
    // Ix = Iy = 3m/20*r^2 + 3m/80*h^2, Iz = 3m/10*r^2
    const float cone_Ixy =
        3.0f * cone_mass / 20.0f * cone_radius * cone_radius + 3.0f * cone_mass / 80.0f * cone_height * cone_height;
    const float cone_Iz = 3.0f * cone_mass / 10.0f * cone_radius * cone_radius;
    const float3 cone_moi = make_float3(cone_Ixy, cone_Ixy, cone_Iz);

    // Orbit parameters
    const float orbit_omega = 2.0f;        // angular speed of orbit (rad/s)
    const float tip_penetration = 0.005f;  // initial tip depth into sphere surface (m)

    // The cone centroid is at 3/4 of height from tip (in original mesh coordinates).
    // After Scale(cone_scale), centroid is at z = 0.75 * cone_scale above the tip.
    const float centroid_above_tip = 0.75f * cone_scale;

    // The cone centroid orbits at a radius such that the tip is at
    // (sphere_scale - tip_penetration) from the sphere center.
    const float orbit_radius = sphere_scale - tip_penetration + centroid_above_tip;

    // Tangential speed of the orbiting centroid
    const float orbit_vel = orbit_radius * orbit_omega;

    // Simulation time settings
    const float step_size = 5e-6f;
    const float frame_time = 0.2f;
    const int n_sub_samples = 500;                                        // sub-samples per frame for force averaging
    const float sub_dt = frame_time / static_cast<float>(n_sub_samples);  // sub-step size
    const float total_time = 2.0f;

    std::cout << "=====================================================" << std::endl;
    std::cout << "DEM Cone-Tip Grazing Sphere Test" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "Sphere radius:     " << sphere_scale << " m (fixed)" << std::endl;
    std::cout << "Cone scale:        " << cone_scale << " m" << std::endl;
    std::cout << "Cone mass:         " << cone_mass << " kg" << std::endl;
    std::cout << "Orbit radius:      " << orbit_radius << " m (centroid)" << std::endl;
    std::cout << "Tip-sphere gap:    " << -(tip_penetration) << " m (negative = penetration)" << std::endl;
    std::cout << "Orbit omega:       " << orbit_omega << " rad/s" << std::endl;
    std::cout << "Orbit speed:       " << orbit_vel << " m/s" << std::endl;
    std::cout << "Total graze angle: " << orbit_omega * total_time << " rad" << std::endl;
    std::cout << "=====================================================" << std::endl;

    // =========================================================================
    // Solver setup
    // =========================================================================
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.InstructBoxDomainDimension(5, 5, 5);
    // No gravity: we are controlling the cone motion explicitly
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, 0));
    // Enable mesh-mesh contacts (required for cone-sphere mesh contact)
    DEMSim.SetMeshUniversalContact(true);
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.SetExpandSafetyAdder(orbit_vel);

    auto mat = DEMSim.LoadMaterial({{"E", E}, {"nu", nu}, {"CoR", CoR}, {"mu", mu}, {"Crr", 0.0f}});

    // =========================================================================
    // Sphere mesh: sphere.obj (fixed at origin)
    // sphere.obj is a unit sphere centered at (0,0,0).
    // =========================================================================
    auto sphere = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/sphere.obj").string(), mat);
    sphere->Scale(sphere_scale);
    sphere->SetMass(1000.0f);
    sphere->SetMOI(make_float3(1000.0f, 1000.0f, 1000.0f));
    sphere->SetInitPos(make_float3(0.f, 0.f, 0.f));
    sphere->SetFamily(100);
    DEMSim.SetFamilyFixed(100);

    // =========================================================================
    // Cone mesh: cone.obj (prescribed circular orbit)
    // cone.obj: tip at origin (0,0,0), cone axis along +z, base at z=1, radius=1.
    //
    // We orient the cone so its axis (from tip to base) points in the +x direction
    // at t=0, which means the tip is on the -x side (toward the sphere center).
    // This requires rotating the cone's local +z axis to align with world +x:
    //   rotation about y-axis by +90 deg -> Q = (0, sin(pi/4), 0, cos(pi/4))
    //
    // Initial centroid position: (orbit_radius, 0, 0).
    //   Tip position = centroid - centroid_above_tip * axis_dir
    //                = (orbit_radius - centroid_above_tip, 0, 0)
    //                = (sphere_scale - tip_penetration, 0, 0)  [inside sphere surface]
    //
    // Prescribed motion:
    //   - Linear velocity: circular orbit in the xy-plane
    //       vx = -orbit_vel * sin(orbit_omega * t)
    //       vy =  orbit_vel * cos(orbit_omega * t)
    //   - Angular velocity: constant rotation about world z-axis at orbit_omega,
    //     keeping the cone axis (from tip to base) always pointing radially outward.
    // =========================================================================
    const float q_y = static_cast<float>(std::sqrt(2.0) / 2.0);  // sin(pi/4) = cos(pi/4)
    float4 cone_init_q = make_float4(0.f, q_y, 0.f, q_y);        // float4(qx,qy,qz,qw)

    auto cone = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cone.obj").string(), mat);
    cone->InformCentroidPrincipal(make_float3(0.f, 0.f, 0.75f), make_float4(0.f, 0.f, 0.f, 1.f));
    cone->Scale(cone_scale);
    cone->SetMass(cone_mass);
    cone->SetMOI(cone_moi);
    cone->SetInitPos(make_float3(orbit_radius, 0.f, 0.f));
    cone->SetInitQuat(cone_init_q);
    cone->SetFamily(1);
    auto cone_tracker = DEMSim.Track(cone);

    // Prescribe circular orbit: tangential velocity as a function of time.
    // The variable 't' represents simulation time in expression strings.
    std::string vx_expr =
        to_string_with_precision(-orbit_vel) + " * sin(" + to_string_with_precision(orbit_omega) + " * t)";
    std::string vy_expr =
        to_string_with_precision(orbit_vel) + " * cos(" + to_string_with_precision(orbit_omega) + " * t)";
    DEMSim.SetFamilyPrescribedLinVel(1, vx_expr, vy_expr, "0");
    // Prescribe constant angular velocity about world z-axis to keep cone
    // pointed radially outward as it orbits.
    DEMSim.SetFamilyPrescribedAngVel(1, to_string_with_precision(-orbit_omega), "0", "0");

    // =========================================================================
    // Initialize and run
    // =========================================================================
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.Initialize();

    // Set initial velocity to match the prescribed formula at t=0.
    // At t=0: vx = -orbit_vel*sin(0) = 0, vy = orbit_vel*cos(0) = orbit_vel.
    cone_tracker->SetVel(make_float3(0.f, orbit_vel, 0.f));

    const float tip_init_x = orbit_radius - centroid_above_tip;
    std::cout << "\nCone centroid init pos: (" << orbit_radius << ", 0, 0)" << std::endl;
    std::cout << "Cone tip init pos:      (" << tip_init_x << ", 0, 0) m  (sphere surface at r=" << sphere_scale << ")"
              << std::endl;
    std::cout << "Running simulation for " << total_time << " s ..." << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    path out_dir = current_path();
    out_dir /= "DEMTest_ConeGrazingSphere";
    create_directory(out_dir);

    std::vector<double> force_mags;
    std::vector<double> force_r_components;  // radial (outward) force component
    const int n_frames = static_cast<int>(total_time / frame_time);

    for (int i = 1; i <= n_frames; i++) {
        // Output
        char meshfilename[200];
        sprintf(meshfilename, "DEMTest_mesh_%04d.vtk", i);
        DEMSim.WriteMeshFile(out_dir / meshfilename);

        // Advance simulation in sub-steps and accumulate contact force readings
        // to compute the average force over this frame interval.
        double sum_fx = 0.0, sum_fy = 0.0, sum_fz = 0.0;
        for (int s = 0; s < n_sub_samples; s++) {
            DEMSim.DoDynamics(sub_dt);
            float3 cnt_acc = cone_tracker->ContactAcc();
            sum_fx += cnt_acc.x;
            sum_fy += cnt_acc.y;
            sum_fz += cnt_acc.z;
        }

        // Average contact force = (average contact acceleration) * mass
        float3 avg_cnt_force;
        avg_cnt_force.x = static_cast<float>(sum_fx / n_sub_samples) * cone_mass;
        avg_cnt_force.y = static_cast<float>(sum_fy / n_sub_samples) * cone_mass;
        avg_cnt_force.z = static_cast<float>(sum_fz / n_sub_samples) * cone_mass;
        double force_mag =
            std::sqrt((double)avg_cnt_force.x * avg_cnt_force.x + (double)avg_cnt_force.y * avg_cnt_force.y +
                      (double)avg_cnt_force.z * avg_cnt_force.z);
        force_mags.push_back(force_mag);

        // Radial direction at the center of this frame interval: cone is at angle
        // orbit_omega * (i - 0.5) * frame_time from the +x axis.
        float t_center = (i - 0.5f) * frame_time;
        float radial_x = std::cos(orbit_omega * t_center);
        float radial_y = std::sin(orbit_omega * t_center);
        // Radial force component (positive = pushes cone outward away from sphere)
        double force_r = avg_cnt_force.x * radial_x + avg_cnt_force.y * radial_y;
        force_r_components.push_back(force_r);

        float3 pos = cone_tracker->Pos();
        // Tip position = centroid - centroid_above_tip * axis_dir
        float t_end = static_cast<float>(i) * frame_time;
        float tip_r = std::sqrt(pos.x * pos.x + pos.y * pos.y) - centroid_above_tip;

        std::cout << "t=" << t_end << "s (avg over " << n_sub_samples << " sub-steps)"
                  << "  centroid_r=" << std::sqrt(pos.x * pos.x + pos.y * pos.y) << " m"
                  << "  tip_r~" << tip_r << " m"
                  << "  |F_cnt_avg|=" << force_mag << " N";

        if (force_mag > 1e-3) {
            float inv_f = static_cast<float>(1.0 / force_mag);
            std::cout << "  F_dir=(" << avg_cnt_force.x * inv_f << "," << avg_cnt_force.y * inv_f << ","
                      << avg_cnt_force.z * inv_f << ")";
        }
        std::cout << std::endl;
    }

    // =========================================================================
    // Statistics over the contact force time series
    // =========================================================================
    if (force_mags.empty()) {
        std::cout << "\nNo force data collected." << std::endl;
        return 0;
    }
    double sum_f = 0.0, min_f = force_mags[0], max_f = force_mags[0];
    double sum_fr = 0.0, min_fr = force_r_components[0], max_fr = force_r_components[0];
    for (size_t k = 0; k < force_mags.size(); k++) {
        double f = force_mags[k];
        double fr = force_r_components[k];
        sum_f += f;
        sum_fr += fr;
        if (f < min_f)
            min_f = f;
        if (f > max_f)
            max_f = f;
        if (fr < min_fr)
            min_fr = fr;
        if (fr > max_fr)
            max_fr = fr;
    }
    double mean_f = sum_f / static_cast<double>(force_mags.size());
    double mean_fr = sum_fr / static_cast<double>(force_r_components.size());
    double var = 0.0;
    for (double f : force_mags) {
        double d = f - mean_f;
        var += d * d;
    }
    double stddev_f = std::sqrt(var / static_cast<double>(force_mags.size()));

    std::cout << "\n=== Contact Force Statistics (over " << force_mags.size() << " frames, each averaged over "
              << n_sub_samples << " sub-steps) ===" << std::endl;
    std::cout << "  Total force magnitude:" << std::endl;
    std::cout << "    Mean:   " << mean_f << " N" << std::endl;
    std::cout << "    Min:    " << min_f << " N" << std::endl;
    std::cout << "    Max:    " << max_f << " N" << std::endl;
    std::cout << "    StdDev: " << stddev_f << " N" << std::endl;
    std::cout << "  Radial (outward) component (should be positive = pushing cone away from sphere):" << std::endl;
    std::cout << "    Mean:   " << mean_fr << " N" << std::endl;
    std::cout << "    Min:    " << min_fr << " N" << std::endl;
    std::cout << "    Max:    " << max_fr << " N" << std::endl;

    DEMSim.ShowTimingStats();
    std::cout << "=====================================================" << std::endl;
    std::cout << "DEMTest_ConeGrazingSphere exiting..." << std::endl;
    return 0;
}
