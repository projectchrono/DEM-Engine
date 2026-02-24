//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Sharp-point grazing plane test: a cone mesh is driven at prescribed
// horizontal velocity so that its tip grazes along a flat plane mesh (no gravity).
// Uses cone.obj and plane_20by20.obj.
// We inspect the time evolution of the contact force magnitude and direction
// to verify stability of the mesh-mesh contact during grazing.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/HostSideHelpers.hpp>

#include <cmath>
#include <iostream>
#include <vector>
#include <filesystem>
#include <system_error>

using namespace std::filesystem;
using namespace deme;

int main() {
    // =========================================================================
    // Parameters
    // =========================================================================
    // Material properties (same for cone and plane)
    const float mu = 0.3f;   // friction coefficient
    const float CoR = 0.3f;  // coefficient of restitution
    const float E = 1e8f;    // Young's modulus (Pa)
    const float nu = 0.3f;   // Poisson's ratio

    // Cone geometry (cone.obj: tip at origin, base at z=1, base radius=1)
    const float cone_scale = 0.2f;         // scale factor: tip stays at origin
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

    // Grazing motion
    const float graze_speed = 0.5f;        // prescribed x-velocity (m/s)
    const float tip_penetration = 0.005f;  // initial tip depth below plane (m)

    // The cone centroid is at 3/4 of height from tip (in original mesh coordinates).
    // After Scale(cone_scale), centroid is at z = 0.75 * cone_scale above the tip.
    const float centroid_above_tip = 0.75f * cone_scale;

    // Simulation time settings
    const float step_size = 1e-5f;
    const float frame_time = 0.2f;
    const float total_time = 2.0f;

    std::cout << "=====================================================" << std::endl;
    std::cout << "DEM Sharp-Point Cone Grazing Plane Test" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "Cone scale:        " << cone_scale << " m" << std::endl;
    std::cout << "Cone mass:         " << cone_mass << " kg" << std::endl;
    std::cout << "Graze speed:       " << graze_speed << " m/s" << std::endl;
    std::cout << "Tip penetration:   " << tip_penetration << " m" << std::endl;
    std::cout << "Total graze dist:  " << graze_speed * total_time << " m" << std::endl;
    std::cout << "=====================================================" << std::endl;

    // =========================================================================
    // Solver setup
    // =========================================================================
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.InstructBoxDomainDimension(30, 30, 5);
    // No gravity: we are controlling the cone position/velocity explicitly
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, 0));
    // Enable mesh-mesh contacts (required for cone-plane mesh contact)
    DEMSim.SetMeshUniversalContact(true);
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.SetExpandSafetyAdder(graze_speed);

    auto mat = DEMSim.LoadMaterial({{"E", E}, {"nu", nu}, {"CoR", CoR}, {"mu", mu}, {"Crr", 0.0f}});

    // =========================================================================
    // Flat plane mesh: plane_20by20.obj (fixed)
    // The plane has vertices at z=0 with normal (0,0,1).
    // =========================================================================
    auto plane = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/plane_20by20.obj").string(), mat);
    plane->Scale(0.1f);
    plane->SetMass(1.0f);
    plane->SetMOI(make_float3(1.0f, 1.0f, 1.0f));
    plane->SetInitPos(make_float3(0.f, 0.f, 0.f));
    plane->SetFamily(100);
    DEMSim.SetFamilyFixed(100);

    // =========================================================================
    // Cone mesh: cone.obj (prescribed horizontal velocity)
    // cone.obj: tip at origin (0,0,0), cone axis along +z, base at z=1, radius=1.
    // After InformCentroidPrincipal, centroid is the reference; tip is at
    //   (0, 0, -0.75) in local frame.
    // After Scale(cone_scale), tip is at (0, 0, -0.75*cone_scale) in local frame.
    // To place tip at z = -tip_penetration (below plane at z=0):
    //   SetInitPos z = -tip_penetration + centroid_above_tip
    // Cone grazes from x=-0.2 toward x = init_x + graze_speed * total_time.
    // =========================================================================
    const float cone_init_x = -0.2f;
    const float cone_init_z = -tip_penetration + centroid_above_tip;

    auto cone = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cone.obj").string(), mat);
    // Set centroid location (3/4 of cone height from tip) and principal axes
    // before scaling, in original mesh coordinates.
    cone->InformCentroidPrincipal(make_float3(0.f, 0.f, 0.75f), make_float4(0.f, 0.f, 0.f, 1.f));
    cone->Scale(cone_scale);
    cone->SetMass(cone_mass);
    cone->SetMOI(cone_moi);
    cone->SetInitPos(make_float3(cone_init_x, 0.f, cone_init_z));
    cone->SetFamily(1);
    auto cone_tracker = DEMSim.Track(cone);

    // Prescribe: constant horizontal velocity, no vertical motion, no rotation.
    DEMSim.SetFamilyPrescribedLinVel(1, to_string_with_precision(graze_speed), "0", "0");
    DEMSim.SetFamilyPrescribedAngVel(1, "0", "0", "0");

    // =========================================================================
    // Initialize and run
    // =========================================================================
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.Initialize();

    // After initialization, set cone velocity (prescribed family uses expression
    // strings, but tracker SetVel provides an initial push for the first step).
    cone_tracker->SetVel(make_float3(graze_speed, 0.f, 0.f));

    std::cout << "\nCone centroid init pos: (" << cone_init_x << ", 0, " << cone_init_z << ")" << std::endl;
    std::cout << "Cone tip init z:        " << cone_init_z - centroid_above_tip << " m  (plane at z=0)" << std::endl;
    std::cout << "Running simulation for " << total_time << " s ..." << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    path out_dir = current_path() / "modular_test_output" / "DEMTest_ConeGrazingPlane";
    std::error_code dir_ec;
    create_directories(out_dir, dir_ec);
    if (dir_ec || !is_directory(out_dir)) {
        std::cerr << "Failed to create output directory: " << out_dir << " (" << dir_ec.message() << ")"
                  << std::endl;
        return 1;
    }

    std::vector<double> force_mags;
    std::vector<double> force_z_components;
    const int n_frames = static_cast<int>(total_time / frame_time);

    for (int i = 1; i <= n_frames; i++) {
        // Output
        char meshfilename[200];
        sprintf(meshfilename, "DEMTest_mesh_%04d.vtk", i);
        DEMSim.WriteMeshFile(out_dir / meshfilename);

        DEMSim.DoDynamics(frame_time);

        // Contact force = contact acceleration * mass
        float3 cnt_acc = cone_tracker->ContactAcc();
        float3 cnt_force = cnt_acc * cone_mass;
        double force_mag = std::sqrt((double)cnt_force.x * cnt_force.x + (double)cnt_force.y * cnt_force.y +
                                     (double)cnt_force.z * cnt_force.z);
        force_mags.push_back(force_mag);
        force_z_components.push_back(static_cast<double>(cnt_force.z));

        float3 pos = cone_tracker->Pos();
        // Tip position = centroid position - centroid_above_tip in z
        float tip_x = pos.x;
        float tip_z = pos.z - centroid_above_tip;

        std::cout << "t=" << i * frame_time << "s"
                  << "  tip_x=" << tip_x << " m"
                  << "  tip_z=" << tip_z << " m"
                  << "  |F_cnt|=" << force_mag << " N";

        if (force_mag > 1e-3) {
            float inv_f = static_cast<float>(1.0 / force_mag);
            std::cout << "  F_dir=(" << cnt_force.x * inv_f << "," << cnt_force.y * inv_f << "," << cnt_force.z * inv_f
                      << ")";
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
    double sum_fz = 0.0, min_fz = force_z_components[0], max_fz = force_z_components[0];
    for (size_t k = 0; k < force_mags.size(); k++) {
        double f = force_mags[k];
        double fz = force_z_components[k];
        sum_f += f;
        sum_fz += fz;
        if (f < min_f)
            min_f = f;
        if (f > max_f)
            max_f = f;
        if (fz < min_fz)
            min_fz = fz;
        if (fz > max_fz)
            max_fz = fz;
    }
    double mean_f = sum_f / static_cast<double>(force_mags.size());
    double mean_fz = sum_fz / static_cast<double>(force_z_components.size());
    double var = 0.0;
    for (double f : force_mags) {
        double d = f - mean_f;
        var += d * d;
    }
    double stddev_f = std::sqrt(var / static_cast<double>(force_mags.size()));

    std::cout << "\n=== Contact Force Statistics (over " << force_mags.size() << " frames) ===" << std::endl;
    std::cout << "  Total force magnitude:" << std::endl;
    std::cout << "    Mean:   " << mean_f << " N" << std::endl;
    std::cout << "    Min:    " << min_f << " N" << std::endl;
    std::cout << "    Max:    " << max_f << " N" << std::endl;
    std::cout << "    StdDev: " << stddev_f << " N" << std::endl;
    std::cout << "  Normal (z) component (should be positive = pushing cone up):" << std::endl;
    std::cout << "    Mean:   " << mean_fz << " N" << std::endl;
    std::cout << "    Min:    " << min_fz << " N" << std::endl;
    std::cout << "    Max:    " << max_fz << " N" << std::endl;

    DEMSim.ShowTimingStats();
    std::cout << "=====================================================" << std::endl;
    std::cout << "DEMTest_ConeGrazingPlane exiting..." << std::endl;
    return 0;
}
