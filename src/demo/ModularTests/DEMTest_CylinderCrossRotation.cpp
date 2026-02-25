//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Cylinder-cylinder cross-to-co-axial rotation test: Two cylinders
// (cyl_r1_h2.obj) start in a "cross" configuration (axes perpendicular,
// intersecting at the origin). Cylinder A is fixed; Cylinder B is slowly
// rotated about the world y-axis at a constant angular rate until both
// cylinders are co-axial (axes parallel along z, with axis-direction overlap).
// At each frame we record the average contact-force magnitude and the averaged
// contact normal direction to show that both metrics evolve smoothly through
// the full 90-degree rotation, demonstrating the stability of the mesh-mesh
// contact derivation.
// Uses cyl_r1_h2.obj.
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
    // Material properties
    const float mu = 0.3f;   // friction coefficient
    const float CoR = 0.3f;  // coefficient of restitution
    const float E = 1e8f;    // Young's modulus (Pa)
    const float nu = 0.3f;   // Poisson's ratio

    // cyl_r1_h2.obj: cylinder with radius=1, height=2 (axis along z, z in [-1,1])
    const float cyl_scale = 0.1f;                      // scale factor
    const float cyl_radius = cyl_scale;                // 0.1 m after scaling
    const float cyl_height = 2.0f * cyl_scale;         // 0.2 m after scaling
    const float cyl_density = 2600.0f;                 // kg/m^3
    const float cyl_volume = (float)PI * cyl_radius * cyl_radius * cyl_height;
    const float cyl_mass = cyl_density * cyl_volume;
    // MOI of solid cylinder about centroid:
    //   Ixx = Iyy = m*(3*r^2 + h^2)/12,  Izz = m*r^2/2
    const float Ixy = cyl_mass * (3.0f * cyl_radius * cyl_radius + cyl_height * cyl_height) / 12.0f;
    const float Iz = cyl_mass * cyl_radius * cyl_radius / 2.0f;
    const float3 cyl_moi = make_float3(Ixy, Ixy, Iz);

    // Rotation: Cylinder B rotates from 90 deg (cross) to 0 deg (co-axial)
    // about the world y-axis. omega_y < 0 takes the axis from +x toward +z.
    const float total_time = 1.0f;  // s
    const float omega_y = -(float)(PI / 2.0) / total_time;  // rad/s

    // Simulation time settings
    const float step_size = 1e-5f;
    const float frame_time = 0.05f;  // 20 frames total
    const int n_sub_samples = 500;   // sub-samples per frame for force averaging
    const float sub_dt = frame_time / static_cast<float>(n_sub_samples);
    const int n_frames = static_cast<int>(total_time / frame_time);

    std::cout << "=====================================================" << std::endl;
    std::cout << "DEM Cylinder-Cylinder Cross-to-Co-axial Rotation Test" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "Cylinder radius:  " << cyl_radius << " m" << std::endl;
    std::cout << "Cylinder height:  " << cyl_height << " m" << std::endl;
    std::cout << "Cylinder mass:    " << cyl_mass << " kg" << std::endl;
    std::cout << "omega_y:          " << omega_y << " rad/s" << std::endl;
    std::cout << "Total rotation:   " << omega_y * total_time * 180.0f / (float)PI << " deg over " << total_time
              << " s" << std::endl;
    std::cout << "=====================================================" << std::endl;

    // =========================================================================
    // Solver setup
    // =========================================================================
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.InstructBoxDomainDimension(2, 2, 2);
    // No gravity: motion is fully prescribed
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, 0));
    // Enable mesh-mesh contacts (required for cylinder-cylinder contact)
    DEMSim.SetMeshUniversalContact(true);
    DEMSim.SetExpandSafetyType("auto");
    // Enable contact-normal output for the detailed info query
    DEMSim.SetContactOutputContent({"FORCE", "NORMAL", "OWNER"});

    auto mat = DEMSim.LoadMaterial({{"E", E}, {"nu", nu}, {"CoR", CoR}, {"mu", mu}, {"Crr", 0.0f}});

    // =========================================================================
    // Cylinder A (fixed): axis along z, centered at origin.
    // cyl_r1_h2.obj already has its axis along z, centered at origin.
    // =========================================================================
    auto cyl_a = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cyl_r1_h2.obj").string(), mat);
    cyl_a->Scale(cyl_scale);
    cyl_a->SetMass(cyl_mass);
    cyl_a->SetMOI(cyl_moi);
    cyl_a->SetInitPos(make_float3(0.f, 0.f, 0.f));
    cyl_a->SetFamily(100);
    DEMSim.SetFamilyFixed(100);

    // =========================================================================
    // Cylinder B (rotating): starts with axis along +x (cross configuration).
    // In the OBJ the axis is along +z, so we rotate +90 deg about the world
    // y-axis to align the axis with +x.  Then we prescribe a constant angular
    // velocity omega_y about y to rotate the axis back to +z (co-axial).
    // The centre stays at the origin (linear velocity locked to zero).
    // =========================================================================
    float4 cyl_b_init_q = make_float4(0.f, 0.f, 0.f, 1.f);
    cyl_b_init_q = RotateQuat(cyl_b_init_q, make_float3(0.f, 1.f, 0.f), (float)(PI / 2.0));

    auto cyl_b = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cyl_r1_h2.obj").string(), mat);
    cyl_b->Scale(cyl_scale);
    cyl_b->SetMass(cyl_mass);
    cyl_b->SetMOI(cyl_moi);
    cyl_b->SetInitPos(make_float3(0.f, 0.f, 0.f));
    cyl_b->SetInitQuat(cyl_b_init_q);
    cyl_b->SetFamily(1);
    auto cyl_b_tracker = DEMSim.Track(cyl_b);

    // Cylinder B: fixed centre, constant rotation about world y-axis
    DEMSim.SetFamilyPrescribedLinVel(1, "0", "0", "0");
    DEMSim.SetFamilyPrescribedAngVel(1, "0", to_string_with_precision(omega_y), "0");

    // =========================================================================
    // Initialize and run
    // =========================================================================
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.Initialize();

    // The two cylinders start intersecting; tell the solver the expected
    // initial tri-tri penetration so it can handle pre-existing overlap.
    // Must be called after Initialize().
    DEMSim.SetTriTriPenetration(cyl_radius * 0.5);

    std::cout << "\nRunning simulation for " << total_time << " s ..." << std::endl;
    std::cout << "Cylinder B axis starts along +x (90 deg from z) and ends along +z (co-axial)." << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    path out_dir = current_path();
    out_dir /= "DEMTest_CylinderCrossRotation";
    create_directory(out_dir);

    std::vector<double> force_mags;
    std::vector<float3> avg_normals;

    for (int i = 1; i <= n_frames; i++) {
        // Write mesh snapshot at the start of this interval
        char meshfilename[200];
        sprintf(meshfilename, "DEMTest_mesh_%04d.vtk", i);
        DEMSim.WriteMeshFile(out_dir / meshfilename);

        // Advance simulation in sub-steps; accumulate contact-force readings
        double sum_fx = 0.0, sum_fy = 0.0, sum_fz = 0.0;
        for (int s = 0; s < n_sub_samples; s++) {
            DEMSim.DoDynamics(sub_dt);
            float3 cnt_acc = cyl_b_tracker->ContactAcc();
            sum_fx += cnt_acc.x;
            sum_fy += cnt_acc.y;
            sum_fz += cnt_acc.z;
        }

        // Average contact force = (avg contact acceleration) * mass
        float3 avg_force;
        avg_force.x = static_cast<float>(sum_fx / n_sub_samples) * cyl_mass;
        avg_force.y = static_cast<float>(sum_fy / n_sub_samples) * cyl_mass;
        avg_force.z = static_cast<float>(sum_fz / n_sub_samples) * cyl_mass;
        double force_mag = std::sqrt((double)avg_force.x * avg_force.x + (double)avg_force.y * avg_force.y +
                                     (double)avg_force.z * avg_force.z);
        force_mags.push_back(force_mag);

        // Query the detailed contact info to get average contact normal
        // (0.0 threshold: include all contacts, even those with zero force)
        auto cnt_info = DEMSim.GetContactDetailedInfo(0.0f);
        float3 avg_normal = make_float3(0.f, 0.f, 0.f);
        int mm_cnt = 0;
        const auto& cnt_types = cnt_info->GetContactType();
        const auto& cnt_normals = cnt_info->GetNormal();
        for (size_t k = 0; k < cnt_types.size(); k++) {
            if (cnt_types[k] == "MM") {  // mesh-mesh (triangle-triangle) contacts
                avg_normal.x += cnt_normals[k].x;
                avg_normal.y += cnt_normals[k].y;
                avg_normal.z += cnt_normals[k].z;
                mm_cnt++;
            }
        }
        if (mm_cnt > 0) {
            float inv_n = 1.0f / static_cast<float>(mm_cnt);
            avg_normal.x *= inv_n;
            avg_normal.y *= inv_n;
            avg_normal.z *= inv_n;
        }
        avg_normals.push_back(avg_normal);

        // Current angle of Cylinder B axis from the z-axis
        float angle_deg = 90.0f + omega_y * static_cast<float>(i) * frame_time * 180.0f / (float)PI;

        std::cout << "t=" << i * frame_time << "s"
                  << "  axis_angle_from_z=" << angle_deg << " deg"
                  << "  |F_cnt_avg|=" << force_mag << " N"
                  << "  #MM_contacts=" << mm_cnt;

        if (mm_cnt > 0) {
            float nm = std::sqrt(avg_normal.x * avg_normal.x + avg_normal.y * avg_normal.y +
                                 avg_normal.z * avg_normal.z);
            if (nm > 1e-6f) {
                float inv_nm = 1.0f / nm;
                std::cout << "  avg_normal=(" << avg_normal.x * inv_nm << "," << avg_normal.y * inv_nm << ","
                          << avg_normal.z * inv_nm << ")";
            }
        }
        std::cout << std::endl;
    }

    // =========================================================================
    // Summary statistics
    // =========================================================================
    if (force_mags.empty()) {
        std::cout << "\nNo force data collected." << std::endl;
        return 0;
    }

    double sum_f = 0.0, min_f = force_mags[0], max_f = force_mags[0];
    for (double f : force_mags) {
        sum_f += f;
        if (f < min_f)
            min_f = f;
        if (f > max_f)
            max_f = f;
    }
    double mean_f = sum_f / static_cast<double>(force_mags.size());
    double var = 0.0;
    for (double f : force_mags) {
        double d = f - mean_f;
        var += d * d;
    }
    double stddev_f = std::sqrt(var / static_cast<double>(force_mags.size()));

    std::cout << "\n=== Contact Force Statistics (over " << force_mags.size() << " frames, each averaged over "
              << n_sub_samples << " sub-steps) ===" << std::endl;
    std::cout << "  Mean:   " << mean_f << " N" << std::endl;
    std::cout << "  Min:    " << min_f << " N" << std::endl;
    std::cout << "  Max:    " << max_f << " N" << std::endl;
    std::cout << "  StdDev: " << stddev_f << " N" << std::endl;
    std::cout << "  (Smooth force evolution indicates stable contact derivation.)" << std::endl;

    DEMSim.ShowTimingStats();
    std::cout << "=====================================================" << std::endl;
    std::cout << "DEMTest_CylinderCrossRotation exiting..." << std::endl;
    return 0;
}
