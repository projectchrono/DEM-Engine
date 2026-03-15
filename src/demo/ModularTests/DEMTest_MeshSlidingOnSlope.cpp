//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Mesh planar sliding test: a mesh cube slides on a meshed slope under gravity.
// Similar to SphereRollUpIncline in DEMdemo_TestPack, but both the sliding body
// and the slope are mesh objects (cube.obj and thin_plate.obj).
// We inspect the time evolution of the contact force magnitude and direction
// to verify stability of the mesh-mesh contact during sliding.
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

using namespace deme;
using namespace std::filesystem;

int main() {
    // =========================================================================
    // Parameters
    // =========================================================================
    // Slope angle from horizontal (degrees). With mu=0.3 < tan(30deg)~0.577,
    // the cube will slide rather than stay stationary.
    const float slope_angle_deg = 30.0f;
    const float slope_angle_rad = slope_angle_deg * PI / 180.0f;

    // Material properties
    const float mu = 0.3f;   // friction coefficient
    const float CoR = 0.3f;  // coefficient of restitution
    const float E = 1e8f;    // Young's modulus (Pa)
    const float nu = 0.3f;   // Poisson's ratio

    // Cube geometry and mass (cube.obj is a unit cube, side = 1m before scale)
    const float cube_scale = 0.3f;       // scale to 0.3 m per side
    const float cube_density = 2600.0f;  // kg/m^3 (rock-like)
    const float cube_mass = cube_density * cube_scale * cube_scale * cube_scale;
    // MOI of a solid cube: I = m*a^2/6 per axis
    const float cube_moi_val = cube_mass * cube_scale * cube_scale / 6.0f;
    const float3 cube_moi = make_float3(cube_moi_val, cube_moi_val, cube_moi_val);

    // Simulation time settings
    const float step_size = 5e-6f;
    const float frame_time = 0.05f;
    const int n_sub_samples = 250;                                        // sub-samples per frame for force averaging
    const float sub_dt = frame_time / static_cast<float>(n_sub_samples);  // sub-step size
    const float total_time = 1.f;

    std::cout << "=====================================================" << std::endl;
    std::cout << "DEM Mesh Planar Sliding Test" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "Slope angle:          " << slope_angle_deg << " deg" << std::endl;
    std::cout << "Friction coefficient: " << mu << std::endl;
    std::cout << "Cube side:            " << cube_scale << " m" << std::endl;
    std::cout << "Cube mass:            " << cube_mass << " kg" << std::endl;

    const float g = 9.81f;
    const float Fn_expected = cube_mass * g * std::cos(slope_angle_rad);
    const float Ft_expected = mu * Fn_expected;
    const float F_total_expected = std::sqrt(Fn_expected * Fn_expected + Ft_expected * Ft_expected);
    std::cout << "Expected normal force (steady slide): " << Fn_expected << " N" << std::endl;
    std::cout << "Expected friction force:              " << Ft_expected << " N" << std::endl;
    std::cout << "Expected total contact force:         " << F_total_expected << " N" << std::endl;
    std::cout << "=====================================================" << std::endl;

    // =========================================================================
    // Solver setup
    // =========================================================================
    DEMSolver DEMSim;
    DEMSim.SetVerbosity("ERROR");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.InstructBoxDomainDimension(5, 5, 5);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -g));
    // Enable mesh-mesh contacts (required for cube-slope mesh contact)
    DEMSim.SetMeshUniversalContact(true);
    DEMSim.SetExpandSafetyType("auto");

    auto mat = DEMSim.LoadMaterial({{"E", E}, {"nu", nu}, {"CoR", CoR}, {"mu", mu}, {"Crr", 0.0f}});

    // =========================================================================
    // Slope mesh: thin_plate.obj (fixed)
    // thin_plate has its large flat face normal in the +x direction.
    // We rotate it so the face normal aligns with the slope normal:
    //   slope_normal = (-sin(alpha), 0, cos(alpha))
    // This requires a rotation about the y-axis by -(90 + alpha) = -120 deg.
    // Verification: R_y(-120 deg) * (1,0,0) = (-0.5, 0, 0.866) = (-sin30, 0, cos30). OK.
    // =========================================================================
    float4 slope_q = make_float4(0.f, 0.f, 0.f, 1.f);
    slope_q = RotateQuat(slope_q, make_float3(0.f, 1.f, 0.f), static_cast<float>(-(PI / 2.0f + slope_angle_rad)));

    auto slope = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/thin_plate.obj").string(), mat);
    // thin_plate CoM is at (0,0,0) per its construction; set reasonable mass/MOI.
    slope->SetMass(1000.0f);
    slope->SetMOI(make_float3(100.f, 100.f, 200.f));
    slope->SetInitQuat(slope_q);
    slope->SetInitPos(make_float3(0.f, 0.f, 0.f));
    slope->SetFamily(100);
    DEMSim.SetFamilyFixed(100);

    // =========================================================================
    // Sliding box: cube.obj (free to move under gravity)
    // cube.obj is a unit cube centered at origin. After Scale(cube_scale), it
    // is a cube_scale-side cube.
    // Initial position: on the slope surface, near the top, slightly lifted off.
    //   Slope normal:   n = (-sin(alpha), 0, cos(alpha))
    //   Uphill dir:     u = (cos(alpha), 0, sin(alpha))
    //   Cube center = slope_center + uphill*0.5 + n*(half_cube + gap)
    // =========================================================================
    const float3 slope_normal = make_float3(-std::sin(slope_angle_rad), 0.f, std::cos(slope_angle_rad));
    const float3 uphill_dir = make_float3(std::cos(slope_angle_rad), 0.f, std::sin(slope_angle_rad));
    const float half_cube = cube_scale / 2.0f;
    const float initial_gap = 0.025f;

    float3 cube_init_pos = uphill_dir * 0.5f + slope_normal * (half_cube + initial_gap);

    // Orient cube so the face resting on slope is parallel to slope surface.
    // We rotate the cube by -alpha about y, so the cube's local z-axis
    // aligns with the slope normal: R_y(-alpha) * (0,0,1) = (-sin(alpha),0,cos(alpha)). OK.
    float4 cube_q = make_float4(0.f, 0.f, 0.f, 1.f);
    cube_q = RotateQuat(cube_q, make_float3(0.f, 1.f, 0.f), -slope_angle_rad);

    auto cube = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cube.obj").string(), mat);
    cube->Scale(cube_scale);
    cube->SetMass(cube_mass);
    cube->SetMOI(cube_moi);
    cube->SetInitPos(cube_init_pos);
    cube->SetInitQuat(cube_q);
    cube->SetFamily(0);
    auto cube_tracker = DEMSim.Track(cube);

    // =========================================================================
    // Initialize and run
    // =========================================================================
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.Initialize();

    std::cout << "\nCube init position: (" << cube_init_pos.x << ", " << cube_init_pos.y << ", " << cube_init_pos.z
              << ")" << std::endl;
    std::cout << "Running simulation for " << total_time << " s ..." << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    path out_dir = current_path() / "modular_test_output" / "DEMTest_MeshSlidingOnSlope";
    std::error_code dir_ec;
    create_directories(out_dir, dir_ec);
    if (dir_ec || !is_directory(out_dir)) {
        std::cerr << "Failed to create output directory: " << out_dir << " (" << dir_ec.message() << ")" << std::endl;
        return 1;
    }

    std::vector<double> force_mags;
    const int n_frames = static_cast<int>(total_time / frame_time);

    for (int i = 1; i <= n_frames; i++) {
        // Output mesh snapshot at the start of this interval
        char meshfilename[200];
        sprintf(meshfilename, "DEMTest_mesh_%04d.vtk", i);
        DEMSim.WriteMeshFile(out_dir / meshfilename);

        // Advance simulation in sub-steps and accumulate contact force readings
        // to compute the average force over the interval [t_prev, t_center+half_window].
        double sum_fx = 0.0, sum_fy = 0.0, sum_fz = 0.0;
        for (int s = 0; s < n_sub_samples; s++) {
            DEMSim.DoDynamics(sub_dt);
            float3 cnt_acc = cube_tracker->ContactAcc();
            sum_fx += cnt_acc.x;
            sum_fy += cnt_acc.y;
            sum_fz += cnt_acc.z;
        }

        // Average contact force = (average contact acceleration) * mass
        float3 avg_cnt_force;
        avg_cnt_force.x = static_cast<float>(sum_fx / n_sub_samples) * cube_mass;
        avg_cnt_force.y = static_cast<float>(sum_fy / n_sub_samples) * cube_mass;
        avg_cnt_force.z = static_cast<float>(sum_fz / n_sub_samples) * cube_mass;
        double force_mag =
            std::sqrt((double)avg_cnt_force.x * avg_cnt_force.x + (double)avg_cnt_force.y * avg_cnt_force.y +
                      (double)avg_cnt_force.z * avg_cnt_force.z);
        force_mags.push_back(force_mag);

        float3 pos = cube_tracker->Pos();
        float3 vel = cube_tracker->Vel();
        double vel_mag = std::sqrt((double)vel.x * vel.x + (double)vel.y * vel.y + (double)vel.z * vel.z);

        std::cout << "t=" << i * frame_time << "s (avg over " << n_sub_samples << " sub-steps)"
                  << "  pos=(" << pos.x << "," << pos.y << "," << pos.z << ")"
                  << "  |v|=" << vel_mag << " m/s"
                  << "  |F_cnt_avg|=" << force_mag << " N";

        if (force_mag > 1.0) {
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
    std::cout << "  Expected total at steady slide: " << F_total_expected << " N" << std::endl;

    DEMSim.ShowTimingStats();
    std::cout << "=====================================================" << std::endl;
    std::cout << "DEMTest_MeshSlidingOnSlope exiting..." << std::endl;
    return 0;
}
