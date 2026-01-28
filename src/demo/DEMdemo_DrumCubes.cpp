//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Rotating drum centrifuge demo with only cube mesh particles.
// Matches the output style of DEMdemo_Centrifuge but uses 4 mm cubes by default
// (12-triangle mesh) inside an analytically defined cylinder and lids.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/Samplers.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

using namespace deme;
using namespace std::filesystem;

int main() {

    const bool disable_periodic = false;

    DEMSolver DEMSim;
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::FAMILY);
    DEMSim.SetMeshUniversalContact(true);

    auto mat_type_cube = DEMSim.LoadMaterial({{"E", 1e6}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.01}});
    auto mat_type_drum = DEMSim.LoadMaterial({{"E", 2e6}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.01}});
    DEMSim.SetMaterialPropertyPair("mu", mat_type_cube, mat_type_drum, 0.5);

    const float cube_size = 0.004f;
    const float cube_density = 2600.0f;
    const float cube_mass = cube_density * cube_size * cube_size * cube_size;
    const float cube_moi = cube_mass * cube_size * cube_size / 6.0f;
    const float half_diag = 0.5f * cube_size * std::sqrt(3.0f);

    // Load cube mesh template (12 triangles) and scale to desired cube size
    auto cube_template = DEMSim.LoadMeshType((GET_DATA_PATH() / "mesh/cube.obj").string(), mat_type_cube, true, false);
    cube_template->Scale(cube_size);
    cube_template->SetConvex(true);
    cube_template->SetNeverWinner(true);

    // Drum definition
    float3 CylCenter = make_float3(0, 0, 0);
    float3 CylAxis = make_float3(0, 0, 1);
    float CylRad = 0.08f;
    float CylHeight = 0.2f;
    float CylMass = 1.0f;
    float safe_delta = 0.003f;
    float IZZ = CylMass * CylRad * CylRad / 2;
    float IYY = (CylMass / 12) * (3 * CylRad * CylRad + CylHeight * CylHeight);
    auto Drum = DEMSim.AddExternalObject();
    Drum->AddPlanarContactCylinder(CylCenter, CylAxis, CylRad, mat_type_drum, ENTITY_NORMAL_INWARD);
    Drum->SetMass(CylMass);
    Drum->SetMOI(make_float3(IYY, IYY, IZZ));
    auto Drum_tracker = DEMSim.Track(Drum);
    unsigned int drum_family = 100;
    Drum->SetFamily(drum_family);
    const float rpm = 200.0f;
    const float drum_ang_vel = rpm * 2.0f * PI / 60.0f;
    DEMSim.SetFamilyPrescribedAngVel(drum_family, "0", "0", to_string_with_precision(drum_ang_vel));
    auto top_bot_planes = DEMSim.AddExternalObject();
    top_bot_planes->AddPlane(make_float3(0, 0, CylHeight / 2. - safe_delta), make_float3(0, 0, -1), mat_type_drum);
    top_bot_planes->AddPlane(make_float3(0, 0, -CylHeight / 2. + safe_delta), make_float3(0, 0, 1), mat_type_drum);
    top_bot_planes->SetFamily(drum_family);
    auto planes_tracker = DEMSim.Track(top_bot_planes);

    // -------------------------------------------------------------------------
    // Cylindrical periodicity in angular direction (wedge) + robust start pos
    // -------------------------------------------------------------------------
    const float cyl_periodic_start = 0.0f;
    const float cyl_periodic_end = deme::PI / 2.0f;
    const float two_pi = 2.0f * deme::PI;

    float span = cyl_periodic_end - cyl_periodic_start;
    if (span < 0.0f) span += two_pi;
    const float wedge_clear = half_diag + safe_delta;
    float cyl_min_radius_geom = 0.0f;
    if (span > 1e-6f && std::sin(0.5f * span) > 1e-6f) {
        cyl_min_radius_geom = wedge_clear / std::sin(0.5f * span);
    }
    const float cyl_min_radius = std::max(half_diag + safe_delta, cyl_min_radius_geom);

    if (!disable_periodic) {
        DEMSim.SetCylindricalPeriodicity(SPATIAL_DIR::Z, cyl_periodic_start, cyl_periodic_end, cyl_min_radius);
    }

    // Place cubes on a grid inside the drum, limited to the periodic wedge (mit Randabstand)
    const unsigned int target_cubes = 5000;
    float sample_radius = CylRad - half_diag - safe_delta;
    float sample_halfheight = CylHeight / 2.0f - half_diag - safe_delta;
    float fill_spacing = cube_size * 1.25f;  // leave gap so meshes don't start in contact
    std::mt19937 rng(42);
    unsigned int created = 0;

    auto in_periodic_wedge_with_margin = [&](float x, float y) {
        const float r2 = x * x + y * y;
        if ((!disable_periodic) &&
            (r2 < cyl_min_radius * cyl_min_radius))
            return false;
        if (disable_periodic) return true;

        const float r = std::sqrt(r2);
        float angle = std::atan2(y, x);
        if (angle < 0.0f) angle += two_pi;

        float rel = angle - cyl_periodic_start;
        if (rel < 0.0f) rel += two_pi;

        if (rel > span) return false;

        // Abstand zu beiden radialen Keilflächen: r*sin(rel) und r*sin(span-rel)
        const float d_start = r * std::sin(rel);
        const float d_end   = r * std::sin(span - rel);
        return (d_start >= wedge_clear) && (d_end >= wedge_clear);
    };

    std::vector<float3> candidate_positions;
    for (float z = -sample_halfheight; z <= sample_halfheight; z += fill_spacing) {
        for (float y = -sample_radius; y <= sample_radius; y += fill_spacing) {
            for (float x = -sample_radius; x <= sample_radius; x += fill_spacing) {
                const float r2 = x * x + y * y;
                if (r2 > sample_radius * sample_radius) continue;
                if (!in_periodic_wedge_with_margin(x, y)) continue;
                candidate_positions.push_back(make_float3(x, y, z));
            }
        }
    }
    std::shuffle(candidate_positions.begin(), candidate_positions.end(), rng);

    const unsigned int n_to_place =
        std::min<unsigned int>(target_cubes, static_cast<unsigned int>(candidate_positions.size()));
    std::vector<std::shared_ptr<DEMTracker>> cube_trackers;
    cube_trackers.reserve(n_to_place);
    for (unsigned int i = 0; i < n_to_place; i++) {
        const float3 pos = candidate_positions[i];
        auto cube = DEMSim.AddMeshFromTemplate(cube_template, pos);
        cube->SetFamily(1);
        cube->SetMass(cube_mass);
        cube->SetMOI(make_float3(cube_moi, cube_moi, cube_moi));
        cube->SetInitQuat(make_float4(0.f, 0.f, 0.f, 1.0f));
        cube_trackers.push_back(DEMSim.Track(cube));
    }
    created = n_to_place;
    if (n_to_place < target_cubes) {
        std::cout << "Warning: target_cubes=" << target_cubes
                  << " exceeds available non-overlapping slots (" << n_to_place << ")." << std::endl;
    }
    std::cout << "Placed " << created << " cubes inside the drum." << std::endl;

    auto max_v_finder = DEMSim.CreateInspector("max_absv");
    float max_v =0.f;

    DEMSim.InstructBoxDomainDimension(0.4, 0.4, 0.4);
    float step_size = 1e-4f;
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGPUTimersEnabled(true);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.SetExpandSafetyAdder(drum_ang_vel * CylRad);
    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir /= "DemoOutput_DrumCubes";
    create_directory(out_dir);

    float time_end = 3.0f;
    unsigned int fps = 20;
    float frame_time = 1.0f / fps;

    std::cout << "Output at " << fps << " FPS" << std::endl;
    std::cout << "DrumCubes config: cube_size=" << cube_size << "m, target_cubes=" << target_cubes
              << ", periodic=" << (disable_periodic ? "off" : "on") << std::endl;
    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    double dynamics_wall = 0.0;
    for (double t = 0; t < (double)time_end; t += frame_time, curr_step++) {
        std::cout << "Frame: " << currframe << std::endl;
        DEMSim.ShowThreadCollaborationStats();
        char filename[100];
        sprintf(filename, "DEMdemo_output_%04d.vtk", currframe);
        DEMSim.WriteMeshFile(out_dir / filename);
        currframe++;
        max_v = max_v_finder->GetValue();
        std::cout << "Max velocity of any point in simulation is " << max_v << std::endl;
        float3 drum_moi = Drum_tracker->MOI();
        float3 drum_acc = Drum_tracker->ContactAngAccLocal();
        float3 drum_torque = drum_acc * drum_moi;
        std::cout << "Contact torque on the side walls is " << drum_torque.x << ", " << drum_torque.y << ", "
                    << drum_torque.z << std::endl;

        float3 force_on_BC = planes_tracker->ContactAcc() * planes_tracker->Mass();
        std::cout << "Contact force on bottom plane is " << std::abs(force_on_BC.z) << std::endl;

        auto dyn_start = std::chrono::high_resolution_clock::now();
        DEMSim.DoDynamics(frame_time);
        auto dyn_end = std::chrono::high_resolution_clock::now();
        dynamics_wall += std::chrono::duration_cast<std::chrono::duration<double>>(dyn_end - dyn_start).count();
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << (time_sec.count()) / time_end * 10.0 << " seconds (wall time) to finish 10 seconds' simulation"
              << std::endl;
    std::cout << (dynamics_wall) / time_end * 10.0
              << " seconds (wall time, DoDynamics only) to finish 10 seconds' simulation" << std::endl;
    DEMSim.ShowThreadCollaborationStats();
    DEMSim.ClearThreadCollaborationStats();

    DEMSim.ShowTimingStats();
    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "DEMdemo_DrumCubes exiting..." << std::endl;
    return 0;
}
