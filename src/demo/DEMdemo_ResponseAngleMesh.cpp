//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Rotating angle of response drum demo with for mesh particles with STL inputs.
// Meshed drum put analytical planes on the side
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/Samplers.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <limits>
#include <utility>
#include <algorithm>

using namespace deme;
using namespace std::filesystem;

namespace {

/// Load an STL mesh, scale it, attach material and register it as a template.
std::shared_ptr<DEMMesh> LoadStlTemplate(DEMSolver& sim,
                                         const path& file,
                                         const std::shared_ptr<DEMMaterial>& mat,
                                         float scale) {
    DEMMesh mesh;
    bool ok = mesh.LoadSTLMesh(file.string());
    if (!ok) {
        DEME_ERROR("Failed to load STL mesh template %s", file.string().c_str());
    }
    mesh.SetMaterial(mat);
    mesh.Scale(scale);
    return sim.LoadMeshType(mesh);
}

/// Load an STL mesh, scale it, attach material and place it directly in the scene.
std::shared_ptr<DEMMesh> LoadStlMesh(DEMSolver& sim,
                                     const path& file,
                                     const std::shared_ptr<DEMMaterial>& mat,
                                     float scale) {
    DEMMesh mesh;
    bool ok = mesh.LoadSTLMesh(file.string());
    if (!ok) {
        DEME_ERROR("Failed to load STL mesh %s", file.string().c_str());
    }
    mesh.SetMaterial(mat);
    mesh.Scale(scale);
    return sim.AddMesh(mesh);
}

std::pair<float3, float3> ComputeBounds(const std::vector<float3>& vertices) {
    float3 vmin = make_float3(std::numeric_limits<float>::max());
    float3 vmax = make_float3(std::numeric_limits<float>::lowest());
    for (const auto& v : vertices) {
        vmin.x = std::min(vmin.x, v.x);
        vmin.y = std::min(vmin.y, v.y);
        vmin.z = std::min(vmin.z, v.z);
        vmax.x = std::max(vmax.x, v.x);
        vmax.y = std::max(vmax.y, v.y);
        vmax.z = std::max(vmax.z, v.z);
    }
    return {vmin, vmax};
}

}  // namespace

int main() {
    DEMSolver DEMSim;
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::FAMILY);
    DEMSim.SetMeshOutputFormat("STL");
    DEMSim.SetNoForceRecord();
    DEMSim.SetMeshUniversalContact(true);
    const float mm_to_m = 0.001f;
    const float drum_inner_radius = 0.1f;  // 200 mm diameter
    const float wall_clearance = 0.002f;   // leave a small gap to the mantle
    const float rpm = 40.0f;
    const float drum_ang_vel = rpm * 2.0f * PI / 60.0f;

    auto mat_type_particle =
        DEMSim.LoadMaterial({{"E", 1e6}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.01}});
    auto mat_type_drum = DEMSim.LoadMaterial({{"E", 2e6}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.01}});
    DEMSim.SetMaterialPropertyPair("mu", mat_type_particle, mat_type_drum, 0.5);

    // Load particle mesh template from STL (approx. 4 mm triangular prism)
    path tri_path = GET_DATA_PATH() / "mesh" / "simpleTriangleShape4mm.stl";
    auto tri_template = LoadStlTemplate(DEMSim, tri_path, mat_type_particle, mm_to_m);
    auto [tri_min, tri_max] = ComputeBounds(tri_template->GetCoordsVertices());
    const float3 tri_dims = tri_max - tri_min;
    const float tri_diag = std::sqrt(tri_dims.x * tri_dims.x + tri_dims.y * tri_dims.y + tri_dims.z * tri_dims.z);
    const float tri_radius = 0.5f * tri_diag;
    const float particle_density = 2600.0f;
    double tri_volume = 0.0;
    float3 tri_center = make_float3(0, 0, 0);
    float3 tri_inertia = make_float3(0, 0, 0);
    tri_template->ComputeMassProperties(tri_volume, tri_center, tri_inertia);
    const float particle_mass = static_cast<float>(tri_volume * particle_density);
    const float3 particle_moi = tri_inertia * particle_density;
    std::cout << "Particle STL volume (m^3): " << tri_volume << std::endl;
    std::cout << "Particle STL MOI (unit density, CoM): " << tri_inertia.x << ", " << tri_inertia.y << ", "
              << tri_inertia.z << std::endl;
    const double cube_vol = std::pow(4.0e-3, 3);
    std::cout << "Particle mass (kg): " << particle_mass << std::endl;

    // Load drum mantle from STL; STL units are mm with z in [0, 100]
    path drum_path = GET_DATA_PATH() / "mesh" / "drum.stl";
    auto drum_mesh = LoadStlMesh(DEMSim, drum_path, mat_type_drum, mm_to_m);
    auto [drum_min, drum_max] = ComputeBounds(drum_mesh->GetCoordsVertices());
    const float drum_height = drum_max.z - drum_min.z;
    unsigned int drum_family = 100;
    drum_mesh->SetFamily(drum_family);
    const float drum_density = 2600.0f;
    double drum_volume = 0.0;
    float3 drum_center = make_float3(0, 0, 0);
    float3 drum_inertia = make_float3(0, 0, 0);
    drum_mesh->ComputeMassProperties(drum_volume, drum_center, drum_inertia);
    const float drum_mass = static_cast<float>(drum_volume * drum_density);
    drum_mesh->SetMass(drum_mass);
    drum_mesh->SetMOI(drum_inertia * drum_density);
    std::cout << "Drum STL volume (m^3): " << drum_volume << std::endl;
    std::cout << "Drum STL MOI (unit density, CoM): " << drum_inertia.x << ", " << drum_inertia.y << ", "
              << drum_inertia.z << std::endl;
    std::cout << "Drum mass (kg): " << drum_mass << std::endl;
    DEMSim.SetFamilyPrescribedAngVel(drum_family, "0", "0", to_string_with_precision(drum_ang_vel));

    // Add top and bottom planes at z = 0 and z = 0.1 m. They rotate with the drum family (axis-aligned so rotation
    // does not change their normals).
    auto end_caps = DEMSim.AddExternalObject();
    end_caps->AddPlane(make_float3(0, 0, drum_max.z), make_float3(0, 0, -1), mat_type_drum);
    end_caps->AddPlane(make_float3(0, 0, drum_min.z), make_float3(0, 0, 1), mat_type_drum);
    end_caps->SetFamily(drum_family);

    auto drum_tracker = DEMSim.Track(drum_mesh);
    auto cap_tracker = DEMSim.Track(end_caps);

    // Sample 5000 particles inside the cylindrical volume with a small wall clearance.
    const unsigned int target_particles = 5000;
    const float sample_radius = drum_inner_radius - wall_clearance - tri_radius;
    const float sample_halfheight = drum_height / 2.0f - wall_clearance - tri_radius;
    HCPSampler sampler(tri_diag * 1.05f);
    auto candidate_pos =
        sampler.SampleCylinderZ(make_float3(0, 0, drum_min.z + drum_height / 2.0f), sample_radius, sample_halfheight);
    if (candidate_pos.size() < target_particles) {
        DEME_WARNING("Sampler produced fewer points (%zu) than requested (%u). Using all generated points.",
                     candidate_pos.size(), target_particles);
    }
    std::mt19937 rng(42);
    std::shuffle(candidate_pos.begin(), candidate_pos.end(), rng);
    if (candidate_pos.size() > target_particles) {
        candidate_pos.resize(target_particles);
    }

    for (const auto& pos : candidate_pos) {
        auto tri = DEMSim.AddMeshFromTemplate(tri_template, pos);
        tri->SetFamily(1);
        tri->SetMass(particle_mass);
        tri->SetMOI(particle_moi);
        tri->SetInitQuat(make_float4(0.f, 0.f, 0.f, 1.0f));
    }
    std::cout << "Placed " << candidate_pos.size() << " STL particles inside the drum." << std::endl;

    auto max_v_finder = DEMSim.CreateInspector("max_absv");
    float max_v;

    DEMSim.InstructBoxDomainDimension(0.3, 0.3, 0.2);
    float step_size = 1e-5f;
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGPUTimersEnabled(true);
    DEMSim.SetGravitationalAcceleration(make_float3(0, -9.81, 0));
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.SetExpandSafetyAdder(drum_ang_vel * drum_inner_radius);
    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir /= "DemoOutput_ResponseAngleMesh";
    create_directory(out_dir);

    float time_end = 3.0f;
    unsigned int fps = 20;
    float frame_time = 1.0f / fps;

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (double t = 0; t < (double)time_end; t += frame_time, curr_step++) {
        std::cout << "Frame: " << currframe << std::endl;
        DEMSim.ShowThreadCollaborationStats();
        char filename[100];
        sprintf(filename, "DEMdemo_output_%04d.stl", currframe);
        DEMSim.WriteMeshFile(out_dir / filename);
        currframe++;
        max_v = max_v_finder->GetValue();
        std::cout << "Max velocity of any point in simulation is " << max_v << std::endl;

        float3 drum_moi = drum_tracker->MOI();
        float3 drum_acc = drum_tracker->ContactAngAccLocal();
        float3 drum_torque = drum_acc * drum_moi;
        std::cout << "Contact torque on the side walls is " << drum_torque.x << ", " << drum_torque.y << ", "
                  << drum_torque.z << std::endl;

        float3 force_on_BC = cap_tracker->ContactAcc() * cap_tracker->Mass();
        std::cout << "Contact force on bottom plane is " << std::abs(force_on_BC.z) << std::endl;

        DEMSim.DoDynamics(frame_time);
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << (time_sec.count()) / time_end * 10.0 << " seconds (wall time) to finish 10 seconds' simulation"
              << std::endl;
    DEMSim.ShowThreadCollaborationStats();
    DEMSim.ClearThreadCollaborationStats();

    DEMSim.ShowTimingStats();
    std::cout << "----------------------------------------" << std::endl;
    DEMSim.ShowMemStats();
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "DEMdemo_ResponseAngleMesh exiting..." << std::endl;
    return 0;
}
