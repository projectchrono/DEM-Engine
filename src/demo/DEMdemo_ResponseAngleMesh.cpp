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
#include <cctype>
#include <filesystem>
#include <random>
#include <limits>
#include <utility>
#include <algorithm>

using namespace deme;
using namespace std::filesystem;

namespace {

std::string ToLower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

/// Load a mesh (STL or OBJ), scale it, attach material and register it as a template.
std::shared_ptr<DEMMesh> LoadMeshTemplate(DEMSolver& sim,
                                          const path& file,
                                          const std::shared_ptr<DEMMaterial>& mat,
                                          float scale) {
    DEMMesh mesh;
    std::string ext = ToLower(file.extension().string());
    bool ok = false;
    if (ext == ".stl") {
        ok = mesh.LoadSTLMesh(file.string());
    } else if (ext == ".obj") {
        ok = mesh.LoadWavefrontMesh(file.string());
    } else {
        DEME_ERROR("Unsupported mesh format: %s (only .stl or .obj)", ext.c_str());
    }
    if (!ok) {
        DEME_ERROR("Failed to load mesh template %s", file.string().c_str());
    }
    mesh.SetMaterial(mat);
    mesh.Scale(scale);
    return sim.LoadMeshType(mesh);
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
    DEMSim.SetMeshOutputFormat("VTK");
    DEMSim.SetNoForceRecord();
    DEMSim.SetMeshUniversalContact(true);
    const float mm_to_m = 0.001f;
    const float drum_inner_radius = 0.1f;  // 200 mm diameter
    const float wall_clearance = 0.001f;   // leave a small gap to the mantle
    const float rpm = 40.0f;
    const float drum_ang_vel = rpm * 2.0f * PI / 60.0f;

    auto mat_type_particle =
        DEMSim.LoadMaterial({{"E", 1e6}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.00}});
    auto mat_type_drum = DEMSim.LoadMaterial({{"E", 2e6}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.00}});
    DEMSim.SetMaterialPropertyPair("mu", mat_type_particle, mat_type_drum, 0.5);

    // --------------------- Particle settings block ---------------------
    // Mesh file can be .stl or .obj (path is relative to data/mesh).
    const path particle_mesh_file = GET_DATA_PATH() / "mesh" / "cross_fine.stl"; // "simpleTriangleShape4mm.stl"
    const float particle_mesh_scale = mm_to_m * 0.5f; // 1.0f for STLs in mm size
    const unsigned int target_particles = 5000;
    // -------------------------------------------------------------------

    auto tri_template = LoadMeshTemplate(DEMSim, particle_mesh_file, mat_type_particle, particle_mesh_scale);
    auto [tri_min, tri_max] = ComputeBounds(tri_template->GetCoordsVertices());
    const float3 tri_dims = tri_max - tri_min;
    const float tri_diag = std::sqrt(tri_dims.x * tri_dims.x + tri_dims.y * tri_dims.y + tri_dims.z * tri_dims.z);
    const float tri_radius = 0.5f * tri_diag;
    const float particle_density = 2600.0f;
    double tri_volume = 0.0;
    float3 tri_center = make_float3(0, 0, 0);
    float3 tri_inertia = make_float3(0, 0, 0);
    tri_template->ComputeMassProperties(tri_volume, tri_center, tri_inertia);
    // tri_template->SetConvex(true); // for convex particels only
    // tri_template->SetNeverWinner(true); // if mesh is more coarse the other contacts
    const float particle_mass = static_cast<float>(tri_volume * particle_density);
    const float3 particle_moi = tri_inertia * particle_density;
    std::cout << "Particle volume (m^3): " << tri_volume << ", mass (kg): "<< particle_mass << std::endl;
    std::cout << "Particle MOI (unit density, CoM): " << tri_inertia.x << ", " << tri_inertia.y << ", "
              << tri_inertia.z << std::endl;
    const double cube_vol = std::pow(4.0e-3, 3);

    // Toggle drum mantle type:
    // true  -> STL drum mesh ("after")
    // false -> analytical planar contact cylinder ("before")
    const bool use_stl_drum = true;

    const float drum_mass = 1.0f;
    const unsigned int drum_family = 100;
    float drum_height = 0.1f;
    float drum_bottom_z = 0.0f;
    float drum_top_z = drum_bottom_z + drum_height;
    float drum_center_z = 0.5f * (drum_bottom_z + drum_top_z);
    std::shared_ptr<DEMTracker> drum_tracker;

    if (use_stl_drum) {
        const path drum_mesh_file = GET_DATA_PATH() / "mesh" / "drum.stl";
        DEMMesh drum_mesh_data;
        if (!drum_mesh_data.LoadSTLMesh(drum_mesh_file.string())) {
            DEME_ERROR("Failed to load drum mesh %s", drum_mesh_file.string().c_str());
        }
        drum_mesh_data.SetMaterial(mat_type_drum);
        drum_mesh_data.Scale(mm_to_m);
        auto drum = DEMSim.AddMesh(drum_mesh_data);

        auto [drum_min, drum_max] = ComputeBounds(drum->GetCoordsVertices());
        drum_bottom_z = drum_min.z;
        drum_top_z = drum_max.z;
        drum_height = drum_top_z - drum_bottom_z;
        drum_center_z = 0.5f * (drum_bottom_z + drum_top_z);

        const float drum_outer_radius = std::max(std::max(std::abs(drum_min.x), std::abs(drum_max.x)),
                                                 std::max(std::abs(drum_min.y), std::abs(drum_max.y)));
        const float IZZ = drum_mass * drum_outer_radius * drum_outer_radius / 2.0f;
        const float IYY =
            (drum_mass / 12.0f) * (3.0f * drum_outer_radius * drum_outer_radius + drum_height * drum_height);

        drum->SetFamily(drum_family);
        drum->SetMass(drum_mass);
        drum->SetMOI(make_float3(IYY, IYY, IZZ));
        drum_tracker = DEMSim.Track(drum);
    } else {
        auto drum = DEMSim.AddExternalObject();
        drum->AddPlanarContactCylinder(make_float3(0, 0, drum_center_z), make_float3(0, 0, 1), drum_inner_radius,
                                       mat_type_drum, ENTITY_NORMAL_INWARD);
        const float IZZ = drum_mass * drum_inner_radius * drum_inner_radius / 2.0f;
        const float IYY =
            (drum_mass / 12.0f) * (3.0f * drum_inner_radius * drum_inner_radius + drum_height * drum_height);

        drum->SetFamily(drum_family);
        drum->SetMass(drum_mass);
        drum->SetMOI(make_float3(IYY, IYY, IZZ));
        drum_tracker = DEMSim.Track(drum);
    }
    const std::string drum_ang_pre =
        "float3 omg_g = make_float3(0.f, 0.f, " + to_string_with_precision(drum_ang_vel) + ");"
        "applyOriQToVector3(omg_g.x, omg_g.y, omg_g.z, oriQw, -oriQx, -oriQy, -oriQz);";
    DEMSim.SetFamilyPrescribedAngVel(drum_family, "omg_g.x", "omg_g.y", "omg_g.z", true, drum_ang_pre);

    // Add top and bottom planes. They rotate with the drum family.
    auto end_caps = DEMSim.AddExternalObject();
    end_caps->AddPlane(make_float3(0, 0, drum_top_z), make_float3(0, 0, -1), mat_type_drum);
    end_caps->AddPlane(make_float3(0, 0, drum_bottom_z), make_float3(0, 0, 1), mat_type_drum);
    end_caps->SetFamily(drum_family);

    auto cap_tracker = DEMSim.Track(end_caps);

    // Sample particles inside the cylindrical volume with a small wall clearance.
    const float r_sphere = tri_radius;  // = 0.5 * tri_diag
    // AABB clearance for a cylinder aligned with z:
    // radial clearance uses the half-diagonal in XY; z-clearance uses half-height in Z.
    const float r_xy_aabb = 0.5f * std::sqrt(tri_dims.x * tri_dims.x + tri_dims.y * tri_dims.y);
    const float r_z_aabb  = 0.5f * tri_dims.z;
    // Spacing of the HCP lattice (center-to-center). Keep conservative spacing (uses tri_diag).
    // Clearance model only changes usable container dimensions.
    HCPSampler sampler(tri_diag * 1.01f);
    auto sample_with_clearance = [&](float r_xy, float r_z) {
        const float sample_radius     = drum_inner_radius - wall_clearance - r_xy;
        const float sample_halfheight = drum_height * 0.5f - wall_clearance - r_z;
        // Guard against negative dimensions
        if (sample_radius <= 0.f || sample_halfheight <= 0.f) {
            return std::vector<float3>{};
        }

        return sampler.SampleCylinderZ(make_float3(0, 0, drum_center_z), sample_radius, sample_halfheight);
    };
    // Generate both candidate sets
    auto cand_sphere = sample_with_clearance(r_sphere, r_sphere);
    auto cand_aabb   = sample_with_clearance(r_xy_aabb, r_z_aabb);
    // Pick denser (more points). If equal, prefer sphere for robustness.
    bool use_aabb = cand_aabb.size() > cand_sphere.size();
    auto& candidate_pos = use_aabb ? cand_aabb : cand_sphere;
    std::cout << "Sampling clearance mode: " << (use_aabb ? "AABB" : "Sphere")
              << " (AABB=" << cand_aabb.size() << ", Sphere=" << cand_sphere.size() << ")\n";
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
    const float total_particle_mass = particle_mass * candidate_pos.size();
    std::cout << "Placed " << candidate_pos.size() << " particles with a mass of "<< total_particle_mass <<" kg inside the drum." <<std::endl;

    auto max_v_finder = DEMSim.CreateInspector("max_absv");
    float max_v;

    DEMSim.InstructBoxDomainDimension(0.3, 0.3, 0.2);
    float step_size = 1e-5f;
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGPUTimersEnabled(true);
    DEMSim.SetGravitationalAcceleration(make_float3(0, -9.81, 0));
    DEMSim.SetExpandSafetyType("auto");
    const float vmax_grav = std::sqrt(2.0f * 9.81f * drum_inner_radius);
    const float vmax_rot = drum_ang_vel * drum_inner_radius;
    const float vmax = (vmax_grav > vmax_rot) ? vmax_grav : vmax_rot;
    DEMSim.SetExpandSafetyAdder(vmax);
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
        sprintf(filename, "DEMdemo_output_%04d.vtk", currframe);
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
