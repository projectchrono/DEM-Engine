//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Simple collision test: a cube hits an analytical plane with no gravity.
// Cases:
// 1) Edge-first impact (45 deg rotation)
// 2) Corner-first impact (45 deg around X and Y)
// For each case, run with:
//  a) Single patch cube
//  b) 12-patch cube (one patch per triangle)
// Each scenario is repeated 10 times. We log rebound speed, rebound direction,
// and peak normal force on the plane, plus mean/min/max/std stats.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/HostSideHelpers.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace deme;

namespace {

constexpr bool kUseTriangleParticles = true; // toggle to run the STL-based triangle setup
constexpr float kMmToMeters = 0.001f;
constexpr double kTriangleParticleDensity = 2600.0;

constexpr int kNumRuns = 10;
constexpr double kGap = 0.01;        // 10 mm
constexpr double kSpeed = 1.0;       // 1 m/s
constexpr double kTimeStep = 1e-5;   // seconds
constexpr int kMaxSteps = 200000;    // 2 seconds max
constexpr double kContactEps = 1e-6; // contact force threshold

struct RunResult {
    bool ok = false;
    double rebound_speed = 0.0;
    double peak_normal_force = 0.0;
    float3 rebound_dir = make_float3(0, 0, 0);
};

struct Stats {
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

double vec_length(const float3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

double vec_dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 vec_scale(const float3& v, double s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

Stats calc_stats(const std::vector<double>& values) {
    Stats s;
    if (values.empty()) {
        return s;
    }
    s.min = values.front();
    s.max = values.front();
    double sum = 0.0;
    for (double v : values) {
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
        sum += v;
    }
    s.mean = sum / values.size();
    double var = 0.0;
    for (double v : values) {
        double d = v - s.mean;
        var += d * d;
    }
    s.stddev = std::sqrt(var / values.size());
    return s;
}

double compute_min_z_rotated(const std::shared_ptr<DEMMesh>& mesh, const float4& rotQ) {
    double min_z = std::numeric_limits<double>::max();
    for (const auto& v_in : mesh->m_vertices) {
        float3 v = v_in;
        applyFrameTransformLocalToGlobal(v, make_float3(0, 0, 0), rotQ);
        min_z = std::min(min_z, static_cast<double>(v.z));
    }
    return min_z;
}

void assign_patch_ids(const std::shared_ptr<DEMMesh>& mesh_template,
                      bool per_triangle_patches,
                      const std::shared_ptr<DEMMaterial>& mat_type) {
    if (!mesh_template) {
        return;
    }
    const size_t num_tris = mesh_template->GetNumTriangles();
    std::vector<patchID_t> patch_ids(num_tris, 0);
    if (per_triangle_patches) {
        for (size_t i = 0; i < num_tris; ++i) {
            patch_ids[i] = static_cast<patchID_t>(i);
        }
    }
    mesh_template->SetPatchIDs(patch_ids);
    mesh_template->SetMaterial(mat_type);
}

std::shared_ptr<DEMMesh> load_cube_template(DEMSolver& DEMSim,
                                            const std::shared_ptr<DEMMaterial>& mat_type,
                                            bool per_triangle_patches) {
    auto mesh_template = DEMSim.LoadMeshType((GET_DATA_PATH() / "mesh/cube.obj").string(), mat_type,
                                             true,   // load_normals
                                             false); // load_uv
    if (!mesh_template) {
        return nullptr;
    }

    assign_patch_ids(mesh_template, per_triangle_patches, mat_type);
    return mesh_template;
}

std::shared_ptr<DEMMesh> load_triangle_template(DEMSolver& DEMSim,
                                                const std::shared_ptr<DEMMaterial>& mat_type,
                                                bool per_triangle_patches,
                                                float& out_mass,
                                                float3& out_moi) {
    std::shared_ptr<DEMMesh> mesh_template =
        DEMSim.LoadMeshType((GET_DATA_PATH() / "mesh/simpleTriangleShape4mm.stl").string(), mat_type, true, false);
    if (!mesh_template) {
        return nullptr;
    }
    mesh_template->Scale(kMmToMeters);

    double volume = 0.0;
    float3 center = make_float3(0, 0, 0);
    float3 inertia = make_float3(0, 0, 0);
    mesh_template->ComputeMassProperties(volume, center, inertia);

    out_mass = static_cast<float>(volume * kTriangleParticleDensity);
    out_moi = inertia * static_cast<float>(kTriangleParticleDensity);

    assign_patch_ids(mesh_template, per_triangle_patches, mat_type);
    return mesh_template;
}

RunResult run_single_collision(const float4& init_rot,
                               bool per_triangle_patches,
                               bool use_triangle_particles,
                               const std::string& label,
                               int run_id) {
    RunResult result;

    DEMSolver DEMSim;
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.InstructBoxDomainDimension(5, 5, 5);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, 0));
    DEMSim.SetCDUpdateFreq(0);
    DEMSim.UseAdaptiveUpdateFreq(false);
    DEMSim.SetMeshUniversalContact(true);

    auto mat_type = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.00}});

    float3 plane_normal = make_float3(0, 0, 1);
    auto plane = DEMSim.AddBCPlane(make_float3(0, 0, 0), plane_normal, mat_type);
    auto plane_tracker = DEMSim.Track(plane);
    const char* mesh_desc = use_triangle_particles ? "triangle mesh" : "cube mesh";
    auto mesh_template = std::shared_ptr<DEMMesh>{};
    float particle_mass = 1.0f;
    float3 particle_moi = make_float3(1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f);

    if (use_triangle_particles) {
        mesh_template = load_triangle_template(DEMSim, mat_type, per_triangle_patches, particle_mass, particle_moi);
    } else {
        mesh_template = load_cube_template(DEMSim, mat_type, per_triangle_patches);
    }
    if (!mesh_template) {
        std::cout << "[" << label << "] Run " << run_id << ": failed to load " << mesh_desc << std::endl;
        return result;
    }
    double min_z = compute_min_z_rotated(mesh_template, init_rot);
    double init_z = kGap - min_z;

    auto cube = DEMSim.AddMeshFromTemplate(mesh_template, make_float3(0, 0, 0));
    cube->SetFamily(0);
    cube->SetMass(particle_mass);
    cube->SetMOI(particle_moi);
    cube->SetInitQuat(init_rot);
    cube->SetInitPos(make_float3(0, 0, static_cast<float>(init_z)));
    auto cube_tracker = DEMSim.Track(cube);

    DEMSim.SetInitTimeStep(kTimeStep);
    DEMSim.Initialize();
    cube_tracker->SetVel(make_float3(0, 0, -static_cast<float>(kSpeed)));

    bool contact_started = false;
    bool rebound_captured = false;
    double peak_normal_force = 0.0;

    for (int step = 0; step < kMaxSteps; ++step) {
        DEMSim.DoStepDynamics();

        float3 plane_force = plane_tracker->ContactAcc();
        plane_force = vec_scale(plane_force, plane_tracker->Mass());
        double normal_force = std::abs(vec_dot(plane_force, plane_normal));
        peak_normal_force = std::max(peak_normal_force, normal_force);

        if (normal_force > kContactEps) {
            contact_started = true;
        }

        float3 vel = cube_tracker->Vel();
        double vel_n = vec_dot(vel, plane_normal);

        if (contact_started && normal_force <= kContactEps && vel_n > 0.0) {
            double speed = vec_length(vel);
            float3 dir = make_float3(0, 0, 0);
            if (speed > 0) {
                dir = vec_scale(vel, 1.0 / speed);
            }
            result.ok = true;
            result.rebound_speed = speed;
            result.peak_normal_force = peak_normal_force;
            result.rebound_dir = dir;
            rebound_captured = true;
            break;
        }
    }

    if (!rebound_captured) {
        std::cout << "[" << label << "] Run " << run_id << ": rebound not captured within max steps" << std::endl;
    }

    return result;
}

void print_stats_block(const std::string& label,
                       const std::vector<RunResult>& results) {
    std::vector<double> speeds;
    std::vector<double> forces;
    std::vector<double> dir_x;
    std::vector<double> dir_y;
    std::vector<double> dir_z;

    for (const auto& r : results) {
        if (!r.ok) {
            continue;
        }
        speeds.push_back(r.rebound_speed);
        forces.push_back(r.peak_normal_force);
        dir_x.push_back(r.rebound_dir.x);
        dir_y.push_back(r.rebound_dir.y);
        dir_z.push_back(r.rebound_dir.z);
    }

    Stats s_speed = calc_stats(speeds);
    Stats s_force = calc_stats(forces);
    Stats s_dx = calc_stats(dir_x);
    Stats s_dy = calc_stats(dir_y);
    Stats s_dz = calc_stats(dir_z);

    std::cout << "\n=== " << label << " stats (population stddev) ===" << std::endl;
    std::cout << "Rebound speed [m/s]: mean=" << s_speed.mean << " min=" << s_speed.min << " max=" << s_speed.max
              << " std=" << s_speed.stddev << std::endl;
    std::cout << "Peak normal force [N]: mean=" << s_force.mean << " min=" << s_force.min << " max=" << s_force.max
              << " std=" << s_force.stddev << std::endl;
    std::cout << "Rebound dir X: mean=" << s_dx.mean << " min=" << s_dx.min << " max=" << s_dx.max
              << " std=" << s_dx.stddev << std::endl;
    std::cout << "Rebound dir Y: mean=" << s_dy.mean << " min=" << s_dy.min << " max=" << s_dy.max
              << " std=" << s_dy.stddev << std::endl;
    std::cout << "Rebound dir Z: mean=" << s_dz.mean << " min=" << s_dz.min << " max=" << s_dz.max
              << " std=" << s_dz.stddev << std::endl;
}

float4 edge_quat() {
    float4 q = make_float4(0, 0, 0, 1);
    q = RotateQuat(q, make_float3(1, 0, 0), static_cast<float>(PI / 4.0));
    return q;
}

float4 corner_quat() {
    float4 q = make_float4(0, 0, 0, 1);
    q = RotateQuat(q, make_float3(1, 0, 0), static_cast<float>(PI / 4.0));
    q = RotateQuat(q, make_float3(0, 1, 0), static_cast<float>(PI / 4.0));
    return q;
}

void run_scenario(const std::string& label,
                  const float4& rot,
                  bool per_triangle_patches,
                  bool use_triangle_particles) {
    std::cout << "\n========================================" << std::endl;
    std::cout << label << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Using mesh: " << (use_triangle_particles ? "simpleTriangleShape4mm.stl" : "cube.obj") << std::endl;

    std::vector<RunResult> results;
    results.reserve(kNumRuns);

    for (int i = 0; i < kNumRuns; ++i) {
        RunResult r = run_single_collision(rot, per_triangle_patches, use_triangle_particles, label, i);
        results.push_back(r);
        if (r.ok) {
            std::cout << "Run " << i << ": speed=" << r.rebound_speed << " dir=(" << r.rebound_dir.x << ", "
                      << r.rebound_dir.y << ", " << r.rebound_dir.z << ") force=" << r.peak_normal_force
                      << std::endl;
        }
    }

    print_stats_block(label, results);
}

}  // namespace

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "DEM Simple Collisions Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Particle mesh mode: "
              << (kUseTriangleParticles ? "simpleTriangleShape4mm.stl" : "cube.obj") << std::endl;

    float4 q_edge = edge_quat();
    float4 q_corner = corner_quat();

    run_scenario("Edge impact - single patch", q_edge, false, kUseTriangleParticles);
    run_scenario("Edge impact - 12 patches", q_edge, true, kUseTriangleParticles);
    run_scenario("Corner impact - single patch", q_corner, false, kUseTriangleParticles);
    run_scenario("Corner impact - 12 patches", q_corner, true, kUseTriangleParticles);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test completed" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
