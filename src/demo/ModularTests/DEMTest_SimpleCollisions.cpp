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

constexpr bool kUseTriangleParticles = false; // toggle to run the STL-based triangle setup
constexpr float kMmToMeters = 0.001f;
constexpr double kTriangleParticleDensity = 2600.0;

constexpr int kNumRuns = 10;
constexpr double kGap = 0.005;        // 0.5 mm
constexpr double kSpeed = 1.0;        // 1 m/s magnitude
constexpr double kTimeStep = 1e-5;
constexpr int kMaxSteps = 50000; // oberserve 0.5s fitting with --> kTimeStep
constexpr double kContactEps = 1e-6;

// NEW: impact angle controls
constexpr double kImpactThetaDeg = 0.0;   // 0 = vertical down, 90 = pure lateral
constexpr double kImpactPhiDeg   = 0.0;   // azimuth in XY plane: 0 -> +X, 90 -> +Y

// NEW: multi-impact tracking
constexpr int kMaxImpactsToRecord = 8;

double vmax = kSpeed;

struct ImpactEvent {
    bool has_rebound = false;           // rebound captured at end of this contact episode
    double peak_normal_force = 0.0;     // peak Fn during this episode
    double rebound_speed = 0.0;         // |v| right after separation (if has_rebound)
    float3 rebound_dir = make_float3(0,0,0);
    int start_step = -1;
    int end_step   = -1;
};

struct RunResult {
    bool ok = false;
    std::vector<ImpactEvent> impacts;   // NEW: can contain multiple episodes
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
    if (values.empty()) return s;
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

// NEW: build initial velocity vector from speed + angles (theta from +normal, phi azimuth in plane)
float3 build_velocity(double speed, double theta_deg, double phi_deg) {
    const double theta = theta_deg * PI / 180.0;
    const double phi   = phi_deg   * PI / 180.0;

    // normal component (downwards for approaching)
    const double v_n = -speed * std::cos(theta);
    // tangential magnitude
    const double v_t =  speed * std::sin(theta);

    const double vx = v_t * std::cos(phi);
    const double vy = v_t * std::sin(phi);
    const double vz = v_n;

    return make_float3((float)vx, (float)vy, (float)vz);
}

std::shared_ptr<DEMMesh> load_cube_template(DEMSolver& DEMSim,
                                           const std::shared_ptr<DEMMaterial>& mat_type) {
    auto mesh_template = DEMSim.LoadMeshType((GET_DATA_PATH() / "mesh/cube.obj").string(), mat_type,
                                             true, false);
    if (!mesh_template) return nullptr;
    mesh_template->SetMaterial(mat_type);
    return mesh_template;
}

std::shared_ptr<DEMMesh> load_triangle_template(DEMSolver& DEMSim,
                                               const std::shared_ptr<DEMMaterial>& mat_type,
                                               float& out_mass,
                                               float3& out_moi) {
    std::shared_ptr<DEMMesh> mesh_template =
        DEMSim.LoadMeshType((GET_DATA_PATH() / "mesh/simpleTriangleShape4mm.stl").string(), mat_type, true, false);
    if (!mesh_template) return nullptr;

    mesh_template->Scale(kMmToMeters);

    double volume = 0.0;
    float3 center = make_float3(0, 0, 0);
    float3 inertia = make_float3(0, 0, 0);
    mesh_template->ComputeMassProperties(volume, center, inertia);

    out_mass = static_cast<float>(volume * kTriangleParticleDensity);
    out_moi = inertia * static_cast<float>(kTriangleParticleDensity);

    mesh_template->SetMaterial(mat_type);
    return mesh_template;
}

RunResult run_single_collision(const float4& init_rot,
                               bool use_triangle_particles,
                               const std::string& label,
                               int run_id,
                               const float3& init_vel) {
    RunResult result;

    DEMSolver DEMSim;
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.InstructBoxDomainDimension(5, 5, 5);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, 0));
    DEMSim.SetMeshUniversalContact(true);
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.SetExpandSafetyAdder(vmax);

    auto mat_type = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.00}});

    float3 plane_normal = make_float3(0, 0, 1);
    auto plane = DEMSim.AddBCPlane(make_float3(0, 0, 0), plane_normal, mat_type);
    auto plane_tracker = DEMSim.Track(plane);

    auto mesh_template = std::shared_ptr<DEMMesh>{};
    float particle_mass = 1.0f;
    float3 particle_moi = make_float3(1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f);

    if (use_triangle_particles) {
        mesh_template = load_triangle_template(DEMSim, mat_type, particle_mass, particle_moi);
    } else {
        mesh_template = load_cube_template(DEMSim, mat_type);
    }
    if (!mesh_template) {
        std::cout << "[" << label << "] Run " << run_id << ": failed to load mesh template" << std::endl;
        return result;
    }

    double min_z = compute_min_z_rotated(mesh_template, init_rot);
    double init_z = kGap - min_z;

    auto body = DEMSim.AddMeshFromTemplate(mesh_template, make_float3(0, 0, 0));
    body->SetFamily(0);
    body->SetMass(particle_mass);
    body->SetMOI(particle_moi);
    body->SetInitQuat(init_rot);
    body->SetInitPos(make_float3(0, 0, static_cast<float>(init_z)));
    auto body_tracker = DEMSim.Track(body);

    DEMSim.SetInitTimeStep(kTimeStep);
    DEMSim.Initialize();

    // NEW: angled initial velocity
    body_tracker->SetVel(init_vel);

    bool in_contact = false;
    ImpactEvent current{};
    int impacts_recorded = 0;

    for (int step = 0; step < kMaxSteps; ++step) {
        DEMSim.DoStepDynamics();

        // NOTE: this is your current way to estimate contact force on the plane
        float3 plane_force = plane_tracker->ContactAcc();
        plane_force = vec_scale(plane_force, plane_tracker->Mass());
        double normal_force = std::abs(vec_dot(plane_force, plane_normal));

        // start of a new contact episode
        if (!in_contact && normal_force > kContactEps) {
            in_contact = true;
            current = ImpactEvent{};
            current.start_step = step;
            current.peak_normal_force = normal_force;
        }

        // update peak during contact
        if (in_contact) {
            current.peak_normal_force = std::max(current.peak_normal_force, normal_force);
        }

        // end of contact episode
        if (in_contact && normal_force <= kContactEps) {
            in_contact = false;
            current.end_step = step;

            // capture rebound info if moving away (positive normal velocity)
            float3 vel = body_tracker->Vel();
            double vel_n = vec_dot(vel, plane_normal);

            if (vel_n > 0.0) {
                double speed = vec_length(vel);
                float3 dir = make_float3(0, 0, 0);
                if (speed > 0) {
                    dir = vec_scale(vel, 1.0 / speed);
                }
                current.has_rebound = true;
                current.rebound_speed = speed;
                current.rebound_dir = dir;
            }

            result.impacts.push_back(current);
            impacts_recorded++;

            if (impacts_recorded >= kMaxImpactsToRecord) {
                break;
            }
        }
    }

    result.ok = !result.impacts.empty();
    if (!result.ok) {
        std::cout << "[" << label << "] Run " << run_id << ": no impacts recorded within max steps" << std::endl;
    }

    return result;
}

// Updated stats: by default we evaluate the FIRST rebound episode that has_rebound==true
void print_stats_block(const std::string& label,
                       const std::vector<RunResult>& results) {
    std::vector<double> speeds;
    std::vector<double> forces;
    std::vector<double> dir_x;
    std::vector<double> dir_y;
    std::vector<double> dir_z;
    std::vector<double> n_impacts;

    for (const auto& r : results) {
        if (!r.ok) continue;

        n_impacts.push_back((double)r.impacts.size());

        // pick first episode with rebound
        const ImpactEvent* chosen = nullptr;
        for (const auto& ev : r.impacts) {
            if (ev.has_rebound) { chosen = &ev; break; }
        }
        if (!chosen) {
            // still record peak of first impact if rebound wasn't detected
            forces.push_back(r.impacts.front().peak_normal_force);
            continue;
        }

        speeds.push_back(chosen->rebound_speed);
        forces.push_back(chosen->peak_normal_force);
        dir_x.push_back(chosen->rebound_dir.x);
        dir_y.push_back(chosen->rebound_dir.y);
        dir_z.push_back(chosen->rebound_dir.z);
    }

    Stats s_speed = calc_stats(speeds);
    Stats s_force = calc_stats(forces);
    Stats s_dx = calc_stats(dir_x);
    Stats s_dy = calc_stats(dir_y);
    Stats s_dz = calc_stats(dir_z);
    Stats s_ni = calc_stats(n_impacts);

    std::cout << "\n=== " << label << " stats (population stddev) ===" << std::endl;
    std::cout << "Impacts per run: mean=" << s_ni.mean << " min=" << s_ni.min << " max=" << s_ni.max
              << " std=" << s_ni.stddev << std::endl;
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

// Rotations
float4 flat_quat() {
    return make_float4(0, 0, 0, 1); // NEW: identity
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
                  bool use_triangle_particles,
                  const float3& init_vel) {
    std::cout << "\n========================================" << std::endl;
    std::cout << label << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Using mesh: " << (use_triangle_particles ? "simpleTriangleShape4mm.stl" : "cube.obj") << std::endl;
    std::cout << "Init vel: (" << init_vel.x << ", " << init_vel.y << ", " << init_vel.z << ")"
              << " |v|=" << vec_length(init_vel) << std::endl;

    std::vector<RunResult> results;
    results.reserve(kNumRuns);

    for (int i = 0; i < kNumRuns; ++i) {
        RunResult r = run_single_collision(rot, use_triangle_particles, label, i, init_vel);
        results.push_back(r);

        if (r.ok) {
            std::cout << "Run " << i << ": impacts=" << r.impacts.size();
            // print first rebound episode if exists
            const ImpactEvent* chosen = nullptr;
            for (const auto& ev : r.impacts) { if (ev.has_rebound) { chosen = &ev; break; } }
            if (chosen) {
                std::cout << " rebound_speed=" << chosen->rebound_speed
                          << " dir=(" << chosen->rebound_dir.x << ", " << chosen->rebound_dir.y << ", " << chosen->rebound_dir.z << ")"
                          << " peakFn=" << chosen->peak_normal_force;
            } else {
                std::cout << " (no rebound captured) peakFn_first=" << r.impacts.front().peak_normal_force;
            }
            std::cout << std::endl;
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

    // NEW: build velocity once (same for all scenarios)
    float3 init_vel = build_velocity(kSpeed, kImpactThetaDeg, kImpactPhiDeg);

    float4 q_flat   = flat_quat();
    float4 q_edge   = edge_quat();
    float4 q_corner = corner_quat();

    run_scenario("Flat impact",   q_flat,   kUseTriangleParticles, init_vel);
    run_scenario("Edge impact",   q_edge,   kUseTriangleParticles, init_vel);
    run_scenario("Corner impact", q_corner, kUseTriangleParticles, init_vel);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test completed" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
