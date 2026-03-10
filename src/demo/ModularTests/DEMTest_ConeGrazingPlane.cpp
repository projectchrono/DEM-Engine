//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Grazing-contact reference demo with these primary 6 variants:
//   cone, cone-study, cube, cube-study, sphere, sphere-study
// 
// horizontal velocity so that the contact surface grazes along a flat plane mesh (no gravity).
// Uses cone.obj, cube.obj, a generic (non-mesh) sphere and plane_20by20.obj.
//
//   1) Variant flag for cone mesh, cube mesh, or sphere clump.
//   2) Plane triangle-size study from 2 to 2048 triangles (factor 4 each step).
//   3) Practical geometric scaling flags.
// Additional cube mode flag: --cube-mode edge|tip
//
// Notes:
//   - sphere uses DEMSim.LoadSphereType(...), i.e. no mesh.
//   - cone and cube use meshes.
//   - cone and cube references are DEME-consistent analytical proxies based on
//     the public FullHertzianForceModel for triangle-involved contact:
//       contact_radius = sqrt(overlapArea / pi)
//       F_n = (4/3) * E_cnt * contact_radius * overlapDepth
//   - sphere uses classic Hertz with DEME's identical-material effective modulus.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/HostSideHelpers.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using namespace std::filesystem;
using namespace deme;

namespace {

constexpr double PI_D = 3.14159265358979323846;
constexpr std::array<int, 6> TRIANGLE_STUDY_COUNTS = {2, 8, 32, 128, 512, 2048};

enum class ShapeVariant { CONE, CUBE, SPHERE };
enum class CubeContactMode { EDGE, TIP };
enum class DemoVariant { CONE, CONE_STUDY, CUBE, CUBE_STUDY, SPHERE, SPHERE_STUDY };

struct DemoConfig {
    DemoVariant variant = DemoVariant::CONE;
    ShapeVariant shape = ShapeVariant::CONE;
    bool triangle_study = false;
    CubeContactMode cube_mode = CubeContactMode::EDGE;

    float mu = 0.3f;
    float CoR = 0.3f;
    float E = 1e8f;
    float nu = 0.3f;
    float density = 2600.0f;

    float base_body_size = 0.2f;       // cone h/r, cube edge, sphere diameter
    float base_plane_halfwidth = 1.0f; // 2 m wide plane
    float base_penetration = 0.005f;
    float base_graze_speed = 0.5f;

    float global_scale = 1.0f;
    float body_size_scale = 1.0f;
    float plane_size_scale = 1.0f;
    float penetration_scale = 1.0f;
    float speed_scale = 1.0f;

    float step_size = 1e-5f;
    float frame_time = 0.2f;
    float total_time = 2.0f;
    float csv_fps = 1000.0f;

    int single_run_plane_triangles = 2048;
};

struct AnalyticalReference {
    bool available = false;
    std::string model;
    double effective_modulus = 0.0;
    double normal_force = 0.0;
    double tangential_force = 0.0;
    double total_force = 0.0;
    double half_angle_rad = 0.0;
};

struct RunSummary {
    int plane_triangles = 0;
    int plane_cells_per_side = 0;
    double mean_force = 0.0;
    double min_force = 0.0;
    double max_force = 0.0;
    double stddev_force = 0.0;
    double mean_fz = 0.0;
    double min_fz = 0.0;
    double max_fz = 0.0;
    AnalyticalReference reference;
    std::string label;
    path out_dir;
};

struct MovingBodyState {
    std::shared_ptr<DEMTracker> tracker;
    float mass = 0.0f;
    float reference_above_lowest_point = 0.0f;
    float init_x = 0.0f;
    float init_z = 0.0f;
    unsigned int family = 1;
};

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string VariantName(const DemoConfig& cfg) {
    switch (cfg.variant) {
        case DemoVariant::CONE:
            return "cone";
        case DemoVariant::CONE_STUDY:
            return "cone-study";
        case DemoVariant::CUBE:
            return "cube";
        case DemoVariant::CUBE_STUDY:
            return "cube-study";
        case DemoVariant::SPHERE:
            return "sphere";
        case DemoVariant::SPHERE_STUDY:
            return "sphere-study";
    }
    return "unknown";
}

std::string CubeModeName(CubeContactMode mode) {
    switch (mode) {
        case CubeContactMode::EDGE:
            return "edge";
        case CubeContactMode::TIP:
            return "tip";
    }
    return "edge";
}

std::string CaseName(const DemoConfig& cfg) {
    std::string name = VariantName(cfg);
    if (cfg.shape == ShapeVariant::CUBE) {
        name += "-" + CubeModeName(cfg.cube_mode);
    }
    return name;
}

DemoVariant ParseVariantName(const std::string& value) {
    const std::string v = ToLower(value);
    if (v == "cone")
        return DemoVariant::CONE;
    if (v == "cone-study" || v == "cone_study")
        return DemoVariant::CONE_STUDY;
    if (v == "cube")
        return DemoVariant::CUBE;
    if (v == "cube-study" || v == "cube_study")
        return DemoVariant::CUBE_STUDY;
    if (v == "sphere")
        return DemoVariant::SPHERE;
    if (v == "sphere-study" || v == "sphere_study")
        return DemoVariant::SPHERE_STUDY;
    throw std::runtime_error("Unknown --variant option: " + value);
}

void ApplyVariantToConfig(DemoConfig& cfg, DemoVariant variant) {
    cfg.variant = variant;
    switch (variant) {
        case DemoVariant::CONE:
            cfg.shape = ShapeVariant::CONE;
            cfg.triangle_study = false;
            break;
        case DemoVariant::CONE_STUDY:
            cfg.shape = ShapeVariant::CONE;
            cfg.triangle_study = true;
            break;
        case DemoVariant::CUBE:
            cfg.shape = ShapeVariant::CUBE;
            cfg.triangle_study = false;
            break;
        case DemoVariant::CUBE_STUDY:
            cfg.shape = ShapeVariant::CUBE;
            cfg.triangle_study = true;
            break;
        case DemoVariant::SPHERE:
            cfg.shape = ShapeVariant::SPHERE;
            cfg.triangle_study = false;
            break;
        case DemoVariant::SPHERE_STUDY:
            cfg.shape = ShapeVariant::SPHERE;
            cfg.triangle_study = true;
            break;
    }
}

void RefreshVariantFromFlags(DemoConfig& cfg) {
    if (cfg.shape == ShapeVariant::CONE)
        cfg.variant = cfg.triangle_study ? DemoVariant::CONE_STUDY : DemoVariant::CONE;
    else if (cfg.shape == ShapeVariant::CUBE)
        cfg.variant = cfg.triangle_study ? DemoVariant::CUBE_STUDY : DemoVariant::CUBE;
    else
        cfg.variant = cfg.triangle_study ? DemoVariant::SPHERE_STUDY : DemoVariant::SPHERE;
}

CubeContactMode ParseCubeMode(const std::string& value) {
    const std::string v = ToLower(value);
    if (v == "edge" || v == "kante")
        return CubeContactMode::EDGE;
    if (v == "tip" || v == "vertex" || v == "corner" || v == "spitze")
        return CubeContactMode::TIP;
    throw std::runtime_error("Unknown --cube-mode option: " + value);
}

bool ParseBool(const std::string& value) {
    const std::string v = ToLower(value);
    if (v == "1" || v == "true" || v == "yes" || v == "on")
        return true;
    if (v == "0" || v == "false" || v == "no" || v == "off")
        return false;
    throw std::runtime_error("Expected boolean value, got: " + value);
}

float ParseFloatArg(const std::string& arg_name, const std::string& value) {
    try {
        return std::stof(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid numeric value for " + arg_name + ": " + value);
    }
}

int ParseIntArg(const std::string& arg_name, const std::string& value) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer value for " + arg_name + ": " + value);
    }
}

void PrintUsage(const char* exe_name) {
    std::cout << "Usage:\n"
              << "  " << exe_name << " --variant cone|cone-study|cube|cube-study|sphere|sphere-study\n"
              << "\nAlternative flags:\n"
              << "  --shape cone|cube|sphere   and   --triangle-study true|false\n"
              << "  --cube-mode edge|tip       (only used with cube)\n"
              << "\nScaling flags:\n"
              << "  --global-scale <f>\n"
              << "  --body-size-scale <f>\n"
              << "  --plane-size-scale <f>\n"
              << "  --penetration-scale <f>\n"
              << "  --speed-scale <f>\n"
              << "\nOther flags:\n"
              << "  --single-run-plane-triangles <n>\n"
              << "  --step-size <f> --frame-time <f> --total-time <f>\n"
              << "  --csv-fps <f>\n"
              << std::endl;
}

DemoConfig ParseArguments(int argc, char* argv[]) {
    DemoConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        }

        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value after " + name);
            }
            return argv[++i];
        };

        if (arg == "--variant") {
            ApplyVariantToConfig(cfg, ParseVariantName(require_value(arg)));
        } else if (arg == "--shape") {
            const std::string value = ToLower(require_value(arg));
            if (value == "cone")
                cfg.shape = ShapeVariant::CONE;
            else if (value == "cube")
                cfg.shape = ShapeVariant::CUBE;
            else if (value == "sphere")
                cfg.shape = ShapeVariant::SPHERE;
            else
                throw std::runtime_error("Unknown --shape option: " + value);
            RefreshVariantFromFlags(cfg);
        } else if (arg == "--triangle-study") {
            cfg.triangle_study = ParseBool(require_value(arg));
            RefreshVariantFromFlags(cfg);
        } else if (arg == "--cube-mode") {
            cfg.cube_mode = ParseCubeMode(require_value(arg));
        } else if (arg == "--global-scale") {
            cfg.global_scale = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--body-size-scale") {
            cfg.body_size_scale = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--plane-size-scale") {
            cfg.plane_size_scale = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--penetration-scale") {
            cfg.penetration_scale = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--speed-scale") {
            cfg.speed_scale = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--step-size") {
            cfg.step_size = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--frame-time") {
            cfg.frame_time = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--total-time") {
            cfg.total_time = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--single-run-plane-triangles") {
            cfg.single_run_plane_triangles = ParseIntArg(arg, require_value(arg));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return cfg;
}

int PlaneCellsFromTriangleCount(int tri_count) {
    if (tri_count <= 0 || tri_count % 2 != 0) {
        throw std::runtime_error("Plane triangle count must be positive and even.");
    }
    const double cells_real = std::sqrt(static_cast<double>(tri_count) / 2.0);
    const int cells = static_cast<int>(std::llround(cells_real));
    if (2 * cells * cells != tri_count) {
        throw std::runtime_error("Plane triangle count must satisfy triangles = 2 * N^2.");
    }
    return cells;
}

DEMMeshConnected BuildPlaneMesh(int tri_count, float halfwidth, const std::shared_ptr<DEMMaterial>& mat) {
    const int cells = PlaneCellsFromTriangleCount(tri_count);

    DEMMeshConnected plane;
    plane.Clear();
    plane.m_vertices.reserve(static_cast<size_t>(cells + 1) * static_cast<size_t>(cells + 1));
    plane.m_face_v_indices.reserve(static_cast<size_t>(tri_count));

    for (int j = 0; j <= cells; ++j) {
        const float y = -halfwidth + (2.0f * halfwidth * static_cast<float>(j) / static_cast<float>(cells));
        for (int i = 0; i <= cells; ++i) {
            const float x = -halfwidth + (2.0f * halfwidth * static_cast<float>(i) / static_cast<float>(cells));
            plane.m_vertices.push_back(make_float3(x, y, 0.0f));
        }
    }

    auto vertex_index = [cells](int i, int j) -> int { return j * (cells + 1) + i; };

    for (int j = 0; j < cells; ++j) {
        for (int i = 0; i < cells; ++i) {
            const int v00 = vertex_index(i, j);
            const int v10 = vertex_index(i + 1, j);
            const int v01 = vertex_index(i, j + 1);
            const int v11 = vertex_index(i + 1, j + 1);

            plane.m_face_v_indices.push_back(make_int3(v00, v10, v11));
            plane.m_face_v_indices.push_back(make_int3(v00, v11, v01));
        }
    }

    plane.nTri = plane.m_face_v_indices.size();
    plane.SetMaterial(mat);
    plane.SetMass(1.0f);
    plane.SetMOI(make_float3(1.0f, 1.0f, 1.0f));
    plane.SetInitPos(make_float3(0.0f, 0.0f, 0.0f));
    plane.SetFamily(100);
    return plane;
}

double DEMEEffectiveModulus(float E, float nu) {
    return static_cast<double>(E) / (2.0 * (1.0 - static_cast<double>(nu) * static_cast<double>(nu)));
}

float Dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 Cross3(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float Length3(const float3& v) {
    return std::sqrt(Dot3(v, v));
}

float3 Normalize3(const float3& v) {
    const float len = Length3(v);
    if (len < 1e-12f)
        return make_float3(0.f, 0.f, 0.f);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

float4 NormalizeQuat(const float4& q) {
    const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (n < 1e-12f)
        return make_float4(0.f, 0.f, 0.f, 1.f);
    return make_float4(q.x / n, q.y / n, q.z / n, q.w / n);
}

float4 QuatMul(const float4& a, const float4& b) {
    return make_float4(a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                       a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                       a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                       a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float4 QuatAxisAngle(const float3& axis_raw, float angle_rad) {
    const float3 axis = Normalize3(axis_raw);
    const float s = std::sin(0.5f * angle_rad);
    const float c = std::cos(0.5f * angle_rad);
    return NormalizeQuat(make_float4(axis.x * s, axis.y * s, axis.z * s, c));
}

float4 QuatFromTwoVectors(const float3& from_raw, const float3& to_raw) {
    const float3 from = Normalize3(from_raw);
    const float3 to = Normalize3(to_raw);
    const float dot_v = Dot3(from, to);

    if (dot_v > 1.f - 1e-6f) {
        return make_float4(0.f, 0.f, 0.f, 1.f);
    }

    if (dot_v < -1.f + 1e-6f) {
        float3 axis = Cross3(from, make_float3(1.f, 0.f, 0.f));
        if (Length3(axis) < 1e-6f) {
            axis = Cross3(from, make_float3(0.f, 1.f, 0.f));
        }
        return QuatAxisAngle(axis, static_cast<float>(PI_D));
    }

    const float3 axis = Cross3(from, to);
    const float s = std::sqrt((1.f + dot_v) * 2.f);
    const float inv_s = 1.f / s;
    return NormalizeQuat(make_float4(axis.x * inv_s, axis.y * inv_s, axis.z * inv_s, 0.5f * s));
}

float3 RotateByQuat(const float4& q_raw, const float3& v) {
    const float4 q = NormalizeQuat(q_raw);
    const float4 vq = make_float4(v.x, v.y, v.z, 0.f);
    const float4 q_conj = make_float4(-q.x, -q.y, -q.z, q.w);
    const float4 rq = QuatMul(QuatMul(q, vq), q_conj);
    return make_float3(rq.x, rq.y, rq.z);
}

float CubeLowestPointOffset(float cube_edge, const float4& quat) {
    float min_z = std::numeric_limits<float>::infinity();
    for (int sx : {-1, 1}) {
        for (int sy : {-1, 1}) {
            for (int sz : {-1, 1}) {
                const float3 p = make_float3(0.5f * cube_edge * static_cast<float>(sx),
                                             0.5f * cube_edge * static_cast<float>(sy),
                                             0.5f * cube_edge * static_cast<float>(sz));
                const float3 pr = RotateByQuat(quat, p);
                min_z = std::min(min_z, pr.z);
            }
        }
    }
    return -min_z;
}

AnalyticalReference BuildReferenceCone(float E, float nu, float mu, float radius, float height, float penetration) {
    AnalyticalReference ref;
    ref.available = true;
    ref.model = "DEME mesh area-proxy Hertz for conical mesh on plane + Coulomb friction";
    ref.half_angle_rad = std::atan(static_cast<double>(radius) / static_cast<double>(height));
    ref.effective_modulus = DEMEEffectiveModulus(E, nu);
    ref.normal_force = (4.0 / 3.0) * ref.effective_modulus * std::tan(ref.half_angle_rad) *
                       static_cast<double>(penetration) * static_cast<double>(penetration);
    ref.tangential_force = static_cast<double>(mu) * ref.normal_force;
    ref.total_force = std::sqrt(ref.normal_force * ref.normal_force + ref.tangential_force * ref.tangential_force);
    return ref;
}

AnalyticalReference BuildReferenceSphere(float E, float nu, float mu, float radius, float penetration) {
    AnalyticalReference ref;
    ref.available = true;
    ref.model = "DEME Hertz sphere on plane + Coulomb friction (identical materials)";
    ref.effective_modulus = DEMEEffectiveModulus(E, nu);
    ref.normal_force = (4.0 / 3.0) * ref.effective_modulus * std::sqrt(static_cast<double>(radius)) *
                       std::pow(static_cast<double>(penetration), 1.5);
    ref.tangential_force = static_cast<double>(mu) * ref.normal_force;
    ref.total_force = std::sqrt(ref.normal_force * ref.normal_force + ref.tangential_force * ref.tangential_force);
    return ref;
}

AnalyticalReference BuildReferenceCubeEdge(float E, float nu, float mu, float cube_edge, float penetration) {
    AnalyticalReference ref;
    ref.available = true;
    ref.model = "DEME mesh area-proxy Hertz for finite 90deg wedge edge + Coulomb friction";
    ref.half_angle_rad = PI_D / 4.0;
    ref.effective_modulus = DEMEEffectiveModulus(E, nu);

    const double overlap_area = 2.0 * static_cast<double>(cube_edge) * static_cast<double>(penetration);
    const double contact_radius = std::sqrt(overlap_area / PI_D);
    ref.normal_force = (4.0 / 3.0) * ref.effective_modulus * contact_radius * static_cast<double>(penetration);
    ref.tangential_force = static_cast<double>(mu) * ref.normal_force;
    ref.total_force = std::sqrt(ref.normal_force * ref.normal_force + ref.tangential_force * ref.tangential_force);
    return ref;
}

AnalyticalReference BuildReferenceCubeTip(float E, float nu, float mu, float penetration) {
    AnalyticalReference ref;
    ref.available = true;
    ref.model = "DEME mesh area-proxy Hertz for cube vertex (trihedral tip) + Coulomb friction";
    ref.effective_modulus = DEMEEffectiveModulus(E, nu);

    const double area_coeff = 1.5 * std::sqrt(3.0);  // A = (3*sqrt(3)/2) * delta^2
    const double contact_radius = std::sqrt(area_coeff / PI_D) * static_cast<double>(penetration);
    ref.half_angle_rad = std::atan(std::sqrt(area_coeff / PI_D));  // equivalent cone half-angle for same A(h)
    ref.normal_force = (4.0 / 3.0) * ref.effective_modulus * contact_radius * static_cast<double>(penetration);
    ref.tangential_force = static_cast<double>(mu) * ref.normal_force;
    ref.total_force = std::sqrt(ref.normal_force * ref.normal_force + ref.tangential_force * ref.tangential_force);
    return ref;
}

AnalyticalReference BuildReferenceForShape(const DemoConfig& cfg, float body_size, float penetration) {
    switch (cfg.shape) {
        case ShapeVariant::CONE:
            return BuildReferenceCone(cfg.E, cfg.nu, cfg.mu, body_size, body_size, penetration);
        case ShapeVariant::SPHERE:
            return BuildReferenceSphere(cfg.E, cfg.nu, cfg.mu, 0.5f * body_size, penetration);
        case ShapeVariant::CUBE:
            if (cfg.cube_mode == CubeContactMode::EDGE)
                return BuildReferenceCubeEdge(cfg.E, cfg.nu, cfg.mu, body_size, penetration);
            return BuildReferenceCubeTip(cfg.E, cfg.nu, cfg.mu, penetration);
    }
    AnalyticalReference ref;
    ref.available = false;
    ref.model = "Unsupported shape";
    return ref;
}

void PrintReference(const AnalyticalReference& ref) {
    if (!ref.available) {
        std::cout << "Analytical reference: not reported (" << ref.model << ")" << std::endl;
        return;
    }

    std::cout << "Analytical reference model: " << ref.model << std::endl;
    std::cout << "  E_eff:   " << ref.effective_modulus << " Pa" << std::endl;
    if (ref.half_angle_rad > 0.0) {
        std::cout << "  alpha:   " << ref.half_angle_rad * 180.0 / PI_D << " deg" << std::endl;
    }
    std::cout << "  F_n:     " << ref.normal_force << " N" << std::endl;
    std::cout << "  F_t:     " << ref.tangential_force << " N" << std::endl;
    std::cout << "  |F|:     " << ref.total_force << " N" << std::endl;
}

MovingBodyState AddMovingBody(DEMSolver& DEMSim,
                              const DemoConfig& cfg,
                              const std::shared_ptr<DEMMaterial>& mat,
                              float body_size,
                              float penetration,
                              float init_x) {
    MovingBodyState state;

    if (cfg.shape == ShapeVariant::CONE) {
        const float cone_radius = body_size;
        const float cone_height = body_size;
        const float cone_volume = (1.0f / 3.0f) * static_cast<float>(PI_D) * cone_radius * cone_radius * cone_height;
        const float cone_mass = cfg.density * cone_volume;
        const float cone_Ixy = 3.0f * cone_mass / 20.0f * cone_radius * cone_radius +
                               3.0f * cone_mass / 80.0f * cone_height * cone_height;
        const float cone_Iz = 3.0f * cone_mass / 10.0f * cone_radius * cone_radius;
        const float centroid_above_tip = 0.75f * cone_height;
        const float init_z = -penetration + centroid_above_tip;

        auto cone = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cone.obj").string(), mat);
        cone->InformCentroidPrincipal(make_float3(0.f, 0.f, 0.75f), make_float4(0.f, 0.f, 0.f, 1.f));
        cone->Scale(body_size);
        cone->SetMass(cone_mass);
        cone->SetMOI(make_float3(cone_Ixy, cone_Ixy, cone_Iz));
        cone->SetInitPos(make_float3(init_x, 0.f, init_z));
        cone->SetFamily(1);

        state.tracker = DEMSim.Track(cone);
        state.mass = cone_mass;
        state.reference_above_lowest_point = centroid_above_tip;
        state.init_x = init_x;
        state.init_z = init_z;
        state.family = 1;
        return state;
    }

    if (cfg.shape == ShapeVariant::CUBE) {
        const float cube_edge = body_size;
        const float cube_mass = cfg.density * cube_edge * cube_edge * cube_edge;
        const float cube_I = cube_mass * cube_edge * cube_edge / 6.0f;

        float4 cube_quat = make_float4(0.f, 0.f, 0.f, 1.f);
        if (cfg.cube_mode == CubeContactMode::EDGE) {
            cube_quat = QuatAxisAngle(make_float3(0.f, 1.f, 0.f), static_cast<float>(PI_D / 4.0));
        } else {
            cube_quat = QuatFromTwoVectors(make_float3(-1.f, -1.f, -1.f), make_float3(0.f, 0.f, -1.f));
        }

        const float centroid_above_lowest = CubeLowestPointOffset(cube_edge, cube_quat);
        const float init_z = -penetration + centroid_above_lowest;

        auto cube = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cube.obj").string(), mat);
        cube->Scale(body_size);
        cube->SetMass(cube_mass);
        cube->SetMOI(make_float3(cube_I, cube_I, cube_I));
        cube->SetInitQuat(cube_quat);
        cube->SetInitPos(make_float3(init_x, 0.f, init_z));
        cube->SetFamily(1);

        state.tracker = DEMSim.Track(cube);
        state.mass = cube_mass;
        state.reference_above_lowest_point = centroid_above_lowest;
        state.init_x = init_x;
        state.init_z = init_z;
        state.family = 1;
        return state;
    }

    const float sphere_radius = 0.5f * body_size;
    const float sphere_volume = 4.0f / 3.0f * static_cast<float>(PI_D) * sphere_radius * sphere_radius * sphere_radius;
    const float sphere_mass = cfg.density * sphere_volume;
    const float init_z = sphere_radius - penetration;

    auto sphere_type = DEMSim.LoadSphereType(sphere_mass, sphere_radius, mat);
    auto sphere_batch = DEMSim.AddClumps(sphere_type, make_float3(init_x, 0.f, init_z));
    auto sphere_tracker = DEMSim.Track(sphere_batch);

    state.tracker = sphere_tracker;
    state.mass = sphere_mass;
    state.reference_above_lowest_point = sphere_radius;
    state.init_x = init_x;
    state.init_z = init_z;
    state.family = 0;
    return state;
}

RunSummary RunSingleCase(const DemoConfig& cfg, int plane_triangles, const path& root_out_dir) {
    const float body_size = cfg.base_body_size * cfg.global_scale * cfg.body_size_scale;
    const float plane_halfwidth = cfg.base_plane_halfwidth * cfg.global_scale * cfg.plane_size_scale;
    const float penetration = cfg.base_penetration * cfg.global_scale * cfg.penetration_scale;
    const float graze_speed = cfg.base_graze_speed * cfg.speed_scale;
    const float init_x = -body_size;
    const int cells = PlaneCellsFromTriangleCount(plane_triangles);

    RunSummary summary;
    summary.plane_triangles = plane_triangles;
    summary.plane_cells_per_side = cells;
    summary.label = CaseName(cfg) + "_T" + std::to_string(plane_triangles);
    summary.out_dir = root_out_dir / summary.label;

    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.InstructBoxDomainDimension(30, 30, 5);
    // No gravity: we are controlling the cone position/velocity explicitly
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, 0));
    // Enable mesh-mesh contacts (required for mesh contact)
    DEMSim.SetMeshUniversalContact(true);
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.SetExpandSafetyAdder(graze_speed);

    auto mat = DEMSim.LoadMaterial({{"E", cfg.E}, {"nu", cfg.nu}, {"CoR", cfg.CoR}, {"mu", cfg.mu}, {"Crr", 0.0f}});

    DEMMeshConnected plane_mesh = BuildPlaneMesh(plane_triangles, plane_halfwidth, mat);
    auto plane = DEMSim.AddWavefrontMeshObject(plane_mesh);
    plane->SetFamily(100);
    DEMSim.SetFamilyFixed(100);

    MovingBodyState body = AddMovingBody(DEMSim, cfg, mat, body_size, penetration, init_x);
    DEMSim.SetFamilyPrescribedLinVel(body.family, to_string_with_precision(graze_speed), "0", "0");
    DEMSim.SetFamilyPrescribedAngVel(body.family, "0", "0", "0");

    DEMSim.SetInitTimeStep(cfg.step_size);
    DEMSim.Initialize();
    body.tracker->SetVel(make_float3(graze_speed, 0.f, 0.f));

    std::error_code dir_ec;
    create_directories(summary.out_dir, dir_ec);
    if (dir_ec || !is_directory(summary.out_dir)) {
        throw std::runtime_error("Failed to create output directory: " + summary.out_dir.string());
    }

    summary.reference = BuildReferenceForShape(cfg, body_size, penetration);

    std::cout << "=====================================================\n";
    std::cout << "Variant:             " << VariantName(cfg) << "\n";
    if (cfg.shape == ShapeVariant::CUBE)
        std::cout << "Cube mode:           " << CubeModeName(cfg.cube_mode) << "\n";
    std::cout << "Plane triangles:     " << plane_triangles << " (" << cells << " x " << cells << " cells)\n";
    std::cout << "Body size:           " << body_size << " m\n";
    std::cout << "Plane halfwidth:     " << plane_halfwidth << " m\n";
    std::cout << "Penetration:         " << penetration << " m\n";
    std::cout << "Graze speed:         " << graze_speed << " m/s\n";
    std::cout << "Total graze dist:    " << graze_speed * cfg.total_time << " m\n";
    PrintReference(summary.reference);
    std::cout << "=====================================================\n";

    std::vector<double> force_mags;
    std::vector<double> force_z_components;

    if (cfg.csv_fps <= 0.0f) {
        throw std::runtime_error("csv_fps must be > 0.");
    }

    const double csv_dt = 1.0 / static_cast<double>(cfg.csv_fps);
    const int n_csv_samples = static_cast<int>(std::round(static_cast<double>(cfg.total_time) / csv_dt));
    const int n_frames = static_cast<int>(std::round(cfg.total_time / cfg.frame_time));

    std::ofstream trace_csv(summary.out_dir / "force_trace_1000fps.csv");
    trace_csv << std::setprecision(12);
    trace_csv << "sample,time_s,x_m,y_m,z_m,z_low_m,fx_N,fy_N,fz_N,fmag_N,plane_triangles,plane_cells_per_side";
    if (summary.reference.available) {
        trace_csv << ",ref_total_N,ref_normal_N,rel_err_total_pct,rel_err_fz_pct";
    }
    trace_csv << "\n";

    int visual_frame_idx = 0;
    double next_visual_time = static_cast<double>(cfg.frame_time);
    const double visual_eps = 0.5 * csv_dt;

    for (int sample_idx = 1; sample_idx <= n_csv_samples; ++sample_idx) {
        DEMSim.DoDynamics(csv_dt);

        const float3 cnt_acc = body.tracker->ContactAcc();
        const float3 cnt_force = cnt_acc * body.mass;
        const double force_mag = std::sqrt(static_cast<double>(cnt_force.x) * cnt_force.x +
                                           static_cast<double>(cnt_force.y) * cnt_force.y +
                                           static_cast<double>(cnt_force.z) * cnt_force.z);
        force_mags.push_back(force_mag);
        force_z_components.push_back(static_cast<double>(cnt_force.z));

        const float3 pos = body.tracker->Pos();
        const float body_lowest_z = pos.z - body.reference_above_lowest_point;
        const double time_s = static_cast<double>(sample_idx) * csv_dt;

        trace_csv << sample_idx << ','
                  << time_s << ','
                  << pos.x << ','
                  << pos.y << ','
                  << pos.z << ','
                  << body_lowest_z << ','
                  << cnt_force.x << ','
                  << cnt_force.y << ','
                  << cnt_force.z << ','
                  << force_mag << ','
                  << plane_triangles << ','
                  << cells;

        if (summary.reference.available) {
            const double rel_err_total = 100.0 * (force_mag - summary.reference.total_force) / summary.reference.total_force;
            const double rel_err_fz = 100.0 * (static_cast<double>(cnt_force.z) - summary.reference.normal_force) /
                                      summary.reference.normal_force;
            trace_csv << ',' << summary.reference.total_force
                      << ',' << summary.reference.normal_force
                      << ',' << rel_err_total
                      << ',' << rel_err_fz;
        }
        trace_csv << '\n';

        if (visual_frame_idx < n_frames && time_s + visual_eps >= next_visual_time) {
            ++visual_frame_idx;

            char meshfilename[256];
            std::snprintf(meshfilename, sizeof(meshfilename), "mesh_%04d.vtk", visual_frame_idx);
            DEMSim.WriteMeshFile(summary.out_dir / meshfilename);

            if (cfg.shape == ShapeVariant::SPHERE) {
                char spherefilename[256];
                std::snprintf(spherefilename, sizeof(spherefilename), "sphere_%04d.csv", visual_frame_idx);
                DEMSim.WriteSphereFile(summary.out_dir / spherefilename);
            }

            std::cout << "t=" << time_s << " s"
                      << "  x=" << pos.x
                      << "  z_low=" << body_lowest_z
                      << "  |F_cnt|=" << force_mag << " N";

            if (force_mag > 1e-12) {
                const double inv_f = 1.0 / force_mag;
                std::cout << "  F_dir=(" << cnt_force.x * inv_f << "," << cnt_force.y * inv_f << ","
                          << cnt_force.z * inv_f << ")";
            }
            std::cout << std::endl;

            next_visual_time += static_cast<double>(cfg.frame_time);
        }
    }

    if (force_mags.empty()) {
        throw std::runtime_error("No force data collected.");
    }

    summary.mean_force = 0.0;
    summary.min_force = force_mags.front();
    summary.max_force = force_mags.front();
    summary.mean_fz = 0.0;
    summary.min_fz = force_z_components.front();
    summary.max_fz = force_z_components.front();

    for (size_t i = 0; i < force_mags.size(); ++i) {
        summary.mean_force += force_mags[i];
        summary.mean_fz += force_z_components[i];
        summary.min_force = std::min(summary.min_force, force_mags[i]);
        summary.max_force = std::max(summary.max_force, force_mags[i]);
        summary.min_fz = std::min(summary.min_fz, force_z_components[i]);
        summary.max_fz = std::max(summary.max_fz, force_z_components[i]);
    }

    summary.mean_force /= static_cast<double>(force_mags.size());
    summary.mean_fz /= static_cast<double>(force_z_components.size());

    double variance = 0.0;
    for (double f : force_mags) {
        const double d = f - summary.mean_force;
        variance += d * d;
    }
    summary.stddev_force = std::sqrt(variance / static_cast<double>(force_mags.size()));

    std::cout << "\n=== Statistics for " << summary.label << " ===" << std::endl;
    std::cout << "  Mean |F|:   " << summary.mean_force << " N" << std::endl;
    std::cout << "  Min |F|:    " << summary.min_force << " N" << std::endl;
    std::cout << "  Max |F|:    " << summary.max_force << " N" << std::endl;
    std::cout << "  StdDev |F|: " << summary.stddev_force << " N" << std::endl;
    std::cout << "  Mean Fz:    " << summary.mean_fz << " N" << std::endl;
    std::cout << "  Min Fz:     " << summary.min_fz << " N" << std::endl;
    std::cout << "  Max Fz:     " << summary.max_fz << " N" << std::endl;

    if (summary.reference.available) {
        const double rel_err_total = 100.0 * (summary.mean_force - summary.reference.total_force) / summary.reference.total_force;
        const double rel_err_fz = 100.0 * (summary.mean_fz - summary.reference.normal_force) / summary.reference.normal_force;
        std::cout << "  Rel. error |F| vs reference: " << rel_err_total << " %" << std::endl;
        std::cout << "  Rel. error Fz vs reference:  " << rel_err_fz << " %" << std::endl;
    }

    DEMSim.ShowTimingStats();
    std::cout << "=====================================================\n";
    return summary;
}

void WriteStudySummaryCsv(const path& filename, const std::vector<RunSummary>& results) {
    std::ofstream out(filename);
    out << "plane_triangles,plane_cells_per_side,mean_force_N,mean_fz_N,stddev_force_N,reference_total_N,reference_fz_N,delta_vs_finest_pct\n";

    const double finest_force = results.empty() ? 0.0 : results.back().mean_force;
    for (const auto& result : results) {
        double delta_vs_finest = std::numeric_limits<double>::quiet_NaN();
        if (!results.empty() && finest_force != 0.0) {
            delta_vs_finest = 100.0 * (result.mean_force - finest_force) / finest_force;
        }
        out << result.plane_triangles << ','
            << result.plane_cells_per_side << ','
            << result.mean_force << ','
            << result.mean_fz << ','
            << result.stddev_force << ','
            << (result.reference.available ? result.reference.total_force : std::numeric_limits<double>::quiet_NaN()) << ','
            << (result.reference.available ? result.reference.normal_force : std::numeric_limits<double>::quiet_NaN()) << ','
            << delta_vs_finest << '\n';
    }
}

void PrintStudyComparison(const std::vector<RunSummary>& results) {
    if (results.empty())
        return;

    const double finest_force = results.back().mean_force;
    std::cout << "\n================ Triangle-size comparison ================\n";
    std::cout << std::setw(12) << "Triangles"
              << std::setw(14) << "Cells/side"
              << std::setw(18) << "Mean |F| [N]"
              << std::setw(18) << "Mean Fz [N]"
              << std::setw(18) << "StdDev [N]"
              << std::setw(20) << "Delta vs finest [%]" << std::endl;

    for (const auto& result : results) {
        const double delta_vs_finest = 100.0 * (result.mean_force - finest_force) / finest_force;
        std::cout << std::setw(12) << result.plane_triangles
                  << std::setw(14) << result.plane_cells_per_side
                  << std::setw(18) << result.mean_force
                  << std::setw(18) << result.mean_fz
                  << std::setw(18) << result.stddev_force
                  << std::setw(20) << delta_vs_finest << std::endl;
    }
    std::cout << "==========================================================\n";
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        DemoConfig cfg = ParseArguments(argc, argv);

        const path root_out_dir = current_path() / "modular_test_output" / "DEMTest_GrazingPlaneVariants" / CaseName(cfg);
        std::error_code dir_ec;
        create_directories(root_out_dir, dir_ec);
        if (dir_ec || !is_directory(root_out_dir)) {
            std::cerr << "Failed to create root output directory: " << root_out_dir << std::endl;
            return 1;
        }

        std::vector<RunSummary> results;
        if (cfg.triangle_study) {
            for (int tri_count : TRIANGLE_STUDY_COUNTS) {
                results.push_back(RunSingleCase(cfg, tri_count, root_out_dir));
            }
            PrintStudyComparison(results);
            WriteStudySummaryCsv(root_out_dir / "triangle_study_summary.csv", results);
        } else {
            results.push_back(RunSingleCase(cfg, cfg.single_run_plane_triangles, root_out_dir));
            WriteStudySummaryCsv(root_out_dir / "single_run_summary.csv", results);
        }

        std::cout << "DEMTest_GrazingPlaneVariants exiting..." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
