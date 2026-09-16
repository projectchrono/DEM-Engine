//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// This modular test pours spheres into three analytical cone/cylinder containers:
//   1. a cone with its corner down, used as a hollow conical wall,
//   2. a cone with its corner up, used as a solid conical obstacle,
//   3. a cone whose analytical tip is below the floor, so the active surface is
//      a conical frustum.
//
// The cone and cylinder contacts are analytical DEM-Engine objects.  The output
// CSV files can be converted into a ParaView time series for visual inspection.
// =============================================================================

#include <DEM/API.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace deme;

namespace {

struct PourCase {
    std::string name;
    std::string title;
    float x_center;
    float3 cone_tip;
    float3 cone_axis;
    float slope;
    float hmin;
    float hmax;
    objNormal_t cone_normal;
    float cylinder_radius;
    float bottom_z;
    float top_z;
    unsigned int sphere_family;
    unsigned int boundary_family;
};

struct SphereSeed {
    size_t sphere_id;
    unsigned int case_id;
    float radius;
    float3 position;
    float3 velocity;
};

void require_condition(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const char* normal_mode_name(objNormal_t normal) {
    return (normal == ENTITY_NORMAL_OUTWARD) ? "outward" : "inward";
}

std::string frame_file_name(const std::string& stem, int frame) {
    std::ostringstream name;
    name << stem << "_" << std::setw(4) << std::setfill('0') << frame << ".csv";
    return name.str();
}

void write_pour_case_metadata(const std::filesystem::path& out_file, const std::vector<PourCase>& cases) {
    std::ofstream metadata(out_file);
    require_condition(static_cast<bool>(metadata), "Unable to open pour case metadata file.");

    metadata << "case_id,name,title,x_center,tip_x,tip_y,tip_z,axis_x,axis_y,axis_z,slope,hmin,hmax,"
             << "normal_mode,cylinder_radius,bottom_z,top_z,sphere_family,boundary_family\n";
    for (size_t i = 0; i < cases.size(); i++) {
        const auto& this_case = cases.at(i);
        metadata << i << "," << this_case.name << "," << this_case.title << "," << this_case.x_center << ","
                 << this_case.cone_tip.x << "," << this_case.cone_tip.y << "," << this_case.cone_tip.z << ","
                 << this_case.cone_axis.x << "," << this_case.cone_axis.y << "," << this_case.cone_axis.z << ","
                 << this_case.slope << "," << this_case.hmin << "," << this_case.hmax << ","
                 << normal_mode_name(this_case.cone_normal) << "," << this_case.cylinder_radius << ","
                 << this_case.bottom_z << "," << this_case.top_z << "," << this_case.sphere_family << ","
                 << this_case.boundary_family << "\n";
    }
}

void write_pour_sphere_metadata(const std::filesystem::path& out_file, const std::vector<SphereSeed>& spheres) {
    std::ofstream metadata(out_file);
    require_condition(static_cast<bool>(metadata), "Unable to open pour sphere metadata file.");

    metadata << "sphere_id,case_id,radius,initial_x,initial_y,initial_z,initial_vx,initial_vy,initial_vz\n";
    for (const auto& sphere : spheres) {
        metadata << sphere.sphere_id << "," << sphere.case_id << "," << sphere.radius << "," << sphere.position.x << ","
                 << sphere.position.y << "," << sphere.position.z << "," << sphere.velocity.x << ","
                 << sphere.velocity.y << "," << sphere.velocity.z << "\n";
    }
}

void write_dynamic_frame_outputs(DEMSolver& DEMSim, const std::filesystem::path& dynamic_dir, int frame) {
    DEMSim.WriteSphereFile(dynamic_dir / frame_file_name("spheres", frame));
    DEMSim.WriteContactFile((dynamic_dir / frame_file_name("contacts_active", frame)).string(), DEME_TINY_FLOAT);
}

std::vector<SphereSeed> make_sphere_seeds(const std::vector<PourCase>& cases, float radius) {
    std::vector<SphereSeed> seeds;
    constexpr float spacing = 0.118f;
    constexpr float downward_speed = -0.95f;

    for (size_t case_id = 0; case_id < cases.size(); case_id++) {
        const auto& this_case = cases.at(case_id);
        for (int layer = 0; layer < 3; layer++) {
            for (int ix = -2; ix <= 2; ix++) {
                for (int iy = -2; iy <= 2; iy++) {
                    const float dx = spacing * ix + 0.018f * ((layer + iy + 8) % 3 - 1);
                    const float dy = spacing * iy + 0.014f * ((layer + ix + 8) % 3 - 1);
                    if (std::sqrt(dx * dx + dy * dy) > this_case.cylinder_radius - 3.2f * radius) {
                        continue;
                    }
                    const float z =
                        this_case.top_z + 0.16f + 0.12f * layer + 0.012f * static_cast<float>((ix + iy + 8) % 3);
                    seeds.push_back({seeds.size(), static_cast<unsigned int>(case_id), radius,
                                     make_float3(this_case.x_center + dx, dy, z),
                                     make_float3(0.0f, 0.0f, downward_speed)});
                }
            }
        }
    }
    return seeds;
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path out_dir =
        (argc > 1) ? std::filesystem::path(argv[1]) : std::filesystem::path("/tmp/deme_cone_pour_validation");
    const std::filesystem::path dynamic_dir = out_dir / "dynamic";
    std::filesystem::create_directories(dynamic_dir);

    DEMSolver DEMSim(1);
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    DEMSim.SetContactOutputContent({"OWNER", "GEO_ID", "FORCE", "POINT", "NORMAL"});
    DEMSim.UseFrictionalHertzianModel();

    auto mat_sphere = DEMSim.LoadMaterial({{"E", 2.0e5f}, {"nu", 0.3f}, {"CoR", 0.2f}, {"mu", 0.45f}, {"Crr", 0.01f}});
    auto mat_wall = DEMSim.LoadMaterial({{"E", 2.0e5f}, {"nu", 0.3f}, {"CoR", 0.2f}, {"mu", 0.45f}, {"Crr", 0.01f}});

    constexpr float cylinder_radius = 0.72f;
    constexpr float bottom_z = 0.0f;
    constexpr float cone_height = 1.25f;
    constexpr float top_z = bottom_z + cone_height;
    constexpr float frustum_tip_below_floor = 1.0f;
    const float frustum_slope = cylinder_radius / (frustum_tip_below_floor + cone_height);

    const std::vector<PourCase> cases = {
        {"corner_down", "Cone corner down", -2.25f, make_float3(-2.25f, 0.0f, bottom_z), make_float3(0.0f, 0.0f, 1.0f),
         cylinder_radius / cone_height, 0.0f, cone_height, ENTITY_NORMAL_INWARD, cylinder_radius, bottom_z, top_z, 10,
         110},
        {"corner_up", "Cone corner up", 0.0f, make_float3(0.0f, 0.0f, top_z), make_float3(0.0f, 0.0f, -1.0f),
         cylinder_radius / cone_height, 0.0f, cone_height, ENTITY_NORMAL_OUTWARD, cylinder_radius, bottom_z, top_z, 11,
         111},
        {"frustum_from_below", "Cone tip below floor visible frustum", 2.25f,
         make_float3(2.25f, 0.0f, bottom_z - frustum_tip_below_floor), make_float3(0.0f, 0.0f, 1.0f), frustum_slope,
         frustum_tip_below_floor, frustum_tip_below_floor + cone_height, ENTITY_NORMAL_INWARD, cylinder_radius,
         bottom_z, top_z, 12, 112},
    };

    for (const auto& this_case : cases) {
        auto container = DEMSim.AddExternalObject();
        container->AddCylinder(make_float3(this_case.x_center, 0.0f, bottom_z), make_float3(0.0f, 0.0f, 1.0f),
                               this_case.cylinder_radius, mat_wall, ENTITY_NORMAL_INWARD);
        container->AddPlane(make_float3(this_case.x_center, 0.0f, this_case.bottom_z), make_float3(0.0f, 0.0f, 1.0f),
                            mat_wall);
        container->AddConeSegment(this_case.cone_tip, this_case.cone_axis, this_case.slope, this_case.hmin,
                                  this_case.hmax, mat_wall, this_case.cone_normal);
        container->SetFamily(this_case.boundary_family);
        DEMSim.SetFamilyFixed(this_case.boundary_family);
    }

    constexpr float sphere_radius = 0.045f;
    constexpr float sphere_density = 1800.0f;
    constexpr float sphere_volume =
        4.0f / 3.0f * 3.14159265358979323846f * sphere_radius * sphere_radius * sphere_radius;
    const float sphere_mass = sphere_density * sphere_volume;
    auto sphere_type = DEMSim.LoadSphereType(sphere_mass, sphere_radius, mat_sphere);
    const std::vector<SphereSeed> seeds = make_sphere_seeds(cases, sphere_radius);

    std::vector<float3> sphere_positions;
    std::vector<float3> sphere_velocities;
    std::vector<unsigned int> sphere_families;
    for (const auto& seed : seeds) {
        sphere_positions.push_back(seed.position);
        sphere_velocities.push_back(seed.velocity);
        sphere_families.push_back(cases.at(seed.case_id).sphere_family);
    }
    auto spheres = DEMSim.AddClumps(sphere_type, sphere_positions);
    spheres->SetFamilies(sphere_families);
    spheres->SetVel(sphere_velocities);

    for (const auto& sphere_case : cases) {
        for (const auto& boundary_case : cases) {
            if (sphere_case.boundary_family != boundary_case.boundary_family) {
                DEMSim.DisableContactBetweenFamilies(sphere_case.sphere_family, boundary_case.boundary_family);
            }
            if (sphere_case.sphere_family != boundary_case.sphere_family) {
                DEMSim.DisableContactBetweenFamilies(sphere_case.sphere_family, boundary_case.sphere_family);
            }
        }
    }

    DEMSim.InstructBoxDomainDimension({-3.35f, 3.35f}, {-0.95f, 0.95f}, {-0.20f, 2.20f});
    DEMSim.InstructBoxDomainBoundingBC("none", mat_wall);
    DEMSim.SetTimeStepSize(2.0e-5);
    DEMSim.SetCDUpdateFreq(10);
    DEMSim.SetGravitationalAcceleration(make_float3(0.0f, 0.0f, -9.81f));
    DEMSim.SetMaxVelocity(80.0f);
    DEMSim.SetErrorOutVelocity(100.0f);

    write_pour_case_metadata(out_dir / "pour_cases.csv", cases);
    write_pour_sphere_metadata(out_dir / "pour_spheres.csv", seeds);

    DEMSim.Initialize();

    std::ofstream frame_times(dynamic_dir / "frame_times.csv");
    require_condition(static_cast<bool>(frame_times), "Unable to open dynamic frame time file.");
    frame_times << "frame,time\n";

    constexpr int dynamic_frames = 250;
    constexpr double dynamic_frame_time = 2.0e-2;
    for (int frame = 0; frame < dynamic_frames; frame++) {
        frame_times << frame << "," << DEMSim.GetSimTime() << "\n";
        write_dynamic_frame_outputs(DEMSim, dynamic_dir, frame);
        DEMSim.DoDynamicsThenSync(dynamic_frame_time);
        if (frame % 30 == 0) {
            std::cout << "Pour frame " << frame << " / " << dynamic_frames << std::endl;
        }
    }

    std::cout << "Analytical cone pouring validation wrote " << dynamic_frames << " frames for " << seeds.size()
              << " spheres in " << out_dir << std::endl;
    return 0;
}
