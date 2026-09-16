//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// This small modular test exercises analytical cone contacts without using
// triangle meshes. It keeps all contacts in one DEM solver initialization:
//   - inward cone side contact,
//   - outward cone side contact,
//   - apex contact for the single-nappe cone tip,
//   - a no-contact control.
// =============================================================================

#include <DEM/API.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace deme;

namespace {

struct ConeValidationCase {
    std::string name;
    std::string title;
    float3 sphere_position;
    float3 cone_tip;
    float3 cone_axis;
    float slope;
    objNormal_t normal;
    unsigned int sphere_family;
    unsigned int cone_family;
    int expected_force_x_sign;
    int expected_force_z_sign;
    bool expected_active_contact;
};

float length3(const float3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

void require_condition(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_force_direction(const std::map<bodyID_t, float3>& forces,
                             bodyID_t owner,
                             const std::string& case_name,
                             int x_sign,
                             int z_sign) {
    auto search = forces.find(owner);
    require_condition(search != forces.end(), case_name + " did not produce an active contact.");

    const float3 force = search->second;
    require_condition(length3(force) > 0.0f, case_name + " contact force is zero.");
    if (x_sign < 0) {
        require_condition(force.x < 0.0f, case_name + " force.x should be negative.");
    } else if (x_sign > 0) {
        require_condition(force.x > 0.0f, case_name + " force.x should be positive.");
    } else {
        require_condition(std::fabs(force.x) < 1.0e-3f * length3(force), case_name + " force.x should be near zero.");
    }

    if (z_sign < 0) {
        require_condition(force.z < 0.0f, case_name + " force.z should be negative.");
    } else if (z_sign > 0) {
        require_condition(force.z > 0.0f, case_name + " force.z should be positive.");
    } else {
        require_condition(std::fabs(force.z) < 1.0e-3f * length3(force), case_name + " force.z should be near zero.");
    }
}

const char* normal_mode_name(objNormal_t normal) {
    return (normal == ENTITY_NORMAL_OUTWARD) ? "outward" : "inward";
}

void write_validation_case_metadata(const std::filesystem::path& out_file,
                                    const std::vector<ConeValidationCase>& cases,
                                    float sphere_radius) {
    std::ofstream metadata(out_file);
    require_condition(static_cast<bool>(metadata), "Unable to open validation case metadata file.");

    metadata << "case_id,name,title,sphere_id,sphere_family,cone_family,sphere_radius,"
             << "sphere_initial_x,sphere_initial_y,sphere_initial_z,"
             << "tip_x,tip_y,tip_z,axis_x,axis_y,axis_z,slope,normal_mode,"
             << "expected_active_contact,expected_force_x_sign,expected_force_z_sign,display_height\n";
    for (size_t i = 0; i < cases.size(); i++) {
        const auto& this_case = cases.at(i);
        // The display height only controls the ParaView inspection shell.  The
        // DEM object itself remains the semi-infinite analytical AddCone used
        // below, so this metadata does not introduce a mesh into the test.
        constexpr float display_height = 2.25f;
        metadata << i << "," << this_case.name << "," << this_case.title << "," << i << "," << this_case.sphere_family
                 << "," << this_case.cone_family << "," << sphere_radius << "," << this_case.sphere_position.x << ","
                 << this_case.sphere_position.y << "," << this_case.sphere_position.z << "," << this_case.cone_tip.x
                 << "," << this_case.cone_tip.y << "," << this_case.cone_tip.z << "," << this_case.cone_axis.x << ","
                 << this_case.cone_axis.y << "," << this_case.cone_axis.z << "," << this_case.slope << ","
                 << normal_mode_name(this_case.normal) << "," << (this_case.expected_active_contact ? 1 : 0) << ","
                 << this_case.expected_force_x_sign << "," << this_case.expected_force_z_sign << "," << display_height
                 << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path out_dir =
        (argc > 1) ? std::filesystem::path(argv[1]) : std::filesystem::path("/tmp/deme_cone_validation");
    std::filesystem::create_directories(out_dir);

    DEMSolver DEMSim(1);
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    DEMSim.SetContactOutputContent({"OWNER", "GEO_ID", "FORCE", "POINT", "NORMAL"});
    DEMSim.UseFrictionlessHertzianModel();

    auto mat_sphere = DEMSim.LoadMaterial({{"E", 1.0e7f}, {"nu", 0.3f}, {"CoR", 0.1f}, {"mu", 0.0f}, {"Crr", 0.0f}});
    auto mat_cone = DEMSim.LoadMaterial({{"E", 1.0e7f}, {"nu", 0.3f}, {"CoR", 0.1f}, {"mu", 0.0f}, {"Crr", 0.0f}});

    constexpr float sphere_radius = 0.2f;
    auto sphere_type = DEMSim.LoadSphereType(1.0f, sphere_radius, mat_sphere);

    const std::vector<ConeValidationCase> cases = {
        {"inward_side", "Inward side contact", make_float3(0.9f, 0.0f, 1.0f), make_float3(0.0f, 0.0f, 0.0f),
         make_float3(0.0f, 0.0f, 1.0f), 1.0f, ENTITY_NORMAL_INWARD, 10, 20, -1, 1, true},
        {"outward_side", "Outward side contact", make_float3(5.1f, 0.0f, 1.0f), make_float3(4.0f, 0.0f, 0.0f),
         make_float3(0.0f, 0.0f, 1.0f), 1.0f, ENTITY_NORMAL_OUTWARD, 11, 21, 1, -1, true},
        {"apex", "Apex contact", make_float3(8.0f, 0.0f, -0.05f), make_float3(8.0f, 0.0f, 0.0f),
         make_float3(0.0f, 0.0f, 1.0f), 1.0f, ENTITY_NORMAL_INWARD, 12, 22, 0, -1, true},
        {"no_contact", "No-contact control", make_float3(12.5f, 0.0f, 1.0f), make_float3(12.0f, 0.0f, 0.0f),
         make_float3(0.0f, 0.0f, 1.0f), 1.0f, ENTITY_NORMAL_INWARD, 13, 23, 0, 0, false}};

    std::vector<float3> sphere_positions;
    std::vector<unsigned int> sphere_families;
    for (const auto& this_case : cases) {
        sphere_positions.push_back(this_case.sphere_position);
        sphere_families.push_back(this_case.sphere_family);
    }
    auto spheres = DEMSim.AddClumps(sphere_type, sphere_positions);
    spheres->SetFamilies(sphere_families);

    for (const auto& this_case : cases) {
        auto cone = DEMSim.AddExternalObject();
        cone->AddCone(this_case.cone_tip, this_case.cone_axis, this_case.slope, mat_cone, this_case.normal);
        cone->SetFamily(this_case.cone_family);
    }

    for (const auto& sphere_case : cases) {
        for (const auto& cone_case : cases) {
            if (sphere_case.cone_family != cone_case.cone_family) {
                DEMSim.DisableContactBetweenFamilies(sphere_case.sphere_family, cone_case.cone_family);
            }
        }
    }

    for (const auto& this_case : cases) {
        DEMSim.SetFamilyFixed(this_case.cone_family);
    }
    DEMSim.InstructBoxDomainDimension({-1.5f, 14.0f}, {-1.5f, 1.5f}, {-1.0f, 2.5f});
    DEMSim.InstructBoxDomainBoundingBC("none", mat_cone);
    DEMSim.SetTimeStepSize(1.0e-6);
    DEMSim.SetCDUpdateFreq(1);
    DEMSim.SetGravitationalAcceleration(make_float3(0.0f));
    DEMSim.SetMaxVelocity(1.0f);

    DEMSim.Initialize();
    DEMSim.DoDynamicsThenSync(1.0e-6);

    auto potential_contacts = DEMSim.GetContactDetailedInfo(-1.0f);
    DEMSim.WriteSphereFile(out_dir / "spheres.csv");
    DEMSim.WriteContactFile((out_dir / "contacts_potential.csv").string(), -1.0f);
    DEMSim.WriteContactFile((out_dir / "contacts_active.csv").string(), DEME_TINY_FLOAT);
    write_validation_case_metadata(out_dir / "validation_cases.csv", cases, sphere_radius);

    std::cout << "Potential analytical contact entries: " << potential_contacts->Size() << std::endl;

    auto contacts = DEMSim.GetContactDetailedInfo(DEME_TINY_FLOAT);
    auto& owners = contacts->GetAOwner();
    auto& forces = contacts->GetForce();
    require_condition(owners.size() == forces.size(), "Contact owner/force arrays have inconsistent lengths.");
    std::cout << "Active analytical contact entries: " << forces.size() << std::endl;
    require_condition(forces.size() == 3, "Expected exactly three active analytical cone contacts.");

    std::map<bodyID_t, float3> force_by_owner;
    for (size_t i = 0; i < forces.size(); i++) {
        force_by_owner[owners[i]] = forces[i];
        std::cout << "owner=" << owners[i] << " force=(" << forces[i].x << ", " << forces[i].y << ", " << forces[i].z
                  << ") normal=(" << contacts->GetNormal()[i].x << ", " << contacts->GetNormal()[i].y << ", "
                  << contacts->GetNormal()[i].z << ") type=" << contacts->GetContactType()[i] << std::endl;
    }

    require_force_direction(force_by_owner, 0, "inward cone side", -1, 1);
    require_force_direction(force_by_owner, 1, "outward cone side", 1, -1);
    require_force_direction(force_by_owner, 2, "cone apex", 0, -1);
    require_condition(force_by_owner.find(3) == force_by_owner.end(), "no-contact control produced an active contact.");

    std::cout << "Analytical cone GPU validation passed. Output directory: " << out_dir << std::endl;
    return 0;
}
