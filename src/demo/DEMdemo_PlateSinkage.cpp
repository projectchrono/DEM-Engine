//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A plate pressure-sinkage test, the classical bevameter experiment: a rigid
// circular plate is pressed at constant speed into a granular bed and the
// pressure it carries is recorded against how deep it has sunk. The output is
// the pressure-sinkage curve, which is how terramechanics characterizes a soil.
//
// The demo runs in two stages, and the split is the part worth copying:
//
//   Stage 1 prepares the bed cohesionless (pour, settle) and saves it with
//   WriteClumpFile. Bed preparation is typically around half the cost of a
//   run, and a saved bed can be reused by every later test that wants the
//   same packing.
//
//   Stage 2 rebuilds a fresh solver, reloads the bed (positions AND
//   orientations: these are clumped grains, and restoring centers without
//   rotations silently changes the interlocking), switches to a force model
//   WITH cohesion, lets the bed re-equilibrate, and only then presses the
//   plate. Preparing cohesionless and loading cohesive is deliberate: if
//   cohesion acts while the bed is being prepared, it props the packing open
//   and you get a looser, weaker bed than the density you asked for.
//
// The cohesion model is the stock ForceModelWithCohesion.cu that ships with
// DEME: a constant attractive acceleration between contacting grains, set as
// the pairwise material property "Cohesion".
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

using namespace deme;
using namespace std::filesystem;

const double math_PI = 3.14159;

int main() {
    // ---------------------------------------------------------------------
    // Everything a user might want to change, in one place.
    // ---------------------------------------------------------------------
    const float plate_diam = 0.35;      // plate diameter, m
    const float plate_thickness = 0.02;
    const float plate_speed = 0.05;     // press speed, m/s; slow it down for quasi-static work
    const float target_sinkage = 0.06;  // stop once the plate is this deep, m

    const double bin_diameter = 1.05;  // soil bin, 3 plate diameters across
    const float bottom = -0.5;         // bin floor, in world coordinates
    const float fill_height = 0.55;    // loose pour column; settles to roughly 0.28 m of bed

    const float terrain_density = 2.6e3;  // grain material density, kg/m^3
    const float cohesion = 200.;          // stage-2 grain-grain cohesion, m/s^2 (an acceleration:
                                          // the model applies force = Cohesion * effective mass)
    const float step_size = 1e-5;
    const double clump_scale = 0.01;  // scales the stock 3-sphere template to ~22 mm grains,
                                      // about 16 grain diameters under the plate

    path out_dir = current_path();
    out_dir /= "DemoOutput_PlateSinkage";
    create_directory(out_dir);
    const std::string bed_file = (out_dir / "bed.csv").string();
    const std::string curve_file = (out_dir / "pressure_sinkage.csv").string();
    const std::string template_name = "bed_grain";

    float bed_surface = 0.;  // measured at the end of stage 1, reused in stage 2

    // ---------------------------------------------------------------------
    // Stage 1: prepare the bed cohesionless and save it.
    // ---------------------------------------------------------------------
    {
        DEMSolver DEMSim;
        DEMSim.SetVerbosity(INFO);
        DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
        // QUAT matters: the saved file must carry orientations or the reload
        // in stage 2 cannot restore the packing's interlocking.
        DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ | OUTPUT_CONTENT::QUAT | OUTPUT_CONTENT::VEL);

        auto mat_type_terrain = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.4}, {"mu", 0.5}, {"Crr", 0.01}});

        const double world_size = 2;
        DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size);
        DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);
        // A cylindrical bin, so the circular plate sees the same wall distance everywhere.
        auto walls = DEMSim.AddExternalObject();
        walls->AddCylinder(make_float3(0), make_float3(0, 0, 1), bin_diameter / 2., mat_type_terrain, 0);
        walls->AddPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);

        // The stock 3-sphere clump template, same numbers as DEMdemo_ConePenetration.
        double clump_vol = 5.5886717;
        float mass = terrain_density * clump_vol;
        float3 MOI = make_float3(2.928, 2.6029, 3.9908) * terrain_density;
        std::shared_ptr<DEMClumpTemplate> my_template =
            DEMSim.LoadClumpType(mass, MOI, GetDEMEDataFile("clumps/3_clump.csv"), mat_type_terrain);
        my_template->SetVolume(clump_vol);
        my_template->Scale(clump_scale);
        // The restart readers in stage 2 look clumps up by this name.
        my_template->AssignName(template_name);

        // Loose-pour the fill column, then let it settle.
        HCPSampler sampler(clump_scale * 3.);
        float3 fill_center = make_float3(0, 0, bottom + 0.03 + fill_height / 2);
        auto input_xyz = sampler.SampleCylinderZ(fill_center, bin_diameter / 2. - clump_scale * 2., fill_height / 2);
        DEMSim.AddClumps(my_template, input_xyz);
        std::cout << "Stage 1: poured " << input_xyz.size() << " grains" << std::endl;

        auto max_z_finder = DEMSim.CreateInspector("clump_max_z");
        auto total_mass_finder = DEMSim.CreateInspector("clump_mass");

        DEMSim.SetInitTimeStep(step_size);
        DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
        DEMSim.Initialize();

        DEMSim.DoDynamicsThenSync(0.8);

        bed_surface = max_z_finder->GetValue();
        float bed_depth = bed_surface - bottom;
        float bulk_density = total_mass_finder->GetValue() / (math_PI * bin_diameter * bin_diameter / 4. * bed_depth);
        std::cout << "Stage 1: settled bed is " << bed_depth << " m deep, bulk density " << bulk_density << " kg/m^3"
                  << std::endl;

        DEMSim.WriteClumpFile(bed_file);
        std::cout << "Stage 1: bed saved to " << bed_file << std::endl;
    }

    // ---------------------------------------------------------------------
    // Stage 2: reload the bed, switch cohesion on, press the plate.
    // ---------------------------------------------------------------------
    {
        DEMSolver DEMSim;
        DEMSim.SetVerbosity(INFO);
        DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
        DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ | OUTPUT_CONTENT::ABSV);

        // The stock cohesion force model. "Cohesion" is a pairwise material
        // property; grain-grain gets the real value, plate-grain gets zero so
        // the plate face does not adhere (set it nonzero for a sticky soil).
        auto my_force_model = DEMSim.ReadContactForceModel("ForceModelWithCohesion.cu");
        my_force_model->SetMustHaveMatProp({"E", "nu", "CoR", "mu", "Crr", "Cohesion"});
        my_force_model->SetMustPairwiseMatProp({"CoR", "mu", "Crr", "Cohesion"});
        my_force_model->SetPerContactWildcards({"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z"});

        auto mat_type_terrain = DEMSim.LoadMaterial(
            {{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.4}, {"mu", 0.5}, {"Crr", 0.01}, {"Cohesion", cohesion}});
        auto mat_type_plate =
            DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.4}, {"mu", 0.5}, {"Crr", 0.01}, {"Cohesion", 0.}});
        DEMSim.SetMaterialPropertyPair("Cohesion", mat_type_plate, mat_type_terrain, 0.);

        const double world_size = 2;
        DEMSim.InstructBoxDomainDimension(world_size, world_size, world_size);
        DEMSim.InstructBoxDomainBoundingBC("top_open", mat_type_terrain);
        auto walls = DEMSim.AddExternalObject();
        walls->AddCylinder(make_float3(0), make_float3(0, 0, 1), bin_diameter / 2., mat_type_terrain, 0);
        walls->AddPlane(make_float3(0, 0, bottom), make_float3(0, 0, 1), mat_type_terrain);

        // Same template, same name, so the saved clumps map back onto it.
        double clump_vol = 5.5886717;
        float mass = terrain_density * clump_vol;
        float3 MOI = make_float3(2.928, 2.6029, 3.9908) * terrain_density;
        std::shared_ptr<DEMClumpTemplate> my_template =
            DEMSim.LoadClumpType(mass, MOI, GetDEMEDataFile("clumps/3_clump.csv"), mat_type_terrain);
        my_template->SetVolume(clump_vol);
        my_template->Scale(clump_scale);
        my_template->AssignName(template_name);

        // Reload the saved bed: positions AND orientations. The readers are
        // keyed by clump-type name and return an empty vector for an unknown
        // key, so guard against that rather than pressing a plate into vacuum.
        auto xyz_map = DEMSim.ReadClumpXyzFromCsv(bed_file);
        auto quat_map = DEMSim.ReadClumpQuatFromCsv(bed_file);
        auto& in_xyz = xyz_map[template_name];
        auto& in_quat = quat_map[template_name];
        if (in_xyz.empty() || in_quat.size() != in_xyz.size()) {
            std::cerr << "Reloading " << bed_file << " failed: got " << in_xyz.size() << " positions and "
                      << in_quat.size() << " orientations for clump type '" << template_name << "'." << std::endl;
            return 1;
        }
        DEMClumpBatch bed_batch(in_xyz.size());
        bed_batch.SetTypes(std::vector<std::shared_ptr<DEMClumpTemplate>>(in_xyz.size(), my_template));
        bed_batch.SetPos(in_xyz);
        bed_batch.SetOriQ(in_quat);
        bed_batch.SetFamily(0);
        DEMSim.AddClumps(bed_batch);
        std::cout << "Stage 2: reloaded " << in_xyz.size() << " grains from " << bed_file << std::endl;

        // The plate: the stock unit cylinder mesh (radius 1, height 2),
        // scaled into a disc. Parked just above the stage-1 bed surface.
        auto plate = DEMSim.AddWavefrontMeshObject(GetDEMEDataFile("mesh/cyl_r1_h2.obj"), mat_type_plate);
        plate->Scale(make_float3(plate_diam / 2., plate_diam / 2., plate_thickness / 2.));
        float plate_mass = 7.8e3 * math_PI * (plate_diam / 2.) * (plate_diam / 2.) * plate_thickness;
        plate->SetMass(plate_mass);
        plate->SetMOI(make_float3(plate_mass * (3 * plate_diam * plate_diam / 4. + plate_thickness * plate_thickness) / 12.,
                                  plate_mass * (3 * plate_diam * plate_diam / 4. + plate_thickness * plate_thickness) / 12.,
                                  plate_mass * plate_diam * plate_diam / 8.));
        float park_z = bed_surface + plate_thickness / 2. + 0.01;
        plate->SetInitPos(make_float3(0, 0, park_z));
        plate->SetFamily(2);
        auto plate_tracker = DEMSim.Track(plate);

        // Family 2 holds still while the bed re-equilibrates under cohesion;
        // family 1 descends at the press speed. The switch starts the test.
        DEMSim.SetFamilyFixed(2);
        DEMSim.SetFamilyPrescribedLinVel(1, "0", "0", "-" + to_string_with_precision(plate_speed));

        auto max_z_finder = DEMSim.CreateInspector("clump_max_z");

        DEMSim.SetInitTimeStep(step_size);
        DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
        DEMSim.Initialize();

        // Cohesion just switched on and tangential contact history is not
        // carried across the two stages, so give the bed a moment to grip.
        DEMSim.DoDynamicsThenSync(0.3);
        bed_surface = max_z_finder->GetValue();
        std::cout << "Stage 2: bed re-equilibrated, surface at z = " << bed_surface << std::endl;

        DEMSim.ChangeFamily(2, 1);

        // March down, recording pressure against sinkage. The sinkage datum
        // is first contact, taken as the depth where the mean plate pressure
        // first reaches 1 kPa; before that the plate is falling through air.
        const float plate_area = math_PI * (plate_diam / 2.) * (plate_diam / 2.);
        const float contact_threshold = 1e3;  // Pa
        const float sample_every = 2e-3;      // s
        std::ofstream curve(curve_file);
        curve << "sinkage_m,pressure_Pa" << std::endl;
        double contact_z = 0.;
        bool in_contact = false;
        float next_print = 0.;
        std::cout << "Stage 2: pressing at " << plate_speed << " m/s" << std::endl;
        std::cout << "  sinkage (mm)   pressure (kPa)" << std::endl;

        while (true) {
            DEMSim.DoDynamics(sample_every);
            float3 plate_pos = plate_tracker->Pos();
            // ContactAcc returns the contact acceleration on the owner, so
            // multiply by the mass we assigned to get the contact force.
            float3 plate_force = plate_tracker->ContactAcc() * plate_mass;
            float pressure = plate_force.z / plate_area;

            if (!in_contact) {
                if (pressure >= contact_threshold) {
                    in_contact = true;
                    contact_z = plate_pos.z;
                }
                continue;
            }
            float sinkage = contact_z - plate_pos.z;
            curve << sinkage << "," << pressure << std::endl;
            if (sinkage * 1000. >= next_print) {
                std::printf("      %6.1f      %9.2f\n", sinkage * 1000., pressure / 1000.);
                next_print += 5.;
            }
            if (sinkage >= target_sinkage)
                break;
        }
        DEMSim.DoDynamicsThenSync(0.);
        std::cout << "Stage 2: curve written to " << curve_file << std::endl;
        DEMSim.ShowTimingStats();
    }

    std::cout << "DEMdemo_PlateSinkage exiting..." << std::endl;
    return 0;
}
