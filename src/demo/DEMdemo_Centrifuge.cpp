//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// This demo features an analytical boundary-represented fast rotating container
// with particles of various shapes pulled into it. Different types of particles
// are marked with different family numbers (identification numbers) for easier
// visualizations.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <chrono>
#include <filesystem>

using namespace deme;
using namespace std::filesystem;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    // Output family numbers (used to identify the centrifuging effect)
    DEMSim.SetOutputContent(OUTPUT_CONTENT::FAMILY);
    // DEMSim.SetVerbosity(STEP_METRIC);

    // If you don't need individual force information, then this option makes the solver run a bit faster.
    DEMSim.SetNoForceRecord();

    // What will be loaded from the file, is a template for ellipsoid with b = c = 1 and a = 2, where Z is the long axis
    DEMClumpTemplate ellipsoid;
    ellipsoid.ReadComponentFromFile((GET_DATA_PATH() / "clumps/ellipsoid_2_1_1.csv").string());
    // Calculate its mass and MOI
    float mass = 2.6e3 * 4. / 3. * PI * 2 * 1 * 1;
    float3 MOI = make_float3(1. / 5. * mass * (1 * 1 + 2 * 2), 1. / 5. * mass * (1 * 1 + 2 * 2),
                             1. / 5. * mass * (1 * 1 + 1 * 1));
    // We can scale this general template to make it smaller, like a DEM particle that you would actually use
    float scaling = 0.01;
    // Scale the template we just created
    mass *= scaling * scaling * scaling;
    MOI *= scaling * scaling * scaling * scaling * scaling;
    ellipsoid.mass = mass;
    ellipsoid.MOI = MOI;
    std::for_each(ellipsoid.radii.begin(), ellipsoid.radii.end(), [scaling](float& r) { r *= scaling; });
    std::for_each(ellipsoid.relPos.begin(), ellipsoid.relPos.end(), [scaling](float3& r) { r *= scaling; });

    auto mat_type_sand = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.01}});
    auto mat_type_drum = DEMSim.LoadMaterial({{"E", 2e9}, {"nu", 0.3}, {"CoR", 0.6}, {"mu", 0.5}, {"Crr", 0.01}});
    // Since two types of materials have the same mu, this following call does not change the default mu for their
    // interaction, it's still 0.5.
    DEMSim.SetMaterialPropertyPair("mu", mat_type_sand, mat_type_drum, 0.5);

    // Define material type for the particles (on a per-sphere-component basis)
    ellipsoid.materials = std::vector<std::shared_ptr<DEMMaterial>>(ellipsoid.nComp, mat_type_sand);

    // Create some random clump templates for the filling materials
    // An array to store these generated clump templates
    std::vector<std::shared_ptr<DEMClumpTemplate>> clump_types;
    // Then randomly create some clump templates for filling the drum
    for (int i = 0; i < 3; i++) {
        // A multiplier is added to the masses of different clumps, so that centrifuging separate those types. Consider
        // it separating materials with different densities.
        float mult = std::pow(1.5, i);
        // Then make a new copy of the template then do the scaling of mass
        DEMClumpTemplate ellipsoid_template = ellipsoid;
        ellipsoid_template.mass *= mult;
        ellipsoid_template.MOI *= mult;

        // Load a (ellipsoid-shaped) clump and a sphere
        clump_types.push_back(DEMSim.LoadClumpType(ellipsoid_template));
        clump_types.push_back(DEMSim.LoadSphereType(ellipsoid_template.mass, std::cbrt(2.0) * scaling, mat_type_sand));
    }

    // Add the centrifuge
    float3 CylCenter = make_float3(0, 0, 0);
    float3 CylAxis = make_float3(0, 0, 1);
    float CylRad = 2.0;
    float CylHeight = 1.0;
    float CylMass = 1.0;
    float safe_delta = 0.03;
    float IZZ = CylMass * CylRad * CylRad / 2;
    float IYY = (CylMass / 12) * (3 * CylRad * CylRad + CylHeight * CylHeight);
    auto Drum = DEMSim.AddExternalObject();
    Drum->AddCylinder(CylCenter, CylAxis, CylRad, mat_type_drum, 0);
    Drum->SetMass(CylMass);
    Drum->SetMOI(make_float3(IYY, IYY, IZZ));
    auto Drum_tracker = DEMSim.Track(Drum);
    // Drum is family 100
    unsigned int drum_family = 100;
    Drum->SetFamily(drum_family);
    // The drum rotates (facing Z direction)
    DEMSim.SetFamilyPrescribedAngVel(drum_family, "0", "0", "6.0");
    // Then add planes to `close up' the drum. We add it as another object b/c we want to track the force on it
    // separately.
    auto top_bot_planes = DEMSim.AddExternalObject();
    top_bot_planes->AddPlane(make_float3(0, 0, CylHeight / 2. - safe_delta), make_float3(0, 0, -1), mat_type_drum);
    top_bot_planes->AddPlane(make_float3(0, 0, -CylHeight / 2. + safe_delta), make_float3(0, 0, 1), mat_type_drum);
    // Planes should rotate together with the drum wall.
    top_bot_planes->SetFamily(drum_family);
    auto planes_tracker = DEMSim.Track(top_bot_planes);

    // Then sample some particles inside the drum
    std::vector<std::shared_ptr<DEMClumpTemplate>> input_template_type;
    std::vector<float3> input_xyz;
    std::vector<unsigned int> family_code;
    float3 sample_center = make_float3(0, 0, 0);
    float sample_halfheight = CylHeight / 2.0 - 3.0 * safe_delta;
    float sample_halfwidth = CylRad / 1.5;
    auto input_material_xyz =
        DEMBoxGridSampler(sample_center, make_float3(sample_halfwidth, sample_halfwidth, sample_halfheight),
                          scaling * std::cbrt(2.0) * 2.1, scaling * std::cbrt(2.0) * 2.1, scaling * 2 * 2.1);
    input_xyz.insert(input_xyz.end(), input_material_xyz.begin(), input_material_xyz.end());
    unsigned int num_clumps = input_material_xyz.size();
    // Casually select from generated clump types
    for (unsigned int i = 0; i < num_clumps; i++) {
        input_template_type.push_back(clump_types.at(i % clump_types.size()));
        // Every clump type that has a unique mass, gets a unique family number
        family_code.push_back((i % clump_types.size()) / 2);
    }
    auto particles = DEMSim.AddClumps(input_template_type, input_xyz);
    particles->SetFamilies(family_code);

    // Keep tab of the max velocity in simulation
    auto max_v_finder = DEMSim.CreateInspector("clump_max_absv");
    float max_v;

    // Make the domain large enough
    DEMSim.InstructBoxDomainDimension(5, 5, 5);
    float step_size = 5e-6;
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.81));
    DEMSim.SetExpandSafetyType("auto");
    // If there is a velocity that an analytical object (i.e. the drum) has that you'd like the solver to take into
    // account in consideration of adding contact margins, you have to specify it here, since the solver's automatic max
    // velocity derivation algorithm currently cannot take analytical object's angular velocity-induced velocity into
    // account.
    DEMSim.SetExpandSafetyAdder(6.0);
    DEMSim.Initialize();

    path out_dir = current_path();
    out_dir /= "DemoOutput_Centrifuge";
    create_directory(out_dir);

    float time_end = 20.0;
    unsigned int fps = 20;
    unsigned int out_steps = (unsigned int)(1.0 / (fps * step_size));

    std::cout << "Output at " << fps << " FPS" << std::endl;
    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (double t = 0; t < (double)time_end; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            std::cout << "Frame: " << currframe << std::endl;
            DEMSim.ShowThreadCollaborationStats();
            char filename[100];
            sprintf(filename, "DEMdemo_output_%04d.csv", currframe);
            DEMSim.WriteSphereFile(out_dir / filename);
            currframe++;
            max_v = max_v_finder->GetValue();
            std::cout << "Max velocity of any point in simulation is " << max_v << std::endl;

            // Torque on the side walls are?
            float3 drum_moi = Drum_tracker->MOI();
            float3 drum_acc = Drum_tracker->ContactAngAccLocal();
            float3 drum_torque = drum_acc * drum_moi;
            std::cout << "Contact torque on the side walls is " << drum_torque.x << ", " << drum_torque.y << ", "
                      << drum_torque.z << std::endl;

            // The force on the bottom plane?
            float3 force_on_BC = planes_tracker->ContactAcc() * planes_tracker->Mass();
            std::cout << "Contact force on bottom plane is " << force_on_BC.z << std::endl;
        }

        DEMSim.DoDynamics(step_size);
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

    std::cout << "DEMdemo_Centrifuge exiting..." << std::endl;
    return 0;
}
