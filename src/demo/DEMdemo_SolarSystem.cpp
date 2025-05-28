//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// The simulation of the solar system, all the planets and the sun. The length
// unit here is AU and time unit day, and the celestial bodies' size is
// not to the scale; others are to the scale and in SI.
// NOTE: The important thing about this demo is that its custom force model
// computes a non-local gravitational force. You can make it take effect even
// when bodies are not in contact like in this demo. But keep in mind, when you
// use this method in your simulation for adding something like electrostatic
// force, you should use SetFamilyExtraMargin with a relatively small value,
// since you do not want to keep element interaction still relatively local,
// or you know, having n^2 contact pairs can be problematic for any simulations
// with serious sizes. Also in most cases, you just need to use geometry
// wildcards (AddGeometryWildcard) to associate extra properties to be used
// in the force model. This is because DEME resolves contact pairs based on
// geometry entities, such as spheres and triangles, so you want to attach force
// model-related quatities directly to them, rather than at clump level.
// =============================================================================

#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <random>

using namespace deme;

const double math_PI = 3.14159;
// In AU and day
const double G = 6.674e-11 * 86400 * 86400 / 1.496e+11 / 1.496e+11 / 1.496e+11;

int main() {
    DEMSolver DEMSim;
    DEMSim.SetVerbosity(INFO);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::ABSV);
    DEMSim.EnsureKernelErrMsgLineNum();
    DEMSim.SetNoForceRecord();

    // Material for the planets, but they ultimately do not matter at all.
    auto mat_type = DEMSim.LoadMaterial({{"E", 10e9}, {"nu", 0.3}, {"CoR", 0.8}});
    // A custom force model that has GMm/R^2 force in it.
    auto my_force_model = DEMSim.ReadContactForceModel("ForceModelWithGravity.cu");
    // Those 2 following lines are needed.
    my_force_model->SetMustHaveMatProp({"E", "nu", "CoR"});
    my_force_model->SetMustPairwiseMatProp({"CoR"});
    // Gravitational pull will be calcuated using this extra quatity that we associate with each sphere.
    // Now, why don't we use the mass property in the clump templates? Since that is at clump level, and if
    // what you have here are, say 3-sphere clumps, rather than single sphere clumps, then the gravitational
    // pull may be double- or triple-counted. In the case of using 3-sphere clumps, maybe you should assign
    // each sphere say 1/3 of the total clump mass, then use this geometry wildcard to calculate the gravity.
    // In the force model script ForceModelWithGravity.cu, you can now use my_mass_A and my_mass_B to refer
    // to this wildcard and use it in force calculations.
    my_force_model->SetPerGeometryWildcards({"my_mass"});

    // For all the celestial bodies, define as templates. Their radii is just for visialization.
    float vis_size_scaler = 3;
    auto Sun_template = DEMSim.LoadSphereType(1.989e30, 0.1 * vis_size_scaler, mat_type);
    auto Mercury_template = DEMSim.LoadSphereType(3.3011e23, 0.02 * vis_size_scaler, mat_type);
    auto Venus_template = DEMSim.LoadSphereType(4.8675e24, 0.035 * vis_size_scaler, mat_type);
    float Earth_radius = 0.04 * vis_size_scaler;
    float Earth_mass = 5.9724e24;
    auto Earth_template = DEMSim.LoadSphereType(Earth_mass, Earth_radius, mat_type);
    auto Moon_template = DEMSim.LoadSphereType(7.342e22, 0.01 * vis_size_scaler, mat_type);
    auto Mars_template = DEMSim.LoadSphereType(6.4171e23, 0.03 * vis_size_scaler, mat_type);
    float Jupiter_radius = 0.08 * vis_size_scaler;
    float Jupiter_mass = 1.8982e27;
    auto Jupiter_template = DEMSim.LoadSphereType(Jupiter_mass, Jupiter_radius, mat_type);
    auto Io_template = DEMSim.LoadSphereType(8.9319e22, 0.004 * vis_size_scaler, mat_type);
    auto Europa_template = DEMSim.LoadSphereType(4.7998e22, 0.004 * vis_size_scaler, mat_type);
    auto Ganymede_template = DEMSim.LoadSphereType(1.4819e23, 0.004 * vis_size_scaler, mat_type);
    auto Callisto_template = DEMSim.LoadSphereType(1.0759e23, 0.004 * vis_size_scaler, mat_type);
    auto Saturn_template = DEMSim.LoadSphereType(5.6834e26, 0.07 * vis_size_scaler, mat_type);
    auto Uranus_template = DEMSim.LoadSphereType(8.6810e25, 0.055 * vis_size_scaler, mat_type);
    auto Neptune_template = DEMSim.LoadSphereType(1.0241e26, 0.06 * vis_size_scaler, mat_type);

    // Now load the bodies
    auto Sun = DEMSim.AddClumps(Sun_template, make_float3(0, 0, 0));
    // Sun is fixed
    Sun->SetFamily(1);
    DEMSim.SetFamilyFixed(1);
    Sun->AddGeometryWildcard("my_mass", 1.989e30);

    // The rest of the bodies: Defaulted to family 0, and they are free to move
    auto Mercury = DEMSim.AddClumps(Mercury_template, make_float3(0, 0.3871, 0));
    Mercury->SetVel(make_float3(47.87 / 1731, 0, 0));
    Mercury->AddGeometryWildcard("my_mass", 3.3011e23);

    auto Venus = DEMSim.AddClumps(Venus_template, make_float3(0, 0.7233, 0));
    Venus->SetVel(make_float3(35.02 / 1731, 0, 0));
    Venus->AddGeometryWildcard("my_mass", 4.8675e24);

    auto Earth = DEMSim.AddClumps(Earth_template, make_float3(0, 1, 0));
    Earth->SetVel(make_float3(29.78 / 1731, 0, 0));
    Earth->AddGeometryWildcard("my_mass", Earth_mass);

    auto Mars = DEMSim.AddClumps(Mars_template, make_float3(0, 1.5237, 0));
    Mars->SetVel(make_float3(24.08 / 1731, 0, 0));
    Mars->AddGeometryWildcard("my_mass", 6.4171e23);

    float Jupiter_orbit_rad = 5.2028;
    auto Jupiter = DEMSim.AddClumps(Jupiter_template, make_float3(0, Jupiter_orbit_rad, 0));
    Jupiter->SetVel(make_float3(13.07 / 1731, 0, 0));
    Jupiter->AddGeometryWildcard("my_mass", Jupiter_mass);

    auto Saturn = DEMSim.AddClumps(Saturn_template, make_float3(0, 9.5388, 0));
    Saturn->SetVel(make_float3(9.69 / 1731, 0, 0));
    Saturn->AddGeometryWildcard("my_mass", 5.6834e26);

    // auto Uranus = DEMSim.AddClumps(Uranus_template, make_float3(0,19.1914,0));
    // Uranus->SetVel(make_float3(6.81/1731,0,0));
    // Uranus->AddGeometryWildcard("my_mass", 8.6810e25);
    // auto Neptune = DEMSim.AddClumps(Neptune_template, make_float3(0,30.0611,0));
    // Neptune->SetVel(make_float3(5.43/1731,0,0));
    // Neptune->AddGeometryWildcard("my_mass", 1.0241e26);

    /*
    // The moons of Jupiter and of Earth (too close to the main planet, cannot reasonably render)
    auto Io = DEMSim.AddClumps(Io_template, make_float3(0,Jupiter_orbit_rad+Jupiter_radius+0.08,0));
    float Io_vel = 3.545*(Jupiter_radius+0.08);
    Io->SetVel(make_float3(Io_vel,0,0));
    Io->AddGeometryWildcard("my_mass", 8.9319e22);
    auto Europa = DEMSim.AddClumps(Europa_template, make_float3(0,Jupiter_orbit_rad+Jupiter_radius+0.16,0));
    float Europa_vel = 1.766*(Jupiter_radius+0.16);
    Europa->SetVel(make_float3(Europa_vel,0,0));
    Europa->AddGeometryWildcard("my_mass", 4.7998e22);
    auto Ganymede = DEMSim.AddClumps(Ganymede_template, make_float3(0,Jupiter_orbit_rad+Jupiter_radius+0.24,0));
    float Ganymede_vel = 0.835*(Jupiter_radius+0.24);
    Ganymede->SetVel(make_float3(Ganymede_vel,0,0));
    Ganymede->AddGeometryWildcard("my_mass", 1.4819e23);
    auto Callisto = DEMSim.AddClumps(Callisto_template, make_float3(0,Jupiter_orbit_rad+Jupiter_radius+0.3,0));
    float Callisto_vel = 0.558*(Jupiter_radius+0.3);
    Callisto->SetVel(make_float3(Callisto_vel,0,0));
    Callisto->AddGeometryWildcard("my_mass", 1.0759e23);
    auto Moon = DEMSim.AddClumps(Moon_template, make_float3(0,1.2,0));
    Moon->SetVel(make_float3(0.234*0.2,0,0)); // 0.234 rad/day * 0.2AU (0.2AU is large, but it's for visualization)
    Moon->AddGeometryWildcard("my_mass", 7.342e22);
    */

    float step_size = 0.001;
    DEMSim.SetInitTimeStep(step_size);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, 0));
    DEMSim.InstructBoxDomainDimension({-50, 50}, {-50, 50}, {-50, 50});
    DEMSim.SetInitBinNumTarget(10);

    // By adding a contact margin that is as large as the simulation domain, we ensure everything is in contact with
    // everything, so all the gravitational force pair is calculated. NOTE HOWEVER! In serious simulations where
    // hundreds of thousands of elements participate, you should be responsible about the size of this extra margin, and
    // it should be something like the same magnitude of element sizes, so contacts are still more or less local,
    // instead of that you have n^2 contacts and have no hope to resolve all.
    DEMSim.SetFamilyExtraMargin(0, 100.);
    DEMSim.SetFamilyExtraMargin(1, 100.);

    // If you know your extra margin policy gonna make the average number of contacts per sphere huge, then set this so
    // that the solver does not error out when checking it.
    DEMSim.SetErrorOutAvgContacts(200);

    DEMSim.Initialize();

    std::filesystem::path out_dir = std::filesystem::current_path();
    out_dir /= "DemoOutput_SolarSystem";
    std::filesystem::create_directory(out_dir);

    // Settle phase
    unsigned int currframe = 0;
    unsigned int curr_step = 0;
    unsigned int frame_per_day = 1;
    unsigned int out_steps = (unsigned int)(1.0 / (frame_per_day * step_size));

    for (double t = 0; t < 2000.; t += step_size, curr_step++) {
        if (curr_step % out_steps == 0) {
            char filename[100];
            sprintf(filename, "DEMdemo_output_%04d.csv", currframe);
            DEMSim.WriteSphereFile(out_dir / filename);
            std::cout << "Day " << currframe << "..." << std::endl;
            currframe++;
            // std::cout << "Num contacts: " << DEMSim.GetNumContacts() << std::endl;
        }

        DEMSim.DoDynamics(step_size);
    }

    DEMSim.ShowThreadCollaborationStats();
    std::cout << "SolarSystem demo exiting..." << std::endl;
    return 0;
}