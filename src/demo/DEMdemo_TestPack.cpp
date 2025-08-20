//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// A collection of DEME validation tests...
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <cstdio>
#include <time.h>
#include <filesystem>

using namespace deme;
using namespace std::filesystem;

inline bool is_near(float a, float b, float t = 1e-6) {
    if (std::abs(a - b) < t) {
        return true;
    }
    return false;
}

void SetSolverProp(DEMSolver& DEMSim) {
    DEMSim.SetVerbosity(DEBUG);
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);

    DEMSim.InstructBoxDomainDimension(25, 25, 10);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, -9.8));
    DEMSim.SetCDUpdateFreq(0);
    // Must disable this if you want to run dT and kT synchronizely, or the solver will automatically find a non-zero
    // update frequency that it sees fit to run it in an async fashion.
    DEMSim.UseAdaptiveUpdateFreq(false);

    // Testing non-jitified clump templates here in this demo...
    DEMSim.DisableJitifyClumpTemplates();
    DEMSim.DisableJitifyMassProperties();
    DEMSim.UseCubForceCollection();
}

void EllpsiodFallingOver() {
    DEMSolver DEMSim;
    SetSolverProp(DEMSim);
    // An ellipsoid a,b,c = 0.2,0.2,0.5, represented several sphere components
    std::vector<float> radii = {0.095, 0.136, 0.179, 0.204, 0.204, 0.179, 0.136, 0.095};
    std::vector<float3> relPos = {make_float3(0, 0, 0.4),    make_float3(0, 0, 0.342),  make_float3(0, 0, 0.228),
                                  make_float3(0, 0, 0.071),  make_float3(0, 0, -0.071), make_float3(0, 0, -0.228),
                                  make_float3(0, 0, -0.342), make_float3(0, 0, -0.4)};
    // Then calculate mass and MOI
    float mass = 5.0;
    // E, nu, CoR, mu, Crr
    auto mat_type_1 = DEMSim.LoadMaterial({{"E", 1e8}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.25}, {"Crr", 0.2}});
    float3 MOI = make_float3(1. / 5. * mass * (0.2 * 0.2 + 0.5 * 0.5), 1. / 5. * mass * (0.2 * 0.2 + 0.5 * 0.5),
                             1. / 5. * mass * (0.2 * 0.2 + 0.2 * 0.2));
    auto ellipsoid_template = DEMSim.LoadClumpType(mass, MOI, radii, relPos, mat_type_1);

    // Add the ground
    float3 normal_dir = make_float3(0, 0, 1);
    float3 tang_dir = make_float3(0, 1, 0);
    DEMSim.AddBCPlane(make_float3(0, 0, 0), normal_dir, mat_type_1);

    // Add an ellipsoid with init vel
    auto ellipsoid = DEMSim.AddClumps(ellipsoid_template, normal_dir * 0.5);
    ellipsoid->SetVel(tang_dir * 0.3);
    auto ellipsoid_tracker = DEMSim.Track(ellipsoid);

    DEMSim.SetInitTimeStep(1e-4);
    // DEMSim.SetIntegrator(TIME_INTEGRATOR::FORWARD_EULER);
    DEMSim.Initialize();

    float frame_time = 1e-1;
    path out_dir = current_path();
    out_dir /= "DemoOutput_TestPack";
    create_directory(out_dir);
    for (int i = 0; i < 6.0 / frame_time; i++) {
        char filename[100];
        sprintf(filename, "DEMdemo_output_%04d.csv", i);
        DEMSim.WriteSphereFile(out_dir / filename);
        std::cout << "Frame: " << i << std::endl;
        float4 oriQ = ellipsoid_tracker->OriQ();
        float3 angVel = ellipsoid_tracker->AngVelLocal();
        std::cout << "Time: " << frame_time * i << std::endl;
        std::cout << "Quaternion of the ellipsoid: " << oriQ.w << ", " << oriQ.x << ", " << oriQ.y << ", " << oriQ.z
                  << std::endl;
        std::cout << "Angular velocity of the ellipsoid: " << angVel.x << ", " << angVel.y << ", " << angVel.z
                  << std::endl;

        DEMSim.DoDynamics(frame_time);
    }
}

void SphereRollUpIncline() {
    // First, test the case when alpha = 35
    float sphere_rad = 0.2;
    float mass = 5.0;
    float mu = 0.25;
    {
        DEMSolver DEMSim;
        SetSolverProp(DEMSim);

        auto mat_type_1 = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", mu}, {"Crr", 0.15}});
        // A ball
        auto sphere_template = DEMSim.LoadSphereType(mass, sphere_rad, mat_type_1);

        // Incline angle
        float alpha = 35.;
        // Add the incline
        float3 normal_dir = make_float3(-std::sin(2. * PI * (alpha / 360.)), 0., std::cos(2. * PI * (alpha / 360.)));
        float3 tang_dir = make_float3(std::cos(2. * PI * (alpha / 360.)), 0., std::sin(2. * PI * (alpha / 360.)));
        DEMSim.AddBCPlane(make_float3(0, 0, 0), normal_dir, mat_type_1);

        // Add a ball rolling
        auto sphere = DEMSim.AddClumps(sphere_template, normal_dir * sphere_rad);
        sphere->SetVel(tang_dir * 0.5);
        auto sphere_tracker = DEMSim.Track(sphere);

        float step_time = 1e-4;
        DEMSim.SetInitTimeStep(step_time);
        DEMSim.Initialize();

        path out_dir = current_path();
        out_dir /= "DemoOutput_TestPack";
        create_directory(out_dir);
        for (int i = 0; i < 0.15 / step_time; i++) {
            char filename[100];
            sprintf(filename, "DEMdemo_output_%04d.csv", i);
            // if (i % 100 == 0) {
            //     DEMSim.WriteSphereFile(out_dir / filename);
            // }
            std::cout << "Frame: " << i << std::endl;
            float3 vel = sphere_tracker->Vel();
            float3 angVel = sphere_tracker->AngVelLocal();
            std::cout << "Time: " << step_time * i << std::endl;
            std::cout << "Velocity of the sphere: " << vel.x << ", " << vel.y << ", " << vel.z << std::endl;
            std::cout << "Angular velocity of the sphere: " << angVel.x << ", " << angVel.y << ", " << angVel.z
                      << std::endl;

            DEMSim.DoStepDynamics();
        }
    }

    std::cout << "======================================" << std::endl;
    // Then try to generate the plot (alpha = [1, 60], Crr = [0.0, 0.3])
    float run_time = 1.0;
    unsigned int i = 0;
    for (float alpha = 60; alpha >= 1; alpha -= 1) {
        for (float Crr = 0.0; Crr <= 0.3; Crr += 0.01) {
            DEMSolver DEMSim;
            SetSolverProp(DEMSim);
            DEMSim.SetVerbosity(QUIET);

            auto mat_type_1 = DEMSim.LoadMaterial({{"E", 1e9}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", mu}, {"Crr", Crr}});
            // A ball
            auto sphere_template = DEMSim.LoadSphereType(mass, sphere_rad, mat_type_1);

            // Add the incline
            float3 normal_dir =
                make_float3(-std::sin(2. * PI * (alpha / 360.)), 0., std::cos(2. * PI * (alpha / 360.)));
            float3 tang_dir = make_float3(std::cos(2. * PI * (alpha / 360.)), 0., std::sin(2. * PI * (alpha / 360.)));
            DEMSim.AddBCPlane(make_float3(0, 0, 0), normal_dir, mat_type_1);

            // Add a ball rolling
            auto sphere = DEMSim.AddClumps(sphere_template, normal_dir * sphere_rad);
            sphere->SetVel(tang_dir * 0.5);
            auto sphere_tracker = DEMSim.Track(sphere);

            float step_time = 1e-4;
            DEMSim.SetInitTimeStep(step_time);
            DEMSim.SetCDUpdateFreq(50);
            DEMSim.SetMaxVelocity(2.0);
            DEMSim.Initialize();

            DEMSim.DoDynamicsThenSync(run_time);
            float3 vel = sphere_tracker->Vel();
            float3 angVel = sphere_tracker->AngVelLocal();
            float vel_mag = length(vel);
            float angVel_mag = length(angVel);
            std::cout << "Angle of incline: " << alpha << std::endl;
            std::cout << "Rolling resistance: " << Crr << std::endl;
            std::cout << "Velocity (mag) of the sphere: " << vel_mag << std::endl;
            std::cout << "Angular velocity (mag) of the sphere: " << angVel_mag << std::endl;
            if (vel_mag < 1e-2) {
                std::cout << "It is stationary" << std::endl;
            } else if (is_near(angVel_mag * sphere_rad, vel_mag, 1e-2)) {
                std::cout << "It is pure rolling" << std::endl;
            } else if (angVel_mag * sphere_rad < 1e-2) {
                std::cout << "It is pure slipping" << std::endl;
            } else if (vel_mag > angVel_mag * sphere_rad) {
                std::cout << "It is rolling with slipping" << std::endl;
            } else {
                std::cout << "WARNING!!! I do not know what happened!!" << std::endl;
            }
            i++;
            std::cout << "=============== " << i << "-th run ===================" << std::endl;
        }
    }
}

void SphereStack() {
    float sphere_rad = 0.15;
    float m_bot = 1.0;
    float mu = 0.2;
    unsigned int run_num = 0;

    for (float gap = 0.2 * sphere_rad; gap <= 0.4 * sphere_rad; gap += 0.05 * sphere_rad) {
        for (float Crr = 0.03; Crr <= 0.3; Crr += 0.01) {
            bool found = false;
            for (float m_top = 0.1; m_top <= 50.0; m_top += 0.02) {
                DEMSolver DEMSim;
                SetSolverProp(DEMSim);
                DEMSim.SetVerbosity("ERROR");

                auto mat_type_1 =
                    DEMSim.LoadMaterial({{"E", 2e6}, {"nu", 0.3}, {"CoR", 0.4}, {"mu", mu}, {"Crr", Crr}});
                // 2 types of spheres
                auto sphere_top_template = DEMSim.LoadSphereType(m_top, sphere_rad, mat_type_1);
                auto sphere_bot_template = DEMSim.LoadSphereType(m_bot, sphere_rad, mat_type_1);

                // Add the bottom plane
                float3 normal_dir = make_float3(0, 0, 1);
                float3 tang_dir = make_float3(1, 0, 0);
                DEMSim.AddBCPlane(make_float3(0, 0, 0), normal_dir, mat_type_1);

                // Add 3 stacking spheres
                auto sphere_bot_1 = DEMSim.AddClumps(sphere_bot_template,
                                                     -tang_dir * (gap / 2. + sphere_rad) + normal_dir * sphere_rad);
                auto sphere_bot_2 =
                    DEMSim.AddClumps(sphere_bot_template, tang_dir * (gap / 2. + sphere_rad) + normal_dir * sphere_rad);
                auto sphere_top = DEMSim.AddClumps(
                    sphere_top_template,
                    normal_dir *
                        (std::sqrt(std::pow(2. * sphere_rad, 2) - std::pow(gap / 2. + sphere_rad, 2)) + sphere_rad));
                auto sphere_tracker = DEMSim.Track(sphere_top);

                float step_time = 1e-5;
                DEMSim.SetInitTimeStep(step_time);
                // Just do CD once and we are all good
                DEMSim.SetCDUpdateFreq(-1);
                DEMSim.SetMaxVelocity(1.0);
                DEMSim.Initialize();

                float frame_time = 1e-1;
                float top_sp_Z = 99999.9;
                float3 pos = make_float3(0);
                path out_dir = current_path();
                out_dir /= "DemoOutput_TestPack";
                create_directory(out_dir);
                int i = 0;
                while (!is_near(pos.z, top_sp_Z, 1e-4)) {
                    top_sp_Z = pos.z;
                    if (run_num == 0) {
                        char filename[100];
                        sprintf(filename, "DEMdemo_output_%04d.csv", i);
                        DEMSim.WriteSphereFile(out_dir / filename);
                    }
                    DEMSim.DoDynamics(frame_time);
                    pos = sphere_tracker->Pos();
                    i++;
                }

                run_num++;
                std::cout << "Test No. " << run_num << std::endl;
                std::cout << "Final Z coord: " << pos.z << std::endl;
                if (top_sp_Z <= sphere_rad) {
                    std::cout << "Sphere mass: " << m_top << std::endl;
                    std::cout << "Rolling resistance: " << Crr << std::endl;
                    std::cout << "Init gap: " << gap << std::endl;
                    std::cout << "Time it takes: " << frame_time * i << std::endl;
                    std::cout << "========== Pile collapse with these params ==========" << std::endl;
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << "WARNING!!! Even with largest mass it did not collapse!" << std::endl;
            }
            if (Crr > 0.1) {
                Crr += 0.02;  // Make it runs less data points so faster
            }
        }
    }
}

int main() {
    // Choose a validation test by uncommenting it
    SphereRollUpIncline();
    // EllpsiodFallingOver();
    // SphereStack();

    std::cout << "DEMdemo_TestPack exiting..." << std::endl;
    return 0;
}
