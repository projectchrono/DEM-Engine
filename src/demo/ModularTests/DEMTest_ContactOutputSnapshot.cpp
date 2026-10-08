// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifndef _WIN32
    #include <sys/stat.h>
    #include <unistd.h>
#endif

#include "DEM/API.h"

using namespace deme;

// A FIFO keeps the writer blocked opening its destination until after dynamics changes the contact list.
// This tests snapshot ownership without sleeps or assumptions about background-thread scheduling.
int main() try {
#ifdef _WIN32
    std::cout << "SKIP: deterministic delayed-output test requires POSIX FIFOs.\n";
    return 0;
#else
    DEMSolver solver(1);
    solver.SetVerbosity("ERROR");
    solver.InstructBoxDomainDimension(4, 4, 4);
    solver.SetGravitationalAcceleration(make_float3(0));
    solver.SetTimeStepSize(1.e-3);
    solver.SetExpandFactor(1.e-4f);
    solver.SetCDUpdateFreq(1);
    solver.DefineContactForceModel("force = make_float3(0.f, 0.f, 1.f); sample_count += 1.f;");
    solver.SetContactWildcards({"sample_count"});
    solver.SetContactOutputContent({"OWNER", "GEO_ID", "FORCE", "POINT", "NORMAL", "TORQUE", "CNT_WILDCARD"});
    auto material = solver.LoadMaterial({{"E", 1.e7}, {"nu", 0.3}, {"CoR", 0.5}, {"mu", 0.4}});
    auto sphere = solver.LoadSphereType(1, 0.1f, material);
    auto spheres =
        solver.AddClumps(sphere, std::vector<float3>{make_float3(-0.5f, 0, 0.09f), make_float3(0.5f, 0, 0.115f)});
    spheres->SetVel(std::vector<float3>{make_float3(0), make_float3(0, 0, -1)});
    solver.AddBCPlane(make_float3(0), make_float3(0, 0, 1), material);
    solver.Initialize();
    solver.DoDynamicsThenSync(1.e-3);
    const size_t before = solver.GetNumContacts();
    if (before != 1)
        throw std::runtime_error("Expected one initial contact, got " + std::to_string(before));

    const auto dir = std::filesystem::temp_directory_path() / ("deme_contact_snapshot_" + std::to_string(getpid()));
    std::filesystem::create_directory(dir);
    // Exercise normal CSV, the binary-to-CSV fallback, and a force threshold that filters out every row.
    // Each output must match a synchronous reference from the same frame even after the solver advances.
    const auto check_delayed_output = [&](OUTPUT_FORMAT format, float threshold, double duration) {
        solver.SetContactOutputFormat(format);
        solver.WriteContactFile((dir / "reference.csv").string(), threshold);
        // Snapshot ownership permits dynamics overlap, but the reference file must finish before we read it.
        solver.WaitForPendingOutput();
        std::ifstream reference_file(dir / "reference.csv");
        std::ostringstream reference;
        reference << reference_file.rdbuf();
        if (!reference_file || reference.str().empty())
            throw std::runtime_error("Cannot read reference contact output");
        const auto pipe = dir / "delayed.csv";
        if (mkfifo(pipe.c_str(), 0600) != 0)
            throw std::runtime_error("Cannot create delayed-output FIFO");
        solver.WriteContactFile(pipe.string(), threshold);

        // Always release the writer before rethrowing a dynamics error, so solver destruction cannot hang on join().
        std::exception_ptr dynamics_error;
        try {
            solver.DoDynamicsThenSync(duration);
        } catch (...) {
            dynamics_error = std::current_exception();
        }
        std::ifstream delayed_file(pipe);
        std::ostringstream delayed;
        delayed << delayed_file.rdbuf();
        // Join the writer before cleaning up its destination; dynamics already ran while output was pending.
        solver.WaitForPendingOutput();
        if (dynamics_error)
            std::rethrow_exception(dynamics_error);
        if (delayed.str() != reference.str())
            throw std::runtime_error("Delayed contact output differs from the requested frame");
        std::filesystem::remove(pipe);
    };
    check_delayed_output(OUTPUT_FORMAT::CSV, DEME_TINY_FLOAT, 0.05);
    const size_t after = solver.GetNumContacts();
    if (after <= before)
        throw std::runtime_error("Contact list did not grow during delayed output");
    check_delayed_output(OUTPUT_FORMAT::BINARY, -1.f, 0.01);
    check_delayed_output(OUTPUT_FORMAT::CSV, 1.e6f, 0.01);
    std::filesystem::remove_all(dir);
    std::cout << "PASS: delayed contact output preserves its snapshot while contacts grow from " << before << " to "
              << after << ".\n";
    return 0;
#endif
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
