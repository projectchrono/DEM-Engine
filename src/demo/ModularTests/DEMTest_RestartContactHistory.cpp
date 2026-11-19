// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DEM/API.h"

using namespace deme;

// Restart a settled bed from the contacts and contact wildcards reported by GetContactDetailedInfo, and check that
// the contact history (delta_time of the default force model) continues instead of restarting from zero. Covers
// sphere-sphere pairs, sphere-analytical pairs (box floor and walls) and sphere-mesh pairs (a fixed plate), loaded with
// SetExistingContactTypes, and the default behaviour where untyped pairs are all treated as sphere-sphere.
namespace {

constexpr float kRadius = 0.01f;
constexpr double kStepSize = 5.e-5;
constexpr double kSettleTime = 0.4;
constexpr double kMinContinuedFraction = 0.99;
const std::vector<std::string> kWildcards = {"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z"};

// Everything needed to rebuild the bed in a fresh solver.
struct SavedBed {
    std::vector<float3> pos, vel, angVel;
    std::vector<float4> oriQ;
    std::vector<std::string> types;
    std::vector<std::pair<bodyID_t, bodyID_t>> pairs;
    std::unordered_map<std::string, std::vector<float>> wildcards;
};

// Contact identity within a type. A sphere-sphere pair is stored with the smaller ID first.
using ContactKey = std::tuple<std::string, bodyID_t, bodyID_t>;
ContactKey makeKey(const std::string& type, bodyID_t a, bodyID_t b) {
    if (type == OUTPUT_FILE_SPH_SPH_CONTACT_NAME && b < a)
        std::swap(a, b);
    return {type, a, b};
}

// Identical scene in every solver: same box walls, same mesh, same order, so analytical component and mesh patch IDs
// saved from one solver are valid in the next. The bed batch is added first, so its sphere IDs are global IDs.
std::shared_ptr<DEMTracker> buildScene(DEMSolver& solver, DEMClumpBatch& bed) {
    solver.SetVerbosity("ERROR");
    solver.InstructBoxDomainDimension({-0.1f, 0.1f}, {-0.1f, 0.1f}, {0.f, 0.3f});
    solver.SetGravitationalAcceleration(make_float3(0, 0, -9.81f));
    solver.SetTimeStepSize(kStepSize);
    solver.SetCDUpdateFreq(10);
    solver.SetMaxVelocity(3.f);
    solver.SetExpandSafetyMultiplier(1.1f);
    solver.SetContactOutputContent({"OWNER", "GEO_ID", "FORCE", "CNT_WILDCARD"});
    auto material = solver.LoadMaterial({{"E", 1.e7}, {"nu", 0.3}, {"CoR", 0.3}, {"mu", 0.5}, {"Crr", 0.0}});
    solver.InstructBoxDomainBoundingBC("top_open", material);

    auto sphere = solver.LoadSphereType(2600.f * 4.f / 3.f * PI * kRadius * kRadius * kRadius, kRadius, material);
    bed.SetTypes(sphere);
    auto handle = solver.AddClumps(bed);

    // A fixed two-triangle plate slightly above the floor on the +x half, so part of the bed rests on a mesh.
    DEMMesh plate;
    const float z = 0.003f;
    plate.m_vertices = {make_float3(0.f, -0.095f, z), make_float3(0.095f, -0.095f, z), make_float3(0.095f, 0.095f, z),
                        make_float3(0.f, 0.095f, z)};
    plate.m_face_v_indices = {make_int3(0, 1, 2), make_int3(0, 2, 3)};
    plate.nTri = 2;
    plate.SetMaterial(material);
    plate.SetMass(1.f);
    plate.SetMOI(make_float3(1.f));
    auto plate_handle = solver.AddMesh(plate);
    plate_handle->SetFamily(10);
    solver.SetFamilyFixed(10);

    auto tracker = solver.Track(handle);
    solver.Initialize();
    return tracker;
}

// delta_time of every contact currently known to the solver, keyed by (type, A, B).
std::map<ContactKey, float> contactAges(DEMSolver& solver) {
    auto info = solver.GetContactDetailedInfo(-1.f);
    const auto& types = info->GetContactType();
    const auto& a = info->GetAGeo();
    const auto& b = info->GetBGeo();
    const auto& age = info->Get<float>("delta_time");
    std::map<ContactKey, float> ages;
    for (size_t i = 0; i < types.size(); i++)
        ages[makeKey(types[i], a[i], b[i])] = age[i];
    return ages;
}

// Settle a small bed of spheres on the floor and the plate, then save its state and all of its contacts.
SavedBed settleAndSave() {
    std::vector<float3> xyz;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> jitter(-0.002f, 0.002f);
    for (int k = 0; k < 6; k++)
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++)
                xyz.push_back(make_float3(-0.077f + 0.022f * i + jitter(rng), -0.077f + 0.022f * j + jitter(rng),
                                          0.025f + 0.022f * k));
    DEMSolver solver(1);
    DEMClumpBatch bed(xyz.size());
    bed.SetPos(xyz);
    auto tracker = buildScene(solver, bed);
    solver.DoDynamicsThenSync(kSettleTime);

    SavedBed saved;
    saved.pos = tracker->Positions();
    saved.vel = tracker->Velocities();
    saved.angVel = tracker->AngularVelocitiesLocal();
    saved.oriQ = tracker->OrientationQuaternions();
    auto info = solver.GetContactDetailedInfo(-1.f);
    saved.types = info->GetContactType();
    const auto& a = info->GetAGeo();
    const auto& b = info->GetBGeo();
    for (size_t i = 0; i < saved.types.size(); i++)
        saved.pairs.emplace_back(a[i], b[i]);
    for (const auto& name : kWildcards)
        saved.wildcards[name] = info->Get<float>(name);
    return saved;
}

// Rebuild the bed in a fresh solver from the saved state and contacts, run exactly one step, return contact ages.
std::map<ContactKey, float> reloadAndStep(const SavedBed& saved, bool with_types) {
    std::vector<std::pair<bodyID_t, bodyID_t>> pairs;
    std::vector<std::string> types;
    std::unordered_map<std::string, std::vector<float>> wildcards;
    for (size_t i = 0; i < saved.types.size(); i++) {
        if (!with_types && saved.types[i] != OUTPUT_FILE_SPH_SPH_CONTACT_NAME)
            continue;
        pairs.push_back(saved.pairs[i]);
        types.push_back(saved.types[i]);
        for (const auto& name : kWildcards)
            wildcards[name].push_back(saved.wildcards.at(name)[i]);
    }

    DEMSolver solver(1);
    DEMClumpBatch bed(saved.pos.size());
    bed.SetPos(saved.pos);
    bed.SetVel(saved.vel);
    bed.SetAngVel(saved.angVel);
    bed.SetOriQ(saved.oriQ);
    bed.SetExistingContacts(pairs);
    if (with_types)
        bed.SetExistingContactTypes(types);
    bed.SetExistingContactWildcards(wildcards);
    buildScene(solver, bed);
    solver.DoDynamicsThenSync(kStepSize);
    return contactAges(solver);
}

// Among saved contacts of one type that were in contact long enough to tell a continued history (age + one step) from
// a restarted one (about one step), return the fraction that continued, and how many such contacts there were.
std::pair<double, size_t> continuedFraction(const SavedBed& saved,
                                            const std::map<ContactKey, float>& now,
                                            const std::string& type) {
    size_t informative = 0, continued = 0;
    const auto& ages = saved.wildcards.at("delta_time");
    for (size_t i = 0; i < saved.types.size(); i++) {
        const double age = ages[i];
        if (saved.types[i] != type || age < 10. * kStepSize)
            continue;
        informative++;
        auto it = now.find(makeKey(type, saved.pairs[i].first, saved.pairs[i].second));
        // float32 accumulation of delta_time, plus half a step of slack.
        if (it != now.end() && it->second >= age + 0.5 * kStepSize - 1.e-6 * age)
            continued++;
    }
    return {informative ? double(continued) / informative : 0., informative};
}

void require(bool ok, const std::string& what) {
    if (!ok)
        throw std::runtime_error(what);
}

void checkFraction(const SavedBed& saved,
                   const std::map<ContactKey, float>& now,
                   const std::string& type,
                   size_t min_informative,
                   const std::string& label) {
    const auto [fraction, informative] = continuedFraction(saved, now, type);
    std::cout << label << ": " << type << " history continued for " << fraction * 100. << "% of " << informative
              << " established contacts.\n";
    require(informative >= min_informative,
            label + ": too few established " + type + " contacts to test (" + std::to_string(informative) + ")");
    require(fraction > kMinContinuedFraction, label + ": " + type + " contact history was not restored");
}

// Host-side validation of SetExistingContactTypes; needs no device.
void checkTypeValidation() {
    std::cout << "Checking that invalid restart contact types are rejected (two errors are expected below).\n";
    DEMClumpBatch batch(2);
    batch.SetExistingContacts({{0, 1}, {1, 0}});
    bool threw = false;
    try {
        batch.SetExistingContactTypes({"SS"});
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, "SetExistingContactTypes accepted a type list of the wrong length");
    threw = false;
    try {
        batch.SetExistingContactTypes({"SS", "MM"});
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, "SetExistingContactTypes accepted an unsupported contact type");
    batch.SetExistingContactTypes({"SS", "SA"});
    batch.SetExistingContacts({{0, 1}});
    require(batch.contact_types.empty(), "SetExistingContacts did not clear previously supplied contact types");
}

}  // namespace

int main() try {
    checkTypeValidation();

    const SavedBed saved = settleAndSave();

    // (1)+(2): typed reload restores sphere-sphere, sphere-analytical and sphere-mesh histories.
    const auto typed = reloadAndStep(saved, true);
    checkFraction(saved, typed, OUTPUT_FILE_SPH_SPH_CONTACT_NAME, 200, "typed reload");
    checkFraction(saved, typed, OUTPUT_FILE_SPH_ANAL_CONTACT_NAME, 20, "typed reload");
    checkFraction(saved, typed, OUTPUT_FILE_SPH_MESH_CONTACT_NAME, 10, "typed reload");

    // (3): without SetExistingContactTypes, pairs are sphere-sphere as before, and their history is restored too.
    const auto untyped = reloadAndStep(saved, false);
    checkFraction(saved, untyped, OUTPUT_FILE_SPH_SPH_CONTACT_NAME, 200, "untyped reload");

    std::cout << "PASS: restart restores sphere-sphere, sphere-analytical and sphere-mesh contact history.\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
