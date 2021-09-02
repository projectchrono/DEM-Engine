//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <cstring>
#include <iostream>
#include <thread>

#include <core/ApiVersion.h>
#include <core/utils/Macros.h>
#include <core/utils/chpf/particle_writer.hpp>
#include <granular/GranularDefines.h>
#include <granular/PhysicsSystem.h>

namespace sgps {

int kinematicThread::costlyProductionStep(int val) const {
    std::this_thread::sleep_for(std::chrono::milliseconds(kinematicAverageTime));
    return 2 * val + 1;
}

void dynamicThread::allocateManagedArrays(unsigned int nClumpBodies,
                                          unsigned int nSpheresGM,
                                          clumpBodyInertiaOffset_default_t nClumpTopo,
                                          distinctSphereRadiiOffset_default_t nSphereRadii,
                                          distinctSphereRelativePositions_default_t nSphereRelaPos,
                                          materialsOffset_default_t nMatTuples) {
    // Resize those that are as long as the number of clumps
    TRACKED_VECTOR_RESIZE(voxelID, nClumpBodies, "voxelID", 0);
    TRACKED_VECTOR_RESIZE(locX, nClumpBodies, "locX", 0);
    TRACKED_VECTOR_RESIZE(locY, nClumpBodies, "locY", 0);
    TRACKED_VECTOR_RESIZE(locZ, nClumpBodies, "locZ", 0);

    // Resize those that are as long as the number of spheres
    TRACKED_VECTOR_RESIZE(ownerClumpBody, nSpheresGM, "ownerClumpBody", 0);
    TRACKED_VECTOR_RESIZE(sphereRadiusOffset, nSpheresGM, "sphereRadiusOffset", 0);
    TRACKED_VECTOR_RESIZE(sphereRelPosXOffset, nSpheresGM, "sphereRelPosXOffset", 0);
    TRACKED_VECTOR_RESIZE(sphereRelPosYOffset, nSpheresGM, "sphereRelPosYOffset", 0);
    TRACKED_VECTOR_RESIZE(sphereRelPosZOffset, nSpheresGM, "sphereRelPosZOffset", 0);

    // Resize those that are as long as the template lengths
    TRACKED_VECTOR_RESIZE(massClumpBody, nClumpTopo, "massClumpBody", 0);
    TRACKED_VECTOR_RESIZE(radiiSphere, nSphereRadii, "radiiSphere", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereX, nSphereRelaPos, "relPosSphereX", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereY, nSphereRelaPos, "relPosSphereY", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereZ, nSphereRelaPos, "relPosSphereZ", 0);
}

void dynamicThread::populateManagedArrays(
    const std::vector<clumpBodyInertiaOffset_default_t>& input_clump_types,
    const std::vector<float3>& input_clump_xyz,
    const std::set<float>& clumps_mass_types,
    const std::set<float>& clumps_sp_radii_types,
    const std::set<float3>& clumps_sp_location_types,
    const std::vector<clumpBodyInertiaOffset_default_t>& clumps_mass_type_offset,
    const std::vector<std::vector<distinctSphereRadiiOffset_default_t>>& clumps_sp_radii_type_offset,
    const std::vector<std::vector<distinctSphereRelativePositions_default_t>>& clumps_sp_location_type_offset) {
    // Use some temporary hacks to get the info in the managed mem
    // All the input vectors should have the same length, nClumpTopo
    unsigned int k = 0;

    for (auto elem : clumps_sp_radii_types) {
        radiiSphere.at(k) = elem;
        k++;
    }
    k = 0;

    for (auto elem : clumps_sp_location_types) {
        relPosSphereX.at(k) = elem.x;
        relPosSphereY.at(k) = elem.y;
        relPosSphereZ.at(k) = elem.z;
        k++;
    }
    k = 0;

    for (size_t i = 0; i < input_clump_types.size(); i++) {
        auto type_of_this_clump = input_clump_types.at(i);
        auto this_CoM_coord = input_clump_xyz.at(i);
        auto this_clump_no_sp_radii_offsets = clumps_sp_radii_type_offset.at(type_of_this_clump);
        auto this_clump_no_sp_loc_offsets = clumps_sp_location_type_offset.at(type_of_this_clump);

        for (size_t j = 0; j < this_clump_no_sp_radii_offsets.size(); j++) {
            sphereRadiusOffset.at(k) = this_clump_no_sp_radii_offsets.at(j);
            ownerClumpBody.at(k) = i;
            sphereRelPosXOffset.at(k) = this_clump_no_sp_loc_offsets.at(j);
            sphereRelPosYOffset.at(k) = this_clump_no_sp_loc_offsets.at(j);
            sphereRelPosZOffset.at(k) = this_clump_no_sp_loc_offsets.at(j);
            k++;
        }
    }
}

void dynamicThread::WriteCsvAsSpheres(std::ofstream& ptFile) const {
    ParticleFormatWriter pw;
    // pw.write(ptFile, ParticleFormatWriter::CompressionType::NONE, mass);
}

}  // namespace sgps
