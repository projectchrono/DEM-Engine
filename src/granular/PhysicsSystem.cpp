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
    TRACKED_VECTOR_RESIZE(relPosSphereX, nSpheresGM, "relPosSphereX", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereY, nSpheresGM, "relPosSphereY", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereZ, nSpheresGM, "relPosSphereZ", 0);

    // Resize those that are as long as the template lengths
    TRACKED_VECTOR_RESIZE(massClumpBody, nClumpTopo, "massClumpBody", 0);
    TRACKED_VECTOR_RESIZE(radiiSphere, nSphereRadii, "radiiSphere", 0);
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
    voxelID.at(0) = 1;
}

void dynamicThread::WriteCsvAsSpheres(std::ofstream& ptFile) const {
    ParticleFormatWriter pw;
    // pw.write(ptFile, ParticleFormatWriter::CompressionType::NONE, mass);
}

}  // namespace sgps
