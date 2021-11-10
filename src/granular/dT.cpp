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
#include <core/utils/JitHelper.h>
#include <helper_math.cuh>

namespace sgps {

// Put sim data array pointers in place
void DEMDynamicThread::packDataPointers() {
    granData->voxelID = voxelID.data();
    granData->locX = locX.data();
    granData->locY = locY.data();
    granData->locZ = locZ.data();
    granData->h2aX = h2aX.data();
    granData->h2aY = h2aY.data();
    granData->h2aZ = h2aZ.data();
    granData->hvX = hvX.data();
    granData->hvY = hvY.data();
    granData->hvZ = hvZ.data();
    granData->idGeometryA = idGeometryA.data();
    granData->idGeometryB = idGeometryB.data();
    granData->idGeometryA_buffer = idGeometryA_buffer.data();
    granData->idGeometryB_buffer = idGeometryB_buffer.data();
}
void DEMDynamicThread::packTransferPointers(DEMDataKT* kTData) {
    // These are the pointers for sending data to dT
    granData->pKTOwnedBuffer_voxelID = kTData->voxelID_buffer;
    granData->pKTOwnedBuffer_locX = kTData->locX_buffer;
    granData->pKTOwnedBuffer_locY = kTData->locY_buffer;
    granData->pKTOwnedBuffer_locZ = kTData->locZ_buffer;
    granData->pKTOwnedBuffer_oriQ0 = kTData->oriQ0_buffer;
    granData->pKTOwnedBuffer_oriQ1 = kTData->oriQ1_buffer;
    granData->pKTOwnedBuffer_oriQ2 = kTData->oriQ2_buffer;
    granData->pKTOwnedBuffer_oriQ3 = kTData->oriQ3_buffer;
}

void DEMDynamicThread::setSimParams(unsigned char nvXp2,
                                    unsigned char nvYp2,
                                    unsigned char nvZp2,
                                    float l,
                                    double voxelSize,
                                    unsigned int binSize,
                                    float3 LBFPoint,
                                    float3 G,
                                    double ts_size) {
    simParams->nvXp2 = nvXp2;
    simParams->nvYp2 = nvYp2;
    simParams->nvZp2 = nvZp2;
    simParams->l = l;
    simParams->voxelSize = voxelSize;
    simParams->binSize = binSize;
    simParams->LBFX = LBFPoint.x;
    simParams->LBFY = LBFPoint.y;
    simParams->LBFZ = LBFPoint.z;
    simParams->Gx = G.x;
    simParams->Gy = G.y;
    simParams->Gz = G.z;
    simParams->h = ts_size;
}

void DEMDynamicThread::allocateManagedArrays(unsigned int nClumpBodies,
                                             unsigned int nSpheresGM,
                                             unsigned int nClumpTopo,
                                             unsigned int nClumpComponents,
                                             unsigned int nMatTuples) {
    // TODO: Why are they here?? Shouldn't they be in setSimParams?
    simParams->nSpheresGM = nSpheresGM;
    simParams->nClumpBodies = nClumpBodies;
    // Resize to the number of clumps
    TRACKED_VECTOR_RESIZE(voxelID, nClumpBodies, "voxelID", 0);
    TRACKED_VECTOR_RESIZE(locX, nClumpBodies, "locX", 0);
    TRACKED_VECTOR_RESIZE(locY, nClumpBodies, "locY", 0);
    TRACKED_VECTOR_RESIZE(locZ, nClumpBodies, "locZ", 0);
    TRACKED_VECTOR_RESIZE(hvX, nClumpBodies, "hvX", 0);
    TRACKED_VECTOR_RESIZE(hvY, nClumpBodies, "hvY", 0);
    TRACKED_VECTOR_RESIZE(hvZ, nClumpBodies, "hvZ", 0);
    TRACKED_VECTOR_RESIZE(h2aX, nClumpBodies, "h2aX", 0);
    TRACKED_VECTOR_RESIZE(h2aY, nClumpBodies, "h2aY", 0);
    TRACKED_VECTOR_RESIZE(h2aZ, nClumpBodies, "h2aZ", 0);

    // Resize to the number of spheres
    TRACKED_VECTOR_RESIZE(ownerClumpBody, nSpheresGM, "ownerClumpBody", 0);
    TRACKED_VECTOR_RESIZE(clumpComponentOffset, nSpheresGM, "sphereRadiusOffset", 0);

    // Resize to the length of the clump templates
    TRACKED_VECTOR_RESIZE(massClumpBody, nClumpTopo, "massClumpBody", 0);
    TRACKED_VECTOR_RESIZE(mmiXX, nClumpTopo, "mmiXX", 0);
    TRACKED_VECTOR_RESIZE(mmiYY, nClumpTopo, "mmiYY", 0);
    TRACKED_VECTOR_RESIZE(mmiZZ, nClumpTopo, "mmiZZ", 0);
    TRACKED_VECTOR_RESIZE(radiiSphere, nClumpComponents, "radiiSphere", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereX, nClumpComponents, "relPosSphereX", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereY, nClumpComponents, "relPosSphereY", 0);
    TRACKED_VECTOR_RESIZE(relPosSphereZ, nClumpComponents, "relPosSphereZ", 0);

    // Arrays for contact info
    // The length of idGeometry arrays is an estimate
    TRACKED_VECTOR_RESIZE(idGeometryA, nSpheresGM, "idGeometryA", 0);
    TRACKED_VECTOR_RESIZE(idGeometryB, nSpheresGM, "idGeometryB", 0);

    // Transfer buffer arrays
    // The length of idGeometry arrays is an estimate
    TRACKED_VECTOR_RESIZE(idGeometryA_buffer, nSpheresGM, "idGeometryA_buffer", 0);
    TRACKED_VECTOR_RESIZE(idGeometryB_buffer, nSpheresGM, "idGeometryB_buffer", 0);
}

void DEMDynamicThread::populateManagedArrays(const std::vector<clumpBodyInertiaOffset_t>& input_clump_types,
                                             const std::vector<float3>& input_clump_xyz,
                                             const std::vector<float>& clumps_mass_types,
                                             const std::vector<std::vector<float>>& clumps_sp_radii_types,
                                             const std::vector<std::vector<float3>>& clumps_sp_location_types) {
    // Use some temporary hacks to get the info in the managed mem
    // All the input vectors should have the same length, nClumpTopo
    unsigned int k = 0;
    std::vector<unsigned int> prescans;

    prescans.push_back(0);
    for (auto elem : clumps_sp_radii_types) {
        for (auto radius : elem) {
            radiiSphere.at(k) = radius;
            k++;
        }
        prescans.push_back(k);
    }
    prescans.pop_back();
    k = 0;

    for (auto elem : clumps_sp_location_types) {
        for (auto loc : elem) {
            relPosSphereX.at(k) = loc.x;
            relPosSphereY.at(k) = loc.y;
            relPosSphereZ.at(k) = loc.z;
            k++;
        }
        // std::cout << "sphere location types: " << elem.x << ", " << elem.y << ", " << elem.z << std::endl;
    }
    k = 0;

    for (size_t i = 0; i < input_clump_types.size(); i++) {
        auto type_of_this_clump = input_clump_types.at(i);
        float3 LBF;
        LBF.x = simParams->LBFX;
        LBF.y = simParams->LBFY;
        LBF.z = simParams->LBFZ;
        auto this_CoM_coord = input_clump_xyz.at(i) - LBF;
        // std::cout << "CoM position: " << this_CoM_coord.x << ", " << this_CoM_coord.y << ", " << this_CoM_coord.z <<
        // std::endl;
        auto this_clump_no_sp_radii = clumps_sp_radii_types.at(type_of_this_clump);
        auto this_clump_no_sp_relPos = clumps_sp_location_types.at(type_of_this_clump);

        for (size_t j = 0; j < this_clump_no_sp_radii.size(); j++) {
            clumpComponentOffset.at(k) = prescans.at(type_of_this_clump) + j;
            ownerClumpBody.at(k) = i;
            k++;
            // std::cout << "Sphere Rel Pos offset: " << this_clump_no_sp_loc_offsets.at(j) << std::endl;
        }

        voxelID_t voxelNumX = (double)this_CoM_coord.x / simParams->voxelSize;
        voxelID_t voxelNumY = (double)this_CoM_coord.y / simParams->voxelSize;
        voxelID_t voxelNumZ = (double)this_CoM_coord.z / simParams->voxelSize;
        locX.at(i) = ((double)this_CoM_coord.x - (double)voxelNumX * simParams->voxelSize) / simParams->l;
        locY.at(i) = ((double)this_CoM_coord.y - (double)voxelNumY * simParams->voxelSize) / simParams->l;
        locZ.at(i) = ((double)this_CoM_coord.z - (double)voxelNumZ * simParams->voxelSize) / simParams->l;
        // std::cout << "Clump voxel num: " << voxelNumX << ", " << voxelNumY << ", " << voxelNumZ << std::endl;

        voxelID.at(i) += voxelNumX;
        voxelID.at(i) += voxelNumY << simParams->nvXp2;
        voxelID.at(i) += voxelNumZ << (simParams->nvXp2 + simParams->nvYp2);
        // std::cout << "Computed voxel num: " << voxelID.at(i) << std::endl;
    }
}

void DEMDynamicThread::WriteCsvAsSpheres(std::ofstream& ptFile) const {
    ParticleFormatWriter pw;
    // pw.write(ptFile, ParticleFormatWriter::CompressionType::NONE, mass);
    std::vector<float> posX(simParams->nSpheresGM, 0);
    std::vector<float> posY(simParams->nSpheresGM, 0);
    std::vector<float> posZ(simParams->nSpheresGM, 0);
    std::vector<float> spRadii(simParams->nSpheresGM, 0);
    for (unsigned int i = 0; i < simParams->nSpheresGM; i++) {
        auto this_owner = ownerClumpBody.at(i);
        voxelID_t voxelIDX =
            voxelID.at(this_owner) & (((voxelID_t)1 << simParams->nvXp2) - 1);  // & operation here equals modulo
        voxelID_t voxelIDY = (voxelID.at(this_owner) >> simParams->nvXp2) & (((voxelID_t)1 << simParams->nvYp2) - 1);
        voxelID_t voxelIDZ = (voxelID.at(this_owner)) >> (simParams->nvXp2 + simParams->nvYp2);
        // std::cout << "this owner: " << this_owner << std::endl;
        // std::cout << "Out voxel ID: " << voxelID.at(this_owner) << std::endl;
        // std::cout << "Out voxel ID XYZ: " << voxelIDX << ", " << voxelIDY << ", " << voxelIDZ << std::endl;

        auto this_sp_deviation_x = relPosSphereX.at(clumpComponentOffset.at(i));
        auto this_sp_deviation_y = relPosSphereY.at(clumpComponentOffset.at(i));
        auto this_sp_deviation_z = relPosSphereZ.at(clumpComponentOffset.at(i));
        posX.at(i) = voxelIDX * simParams->voxelSize + locX.at(this_owner) * simParams->l + this_sp_deviation_x +
                     simParams->LBFX;
        posY.at(i) = voxelIDY * simParams->voxelSize + locY.at(this_owner) * simParams->l + this_sp_deviation_y +
                     simParams->LBFY;
        posZ.at(i) = voxelIDZ * simParams->voxelSize + locZ.at(this_owner) * simParams->l + this_sp_deviation_z +
                     simParams->LBFZ;
        // std::cout << "Sphere Pos: " << posX.at(i) << ", " << posY.at(i) << ", " << posZ.at(i) << std::endl;

        spRadii.at(i) = radiiSphere.at(clumpComponentOffset.at(i));
    }
    pw.write(ptFile, ParticleFormatWriter::CompressionType::NONE, posX, posY, posZ, spRadii);
}

inline void DEMDynamicThread::unpackMyBuffer() {
    cudaMemcpy(granData->idGeometryA, granData->idGeometryA_buffer, N_MANUFACTURED_ITEMS * sizeof(bodyID_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(granData->idGeometryA, granData->idGeometryA_buffer, N_MANUFACTURED_ITEMS * sizeof(bodyID_t),
               cudaMemcpyDeviceToDevice);
}

inline void DEMDynamicThread::sendToTheirBuffer() {
    cudaMemcpy(granData->pKTOwnedBuffer_voxelID, granData->voxelID, simParams->nClumpBodies * sizeof(voxelID_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(granData->pKTOwnedBuffer_locX, granData->locX, simParams->nClumpBodies * sizeof(subVoxelPos_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(granData->pKTOwnedBuffer_locY, granData->locY, simParams->nClumpBodies * sizeof(subVoxelPos_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(granData->pKTOwnedBuffer_locZ, granData->locZ, simParams->nClumpBodies * sizeof(subVoxelPos_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(granData->pKTOwnedBuffer_oriQ0, granData->oriQ0, simParams->nClumpBodies * sizeof(oriQ_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(granData->pKTOwnedBuffer_oriQ1, granData->oriQ1, simParams->nClumpBodies * sizeof(oriQ_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(granData->pKTOwnedBuffer_oriQ2, granData->oriQ2, simParams->nClumpBodies * sizeof(oriQ_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(granData->pKTOwnedBuffer_oriQ3, granData->oriQ3, simParams->nClumpBodies * sizeof(oriQ_t),
               cudaMemcpyDeviceToDevice);
}

inline void DEMDynamicThread::calculateForces() {
    unsigned int nClumps = simParams->nClumpBodies;
    auto cal_force =
        JitHelper::buildProgram("DEMForceKernels", JitHelper::KERNEL_DIR / "DEMForceKernels.cu",
                                std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

    cal_force.kernel("deriveClumpForces")
        .instantiate()
        .configure(dim3(1), dim3(nClumps), 0, streamInfo.stream)
        .launch(simParams, granData);

    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
}

inline void DEMDynamicThread::integrateClumpLinearMotions() {
    unsigned int nClumps = simParams->nClumpBodies;
    auto integrator =
        JitHelper::buildProgram("DEMIntegrationKernels", JitHelper::KERNEL_DIR / "DEMIntegrationKernels.cu",
                                std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});
    integrator.kernel("integrateClumps")
        .instantiate()
        .configure(dim3(1), dim3(nClumps), 0, streamInfo.stream)
        .launch(simParams, granData);

    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
}

void DEMDynamicThread::integrateClumpRotationalMotions() {}

void DEMDynamicThread::workerThread() {
    // Set the gpu for this thread
    cudaSetDevice(streamInfo.device);
    cudaStreamCreate(&streamInfo.stream);
    int totGPU;
    cudaGetDeviceCount(&totGPU);
    printf("Total device: %d\n", totGPU);

    while (!pSchedSupport->dynamicShouldJoin) {
        {
            std::unique_lock<std::mutex> lock(pSchedSupport->dynamicStartLock);
            while (!pSchedSupport->dynamicStarted) {
                pSchedSupport->cv_DynamicStartLock.wait(lock);
            }
            // Ensure that we wait for start signal on next iteration
            pSchedSupport->dynamicStarted = false;
            if (pSchedSupport->dynamicShouldJoin) {
                break;
            }
        }

        // At the beginning of each user call, send kT a work order, b/c dT need results from CD to proceed. After this
        // one instance, kT and dT may work in an async fashion.
        {
            std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
            sendToTheirBuffer();
        }
        pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = true;
        pSchedSupport->schedulingStats.nKinematicUpdates++;
        // Signal the kinematic that it has data for a new work order.
        pSchedSupport->cv_KinematicCanProceed.notify_all();
        // Then dT will wait for kT to finish one initial run
        {
            std::unique_lock<std::mutex> lock(pSchedSupport->dynamicCanProceed);
            while (!pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
                // loop to avoid spurious wakeups
                pSchedSupport->cv_DynamicCanProceed.wait(lock);
            }
        }

        for (int cycle = 0; cycle < nDynamicCycles; cycle++) {
            // if the produce is fresh, use it
            if (pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
                {
                    // acquire lock and use the content of the dynamic-owned transfer buffer
                    std::lock_guard<std::mutex> lock(pSchedSupport->dynamicOwnedBuffer_AccessCoordination);
                    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_GRANULARITY_MS));
                    // unpackMyBuffer();
                }
                pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = false;
                pSchedSupport->stampLastUpdateOfDynamic = cycle;
            }

            // if it's the case, it's important at this point to let the kinematic know
            // that this is the last dynamic cycle; this is important otherwise the
            // kinematic will hang waiting for communication swith the dynamic
            if (cycle == (nDynamicCycles - 1))
                pSchedSupport->dynamicDone = true;

            // if the kinematic is idle, give it the opportunity to get busy again
            if (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh) {
                // acquire lock and refresh the work order for the kinematic
                {
                    std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
                    sendToTheirBuffer();
                }
                pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = true;
                pSchedSupport->schedulingStats.nKinematicUpdates++;
                // signal the kinematic that it has data for a new work order
                pSchedSupport->cv_KinematicCanProceed.notify_all();
            }

            /* Currently no work at all
            // this is the fake place where produce is used;
            for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
                outcome[j] += this->localUse(j);
            }
            */

            calculateForces();
            integrateClumpLinearMotions();

            integrateClumpRotationalMotions();

            std::cout << "Dynamic side values. Cycle: " << cycle << std::endl;

            // dynamic wrapped up one cycle
            pSchedSupport->currentStampOfDynamic++;

            // check if we need to wait; i.e., if dynamic drifted too much into future, then we must wait a bit before
            // the next cycle begins
            if (pSchedSupport->dynamicShouldWait()) {
                // wait for a signal from the kinematic to indicate that
                // the kinematic has caught up
                pSchedSupport->schedulingStats.nTimesDynamicHeldBack++;
                std::unique_lock<std::mutex> lock(pSchedSupport->dynamicCanProceed);
                while (!pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
                    // loop to avoid spurious wakeups
                    pSchedSupport->cv_DynamicCanProceed.wait(lock);
                }
            }
        }

        // When getting here, dT has finished one user call (although perhaps not at the end of the user script).
        userCallDone = true;
    }
}

void DEMDynamicThread::startThread() {
    std::lock_guard<std::mutex> lock(pSchedSupport->dynamicStartLock);
    pSchedSupport->dynamicStarted = true;
    pSchedSupport->cv_DynamicStartLock.notify_one();
}

bool DEMDynamicThread::isUserCallDone() {
    // return true if done, false if not
    return userCallDone;
}

void DEMDynamicThread::resetUserCallStat() {
    userCallDone = false;
    // Reset last kT-side data receiving cycle time stamp.
    pSchedSupport->stampLastUpdateOfDynamic = -1;
    pSchedSupport->currentStampOfDynamic = 0;
    // Reset dT stats variables, making ready for next user call
    pSchedSupport->dynamicDone = false;
    pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = false;
}

int DEMDynamicThread::localUse(int val) {
    cudaSetDevice(streamInfo.device);
    // std::this_thread::sleep_for(std::chrono::milliseconds(dynamicAverageTime));

    // dynamicTestKernel<<<1, 1>>>();
    auto gpu_program =
        JitHelper::buildProgram("gpuKernels", JitHelper::KERNEL_DIR / "gpuKernels.cu", std::vector<JitHelper::Header>(),
                                {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

    gpu_program.kernel("dynamicTestKernel").instantiate().configure(dim3(1), dim3(1), 0, streamInfo.stream).launch();
    // cudaLaunchKernel((void*)&dynamicTestKernel, dim3(1), dim3(1), NULL, 0, streamInfo.stream);
    cudaDeviceSynchronize();
    return 2 * val;
}

}  // namespace sgps
