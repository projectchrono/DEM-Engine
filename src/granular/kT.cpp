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

namespace sgps {

int DEMKinematicThread::costlyProductionStep(int val) const {
    std::this_thread::sleep_for(std::chrono::milliseconds(kinematicAverageTime));
    return 2 * val + 1;
}

inline void DEMKinematicThread::contactDetection() {
    // auto data_arg = voxelID.data();
    auto bin_occupation =
        JitHelper::buildProgram("DEMContactKernels", JitHelper::KERNEL_DIR / "DEMContactKernels.cu",
                                std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

    bin_occupation.kernel("calculateSphereBinOverlap")
        .instantiate()
        .configure(dim3(1), dim3(simParams->nSpheresGM), 0, streamInfo.stream)
        .launch(simParams, granData);

    GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
}

void DEMKinematicThread::workerThread() {
    // Set the device for this thread
    cudaSetDevice(streamInfo.device);
    cudaStreamCreate(&streamInfo.stream);

    while (!pSchedSupport->kinematicShouldJoin) {
        {
            std::unique_lock<std::mutex> lock(pSchedSupport->kinematicStartLock);
            while (!pSchedSupport->kinematicStarted) {
                pSchedSupport->cv_KinematicStartLock.wait(lock);
            }
            // Ensure that we wait for start signal on next iteration
            pSchedSupport->kinematicStarted = false;
            if (pSchedSupport->kinematicShouldJoin) {
                break;
            }
        }
        // run a while loop producing stuff in each iteration;
        // once produced, it should be made available to the dynamic via memcpy
        while (!pSchedSupport->dynamicDone) {
            // before producing something, a new work order should be in place. Wait on
            // it
            if (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh) {
                pSchedSupport->schedulingStats.nTimesKinematicHeldBack++;
                std::unique_lock<std::mutex> lock(pSchedSupport->kinematicCanProceed);

                while (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh) {
                    // loop to avoid spurious wakeups
                    pSchedSupport->cv_KinematicCanProceed.wait(lock);
                }

                // getting here means that new "work order" data has been provided
                {
                    // acquire lock and get the work order
                    std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
                    cudaMemcpy(granData->voxelID, granData->voxelID_buffer, N_INPUT_ITEMS * sizeof(voxelID_t),
                               cudaMemcpyDeviceToDevice);
                }
            }

            // figure out the amount of shared mem
            // cudaDeviceGetAttribute.cudaDevAttrMaxSharedMemoryPerBlock

            // produce something here; fake stuff for now
            // cudaStream_t currentStream;
            // cudaStreamCreate(&currentStream);pSchedSupport->dynamicShouldWait()

            // TODO: crash on reaching conditioanl variable if the other thread is in kernel??
            // contactDetection();

            /* for the reference
            for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
                // kinematicTestKernel<<<1, 1, 0, kinematicStream.stream>>>();

                // use cudaLaunchKernel
                // cudaLaunchKernel((void*)&kinematicTestKernel, dim3(1), dim3(1), NULL, 0, stream_id);
                // example argument list:
                //  args = { &arg1, &arg2, ... &argN };
                // cudaLaunchKernel((void*)&kinematicTestKernelWithArgs, dim3(1), dim3(1), &args, 0, stream_id);
                kinematicTestKernel<<<1, 1>>>();
                cudaDeviceSynchronize();
                pSchedSupport->dynamicShouldWait()
                int indx = j % N_INPUT_ITEMS;
                product[j] += this->costlyProductionStep(j) + inputData[indx];
            }
            */

            // make it clear that the data for most recent work order has
            // been used, in case there is interest in updating it
            pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = false;

            {
                // acquire lock and supply the dynamic with fresh produce
                std::lock_guard<std::mutex> lock(pSchedSupport->dynamicOwnedBuffer_AccessCoordination);
                cudaMemcpy(granData->pDTOwnedBuffer_idGeometryA, granData->idGeometryA,
                           N_MANUFACTURED_ITEMS * sizeof(bodyID_t), cudaMemcpyDeviceToDevice);
            }
            pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = true;
            pSchedSupport->schedulingStats.nDynamicUpdates++;

            // signal the dynamic that it has fresh produce
            pSchedSupport->cv_DynamicCanProceed.notify_all();
        }

        // in case the dynamic is hanging in there...
        pSchedSupport->cv_DynamicCanProceed.notify_all();

        // When getting here, dT has finished one user call (although perhaps not at the end of the user script).
        userCallDone = true;
    }
}

void DEMKinematicThread::startThread() {
    std::lock_guard<std::mutex> lock(pSchedSupport->kinematicStartLock);
    pSchedSupport->kinematicStarted = true;
    pSchedSupport->cv_KinematicStartLock.notify_one();
}

bool DEMKinematicThread::isUserCallDone() {
    // return true if done, false if not
    return userCallDone;
}

void DEMKinematicThread::resetUserCallStat() {
    userCallDone = false;
    // Reset kT stats variables, making ready for next user call
    pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh = false;
}

// Put sim data array pointers in place
void DEMKinematicThread::packDataPointers() {
    granData->voxelID = voxelID.data();
    granData->locX = locX.data();
    granData->locY = locY.data();
    granData->locZ = locZ.data();
    granData->oriQ0 = oriQ0.data();
    granData->oriQ1 = oriQ1.data();
    granData->oriQ2 = oriQ2.data();
    granData->oriQ3 = oriQ3.data();
    granData->idGeometryA = idGeometryA.data();
    granData->idGeometryB = idGeometryB.data();

    // for kT, those state vectors are fed by dT, so each has a buffer
    granData->voxelID = voxelID_buffer.data();
    granData->locX = locX_buffer.data();
    granData->locY = locY_buffer.data();
    granData->locZ = locZ_buffer.data();
    granData->oriQ0 = oriQ0_buffer.data();
    granData->oriQ1 = oriQ1_buffer.data();
    granData->oriQ2 = oriQ2_buffer.data();
    granData->oriQ3 = oriQ3_buffer.data();
}
void DEMKinematicThread::packTransferPointers(DEMDataDT* dTData) {
    // Set the pointers to dT owned buffers
    granData->pDTOwnedBuffer_idGeometryA = dTData->idGeometryA_buffer;
    granData->pDTOwnedBuffer_idGeometryB = dTData->idGeometryB_buffer;
}

void DEMKinematicThread::setSimParams(unsigned char nvXp2,
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

void DEMKinematicThread::allocateManagedArrays(unsigned int nClumpBodies,
                                               unsigned int nSpheresGM,
                                               unsigned int nClumpTopo,
                                               unsigned int nClumpComponents,
                                               unsigned int nMatTuples) {
    simParams->nSpheresGM = nSpheresGM;
    simParams->nClumpBodies = nClumpBodies;

    // Resize to the number of clumps
    TRACKED_VECTOR_RESIZE(voxelID, nClumpBodies, "voxelID", 0);
    TRACKED_VECTOR_RESIZE(locX, nClumpBodies, "locX", 0);
    TRACKED_VECTOR_RESIZE(locY, nClumpBodies, "locY", 0);
    TRACKED_VECTOR_RESIZE(locZ, nClumpBodies, "locZ", 0);
    TRACKED_VECTOR_RESIZE(oriQ0, nClumpBodies, "oriQ0", 0);
    TRACKED_VECTOR_RESIZE(oriQ1, nClumpBodies, "oriQ1", 0);
    TRACKED_VECTOR_RESIZE(oriQ2, nClumpBodies, "oriQ2", 0);
    TRACKED_VECTOR_RESIZE(oriQ3, nClumpBodies, "oriQ3", 0);

    // Transfer buffer arrays
    TRACKED_VECTOR_RESIZE(voxelID_buffer, nClumpBodies, "voxelID_buffer", 0);
    TRACKED_VECTOR_RESIZE(locX_buffer, nClumpBodies, "locX_buffer", 0);
    TRACKED_VECTOR_RESIZE(locY_buffer, nClumpBodies, "locY_buffer", 0);
    TRACKED_VECTOR_RESIZE(locZ_buffer, nClumpBodies, "locZ_buffer", 0);
    TRACKED_VECTOR_RESIZE(oriQ0_buffer, nClumpBodies, "oriQ0_buffer", 0);
    TRACKED_VECTOR_RESIZE(oriQ1_buffer, nClumpBodies, "oriQ1_buffer", 0);
    TRACKED_VECTOR_RESIZE(oriQ2_buffer, nClumpBodies, "oriQ2_buffer", 0);
    TRACKED_VECTOR_RESIZE(oriQ3_buffer, nClumpBodies, "oriQ3_buffer", 0);
}

void DEMKinematicThread::primeDynamic() {
    // transfer produce to dynamic buffer
    // cudaMemcpy(pDTOwnedBuffer_voxelID, voxelID.data(), N_INPUT_ITEMS * sizeof(voxelID_t),
    //            cudaMemcpyDeviceToDevice);
    pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = true;
    pSchedSupport->schedulingStats.nDynamicUpdates++;
}

}  // namespace sgps
