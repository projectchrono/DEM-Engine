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

int kinematicThread::costlyProductionStep(int val) const {
    std::this_thread::sleep_for(std::chrono::milliseconds(kinematicAverageTime));
    return 2 * val + 1;
}

void dynamicThread::setSimParams(unsigned char nvXp2,
                                 unsigned char nvYp2,
                                 unsigned char nvZp2,
                                 float l,
                                 double voxelSize,
                                 float3 LBFPoint) {
    simParams->nvXp2 = nvXp2;
    simParams->nvYp2 = nvYp2;
    simParams->nvZp2 = nvZp2;
    simParams->l = l;
    simParams->voxelSize = voxelSize;
    simParams->LBFX = LBFPoint.x;
    simParams->LBFY = LBFPoint.y;
    simParams->LBFZ = LBFPoint.z;
}

void dynamicThread::allocateManagedArrays(unsigned int nClumpBodies,
                                          unsigned int nSpheresGM,
                                          clumpBodyInertiaOffset_default_t nClumpTopo,
                                          distinctSphereRadiiOffset_default_t nSphereRadii,
                                          distinctSphereRelativePositions_default_t nSphereRelaPos,
                                          materialsOffset_default_t nMatTuples) {
    simParams->nSpheresGM = nSpheresGM;
    simParams->nClumpBodies = nClumpBodies;
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
    TRACKED_VECTOR_RESIZE(sphereRelPosZOffset, nSpheresGM, "sphereRelPosZOffset",
                          0);  // TODO: these relPosOffsets should be reduced to 1

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
        auto this_clump_no_sp_radii_offsets = clumps_sp_radii_type_offset.at(type_of_this_clump);
        auto this_clump_no_sp_loc_offsets = clumps_sp_location_type_offset.at(type_of_this_clump);

        for (size_t j = 0; j < this_clump_no_sp_radii_offsets.size(); j++) {
            sphereRadiusOffset.at(k) = this_clump_no_sp_radii_offsets.at(j);
            ownerClumpBody.at(k) = i;
            sphereRelPosXOffset.at(k) = this_clump_no_sp_loc_offsets.at(j);
            sphereRelPosYOffset.at(k) = this_clump_no_sp_loc_offsets.at(j);
            sphereRelPosZOffset.at(k) = this_clump_no_sp_loc_offsets.at(j);
            k++;
            // std::cout << "Sphere Rel Pos offset: " << this_clump_no_sp_loc_offsets.at(j) << std::endl;
        }

        unsigned int voxelNumX = (unsigned int)((double)this_CoM_coord.x / simParams->voxelSize);
        unsigned int voxelNumY = (unsigned int)((double)this_CoM_coord.y / simParams->voxelSize);
        unsigned int voxelNumZ = (unsigned int)((double)this_CoM_coord.z / simParams->voxelSize);
        locX.at(i) =
            (unsigned int)(((double)this_CoM_coord.x - (double)voxelNumX * simParams->voxelSize) / simParams->l);
        locY.at(i) =
            (unsigned int)(((double)this_CoM_coord.y - (double)voxelNumY * simParams->voxelSize) / simParams->l);
        locZ.at(i) =
            (unsigned int)(((double)this_CoM_coord.z - (double)voxelNumZ * simParams->voxelSize) / simParams->l);
        // std::cout << "Clump voxel num: " << voxelNumX << ", " << voxelNumY << ", " << voxelNumZ << std::endl;

        voxelID.at(i) += voxelNumX;
        voxelID.at(i) += voxelNumY << simParams->nvXp2;
        voxelID.at(i) += voxelNumZ << (simParams->nvXp2 + simParams->nvYp2);
        // std::cout << "Computed voxel num: " << voxelID.at(i) << std::endl;
    }
}

void dynamicThread::WriteCsvAsSpheres(std::ofstream& ptFile) const {
    ParticleFormatWriter pw;
    // pw.write(ptFile, ParticleFormatWriter::CompressionType::NONE, mass);
    std::vector<float> posX(simParams->nSpheresGM, 0);
    std::vector<float> posY(simParams->nSpheresGM, 0);
    std::vector<float> posZ(simParams->nSpheresGM, 0);
    std::vector<float> spRadii(simParams->nSpheresGM, 0);
    for (unsigned int i = 0; i < simParams->nSpheresGM; i++) {
        auto this_owner = ownerClumpBody.at(i);
        unsigned int voxelIDX =
            voxelID.at(this_owner) & ((1u << simParams->nvXp2) - 1);  // & operation here equals modulo
        unsigned int voxelIDY = (voxelID.at(this_owner) >> simParams->nvXp2) & ((1u << simParams->nvYp2) - 1);
        unsigned int voxelIDZ = (voxelID.at(this_owner)) >> (simParams->nvXp2 + simParams->nvYp2);
        // std::cout << "this owner: " << this_owner << std::endl;
        // std::cout << "Out voxel ID: " << voxelID.at(this_owner) << std::endl;
        // std::cout << "Out voxel ID XYZ: " << voxelIDX << ", " << voxelIDY << ", " << voxelIDZ << std::endl;

        auto this_sp_deviation_x = relPosSphereX.at(sphereRelPosXOffset.at(i));
        auto this_sp_deviation_y = relPosSphereY.at(sphereRelPosYOffset.at(i));
        auto this_sp_deviation_z = relPosSphereZ.at(sphereRelPosZOffset.at(i));
        posX.at(i) = voxelIDX * simParams->voxelSize + locX.at(this_owner) * simParams->l + this_sp_deviation_x +
                     simParams->LBFX;
        posY.at(i) = voxelIDY * simParams->voxelSize + locY.at(this_owner) * simParams->l + this_sp_deviation_y +
                     simParams->LBFY;
        posZ.at(i) = voxelIDZ * simParams->voxelSize + locZ.at(this_owner) * simParams->l + this_sp_deviation_z +
                     simParams->LBFZ;
        // std::cout << "Sphere Pos: " << posX.at(i) << ", " << posY.at(i) << ", " << posZ.at(i) << std::endl;

        spRadii.at(i) = radiiSphere.at(sphereRadiusOffset.at(i));
    }
    pw.write(ptFile, ParticleFormatWriter::CompressionType::NONE, posX, posY, posZ, spRadii);
}

void kinematicThread::operator()() {
    // Set the device for this thread
    cudaSetDevice(streamInfo.device);

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
                // acquire lock and supply the dynamic with fresh produce
                std::lock_guard<std::mutex> lock(pSchedSupport->kinematicOwnedBuffer_AccessCoordination);
                cudaMemcpy(voxelID.data(), transferBuffer_voxelID.data(), N_INPUT_ITEMS * sizeof(voxelID_default_t),
                           cudaMemcpyDeviceToDevice);
            }
        }

        // figure out the amount of shared mem
        // cudaDeviceGetAttribute.cudaDevAttrMaxSharedMemoryPerBlock

        // produce something here; fake stuff for now
        // cudaStream_t currentStream;
        // cudaStreamCreate(&currentStream);

        auto data_arg = voxelID.data();

        auto gpu_program =
            JitHelper::buildProgram("gpuKernels", JitHelper::KERNEL_DIR / "gpuKernels.cu",
                                    std::vector<JitHelper::Header>(), {"-I" + (JitHelper::KERNEL_DIR / "..").string()});

        gpu_program.kernel("kinematicTestKernel")
            .instantiate()
            .configure(dim3(1), dim3(N_INPUT_ITEMS), 0, streamInfo.stream)
            .launch((void*)(&data_arg));

        cudaDeviceSynchronize();
        // cudaStreamDestroy(currentStream);

        /* for reference staff
        for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
            // kinematicTestKernel<<<1, 1, 0, kinematicStream.stream>>>();

            // use cudaLaunchKernel
            // cudaLaunchKernel((void*)&kinematicTestKernel, dim3(1), dim3(1), NULL, 0, stream_id);
            // example argument list:
            //  args = { &arg1, &arg2, ... &argN };
            // cudaLaunchKernel((void*)&kinematicTestKernelWithArgs, dim3(1), dim3(1), &args, 0, stream_id);
            kinematicTestKernel<<<1, 1>>>();
            cudaDeviceSynchronize();

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
            cudaMemcpy(pDynamicOwnedBuffer_voxelID, voxelID.data(), N_MANUFACTURED_ITEMS * sizeof(voxelID_default_t),
                       cudaMemcpyDeviceToDevice);
        }
        pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = true;
        pSchedSupport->schedulingStats.nDynamicUpdates++;

        // signal the dynamic that it has fresh produce
        pSchedSupport->cv_DynamicCanProceed.notify_all();
    }

    // in case the dynamic is hanging in there...
    pSchedSupport->cv_DynamicCanProceed.notify_all();
}

void dynamicThread::operator()() {
    // Set the gpu for this thread
    cudaSetDevice(streamInfo.device);

    // acquire lock to prevent the kinematic to mess up
    // with the transfer buffer while the latter is used

    for (int cycle = 0; cycle < nDynamicCycles; cycle++) {
        // if the produce is fresh, use it
        if (pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh) {
            {
                // acquire lock and use the content of the dynamic-owned transfer
                // buffer
                std::lock_guard<std::mutex> lock(pSchedSupport->dynamicOwnedBuffer_AccessCoordination);
                cudaMemcpy(voxelID.data(), transferBuffer_voxelID.data(),
                           N_MANUFACTURED_ITEMS * sizeof(voxelID_default_t), cudaMemcpyDeviceToDevice);
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
                cudaMemcpy(pKinematicOwnedBuffer_voxelID, voxelID.data(), N_INPUT_ITEMS * sizeof(voxelID_default_t),
                           cudaMemcpyDeviceToDevice);
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

        int totGPU;
        cudaGetDeviceCount(&totGPU);
        printf("Total device: %d\n", totGPU);

        std::cout << "Dynamic side values. Cycle: " << cycle << std::endl;

        // dynamic wrapped up one cycle
        pSchedSupport->currentStampOfDynamic++;

        // check if we need to wait; i.e., if dynamic drifted too much into future, then we must wait a bit before the
        // next cycle begins
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
}

int dynamicThread::localUse(int val) {
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

void kinematicThread::primeDynamic() {
    // transfer produce to dynamic buffer
    cudaMemcpy(pDynamicOwnedBuffer_voxelID, voxelID.data(), N_INPUT_ITEMS * sizeof(voxelID_default_t),
               cudaMemcpyDeviceToDevice);
    pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh = true;
    pSchedSupport->schedulingStats.nDynamicUpdates++;
}

}  // namespace sgps
