//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <mutex>
#include <vector>
#include <thread>
// #include <set>

#include <core/ApiVersion.h>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/ThreadManager.h>
#include <core/utils/GpuManager.h>
#include <helper_math.cuh>
#include <core/utils/GpuError.h>

#include <granular/GranularDefines.h>
#include <granular/GranularStructs.h>
#include <granular/DataStructs.h>

namespace sgps {

// Implementation-level classes
class DEMKinematicThread;
class DEMDynamicThread;
class DEMSolverStateDataKT;

class DEMKinematicThread {
  protected:
    ThreadManager* pSchedSupport;
    GpuManager* pGpuDistributor;

    // The std::thread that binds to this instance
    std::thread th;

    // Object which stores the device and stream IDs for this thread
    GpuManager::StreamInfo streamInfo;

    // A class that contains scratch pad and system status data
    DEMSolverStateDataKT stateOfSolver_resources;

    size_t m_approx_bytes_used = 0;

    // Set to true only when a user AdvanceSimulation call is finished. Set to false otherwise.
    bool userCallDone = false;

    // Pointers to simulation params-related arrays
    // Check DataStructs.h for details
    DEMSimParams* simParams;

    // Pointers to those data arrays defined below, stored in a struct
    DEMDataKT* granData;

    // Pointers to clump template data. In the end, the use of this struct should be replaced by JIT.
    DEMTemplate* granTemplates;

    // The pointers to dT owned buffer arrays themselves. The sole purpose of these pointers are for resizing on kT
    // side.
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>>* pDTOwnedVector_idGeometryA;
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>>* pDTOwnedVector_idGeometryB;

    // Buffer arrays for storing info from the dT side.
    // dT modifies these arrays; kT uses them only.

    // kT gets clump locations and rotations from dT
    // The voxel ID
    std::vector<voxelID_t, ManagedAllocator<voxelID_t>> voxelID_buffer;
    // The XYZ local location inside a voxel
    std::vector<subVoxelPos_t, ManagedAllocator<subVoxelPos_t>> locX_buffer;
    std::vector<subVoxelPos_t, ManagedAllocator<subVoxelPos_t>> locY_buffer;
    std::vector<subVoxelPos_t, ManagedAllocator<subVoxelPos_t>> locZ_buffer;
    // The clump quaternion
    std::vector<oriQ_t, ManagedAllocator<oriQ_t>> oriQ0_buffer;
    std::vector<oriQ_t, ManagedAllocator<oriQ_t>> oriQ1_buffer;
    std::vector<oriQ_t, ManagedAllocator<oriQ_t>> oriQ2_buffer;
    std::vector<oriQ_t, ManagedAllocator<oriQ_t>> oriQ3_buffer;

    // Managed arrays for dT's personal use (not transfer buffer)

    // Those are the smaller ones, the unique, template ones. TODO: These should be jitified into kernels not brought
    // from global mem The distinct sphere radii values
    std::vector<float, ManagedAllocator<float>> radiiSphere;
    // Radii times beta, expressed as the multiple of voxelSize, used only in contact detection (TODO: Remove this
    // unused array) std::vector<unsigned int, ManagedAllocator<unsigned int>> inflatedRadiiVoxelRatio;

    // The distinct sphere local position (wrt CoM) values
    std::vector<float, ManagedAllocator<float>> relPosSphereX;
    std::vector<float, ManagedAllocator<float>> relPosSphereY;
    std::vector<float, ManagedAllocator<float>> relPosSphereZ;

    // The voxel ID (split into 3 parts, representing XYZ location)
    std::vector<voxelID_t, ManagedAllocator<voxelID_t>> voxelID;

    // The XYZ local location inside a voxel
    std::vector<subVoxelPos_t, ManagedAllocator<subVoxelPos_t>> locX;
    std::vector<subVoxelPos_t, ManagedAllocator<subVoxelPos_t>> locY;
    std::vector<subVoxelPos_t, ManagedAllocator<subVoxelPos_t>> locZ;

    // The clump quaternion
    std::vector<oriQ_t, ManagedAllocator<oriQ_t>> oriQ0;
    std::vector<oriQ_t, ManagedAllocator<oriQ_t>> oriQ1;
    std::vector<oriQ_t, ManagedAllocator<oriQ_t>> oriQ2;
    std::vector<oriQ_t, ManagedAllocator<oriQ_t>> oriQ3;

    // Clump's family identification code. Used in determining whether they can be contacts between two families, and
    // whether a family has prescribed motions.
    std::vector<family_t, ManagedAllocator<family_t>> familyID;

    // kT computed contact pair info
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> idGeometryA;
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> idGeometryB;

    // Sphere-related arrays in managed memory
    // Belonged-body ID (default unsigned int type)
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> ownerClumpBody;

    // The ID that maps this sphere's radius and relPos
    std::vector<clumpComponentOffset_t, ManagedAllocator<clumpComponentOffset_t>> clumpComponentOffset;

  public:
    friend class DEMSolver;
    friend class DEMDynamicThread;

    DEMKinematicThread(ThreadManager* pSchedSup, GpuManager* pGpuDist)
        : pSchedSupport(pSchedSup), pGpuDistributor(pGpuDist) {
        GPU_CALL(cudaMallocManaged(&simParams, sizeof(DEMSimParams), cudaMemAttachGlobal));
        GPU_CALL(cudaMallocManaged(&granData, sizeof(DEMDataKT), cudaMemAttachGlobal));
        GPU_CALL(cudaMallocManaged(&granTemplates, sizeof(DEMTemplate), cudaMemAttachGlobal));

        // Get a device/stream ID to use from the GPU Manager
        streamInfo = pGpuDistributor->getAvailableStream();

        pSchedSupport->kinematicShouldJoin = false;
        pSchedSupport->kinematicStarted = false;

        // Launch a worker thread bound to this instance
        th = std::move(std::thread([this]() { this->workerThread(); }));
    }
    ~DEMKinematicThread() {
        // std::cout << "Kinematic thread closing..." << std::endl;
        pSchedSupport->kinematicShouldJoin = true;
        startThread();
        th.join();
        cudaStreamDestroy(streamInfo.stream);
    }

    // buffer exchange methods
    void setDestinationBufferPointers();

    void primeDynamic();

    // Called each time when the user calls LaunchThreads.
    void startThread();

    // The actual kernel things go here.
    // It is called upon construction.
    void workerThread();

    // Query the value of userCallDone
    bool isUserCallDone();
    // Reset userCallDone back to false
    void resetUserCallStat();

    // Resize managed arrays (and perhaps Instruct/Suggest their preferred residence location as well?)
    void allocateManagedArrays(size_t nClumpBodies,
                               size_t nSpheresGM,
                               unsigned int nClumpTopo,
                               unsigned int nClumpComponents,
                               unsigned int nMatTuples);

    // Data type TBD, should come from JITCed headers
    void populateManagedArrays(const std::vector<unsigned int>& input_clump_types,
                               const std::vector<float3>& input_clump_xyz,
                               const std::vector<float3>& input_clump_vel,
                               const std::vector<unsigned int>& input_clump_family,
                               const std::vector<float>& clumps_mass_types,
                               const std::vector<std::vector<float>>& clumps_sp_radii_types,
                               const std::vector<std::vector<float3>>& clumps_sp_location_types);

    // Set SimParams items
    void setSimParams(unsigned char nvXp2,
                      unsigned char nvYp2,
                      unsigned char nvZp2,
                      float l,
                      double voxelSize,
                      double binSize,
                      float3 LBFPoint,
                      float3 G,
                      double ts_size,
                      float expand_factor);

    // Put sim data array pointers in place
    void packDataPointers();
    void packTransferPointers(DEMDynamicThread* dT);

  private:
    // Contact detections
    void contactDetection();
    // Bring kT buffer array data to its working arrays
    void unpackMyBuffer();
    // Send produced data to dT-owned biffers
    void sendToTheirBuffer();
};  // kT ends

}  // namespace sgps