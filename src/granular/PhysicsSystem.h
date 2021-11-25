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
#include <granular/DataStructs.h>

namespace sgps {

/// KinematicThread class
class DEMKinematicThread {
  protected:
    ThreadManager* pSchedSupport;
    GpuManager* pGpuDistributor;

    // The std::thread that binds to this instance
    std::thread th;

    // Object which stores the device and stream IDs for this thread
    GpuManager::StreamInfo streamInfo;

    int kinematicAverageTime;
    int costlyProductionStep(int) const;

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

    // kT computed contact pair info
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> idGeometryA;
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> idGeometryB;

    // Sphere-related arrays in managed memory
    // Belonged-body ID (default unsigned int type)
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> ownerClumpBody;

    // The ID that maps this sphere's radius and relPos
    std::vector<clumpComponentOffset_t, ManagedAllocator<clumpComponentOffset_t>> clumpComponentOffset;

    // A number of kT work arrays
    // The number of bins each sphere touches (also serves as the container for the related prefix scan)
    std::vector<binsSphereTouches_t, ManagedAllocator<binsSphereTouches_t>> numBinsSphereTouches;
    // The IDs of those bins that touch each sphere
    std::vector<binID_t, ManagedAllocator<binID_t>> binIDsEachSphereTouches;
    // The company array of the previous one
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> sphereIDsEachBinTouches;
    // Derived from binIDsEachSphereTouches we have the IDs of the active bins (have at least 2 spheres in it). Then
    // this following array is generated to instruct the kernel where to look in sphereIDsEachBinTouches to find this
    // thread's relevant sphere IDs.
    std::vector<binsSphereTouches_t, ManagedAllocator<binsSphereTouches_t>> sphereIDsLookUpTable;
    // And how far should this thread look into (number of spheres this bin touches)?
    std::vector<spheresBinTouches_t, ManagedAllocator<spheresBinTouches_t>> numSpheresBinTouches;
    // The number of contact pairs in each bin (also serves as the container for the related prefix scan)
    std::vector<contactPairs_t, ManagedAllocator<contactPairs_t>> numContactsInEachBin;

  public:
    friend class DEMSolver;
    friend class DEMDynamicThread;

    DEMKinematicThread(ThreadManager* pSchedSup, GpuManager* pGpuDist)
        : pSchedSupport(pSchedSup), pGpuDistributor(pGpuDist) {
        GPU_CALL(cudaMallocManaged(&simParams, sizeof(DEMSimParams), cudaMemAttachGlobal));
        GPU_CALL(cudaMallocManaged(&granData, sizeof(DEMDataKT), cudaMemAttachGlobal));
        GPU_CALL(cudaMallocManaged(&granTemplates, sizeof(DEMTemplate), cudaMemAttachGlobal));
        kinematicAverageTime = 0;

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

    void setKinematicAverageTime(int val) { kinematicAverageTime = val; }

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
    void packTransferPointers(DEMDataDT* dTData);

  private:
    void contactDetection();

    // Bring kT buffer array data to its working arrays
    void unpackMyBuffer();
    // Send produced data to dT-owned biffers
    void sendToTheirBuffer();
};  // kT ends

/// DynamicThread class
class DEMDynamicThread {
  protected:
    ThreadManager* pSchedSupport;
    GpuManager* pGpuDistributor;

    // The std::thread that binds to this instance
    std::thread th;

    // Object which stores the device and stream IDs for this thread
    GpuManager::StreamInfo streamInfo;

    int dynamicAverageTime;  // time required in the consumption process; fake lag
    int nDynamicCycles;

    // Buffer arrays for storing info from the dT side.
    // kT modifies these arrays; dT uses them only.

    // dT gets contact pair/location info from kT
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> idGeometryA_buffer;
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> idGeometryB_buffer;

    // Pointers to simulation params-related arrays
    // Check DataStructs.h for details
    DEMSimParams* simParams;

    // Pointers to those data arrays defined below, stored in a struct
    DEMDataDT* granData;

    // Pointers to clump template data. In the end, the use of this struct should be replaced by JIT.
    DEMTemplate* granTemplates;
    DEMMaterialProxy* matProxy;

    // Body-related arrays in managed memory, for dT's personal use (not transfer buffer)

    // Those are the smaller ones, the unique, template ones. TODO: These should be jitified into kernels not brought
    // from global mem.
    // The mass values
    std::vector<float, ManagedAllocator<float>> massClumpBody;

    // The components of MOI values
    std::vector<float, ManagedAllocator<float>> mmiXX;
    std::vector<float, ManagedAllocator<float>> mmiYY;
    std::vector<float, ManagedAllocator<float>> mmiZZ;

    // The distinct sphere radii values
    std::vector<float, ManagedAllocator<float>> radiiSphere;

    // The distinct sphere local position (wrt CoM) values
    std::vector<float, ManagedAllocator<float>> relPosSphereX;
    std::vector<float, ManagedAllocator<float>> relPosSphereY;
    std::vector<float, ManagedAllocator<float>> relPosSphereZ;

    // Those are the large ones, ones that have the same length as the number of clumps
    // The mass offsets
    std::vector<clumpBodyInertiaOffset_t, ManagedAllocator<clumpBodyInertiaOffset_t>> inertiaPropOffsets;

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

    // Linear velocity times ts size, as a multiple of l
    std::vector<float, ManagedAllocator<float>> hvX;
    std::vector<float, ManagedAllocator<float>> hvY;
    std::vector<float, ManagedAllocator<float>> hvZ;

    // The angular velocity times ts size: h*omega
    std::vector<float, ManagedAllocator<float>> hOmgBarX;
    std::vector<float, ManagedAllocator<float>> hOmgBarY;
    std::vector<float, ManagedAllocator<float>> hOmgBarZ;

    // Linear acceleration times h^2, as a multiple of l
    std::vector<float, ManagedAllocator<float>> h2aX;
    std::vector<float, ManagedAllocator<float>> h2aY;
    std::vector<float, ManagedAllocator<float>> h2aZ;

    // Angular acceleration times h^2
    std::vector<float, ManagedAllocator<float>> h2AlphaX;
    std::vector<float, ManagedAllocator<float>> h2AlphaY;
    std::vector<float, ManagedAllocator<float>> h2AlphaZ;

    // Contact pair/location, for dT's personal use!!
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> idGeometryA;
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> idGeometryB;

    size_t m_approx_bytes_used = 0;

    // Set to true only when a user AdvanceSimulation call is finished. Set to false otherwise.
    bool userCallDone = false;

    // Sphere-related arrays in managed memory
    // Belonged-body ID (default unsigned int type)
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> ownerClumpBody;

    // The ID that maps this sphere's radius and relPos
    std::vector<clumpComponentOffset_t, ManagedAllocator<clumpComponentOffset_t>> clumpComponentOffset;

    // The ID that maps this sphere's material
    std::vector<materialsOffset_t, ManagedAllocator<materialsOffset_t>> materialTupleOffset;

    int localUse(int val);

  public:
    friend class DEMSolver;
    friend class DEMKinematicThread;

    DEMDynamicThread(ThreadManager* pSchedSup, GpuManager* pGpuDist)
        : pSchedSupport(pSchedSup), pGpuDistributor(pGpuDist) {
        GPU_CALL(cudaMallocManaged(&simParams, sizeof(DEMSimParams), cudaMemAttachGlobal));
        GPU_CALL(cudaMallocManaged(&granData, sizeof(DEMDataDT), cudaMemAttachGlobal));

        nDynamicCycles = 0;
        dynamicAverageTime = 0;

        // Get a device/stream ID to use from the GPU Manager
        streamInfo = pGpuDistributor->getAvailableStream();

        pSchedSupport->dynamicShouldJoin = false;
        pSchedSupport->dynamicStarted = false;

        // Launch a worker thread bound to this instance
        th = std::move(std::thread([this]() { this->workerThread(); }));
    }
    ~DEMDynamicThread() {
        // std::cout << "Dynamic thread closing..." << std::endl;
        pSchedSupport->dynamicShouldJoin = true;
        startThread();
        th.join();
        cudaStreamDestroy(streamInfo.stream);
    }

    void setDynamicAverageTime(int val) { dynamicAverageTime = val; }
    void setNDynamicCycles(int val) { nDynamicCycles = val; }

    // buffer exchange methods
    void setDestinationBufferPointers();

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
                               const std::vector<float>& clumps_mass_types,
                               const std::vector<std::vector<float>>& clumps_sp_radii_types,
                               const std::vector<std::vector<float3>>& clumps_sp_location_types);

    // Put sim data array pointers in place
    void packDataPointers();
    void packTransferPointers(DEMDataKT* kTData);

    void WriteCsvAsSpheres(std::ofstream& ptFile) const;

    // Called each time when the user calls LaunchThreads.
    void startThread();

    // The actual kernel things go here.
    // It is called upon construction.
    void workerThread();

    // Query the value of userCallDone
    bool isUserCallDone();
    // Reset userCallDone back to false
    void resetUserCallStat();

  private:
    // update clump-based acceleration array based on sphere-based force array
    void calculateForces();

    // update clump pos and vel based on acceleration
    void integrateClumpLinearMotions();

    // update clump orientation and angvel based on ang-acceleration
    void integrateClumpRotationalMotions();

    // Bring dT buffer array data to its working arrays
    void unpackMyBuffer();
    // Send produced data to kT-owned biffers
    void sendToTheirBuffer();
};  // dT ends

}  // namespace sgps
