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

#include <core/utils/JitHelper.h>

namespace sgps {

// Implementation-level classes
class DEMKinematicThread;
class DEMDynamicThread;
class DEMSolverStateDataDT;

/// DynamicThread class
class DEMDynamicThread {
  protected:
    ThreadManager* pSchedSupport;
    GpuManager* pGpuDistributor;

    // The std::thread that binds to this instance
    std::thread th;

    // Object which stores the device and stream IDs for this thread
    GpuManager::StreamInfo streamInfo;

    // A class that contains scratch pad and system status data
    DEMSolverStateDataDT stateOfSolver_resources;

    // The number of for iterations dT does for a specific user "run simulation" call
    size_t nDynamicCycles;

    // Buffer arrays for storing info from the dT side.
    // kT modifies these arrays; dT uses them only.

    // dT gets contact pair/location info from kT
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> idGeometryA_buffer;
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> idGeometryB_buffer;
    std::vector<contact_t, ManagedAllocator<contact_t>> contactType_buffer;

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

    // Triangles (templates) are given a special place (unlike other analytical shapes), b/c we expect them to appear
    // frequently as meshes.
    std::vector<float3, ManagedAllocator<float3>> relPosNode1;
    std::vector<float3, ManagedAllocator<float3>> relPosNode2;
    std::vector<float3, ManagedAllocator<float3>> relPosNode3;

    // External object's components may need the following arrays to store some extra defining features of them. We
    // assume there are usually not too many of them in a simulation. Relative position w.r.t. the owner. For example,
    // the following 3 arrays may hold center points for plates, or tip positions for cones.
    std::vector<float, ManagedAllocator<float>> relPosEntityX;
    std::vector<float, ManagedAllocator<float>> relPosEntityY;
    std::vector<float, ManagedAllocator<float>> relPosEntityZ;
    // Some orientation specifiers. For example, the following 3 arrays may hold normal vectors for planes, or center
    // axis vectors for cylinders.
    std::vector<float, ManagedAllocator<float>> oriEntityX;
    std::vector<float, ManagedAllocator<float>> oriEntityY;
    std::vector<float, ManagedAllocator<float>> oriEntityZ;
    // Some size specifiers. For example, the following 3 arrays may hold top, bottom and length information for finite
    // cylinders.
    std::vector<float, ManagedAllocator<float>> sizeEntity1;
    std::vector<float, ManagedAllocator<float>> sizeEntity2;
    std::vector<float, ManagedAllocator<float>> sizeEntity3;

    // Stores the actual stiffness/damping, where the kernels will need offsets to index into them
    std::vector<float, ManagedAllocator<float>> EProxy;
    std::vector<float, ManagedAllocator<float>> GProxy;
    std::vector<float, ManagedAllocator<float>> CoRProxy;

    // Those are the large ones, ones that have the same length as the number of clumps
    // The mass/MOI offsets
    std::vector<clumpBodyInertiaOffset_t, ManagedAllocator<clumpBodyInertiaOffset_t>> inertiaPropOffsets;

    // Clump's family identification code. Used in determining whether they can be contacts between two families, and
    // whether a family has prescribed motions.
    std::vector<family_t, ManagedAllocator<family_t>> familyID;

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

    // Linear velocity
    std::vector<float, ManagedAllocator<float>> vX;
    std::vector<float, ManagedAllocator<float>> vY;
    std::vector<float, ManagedAllocator<float>> vZ;

    // The angular velocity
    std::vector<float, ManagedAllocator<float>> omgBarX;
    std::vector<float, ManagedAllocator<float>> omgBarY;
    std::vector<float, ManagedAllocator<float>> omgBarZ;

    // Linear acceleration
    std::vector<float, ManagedAllocator<float>> aX;
    std::vector<float, ManagedAllocator<float>> aY;
    std::vector<float, ManagedAllocator<float>> aZ;

    // Angular acceleration
    std::vector<float, ManagedAllocator<float>> alphaX;
    std::vector<float, ManagedAllocator<float>> alphaY;
    std::vector<float, ManagedAllocator<float>> alphaZ;

    // Contact pair/location, for dT's personal use!!
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> idGeometryA;
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> idGeometryB;
    std::vector<contact_t, ManagedAllocator<contact_t>> contactType;

    // Some of dT's own work arrays
    // Force of each contact event. It is the force that bodyA feels.
    std::vector<float3, ManagedAllocator<float3>> contactForces;
    // Local position of contact point of contact w.r.t. the reference frame of body A and B
    std::vector<float3, ManagedAllocator<float3>> contactPointGeometryA;
    std::vector<float3, ManagedAllocator<float3>> contactPointGeometryB;

    size_t m_approx_bytes_used = 0;

    // Set to true only when a user AdvanceSimulation call is finished. Set to false otherwise.
    bool userCallDone = false;

    // If true, dT needs to re-process idA- and idB-related data arrays before collecting forces, as those arrays are
    // freshly obtained from kT.
    bool contactPairArr_isFresh = true;

    // Template-related arrays in managed memory
    // Belonged-body ID
    std::vector<bodyID_t, ManagedAllocator<bodyID_t>> ownerClumpBody;

    // The ID that maps this sphere component's geometry-defining parameters, when this component is jitified
    std::vector<clumpComponentOffset_t, ManagedAllocator<clumpComponentOffset_t>> clumpComponentOffset;
    // The ID that maps this sphere component's geometry-defining parameters, when this component is not jitified (too
    // many templates)
    std::vector<clumpComponentOffsetExt_t, ManagedAllocator<clumpComponentOffsetExt_t>> clumpComponentOffsetExt;
    // The ID that maps this triangle component's geometry-defining parameters, when this component is jitified
    std::vector<clumpComponentOffset_t, ManagedAllocator<clumpComponentOffset_t>> triComponentOffset;
    // The ID that maps this triangle component's geometry-defining parameters, when this component is not jitified (too
    // many templates)
    std::vector<clumpComponentOffsetExt_t, ManagedAllocator<clumpComponentOffsetExt_t>> triComponentOffsetExt;
    // The ID that maps this analytical entity component's geometry-defining parameters, when this component is jitified
    std::vector<clumpComponentOffset_t, ManagedAllocator<clumpComponentOffset_t>> analComponentOffset;

    // The ID that maps this entity's material
    std::vector<materialsOffset_t, ManagedAllocator<materialsOffset_t>> materialTupleOffset;

  public:
    friend class DEMSolver;
    friend class DEMKinematicThread;

    DEMDynamicThread(ThreadManager* pSchedSup, GpuManager* pGpuDist)
        : pSchedSupport(pSchedSup), pGpuDistributor(pGpuDist) {
        GPU_CALL(cudaMallocManaged(&simParams, sizeof(DEMSimParams), cudaMemAttachGlobal));
        GPU_CALL(cudaMallocManaged(&granData, sizeof(DEMDataDT), cudaMemAttachGlobal));
        GPU_CALL(cudaMallocManaged(&granTemplates, sizeof(DEMTemplate), cudaMemAttachGlobal));

        nDynamicCycles = 0;

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

    void setNDynamicCycles(size_t val) { nDynamicCycles = val; }

    // buffer exchange methods
    void setDestinationBufferPointers();

    // Set SimParams items
    void setSimParams(unsigned char nvXp2,
                      unsigned char nvYp2,
                      unsigned char nvZp2,
                      float l,
                      double voxelSize,
                      double binSize,
                      binID_t nbX,
                      binID_t nbY,
                      binID_t nbZ,
                      float3 LBFPoint,
                      float3 G,
                      double ts_size,
                      float expand_factor);

    // Compute total KE of all clumps
    float getKineticEnergy();

    // Resize managed arrays (and perhaps Instruct/Suggest their preferred residence location as well?)
    void allocateManagedArrays(size_t nOwnerBodies,
                               size_t nOwnerClumps,
                               unsigned int nExtObj,
                               size_t nTriEntities,
                               size_t nSpheresGM,
                               size_t nTriGM,
                               unsigned int nAnalGM,
                               unsigned int nClumpTopo,
                               unsigned int nClumpComponents,
                               unsigned int nMatTuples);

    // Data type TBD, should come from JITCed headers
    void populateManagedArrays(const std::vector<unsigned int>& input_clump_types,
                               const std::vector<float3>& input_clump_xyz,
                               const std::vector<float3>& input_clump_vel,
                               const std::vector<unsigned int>& input_clump_family,
                               const std::vector<float3>& input_ext_obj_xyz,
                               const std::vector<std::vector<unsigned int>>& input_clumps_sp_mat_ids,
                               const std::vector<float>& clumps_mass_types,
                               const std::vector<float3>& clumps_moi_types,
                               const std::vector<std::vector<float>>& clumps_sp_radii_types,
                               const std::vector<std::vector<float3>>& clumps_sp_location_types,
                               const std::vector<float>& mat_k,
                               const std::vector<float>& mat_g,
                               const std::vector<float>& mat_CoR);

    // Put sim data array pointers in place
    void packDataPointers();
    void packTransferPointers(DEMKinematicThread* kT);

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

    // Jitify dT kernels (at initialization) based on existing knowledge of this run
    void jitifyKernels(const std::unordered_map<std::string, std::string>& templateSubs,
                       const std::unordered_map<std::string, std::string>& simParamSubs,
                       const std::unordered_map<std::string, std::string>& massMatSubs,
                       const std::unordered_map<std::string, std::string>& familySubs,
                       const std::unordered_map<std::string, std::string>& analGeoSubs);

  private:
    // update clump-based acceleration array based on sphere-based force array
    void calculateForces();

    // update clump pos/oriQ and vel/omega based on acceleration
    void integrateClumpMotions();

    // Bring dT buffer array data to its working arrays
    void unpackMyBuffer();
    // Send produced data to kT-owned biffers
    void sendToTheirBuffer();
    // Resize some work arrays based on the number of contact pairs provided by kT
    void contactEventArraysResize(size_t nContactPairs);

    // Just-in-time compiled kernels
    std::shared_ptr<jitify::Program> prep_force;
    std::shared_ptr<jitify::Program> cal_force;
    std::shared_ptr<jitify::Program> collect_force;
    std::shared_ptr<jitify::Program> integrator;
    std::shared_ptr<jitify::Program> quarry_stats;
};  // dT ends

}  // namespace sgps
