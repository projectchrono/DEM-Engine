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
class DEMSolverStateDataKT;

class DEMKinematicThread {
  protected:
    ThreadManager* pSchedSupport;
    GpuManager* pGpuDistributor;

    // The std::thread that binds to this instance
    std::thread th;

    // Friend system DEMDynamicThread
    DEMDynamicThread* dT;

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

    // Those are the template array of the unique component information. Note that these arrays may be thousands-element
    // long, and only a part of it is jitified. The jitified part of it is typically the frequently used clump and maybe
    // triangle tempates; the other part may be the components for a few large clump bodies which are not frequently
    // used. Component sphere's radius
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
    std::vector<contact_t, ManagedAllocator<contact_t>> contactType;

    // Sphere-related arrays in managed memory
    // Owner body ID of this component
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
    // std::vector<clumpComponentOffset_t, ManagedAllocator<clumpComponentOffset_t>> analComponentOffset;

  public:
    friend class DEMSolver;
    friend class DEMDynamicThread;

    DEMKinematicThread(ThreadManager* pSchedSup, GpuManager* pGpuDist, DEMDynamicThread* dT)
        : pSchedSupport(pSchedSup), pGpuDistributor(pGpuDist) {
        GPU_CALL(cudaMallocManaged(&simParams, sizeof(DEMSimParams), cudaMemAttachGlobal));
        GPU_CALL(cudaMallocManaged(&granData, sizeof(DEMDataKT), cudaMemAttachGlobal));
        GPU_CALL(cudaMallocManaged(&granTemplates, sizeof(DEMTemplate), cudaMemAttachGlobal));

        // Get a device/stream ID to use from the GPU Manager
        streamInfo = pGpuDistributor->getAvailableStream();

        // My friend dT
        this->dT = dT;

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
                               const std::vector<unsigned int>& input_clump_family,
                               const std::vector<unsigned int>& input_ext_obj_family,
                               const std::unordered_map<unsigned int, family_t>& family_user_impl_map,
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
                      binID_t nbX,
                      binID_t nbY,
                      binID_t nbZ,
                      float3 LBFPoint,
                      float3 G,
                      double ts_size,
                      float expand_factor);

    // Put sim data array pointers in place
    void packDataPointers();
    void packTransferPointers(DEMDynamicThread* dT);

    // Jitify kT kernels (at initialization) based on existing knowledge of this run
    void jitifyKernels(const std::unordered_map<std::string, std::string>& templateSubs,
                       const std::unordered_map<std::string, std::string>& simParamSubs,
                       const std::unordered_map<std::string, std::string>& massMatSubs,
                       const std::unordered_map<std::string, std::string>& familyMaskSubs,
                       const std::unordered_map<std::string, std::string>& familyPrescribeSubs,
                       const std::unordered_map<std::string, std::string>& analGeoSubs);

  private:
    // Contact detections
    void contactDetection();
    // Bring kT buffer array data to its working arrays
    void unpackMyBuffer();
    // Send produced data to dT-owned biffers
    void sendToTheirBuffer();
    // Resize contact storage arrays based on the number of contact pairs
    void contactEventArraysResize(size_t nContactPairs);
    // Resize dT's buffer arrays based on the number of contact pairs
    inline void transferArraysResize(size_t nContactPairs);

    // Just-in-time compiled kernels
    // jitify::Program bin_occupation = JitHelper::buildProgram("bin_occupation", " ");
    std::shared_ptr<jitify::Program> bin_occupation;
    std::shared_ptr<jitify::Program> contact_detection;

};  // kT ends

}  // namespace sgps