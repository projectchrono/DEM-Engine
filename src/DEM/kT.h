//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_KT
#define DEME_KT

#include <mutex>
#include <vector>
#include <thread>
#include <unordered_map>
// #include <set>

#include "../core/utils/CudaAllocator.hpp"
#include "../core/utils/ThreadManager.h"
#include "../core/utils/GpuManager.h"
#include "../core/utils/DataMigrationHelper.hpp"
#include "../core/utils/GpuError.h"
#include "../core/utils/Timer.hpp"

#include "BdrsAndObjs.h"
#include "Defines.h"
#include "Structs.h"

// Forward declare jitify::Program to avoid downstream dependency
namespace jitify {
class Program;
}

namespace deme {

// Implementation-level classes
class DEMKinematicThread;
class DEMDynamicThread;
class DEMSolverScratchData;

class DEMKinematicThread {
  protected:
    WorkerReportChannel* pPagerToMain;
    ThreadManager* pSchedSupport;
    // GpuManager* pGpuDistributor;

    // kT verbosity
    VERBOSITY verbosity = INFO;

    // Some behavior-related flags
    SolverFlags solverFlags;

    // The std::thread that binds to this instance
    std::thread th;

    // Friend system DEMDynamicThread
    DEMDynamicThread* dT;

    // Object which stores the device and stream IDs for this thread
    GpuManager::StreamInfo streamInfo;

    // Memory usage recorder
    size_t m_approxDeviceBytesUsed = 0;
    size_t m_approxHostBytesUsed = 0;

    // A class that contains scratch pad and system status data (constructed with the number of temp arrays we need)
    DEMSolverScratchData solverScratchSpace = DEMSolverScratchData(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // kT should break out of its inner loop and return to a state where it awaits a `start' call at the outer loop
    bool kTShouldReset = false;

    // Simulation params-related variables
    DualStruct<DEMSimParams> simParams = DualStruct<DEMSimParams>();

    // Pointers to those data arrays defined below, stored in a struct
    DualStruct<DEMDataKT> granData = DualStruct<DEMDataKT>();

    // Log for anomalies in the simulation
    WorkerAnomalies anomalies = WorkerAnomalies();

    // Buffer arrays for storing info from the dT side.
    // dT modifies these arrays; kT uses them only.

    // kT gets entity locations and rotations from dT
    // The voxel ID
    DeviceArray<voxelID_t> voxelID_buffer = DeviceArray<voxelID_t>(&m_approxDeviceBytesUsed);
    // The XYZ local location inside a voxel
    DeviceArray<subVoxelPos_t> locX_buffer = DeviceArray<subVoxelPos_t>(&m_approxDeviceBytesUsed);
    DeviceArray<subVoxelPos_t> locY_buffer = DeviceArray<subVoxelPos_t>(&m_approxDeviceBytesUsed);
    DeviceArray<subVoxelPos_t> locZ_buffer = DeviceArray<subVoxelPos_t>(&m_approxDeviceBytesUsed);
    // The clump quaternion
    DeviceArray<oriQ_t> oriQ0_buffer = DeviceArray<oriQ_t>(&m_approxDeviceBytesUsed);
    DeviceArray<oriQ_t> oriQ1_buffer = DeviceArray<oriQ_t>(&m_approxDeviceBytesUsed);
    DeviceArray<oriQ_t> oriQ2_buffer = DeviceArray<oriQ_t>(&m_approxDeviceBytesUsed);
    DeviceArray<oriQ_t> oriQ3_buffer = DeviceArray<oriQ_t>(&m_approxDeviceBytesUsed);
    DeviceArray<family_t> familyID_buffer = DeviceArray<family_t>(&m_approxDeviceBytesUsed);
    // Triangle-related, for mesh deformation
    DeviceArray<float3> relPosNode1_buffer = DeviceArray<float3>(&m_approxDeviceBytesUsed);
    DeviceArray<float3> relPosNode2_buffer = DeviceArray<float3>(&m_approxDeviceBytesUsed);
    DeviceArray<float3> relPosNode3_buffer = DeviceArray<float3>(&m_approxDeviceBytesUsed);
    // Max vel of entities
    DeviceArray<float> absVel_buffer = DeviceArray<float>(&m_approxDeviceBytesUsed);

    // kT's copy of family map
    // std::unordered_map<unsigned int, family_t> familyUserImplMap;
    // std::unordered_map<family_t, unsigned int> familyImplUserMap;

    // Managed arrays for dT's personal use (not transfer buffer)

    // Those are the template array of the unique component information. Note that these arrays may be thousands-element
    // long, and only a part of it is jitified. The jitified part of it is typically the frequently used clump and maybe
    // triangle tempates; the other part may be the components for a few large clump bodies which are not frequently
    // used.
    // Component sphere's radius
    DualArray<float> radiiSphere = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // The distinct sphere local position (wrt CoM) values
    DualArray<float> relPosSphereX = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> relPosSphereY = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> relPosSphereZ = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Triangles (templates) are given a special place (unlike other analytical shapes), b/c we expect them to appear
    // frequently as meshes.
    DualArray<float3> relPosNode1 = DualArray<float3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float3> relPosNode2 = DualArray<float3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float3> relPosNode3 = DualArray<float3>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // External object's components may need the following arrays to store some extra defining features of them. We
    // assume there are usually not too many of them in a simulation.
    // Relative position w.r.t. the owner. For example, the following 3 arrays may hold center points for plates, or tip
    // positions for cones.
    DualArray<float> relPosEntityX = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> relPosEntityY = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> relPosEntityZ = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Some orientation specifiers. For example, the following 3 arrays may hold normal vectors for planes, or center
    // axis vectors for cylinders.
    DualArray<float> oriEntityX = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> oriEntityY = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> oriEntityZ = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // Some size specifiers. For example, the following 3 arrays may hold top, bottom and length information for finite
    // cylinders.
    DualArray<float> sizeEntity1 = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> sizeEntity2 = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<float> sizeEntity3 = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The voxel ID (split into 3 parts, representing XYZ location)
    DualArray<voxelID_t> voxelID = DualArray<voxelID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The XYZ local location inside a voxel
    DualArray<subVoxelPos_t> locX = DualArray<subVoxelPos_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<subVoxelPos_t> locY = DualArray<subVoxelPos_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<subVoxelPos_t> locZ = DualArray<subVoxelPos_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The clump quaternion
    DualArray<oriQ_t> oriQw = DualArray<oriQ_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<oriQ_t> oriQx = DualArray<oriQ_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<oriQ_t> oriQy = DualArray<oriQ_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<oriQ_t> oriQz = DualArray<oriQ_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // dT-supplied system velocity
    DualArray<float> marginSize = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Clump's family identification code. Used in determining whether they can be contacts between two families, and
    // whether a family has prescribed motions.
    DualArray<family_t> familyID = DualArray<family_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // A long array (usually 32640 elements) registering whether between 2 families there should be contacts
    DualArray<notStupidBool_t> familyMaskMatrix =
        DualArray<notStupidBool_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The amount of contact margin that each family should add to its associated contact geometries. Default is 0, and
    // that means geometries should be considered in contact when they are physically in contact.
    DualArray<float> familyExtraMarginSize = DualArray<float>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // kT computed contact pair info
    DualArray<bodyID_t> idGeometryA = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<bodyID_t> idGeometryB = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<contact_t> contactType = DualArray<contact_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Contact pair info at the previous time step. This is needed by dT so enduring contacts are identified in
    // history-based models.
    DualArray<bodyID_t> previous_idGeometryA = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<bodyID_t> previous_idGeometryB = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<contact_t> previous_contactType = DualArray<contact_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<contactPairs_t> contactMapping =
        DualArray<contactPairs_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // Sphere-related arrays
    // Owner body ID of this component
    DualArray<bodyID_t> ownerClumpBody = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    DualArray<bodyID_t> ownerMesh = DualArray<bodyID_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // The ID that maps this sphere component's geometry-defining parameters, when this component is jitified
    DualArray<clumpComponentOffset_t> clumpComponentOffset =
        DualArray<clumpComponentOffset_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // The ID that maps this sphere component's geometry-defining parameters, when this component is not jitified (too
    // many templates)
    DualArray<clumpComponentOffsetExt_t> clumpComponentOffsetExt =
        DualArray<clumpComponentOffsetExt_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);
    // The ID that maps this analytical entity component's geometry-defining parameters, when this component is jitified
    // DualArray<clumpComponentOffset_t> analComponentOffset;

    // Records if this contact is persistent and serves as kT's work array on treating their persistency.
    DualArray<notStupidBool_t> contactPersistency =
        DualArray<notStupidBool_t>(&m_approxHostBytesUsed, &m_approxDeviceBytesUsed);

    // kT's timers
    std::vector<std::string> timer_names = {"Discretize domain",      "Find contact pairs", "Build history map",
                                            "Unpack updates from dT", "Send to dT buffer",  "Wait for dT update"};
    SolverTimers timers = SolverTimers(timer_names);

    kTStateParams stateParams;

  public:
    friend class DEMSolver;
    friend class DEMDynamicThread;

    DEMKinematicThread(WorkerReportChannel* pPager, ThreadManager* pSchedSup, const GpuManager::StreamInfo& sInfo)
        : pPagerToMain(pPager), pSchedSupport(pSchedSup), streamInfo(sInfo) {
        pPagerToMain->userCallDone = false;
        pSchedSupport->kinematicShouldJoin = false;
        pSchedSupport->kinematicStarted = false;

        // I found creating the stream here is needed (rather than creating it in the child thread).
        // This is because in smaller problems, the array data transfer portion (which needs the stream) could even be
        // reached before the stream is created in the child thread. So we have to create the stream here before
        // spawning the child thread.
        DEME_GPU_CALL(cudaStreamCreate(&streamInfo.stream));

        // Launch a worker thread bound to this instance
        th = std::move(std::thread([this]() { this->workerThread(); }));
    }
    ~DEMKinematicThread() {
        // std::cout << "Kinematic thread closing..." << std::endl;

        // kT as the aux helper thread, it could be hanging waiting for dT updates when the destructor is called, so we
        // release it from that status here
        breakWaitingStatus();

        // It says start thread but it's actually just telling it to join
        pSchedSupport->kinematicShouldJoin = true;
        startThread();
        th.join();

        cudaStreamDestroy(streamInfo.stream);

        // deallocateEverything();
    }

    // buffer exchange methods
    void setDestinationBufferPointers();

    // Break inner loop hanging status and wait in the outer loop. Note we must ensure resetUserCallStat is called
    // shortly after breakWaitingStatus is called, since kinematicOwned_Cons2ProdBuffer_isFresh and kTShouldReset can be
    // vulnerable if kT exited through dynamicsDone rather than control variable-based release.
    void breakWaitingStatus();

    // Called each time when the user calls DoDynamicsThenSync.
    void startThread();

    // The actual kernel things go here.
    // It is called upon construction.
    void workerThread();

    /// Reset kT--dT interaction coordinator stats
    void resetUserCallStat();
    /// Return the approximate RAM usage
    size_t estimateDeviceMemUsage() const;
    size_t estimateHostMemUsage() const;

    /// Resize arrays
    void allocateGPUArrays(size_t nOwnerBodies,
                           size_t nOwnerClumps,
                           unsigned int nExtObj,
                           size_t nTriMeshes,
                           size_t nSpheresGM,
                           size_t nTriGM,
                           unsigned int nAnalGM,
                           size_t nExtraContacts,
                           unsigned int nMassProperties,
                           unsigned int nClumpTopo,
                           unsigned int nClumpComponents,
                           unsigned int nJitifiableClumpComponents,
                           unsigned int nMatTuples);

    // initGPUArrays's components
    void registerPolicies(const std::vector<notStupidBool_t>& family_mask_matrix);
    void populateEntityArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                              const std::vector<unsigned int>& input_ext_obj_family,
                              const std::vector<unsigned int>& input_mesh_obj_family,
                              const std::vector<unsigned int>& input_mesh_facet_owner,
                              const std::vector<DEMTriangle>& input_mesh_facets,
                              const ClumpTemplateFlatten& clump_templates,
                              size_t nExistOwners,
                              size_t nExistSpheres,
                              size_t nExistingFacets);

    /// Initialize arrays
    void initGPUArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                       const std::vector<unsigned int>& input_ext_obj_family,
                       const std::vector<unsigned int>& input_mesh_obj_family,
                       const std::vector<unsigned int>& input_mesh_facet_owner,
                       const std::vector<DEMTriangle>& input_mesh_facets,
                       const std::vector<notStupidBool_t>& family_mask_matrix,
                       const ClumpTemplateFlatten& clump_templates);

    /// Add more clumps and/or meshes into the system, without re-initialization. It must be clump/mesh-addition only,
    /// no other changes to the system.
    void updateClumpMeshArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                               const std::vector<unsigned int>& input_ext_obj_family,
                               const std::vector<unsigned int>& input_mesh_obj_family,
                               const std::vector<unsigned int>& input_mesh_facet_owner,
                               const std::vector<DEMTriangle>& input_mesh_facets,
                               const std::vector<notStupidBool_t>& family_mask_matrix,
                               const ClumpTemplateFlatten& clump_templates,
                               size_t nExistingOwners,
                               size_t nExistingClumps,
                               size_t nExistingSpheres,
                               size_t nExistingTriMesh,
                               size_t nExistingFacets,
                               unsigned int nExistingObj,
                               unsigned int nExistingAnalGM);

    /// Set SimParams items
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
                      float3 user_box_min,
                      float3 user_box_max,
                      float3 G,
                      double ts_size,
                      float expand_factor,
                      float approx_max_vel,
                      float expand_safety_param,
                      float expand_safety_adder,
                      const std::set<std::string>& contact_wildcards,
                      const std::set<std::string>& owner_wildcards,
                      const std::set<std::string>& geo_wildcards);

    // Put sim data array pointers in place
    void packDataPointers();
    void packTransferPointers(DEMDynamicThread*& dT);

    // Move array data to or from device
    void migrateDataToDevice();
    // void migrateDataToHost();

    // Sync my stream
    void syncMemoryTransfer() { DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream)); }

    /// Return timing inforation for this current run
    void getTiming(std::vector<std::string>& names, std::vector<double>& vals);

    /// Reset the timers
    void resetTimers() {
        for (const auto& name : timer_names) {
            timers.GetTimer(name).reset();
        }
    }

    /// Change all entities with (user-level) family number ID_from to have a new number ID_to
    void changeFamily(unsigned int ID_from, unsigned int ID_to);

    /// Change radii and relPos info of these owners (if these owners are clumps)
    void changeOwnerSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors);

    // Jitify kT kernels (at initialization) based on existing knowledge of this run
    void jitifyKernels(const std::unordered_map<std::string, std::string>& Subs,
                       const std::vector<std::string>& JitifyOptions);

    /// Set this owner's family number, for n consecutive items.
    void setOwnerFamily(bodyID_t ownerID, family_t fam, bodyID_t n = 1);

    /// Rewrite the relative positions of the flattened triangle soup, starting from `start', using triangle nodal
    /// positions in `triangles'.
    void setTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& triangles);
    /// Rewrite the relative positions of the flattened triangle soup, starting from `start', using triangle nodal
    /// positions in `triangles', by the amount stipulated in updates.
    void updateTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& updates);

    /// Update (overwrite) kT's previous contact array based on input
    void updatePrevContactArrays(DualStruct<DEMDataDT>& dT_data, size_t nContacts);

    /// Print temporary arrays' memory usage. This is for debugging purposes only.
    void printScratchSpaceUsage() const {
        std::cout << Name << " scratch space usage: " << std::endl;
        solverScratchSpace.printVectorUsage();
    }

  private:
    const std::string Name = "kT";

    // Bring kT buffer array data to its working arrays
    inline void unpackMyBuffer();
    // Send produced data to dT-owned biffers
    void sendToTheirBuffer();
    // Resize dT's buffer arrays based on the number of contact pairs
    inline void transferArraysResize(size_t nContactPairs);
    // Automatic adjustments to sim params
    void calibrateParams();
    // The kT-side allocations that can be done at initialization time
    void initAllocation();
    // Deallocate everything
    void deallocateEverything();

    // Just-in-time compiled kernels
    // jitify::Program bin_sphere_kernels = JitHelper::buildProgram("bin_sphere_kernels", " ");
    std::shared_ptr<jitify::Program> bin_sphere_kernels;
    std::shared_ptr<jitify::Program> bin_triangle_kernels;
    std::shared_ptr<jitify::Program> sphTri_contact_kernels;
    std::shared_ptr<jitify::Program> sphere_contact_kernels;
    std::shared_ptr<jitify::Program> history_kernels;
    std::shared_ptr<jitify::Program> misc_kernels;

    // Adjuster for bin size
    class AccumTimer {
      private:
        double prev_time = DEME_HUGE_FLOAT;
        unsigned int cached_count = 0;
        Timer<double> timer;

      public:
        AccumTimer() { timer = Timer<double>(); }
        ~AccumTimer() {}
        void Begin() { timer.start(); }
        void End() {
            timer.stop();
            cached_count++;
        }

        double GetPrevTime() { return prev_time; }

        void Query(double& prev, double& curr) {
            double avg_time = timer.GetTimeSeconds() / (double)(cached_count);
            prev = prev_time;
            curr = avg_time;
            // Record the time for this run
            prev_time = avg_time;
            timer.reset();
            cached_count = 0;
        }

        bool QueryOn(double& prev, double& curr, unsigned int n) {
            if (cached_count >= n) {
                Query(prev, curr);
                return true;
            } else {
                return false;
            }
        }

        // Return this accumulator to initial state
        void Clear() {
            prev_time = -1.0;
            cached_count = 0;
            timer.reset();
        }
    };

    AccumTimer CDAccumTimer = AccumTimer();

    // A collection of migrate-to-host methods. Bulk migrate-to-host is by nature on-demand only.
    void migrateFamilyToHost();
    void migrateDeviceModifiableInfoToHost();

};  // kT ends

}  // namespace deme

#endif
