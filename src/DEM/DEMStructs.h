//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#ifndef SGPS_DEM_HOST_STRUCTS
#define SGPS_DEM_HOST_STRUCTS

#include <DEM/DEMDefines.h>
#include <core/utils/ManagedAllocator.hpp>
#include <sstream>
#include <exception>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <nvmath/helper_math.cuh>

namespace sgps {
// Structs defined here will be used by some host classes in DEM.
// NOTE: Data structs here need to be those complex ones (such as needing to include ManagedAllocator.hpp), which may
// not be jitifiable.

/// <summary>
/// DEMSolverStateData contains information that pertains the DEM solver dT thread, at a certain point in time. It also
/// contains space allocated as system scratch pad and as thread temporary arrays.
/// </summary>
class DEMSolverStateDataDT {
  private:
    // The vector used by CUB or by anybody else that needs scratch space.
    // Please pay attention to the type the vector stores.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadScratchSpace;

    // The vectors used by threads when they need temporary arrays (very typically, for storing arrays outputted by cub
    // scan or reduce operations).
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector1;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector2;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector3;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector4;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector5;

    // The vectors used to cache some array (typically the result of some pre- or post-processing) which can potentially
    // be used across iterations.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadCachedOwner;

  public:
    // Temp size_t variables that can be reused
    size_t* pTempSizeVar1;

    // Number of contacts in this CD step
    size_t* pNumContacts;
    // Number of contacts in the previous CD step
    size_t* pNumPrevContacts;

    DEMSolverStateDataDT() {
        cudaMallocManaged(&pTempSizeVar1, sizeof(size_t));
        cudaMallocManaged(&pNumContacts, sizeof(size_t));
        cudaMallocManaged(&pNumPrevContacts, sizeof(size_t));
        *pNumContacts = 0;
        *pNumPrevContacts = 0;
    }
    ~DEMSolverStateDataDT() {
        cudaFree(pTempSizeVar1);
        cudaFree(pNumContacts);
        cudaFree(pNumPrevContacts);
    }

    // Return raw pointer to swath of device memory that is at least "sizeNeeded" large
    inline scratch_t* allocateScratchSpace(size_t sizeNeeded) {
        if (threadScratchSpace.size() < sizeNeeded) {
            threadScratchSpace.resize(sizeNeeded);
        }
        return threadScratchSpace.data();
    }

    // TODO: Better way to write this???
    inline scratch_t* allocateTempVector1(size_t sizeNeeded) {
        if (threadTempVector1.size() < sizeNeeded) {
            threadTempVector1.resize(sizeNeeded);
        }
        return threadTempVector1.data();
    }
    inline scratch_t* allocateTempVector2(size_t sizeNeeded) {
        if (threadTempVector2.size() < sizeNeeded) {
            threadTempVector2.resize(sizeNeeded);
        }
        return threadTempVector2.data();
    }
    inline scratch_t* allocateTempVector3(size_t sizeNeeded) {
        if (threadTempVector3.size() < sizeNeeded) {
            threadTempVector3.resize(sizeNeeded);
        }
        return threadTempVector3.data();
    }
    inline scratch_t* allocateTempVector4(size_t sizeNeeded) {
        if (threadTempVector4.size() < sizeNeeded) {
            threadTempVector4.resize(sizeNeeded);
        }
        return threadTempVector4.data();
    }
    inline scratch_t* allocateTempVector5(size_t sizeNeeded) {
        if (threadTempVector5.size() < sizeNeeded) {
            threadTempVector5.resize(sizeNeeded);
        }
        return threadTempVector5.data();
    }

    inline scratch_t* allocateCachedOwner(size_t sizeNeeded) {
        if (threadCachedOwner.size() < sizeNeeded) {
            threadCachedOwner.resize(sizeNeeded);
        }
        return threadCachedOwner.data();
    }
};

/// <summary>
/// DEMSolverStateData contains information that pertains the DEM solver kT thread, at a certain point in time. It also
/// contains space allocated as system scratch pad and as thread temporary arrays.
/// </summary>
class DEMSolverStateDataKT {
  private:
    // The vector used by CUB or by anybody else that needs scratch space.
    // Please pay attention to the type the vector stores.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadScratchSpace;

    // The vectors used by threads when they need temporary arrays (very typically, for storing arrays outputted by cub
    // scan or reduce operations).
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector1;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector2;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector3;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector4;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector5;
    std::vector<scratch_t, ManagedAllocator<scratch_t>> threadTempVector6;
    // In theory you can keep going and invent more vectors here. But I feel these I have here are just enough for me to
    // use conveniently.

  public:
    // Temp size_t variables that can be reused
    size_t* pTempSizeVar1;
    size_t* pTempSizeVar2;

    // Number of contacts in this CD step
    size_t* pNumContacts;
    // Number of contacts in the previous CD step
    size_t* pNumPrevContacts;
    // Number of spheres in the previous CD step (in case user added/removed clumps from the system)
    size_t* pNumPrevSpheres;

    DEMSolverStateDataKT() {
        cudaMallocManaged(&pNumContacts, sizeof(size_t));
        cudaMallocManaged(&pTempSizeVar1, sizeof(size_t));
        cudaMallocManaged(&pTempSizeVar2, sizeof(size_t));
        cudaMallocManaged(&pNumPrevContacts, sizeof(size_t));
        cudaMallocManaged(&pNumPrevSpheres, sizeof(size_t));
        *pNumContacts = 0;
        *pNumPrevContacts = 0;
        *pNumPrevSpheres = 0;
    }
    ~DEMSolverStateDataKT() {
        cudaFree(pNumContacts);
        cudaFree(pTempSizeVar1);
        cudaFree(pTempSizeVar2);
        cudaFree(pNumPrevContacts);
        cudaFree(pNumPrevSpheres);
    }

    // Return raw pointer to swath of device memory that is at least "sizeNeeded" large
    inline scratch_t* allocateScratchSpace(size_t sizeNeeded) {
        if (threadScratchSpace.size() < sizeNeeded) {
            threadScratchSpace.resize(sizeNeeded);
        }
        return threadScratchSpace.data();
    }

    // TODO: Better way to write this???
    inline scratch_t* allocateTempVector1(size_t sizeNeeded) {
        if (threadTempVector1.size() < sizeNeeded) {
            threadTempVector1.resize(sizeNeeded);
        }
        return threadTempVector1.data();
    }
    inline scratch_t* allocateTempVector2(size_t sizeNeeded) {
        if (threadTempVector2.size() < sizeNeeded) {
            threadTempVector2.resize(sizeNeeded);
        }
        return threadTempVector2.data();
    }
    inline scratch_t* allocateTempVector3(size_t sizeNeeded) {
        if (threadTempVector3.size() < sizeNeeded) {
            threadTempVector3.resize(sizeNeeded);
        }
        return threadTempVector3.data();
    }
    inline scratch_t* allocateTempVector4(size_t sizeNeeded) {
        if (threadTempVector4.size() < sizeNeeded) {
            threadTempVector4.resize(sizeNeeded);
        }
        return threadTempVector4.data();
    }
    inline scratch_t* allocateTempVector5(size_t sizeNeeded) {
        if (threadTempVector5.size() < sizeNeeded) {
            threadTempVector5.resize(sizeNeeded);
        }
        return threadTempVector5.data();
    }
    inline scratch_t* allocateTempVector6(size_t sizeNeeded) {
        if (threadTempVector6.size() < sizeNeeded) {
            threadTempVector6.resize(sizeNeeded);
        }
        return threadTempVector6.data();
    }
};

inline std::string pretty_format_bytes(size_t bytes) {
    // set up byte prefixes
    constexpr size_t KIBI = 1024;
    constexpr size_t MEBI = KIBI * KIBI;
    constexpr size_t GIBI = KIBI * KIBI * KIBI;
    float gibival = float(bytes) / GIBI;
    float mebival = float(bytes) / MEBI;
    float kibival = float(bytes) / KIBI;
    std::stringstream ret;
    if (gibival > 1) {
        ret << gibival << " GiB";
    } else if (mebival > 1) {
        ret << mebival << " MiB";
    } else if (kibival > 1) {
        ret << kibival << " KiB";
    } else {
        ret << bytes << " B";
    }
    return ret.str();
}

// =============================================================================
// NOW DEFINING MACRO COMMANDS USED BY THE DEM MODULE
// =============================================================================

#define SGPS_DEM_PRINTF(...)                    \
    {                                           \
        if (verbosity > DEM_VERBOSITY::QUIET) { \
            printf(__VA_ARGS__);                \
            printf("\n");                       \
        }                                       \
    }

#define SGPS_DEM_ERROR(...)                  \
    {                                        \
        char error_message[256];             \
        char func_name[256];                 \
        sprintf(error_message, __VA_ARGS__); \
        sprintf(func_name, __func__);        \
        std::string out = error_message;     \
        out += "\n";                         \
        out += "This happened in ";          \
        out += func_name;                    \
        out += ".\n";                        \
        throw std::runtime_error(out);       \
    }

#define SGPS_DEM_WARNING(...)                      \
    {                                              \
        if (verbosity >= DEM_VERBOSITY::WARNING) { \
            printf("\nWARNING! ");                 \
            printf(__VA_ARGS__);                   \
            printf("\n\n");                        \
        }                                          \
    }

#define SGPS_DEM_INFO(...)                      \
    {                                           \
        if (verbosity >= DEM_VERBOSITY::INFO) { \
            printf(__VA_ARGS__);                \
            printf("\n");                       \
        }                                       \
    }

#define SGPS_DEM_INFO_STEP_STATS(...)                      \
    {                                                      \
        if (verbosity >= DEM_VERBOSITY::INFO_STEP_STATS) { \
            printf(__VA_ARGS__);                           \
            printf("\n");                                  \
        }                                                  \
    }

#define SGPS_DEM_INFO_STEP_WARN(...)                      \
    {                                                     \
        if (verbosity >= DEM_VERBOSITY::INFO_STEP_WARN) { \
            printf(__VA_ARGS__);                          \
            printf("\n");                                 \
        }                                                 \
    }

#define SGPS_DEM_DEBUG_PRINTF(...)               \
    {                                            \
        if (verbosity >= DEM_VERBOSITY::DEBUG) { \
            printf(__VA_ARGS__);                 \
            printf("\n");                        \
        }                                        \
    }

#define SGPS_DEM_DEBUG_EXEC(...)                 \
    {                                            \
        if (verbosity >= DEM_VERBOSITY::DEBUG) { \
            __VA_ARGS__;                         \
        }                                        \
    }

#define SGPS_DEM_TRACKED_RESIZE_NOPRINT(vec, newsize)          \
    {                                                          \
        size_t item_size = sizeof(decltype(vec)::value_type);  \
        size_t old_size = vec.size();                          \
        vec.resize(newsize);                                   \
        size_t new_size = vec.size();                          \
        size_t byte_delta = item_size * (new_size - old_size); \
        m_approx_bytes_used += byte_delta;                     \
    }

#define SGPS_DEM_TRACKED_RESIZE(vec, newsize, name, val)                                                          \
    {                                                                                                             \
        size_t item_size = sizeof(decltype(vec)::value_type);                                                     \
        size_t old_size = vec.size();                                                                             \
        vec.resize(newsize, val);                                                                                 \
        size_t new_size = vec.size();                                                                             \
        size_t byte_delta = item_size * (new_size - old_size);                                                    \
        m_approx_bytes_used += byte_delta;                                                                        \
        SGPS_DEM_INFO_STEP_STATS("Resizing vector %s, old size %zu, new size %zu, byte delta %s", name, old_size, \
                                 new_size, pretty_format_bytes(byte_delta).c_str());                              \
    }

// =============================================================================
// NOW SOME HOST-SIDE SIMPLE STRUCTS USED BY THE DEM MODULE
// =============================================================================

// Manager of the collabortation between the main thread and worker threads
class WorkerReportChannel {
  public:
    std::atomic<bool> userCallDone;
    std::mutex mainCanProceed;
    std::condition_variable cv_mainCanProceed;
    WorkerReportChannel() noexcept { userCallDone = false; }
    ~WorkerReportChannel() {}
};

struct familyPrescription_t {
    unsigned int family;
    std::string linPosX = "none";
    std::string linPosY = "none";
    std::string linPosZ = "none";
    std::string linVelX = "none";
    std::string linVelY = "none";
    std::string linVelZ = "none";

    std::string oriQ = "none";
    std::string rotVelX = "none";
    std::string rotVelY = "none";
    std::string rotVelZ = "none";
    // Is this prescribed motion dictating the motion of the entities (true), or just added on top of the true
    // physics (false)
    bool linVelPrescribed = false;
    bool rotVelPrescribed = false;
    bool rotPosPrescribed = false;
    bool linPosPrescribed = false;
    // This family will receive external updates of velocity and position (overwrites analytical prescription)
    bool externVel = false;
    bool externPos = false;
    // A switch to mark if there is any prescription going on for this family at all
    bool used = false;
};

struct familyPair_t {
    unsigned int ID1;
    unsigned int ID2;
};

struct SolverFlags {
    // Sort contact pair arrays before sending to kT
    bool should_sort_pairs = true;
    // Whether to adopt a contact force calculation strategy where a thread takes care of multiple contacts so shared
    // memory is leveraged
    // NOTE: This is not implemented
    bool use_compact_force_kernel = false;
    // This run is historyless
    bool isHistoryless = false;
    // This run uses contact detection in an async fashion (kT and dT working at different points in simulation time)
    bool isAsync = true;
    // If family number can potentially change (at each time step) during the simulation, because of user intervention
    bool canFamilyChange = false;
    // Some output-related flags
    unsigned int outputFlags = DEM_OUTPUT_CONTENT::QUAT | DEM_OUTPUT_CONTENT::ABSV;
};

struct DEMMaterial {
    float rho = -1.f;  // Density
    float E = 1e8;     // Young's modulus
    float nu = 0.3;    // Poission's ratio
    float CoR = 0.5;   // Coeff of Restitution
    float mu = 0.5;    // Static friction coeff
    float Crr = 0.01;  // Rolling resistance coeff
};

// A struct that defines a `clump' (one of the core concepts of this solver). A clump is typically small which consists
// of several sphere components, but it can be as large as having thousands of spheres.
struct DEMClumpTemplate {
    float mass;
    float3 MOI;
    std::vector<float> radii;
    std::vector<float3> relPos;
    std::vector<std::shared_ptr<DEMMaterial>> materials;
    unsigned int nComp = 0;  // Number of components
    // Position of this clump's CoM, in the frame which is used to report the positions of this clump's component
    // spheres. It is usually all 0, unless the user specifies it, in which case we need to process relPos such that
    // when the system is initialized, everything is still in the clump's CoM frame.
    float3 CoM = make_float3(0);
    // CoM frame's orientation quaternion in the frame which is used to report the positions of this clump's component
    // spheres. Usually unit quaternion.
    float4 CoM_oriQ = make_float4(1.f, 0.f, 0.f, 0.f);
    // Each clump template will have a unique mark number. When clumps are loaded to the system, this mark will help
    // find their type offset.
    unsigned int mark;
    // Whether this is a big clump (not used; jitifiability is determined automatically)
    bool isBigClump = false;
};

// A struct to get or set tracked owner entities
struct DEMTrackedObj {
    // ownerID will be updated by dT on initialization
    bodyID_t ownerID = DEM_NULL_BODYID;
    DEM_ENTITY_TYPE type;
    // A tracker tracks a owner loaded into the system via its respective loading method, so load_order registers
    // the position of this object in the corresponding API-side array
    size_t load_order;
};

}  // namespace sgps

#endif
