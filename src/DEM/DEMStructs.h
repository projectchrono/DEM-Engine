//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef SGPS_DEM_HOST_STRUCTS
#define SGPS_DEM_HOST_STRUCTS

#include <DEM/DEMDefines.h>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/ManagedMemory.hpp>
#include <core/utils/csv.hpp>
#include <core/utils/GpuError.h>
#include <core/utils/Timer.hpp>

#include <sstream>
#include <exception>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <nvmath/helper_math.cuh>
#include <DEM/HostSideHelpers.hpp>

namespace sgps {
// Structs defined here will be used by some host classes in DEM.
// NOTE: Data structs here need to be those complex ones (such as needing to include ManagedAllocator.hpp), which may
// not be jitifiable.

/// <summary>
/// DEMSolverStateData contains information that pertains the DEM solver worker threads, at a certain point in time. It
/// also contains space allocated as system scratch pad and as thread temporary arrays.
/// </summary>
class DEMSolverStateData {
  private:
    const unsigned int numTempArrays;
    // The vector used by CUB or by anybody else that needs scratch space.
    // Please pay attention to the type the vector stores.
    std::vector<scratch_t, ManagedAllocator<scratch_t>> cubScratchSpace;

    // The vectors used by threads when they need temporary arrays (very typically, for storing arrays outputted by cub
    // scan or reduce operations).
    std::vector<std::vector<scratch_t, ManagedAllocator<scratch_t>>,
                ManagedAllocator<std::vector<scratch_t, ManagedAllocator<scratch_t>>>>
        threadTempVectors;
    // You can keep more temp arrays if you construct this class with a different initializer

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

    DEMSolverStateData(unsigned int nArrays) : numTempArrays(nArrays) {
        GPU_CALL(cudaMallocManaged(&pNumContacts, sizeof(size_t)));
        GPU_CALL(cudaMallocManaged(&pTempSizeVar1, sizeof(size_t)));
        GPU_CALL(cudaMallocManaged(&pTempSizeVar2, sizeof(size_t)));
        GPU_CALL(cudaMallocManaged(&pNumPrevContacts, sizeof(size_t)));
        GPU_CALL(cudaMallocManaged(&pNumPrevSpheres, sizeof(size_t)));
        *pNumContacts = 0;
        *pNumPrevContacts = 0;
        *pNumPrevSpheres = 0;
        threadTempVectors.resize(numTempArrays);
    }
    ~DEMSolverStateData() {
        GPU_CALL(cudaFree(pNumContacts));
        GPU_CALL(cudaFree(pTempSizeVar1));
        GPU_CALL(cudaFree(pTempSizeVar2));
        GPU_CALL(cudaFree(pNumPrevContacts));
        GPU_CALL(cudaFree(pNumPrevSpheres));
    }

    // Return raw pointer to swath of device memory that is at least "sizeNeeded" large
    inline scratch_t* allocateScratchSpace(size_t sizeNeeded) {
        if (cubScratchSpace.size() < sizeNeeded) {
            cubScratchSpace.resize(sizeNeeded);
        }
        return cubScratchSpace.data();
    }

    inline scratch_t* allocateTempVector(unsigned int i, size_t sizeNeeded) {
        if (threadTempVectors.at(i).size() < sizeNeeded) {
            threadTempVectors.at(i).resize(sizeNeeded);
        }
        return threadTempVectors.at(i).data();
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
        }                                       \
    }

#define SGPS_DEM_ERROR(...)                  \
    {                                        \
        char error_message[1024];            \
        char func_name[1024];                \
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

#define SGPS_DEM_STEP_STATS(...)                      \
    {                                                 \
        if (verbosity >= DEM_VERBOSITY::STEP_STATS) { \
            printf(__VA_ARGS__);                      \
            printf("\n");                             \
        }                                             \
    }

#define SGPS_DEM_STEP_METRIC(...)                      \
    {                                                  \
        if (verbosity >= DEM_VERBOSITY::STEP_METRIC) { \
            printf(__VA_ARGS__);                       \
            printf("\n");                              \
        }                                              \
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

#define SGPS_DEM_TRACKED_RESIZE(vec, newsize, name, val)                                                               \
    {                                                                                                                  \
        size_t item_size = sizeof(decltype(vec)::value_type);                                                          \
        size_t old_size = vec.size();                                                                                  \
        vec.resize(newsize, val);                                                                                      \
        size_t new_size = vec.size();                                                                                  \
        size_t byte_delta = item_size * (new_size - old_size);                                                         \
        m_approx_bytes_used += byte_delta;                                                                             \
        SGPS_DEM_STEP_STATS("Resizing vector %s, old size %zu, new size %zu, byte delta %s", name, old_size, new_size, \
                            pretty_format_bytes(byte_delta).c_str());                                                  \
    }

//// TODO: this is currently not tracked...
// ptr being a reference to a pointer is crucial
template <typename T>
inline void SGPS_DEM_DEVICE_PTR_ALLOC(T*& ptr, size_t size) {
    cudaPointerAttributes attrib;
    GPU_CALL(cudaPointerGetAttributes(&attrib, ptr));

    if (attrib.type != cudaMemoryType::cudaMemoryTypeUnregistered)
        GPU_CALL(cudaFree(ptr));
    GPU_CALL(cudaMalloc((void**)&ptr, size * sizeof(T)));
}

// Managed advise doesn't seem to do anything...
#define SGPS_DEM_ADVISE_DEVICE(vec, device) \
    { advise(vec, ManagedAdvice::PREFERRED_LOC, device); }
#define SGPS_DEM_MIGRATE_TO_DEVICE(vec, device, stream) \
    { migrate(vec, device, stream); }

// #define DEM_OUTPUT_IF_GPU_FAILS(res) \
//     { gpu_assert((res), __FILE__, __LINE__, false); throw std::runtime_error("GPU Assertion Failed!");}

// =============================================================================
// NOW SOME HOST-SIDE SIMPLE STRUCTS USED BY THE DEM MODULE
// =============================================================================

// Timers used by kT and dT
class SolverTimers {
  private:
    const unsigned int num_timers;
    std::unordered_map<std::string, Timer<double>> m_timers;

  public:
    SolverTimers(const std::vector<std::string>& names) : num_timers(names.size()) {
        for (unsigned int i = 0; i < num_timers; i++) {
            m_timers[names.at(i)] = Timer<double>();
        }
    }
    Timer<double>& GetTimer(const std::string& name) { return m_timers.at(name); }
};

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

enum class DEM_VAR_TS_STRAT { CONST, MAX_VEL, INT_GAP };

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
    // Time step constant-ness and expand factor constant-ness
    bool isStepConst = true;
    bool isExpandFactorFixed = true;
    // The strategy for selecting the variable time step size
    DEM_VAR_TS_STRAT stepSizeStrat = DEM_VAR_TS_STRAT::CONST;
    // Whether instructed to use jitification for mass properties and clump components (default to no and it is
    // recommended)
    bool useClumpJitify = false;
    bool useMassJitify = false;
    // Contact detection uses a thread for a bin, not a block for a bin
    bool useOneBinPerThread = false;
};

struct DEMMaterial {
    float E = 1e8;    // Young's modulus
    float nu = 0.3;   // Poission's ratio
    float CoR = 0.5;  // Coeff of Restitution
    float mu = 0.5;   // Static friction coeff
    float Crr = 0.0;  // Rolling resistance coeff
};

class DEMTriangle {
  public:
    DEMTriangle(float3 pnt1, float3 pnt2, float3 pnt3) : p1(pnt1), p2(pnt2), p3(pnt3) {}
    DEMTriangle() {}
    ~DEMTriangle() {}
    float3 p1;
    float3 p2;
    float3 p3;
};

// A struct that defines a `clump' (one of the core concepts of this solver). A clump is typically small which consists
// of several sphere components, but it can be as large as having thousands of spheres.
class DEMClumpTemplate {
  public:
    float mass = 0;
    float3 MOI = make_float3(0);
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
    float4 CoM_oriQ = host_make_float4(1, 0, 0, 0);
    // Each clump template will have a unique mark number. When clumps are loaded to the system, this mark will help
    // find their type offset.
    unsigned int mark;
    // Whether this is a big clump (not used; jitifiability is determined automatically)
    bool isBigClump = false;
    // A name given by the user. It will be outputted to file to indicate the type of a clump.
    std::string m_name = "NULL";

    /// Retrieve clump's sphere component information from a file
    int ReadComponentFromFile(const std::string filename,
                              const std::string x_id = "x",
                              const std::string y_id = "y",
                              const std::string z_id = "z",
                              const std::string r_id = "r") {
        io::CSVReader<4> in(filename);
        in.read_header(io::ignore_missing_column, x_id, y_id, z_id, r_id);
        float r = 0;
        float3 pos;
        unsigned int count = 0;
        while (in.read_row(pos.x, pos.y, pos.z, r)) {
            radii.push_back(r);
            relPos.push_back(pos);
            count++;
        }
        nComp += count;

        //// TODO: If there is an error while loading, we should report it
        return 0;
    }

    void AssignName(const std::string& some_name) { m_name = some_name; }
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
