//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <vector>
#include <set>
#include <cfloat>

#include <core/ApiVersion.h>
#include <granular/kT.h>
#include <granular/dT.h>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/ThreadManager.h>
#include <core/utils/GpuManager.h>
#include <core/utils/Macros.h>
#include <helper_math.cuh>
#include <granular/GranularDefines.h>

namespace sgps {

// class DEMKinematicThread;
// class DEMDynamicThread;
// class ThreadManager;

class DEMSolver {
  public:
    DEMSolver(unsigned int nGPUs = 2);
    virtual ~DEMSolver();

    /// Instruct the dimension of the ``world'', as well as the origin point of this ``world''. On initialization, this
    /// info will be used to figure out how to assign the num of voxels in each direction. If your ``useful'' domain is
    /// not box-shaped, then define a box that contains your domian. O is the coordinate of the left-bottom-front point
    /// of your simulation ``world''.
    void InstructBoxDomainDimension(float x, float y, float z, float3 O = make_float3(0));

    /// Explicitly instruct the number of voxels (as 2^{x,y,z}) along each direction, as well as the smallest unit
    /// length l. This is usually for test purposes, and will overwrite other size-related definitions of the big
    /// domain.
    void InstructBoxDomainNumVoxel(unsigned char x,
                                   unsigned char y,
                                   unsigned char z,
                                   float len_unit = 1e-10f,
                                   float3 O = make_float3(0));

    /// Set gravity
    void SetGravitationalAcceleration(float3 g);
    /// Set a constant time step size
    void SetTimeStepSize(double ts_size);
    /// Set the number of dT steps before it waits for a contact-pair info update from kT
    void SetCDUpdateFreq(int freq) { m_updateFreq = freq; }
    // TODO: Implement an API that allows setting ts size through a list

    /// A convenient call that sets the origin of your coordinate system to be in the very center of your simulation
    /// ``world''. Useful especially you feel like having this ``world'' large to safely hold everything, and don't
    /// quite care about the amount of accuracy lost by not fine-tuning the ``world'' size. Returns the coordinate of
    /// the left-bottom-front point of your simulation ``world'' after this operation.
    float3 CenterCoordSys();

    /// (Explicitly) set the amount by which the radii of the spheres (and the thickness of the boundaries) are expanded
    /// for the purpose of contact detection (safe, and creates false positives).
    void SetExpandFactor(float beta);
    /// Input the maximum expected particle velocity and simulation time per contact detection (a.k.a per kT run), to
    /// help the solver automatically select a expand factor.
    void SuggestExpandFactor(float max_vel, float max_time_per_CD);
    /// If using constant step size and the step size is set, then inputting only the max expected velocity is fine.
    void SuggestExpandFactor(float max_vel);
    /// Further enlarge the safety perimeter needed by the input amount. Large number means even safer contact detection
    /// (missing no contacts), but creates more false positives, and risks leading to more bodies in a bin than a block
    /// can handle.
    void SuggestExpandSafetyParam(float param);

    /// Load possible clump types into the API-level cache.
    /// Return the index of the clump type just loaded.
    unsigned int LoadClumpType(float mass,
                               float3 moi,
                               const std::vector<float>& sp_radii,
                               const std::vector<float3>& sp_locations_xyz,
                               const std::vector<unsigned int>& sp_material_ids);
    // TODO: need to overload with (vec_distinctSphereRadiiOffset_default_t spheres_component_type, vec_float3
    // location). If this method is called then corresponding sphere_types must have been defined via LoadSphereType.

    /// A simplified version of LoadClumpType: it just loads a one-sphere clump template
    unsigned int LoadClumpSimpleSphere(float mass, float radius, unsigned int material_id);

    /// Load materials properties (Young's modulus, Poisson's ratio, Coeff of Restitution and optionally density) into
    /// the API-level cache. Return the index of the material type just loaded. If CoR is not given then it is assumed
    /// 0; if density is not given then later calculating particle mass from density is not allowed (instead it has to
    /// be explicitly given).
    unsigned int LoadMaterialType(float E, float nu, float CoR, float density);
    unsigned int LoadMaterialType(float E, float nu, float CoR) { return LoadMaterialType(E, nu, CoR, -1.f); }
    unsigned int LoadMaterialType(float E, float nu) { return LoadMaterialType(E, nu, 0.f, -1.f); }

    /// Load input clumps (topology types and initial locations) on a per-pair basis
    /// TODO: Add a overload that takes velocities too
    void SetClumps(const std::vector<unsigned int>& types, const std::vector<float3>& xyz);

    /// Load input clump initial velocities on a per-pair basis. If this is not called (or if this vector is shorter
    /// than the clump location vector, then for the unassigned part) the initial velocity is assumed to be 0.
    void SetClumpVels(const std::vector<float3>& vel);

    /// Instruct each clump the type of prescribed motion it should follow. If this is not called (or if this vector is
    /// shorter than the clump location vector, then for the unassigned part) those clumps are defaulted to type 0,
    /// which is following ``normal'' physics.
    void SetClumpFamily(const std::vector<unsigned int>& code);

    /// Return the voxel ID of a clump by its numbering
    voxelID_t GetClumpVoxelID(unsigned int i) const;

    /// Write current simulation status to a file
    /// Write overlapping spheres, not clumps. It makes the file larger, but less trouble to visualize. Use for test
    /// purposes only.
    void WriteFileAsSpheres(const std::string& outfilename) const;

    int Initialize();

    /// Advance simulation by this amount of time
    int LaunchThreads(double thisCallDuration);

    /// Copy the cached sim params to the GPU-accessible managed memory, so that they are picked up from the next ts of
    /// simulation. Usually used when you want to change simulation parameters after the system is already Intialized.
    void UpdateSimParams();

    /*
      protected:
        DEMSolver() : m_sys(nullptr) {}
        DEMSolver_impl* m_sys;
    */

  private:
    // This is the cached material information.
    // It will be massaged into the managed memory upon Initialize().
    struct DEMMaterial {
        float density;
        float E;
        float nu;
        float CoR;
    };
    std::vector<DEMMaterial> m_sp_materials;
    // Materials info is processed at API level (on initialization) for generating proxy arrays
    std::vector<float> m_E_proxy;
    std::vector<float> m_G_proxy;
    std::vector<float> m_CoR_proxy;

    // This is the cached clump structure information.
    // It will be massaged into kernels upon Initialize.
    std::vector<float> m_template_mass;
    std::vector<float3> m_template_moi;
    std::vector<std::vector<unsigned int>> m_template_sp_mat_ids;
    std::vector<std::vector<float>> m_template_sp_radii;
    std::vector<std::vector<float3>> m_template_sp_relPos;

    /*
    // Dan and Ruochun decided NOT to extract unique input values.
    // Instead, we trust users: we simply store all clump template info users give.
    // So this unique-value-extractor block is disabled and commented.

    // unique clump masses derived from m_template_mass
    std::set<float> m_template_mass_types;
    std::vector<unsigned int> m_template_mass_type_offset;
    // unique sphere radii types derived from m_template_sp_radii
    std::set<float> m_template_sp_radii_types;
    std::vector<std::vector<distinctSphereRadiiOffset_default_t>> m_template_sp_radii_type_offset;
    // unique sphere (local) location types derived from m_template_sp_relPos
    // std::set<float3, float3_less_than> m_clumps_sp_location_types;
    std::set<float3> m_clumps_sp_location_types;
    std::vector<std::vector<distinctSphereRelativePositions_default_t>> m_clumps_sp_location_type_offset;
    */

    // ``World'' size along X dir
    float m_boxX = 0.f;
    // ``World'' size along Y dir
    float m_boxY = 0.f;
    // ``World'' size along Z dir
    float m_boxZ = 0.f;
    // Origin of the ``world''
    float3 m_boxLBF = make_float3(0);
    // Number of voxels in the X direction, expressed as a power of 2
    unsigned char nvXp2;
    // Number of voxels in the Y direction, expressed as a power of 2
    unsigned char nvYp2;
    // Number of voxels in the Z direction, expressed as a power of 2
    unsigned char nvZp2;
    // Gravitational acceleration
    float3 G;
    // Actual (double-precision) size of a voxel
    double m_voxelSize;
    // Time step size
    double m_ts_size = -1.0;
    // If the time step size is a constant (if not, it needs to be supplied with a file or a function)
    bool ts_size_is_const = true;
    // The length unit. Any XYZ we report to the user, is under the hood a multiple of this l.
    float l = FLT_MAX;
    // The edge length of a bin (for contact detection)
    double m_binSize;
    // The amount at which all geometries inflate (for safer contact detection)
    float m_expand_factor = 0.f;
    // When the user suggests the expand factor without explicitly setting it, the ``just right'' amount of expansion is
    // multiplied by this expand_safety_param, so the geometries over-expand for CD purposes. This creates more false
    // positives, and risks leading to more bodies in a bin than a block can handle, but helps prevent contacts being
    // left undiscovered by CD.
    float m_expand_safety_param = 1.f;

    // Total number of spheres
    size_t nSpheresGM;
    // Total number of clump bodies
    size_t nClumpBodies;

    float sphereUU;

    // Whether the number of voxels and length unit l is explicitly given by the user.
    bool explicit_nv_override = false;
    // Whether the GPU-side systems have been initialized
    bool sys_initialized = false;
    // Smallest sphere radius (used to let the user know whether the expand factor is sufficient)
    float m_smallest_radius = FLT_MAX;

    // Right now, the following two are integrated into one, in nDistinctClumpComponents
    // unsigned int nDistinctSphereRadii_computed;
    // unsigned int nDistinctSphereRelativePositions_computed;

    unsigned int nDistinctClumpComponents_computed;
    unsigned int nDistinctClumpBodyTopologies_computed;
    unsigned int nMatTuples_computed;

    // cached state vectors such as the types and locations/velocities of the initial clumps to fill the sim domain with
    std::vector<unsigned int> m_input_clump_types;
    std::vector<float3> m_input_clump_xyz;
    std::vector<float3> m_input_clump_vel;
    // Specify the ``family'' code for each clump. Then you can specify if they should go with some prescribed motion or
    // some special physics (for example, being fixed). The default behavior (without specification) for every family is
    // using ``normal'' physics.
    std::vector<unsigned int> m_input_clump_family;
    // TODO: add APIs to allow specification of prescribed motions for each family. This information is only needed by
    // dT. (Prescribed types: an added force as a function of sim time or location; prescribed velocity/angVel as a
    // function; prescribed location as a function)
    // TODO: add a interaction ``mask'', which clarifies the family codes that a family can interact with. This can be a
    // bit slow to process but it only involves contact detection so needed by kT only, which it's acceptable even if
    // it's somewhat slow.
    // TODO: fixed particles should automatically attain status indicating they don't interact with each other.

    // The number of dT steps before it waits for a kT update. The default value 0 means every dT step will wait for a
    // newly produced contact-pair info (from kT) before proceeding.
    int m_updateFreq = 0;

    GpuManager* dTkT_GpuManager;
    ThreadManager* dTkT_InteractionManager;
    DEMKinematicThread* kT;
    DEMDynamicThread* dT;

    void generateJITResources();
    void jitifyKernels();
    // Figure out the unit length l and corresponding numbers of voxels along each direction, based on domain size X, Y,
    // Z
    void figureOutNV();
    // Set the default bin (for contact detection) size to be the same of the smallest sphere
    void decideDefaultBinSize();
    // Transfer cached sim params to dT (and kT?)
    void transferSimParams();
    // Wait for kT and dT until they are done with the work and a signal is give by them, then the ApiSystem can go on.
    void waitOnThreads();
    // Allocate and populate kT dT managed arrays.
    void initializeArrays();
    // Pack array pointers to a struct so they can be easily used as kernel arguments
    void packDataPointers();
    // Warn users if the data types defined in GranularDefines.h do not blend well with the user inputs (such as when
    // the user inputs a huge amount of clump templates).
    void validateUserInputs();
    // Compute the number of dT for cycles based on the amount of time the user wants to advance the simulation
    inline size_t computeDTCycles(double thisCallDuration);
    // Prepare the material/contact proxy matrix force computation kernels
    void figureOutMaterialProxies();
};

}  // namespace sgps
