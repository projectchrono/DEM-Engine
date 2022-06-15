//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <vector>
#include <set>
#include <cfloat>

#include <core/ApiVersion.h>
#include <DEM/kT.h>
#include <DEM/dT.h>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/ThreadManager.h>
#include <core/utils/GpuManager.h>
#include <nvmath/helper_math.cuh>
#include <DEM/DEMDefines.h>
#include <DEM/DEMStructs.h>
#include <DEM/Boundaries.h>
#include <DEM/DEMModels.h>

namespace sgps {

// class DEMKinematicThread;
// class DEMDynamicThread;
// class ThreadManager;
class DEMTracker;

//////////////////////////////////////////////////////////////
// TODO LIST: 1. Check if anal obj's normal direction is correct, and if applyOriQ2Vec is correct
//            2. AddClumps returns a batch object and allow it to be tracked
//            3. Allow ext obj init CoM setting
//            4. Mass/MOI can be too small? Need to scale
//////////////////////////////////////////////////////////////

class DEMSolver {
  public:
    DEMSolver(unsigned int nGPUs = 2);
    virtual ~DEMSolver();

    /// Set output detail level
    void SetVerbosity(DEM_VERBOSITY verbose) { verbosity = verbose; }

    /// Instruct the dimension of the `world', as well as the origin point of this `world'. On initialization, this
    /// info will be used to figure out how to assign the num of voxels in each direction. If your `useful' domain is
    /// not box-shaped, then define a box that contains your domian. O is the coordinate of the left-bottom-front point
    /// of your simulation `world'.
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
    void SetGravitationalAcceleration(float3 g) { G = g; }
    /// Set a constant time step size
    void SetTimeStepSize(double ts_size) { m_ts_size = ts_size; }
    /// Set the number of dT steps before it waits for a contact-pair info update from kT
    void SetCDUpdateFreq(int freq) { m_updateFreq = freq; }
    // TODO: Implement an API that allows setting ts size through a list

    /// A convenient call that sets the origin of your coordinate system to be in the very center of your simulation
    /// `world'. Useful especially you feel like having this `world' large to safely hold everything, and don't
    /// quite care about the amount of accuracy lost by not fine-tuning the `world' size. Returns the coordinate of
    /// the left-bottom-front point of your simulation `world' after this operation.
    float3 CenterCoordSys();

    /// Explicitly instruct the bin size (for contact detection) that the solver should use
    void InstructBinSize(double bin_size) {
        m_use_user_instructed_bin_size = true;
        m_binSize = bin_size;
    }

    /// Explicitly instruct the sizes for the arrays at initialization time. This is useful when the number of owners
    /// tends to change (especially gradually increase) frequently in the simulation, by reducing the need for
    /// reallocation. Note however, whatever instruction the user gives here it won't affect the correctness of the
    /// simulation, since if the arrays are not long enough they will always be auto-resized.
    void InstructNumOwners(size_t numOwners) { m_instructed_num_owners = numOwners; }

    /// Manually instruct the solver to save time by using historyless contact model (usually not needed to call)
    void SetSolverHistoryless(bool useHistoryless = true);

    /// Instruct the solver to use frictonal (history-based) Hertzian contact force model
    void UseFrictionalHertzianModel();

    /// Instruct the solver to use frictonless Hertzian contact force model
    void UseFrictionlessHertzianModel();

    /// Instruct the solver if contact pair arrays should be sorted before usage. This is needed if history-based model
    /// is in use.
    void SetSortContactPairs(bool use_sort) { kT_should_sort = use_sort; }

    // NOTE: compact force calculation (in the hope to use shared memory) is not implemented
    void UseCompactForceKernel(bool use_compact);

    /// (Explicitly) set the amount by which the radii of the spheres (and the thickness of the boundaries) are expanded
    /// for the purpose of contact detection (safe, and creates false positives).
    void SetExpandFactor(float beta) { m_expand_factor = beta; }
    /// Input the maximum expected particle velocity and simulation time per contact detection (a.k.a per kT run), to
    /// help the solver automatically select a expand factor.
    void SuggestExpandFactor(float max_vel, float max_time_per_CD) { m_expand_factor = max_vel * max_time_per_CD; }
    /// If using constant step size and the step size is set, then inputting only the max expected velocity is fine.
    void SuggestExpandFactor(float max_vel);
    /// Further enlarge the safety perimeter needed by the input amount. Large number means even safer contact detection
    /// (missing no contacts), but creates more false positives, and risks leading to more bodies in a bin than a block
    /// can handle.
    void SuggestExpandSafetyParam(float param) { m_expand_safety_param = param; }

    /// Load possible clump types into the API-level cache
    /// Return the shared ptr to the clump type just loaded
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(float mass,
                                                    float3 moi,
                                                    const std::vector<float>& sp_radii,
                                                    const std::vector<float3>& sp_locations_xyz,
                                                    const std::vector<std::shared_ptr<DEMMaterial>>& sp_materials);
    /// An overload of LoadClumpType where all components use the same material
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(float mass,
                                                    float3 moi,
                                                    const std::vector<float>& sp_radii,
                                                    const std::vector<float3>& sp_locations_xyz,
                                                    const std::shared_ptr<DEMMaterial>& sp_material);
    /// An overload of LoadClumpType where the user builds the DEMClumpTemplate struct themselves then supply it
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(DEMClumpTemplate& clump);

    /// A simplified version of LoadClumpType: it just loads a one-sphere clump template
    std::shared_ptr<DEMClumpTemplate> LoadClumpSimpleSphere(float mass,
                                                            float radius,
                                                            const std::shared_ptr<DEMMaterial>& material);

    /// Load materials properties (Young's modulus, Poisson's ratio, Coeff of Restitution...) into
    /// the API-level cache. Return the ptr of the material type just loaded. If rho is not given then later
    /// calculating particle mass from rho is not allowed (instead it has to be explicitly given).
    std::shared_ptr<DEMMaterial> LoadMaterialType(DEMMaterial& mat);
    std::shared_ptr<DEMMaterial> LoadMaterialType(float E, float nu, float CoR, float mu, float Crr, float rho);
    std::shared_ptr<DEMMaterial> LoadMaterialType(float E, float nu, float CoR, float rho) {
        return LoadMaterialType(E, nu, CoR, 0.5, 0.01, rho);
    }
    std::shared_ptr<DEMMaterial> LoadMaterialType(float E, float nu, float CoR, float mu, float Crr) {
        return LoadMaterialType(E, nu, CoR, mu, Crr, -1.f);
    }
    std::shared_ptr<DEMMaterial> LoadMaterialType(float E, float nu, float CoR) {
        return LoadMaterialType(E, nu, CoR, 0.5, 0.01, -1.f);
    }

    /// Get position of a owner
    float3 GetOwnerPosition(bodyID_t ownerID) const;
    /// Get angular velocity of a owner
    float3 GetOwnerAngVel(bodyID_t ownerID) const;
    /// Get quaternion of a owner
    float4 GetOwnerOriQ(bodyID_t ownerID) const;
    /// Get velocity of a owner
    float3 GetOwnerVelocity(bodyID_t ownerID) const;

    /// Load input clumps (topology types and initial locations) on a per-pair basis. Note that the initial location
    /// means the location of the clumps' CoM coordinates in the global frame. If velocties are not given then they are
    /// assumed 0.
    void AddClumps(const std::vector<unsigned int>& types,
                   const std::vector<float3>& xyz,
                   const std::vector<float3>& vel = std::vector<float3>());
    void AddClumps(const std::vector<std::shared_ptr<DEMClumpTemplate>>& types,
                   const std::vector<float3>& xyz,
                   const std::vector<float3>& vel = std::vector<float3>());

    /// Load a clump into the system, and return a tracker object to the user, so that the user can apply direct
    /// control/modification/quarry to this clump (while dT is hanging)
    std::shared_ptr<DEMTracker> AddClumpTracked(const std::shared_ptr<DEMClumpTemplate>& type,
                                                float3 xyz,
                                                float3 vel = make_float3(0));

    /// Create a DEMTracker to allow direct control/modification/quarry to the argument object
    std::shared_ptr<DEMTracker> Track(std::shared_ptr<DEMExternObj>& obj);
    // std::shared_ptr<DEMTracker> Track(std::shared_ptr<ClumpBatch>& obj);

    /// Instruct each clump the type of prescribed motion it should follow. If this is not called (or if this vector is
    /// shorter than the clump location vector, then for the unassigned part) those clumps are defaulted to type 0,
    /// which is following `normal' physics.
    void SetClumpFamilies(const std::vector<unsigned int>& code);

    /// Instruct the solver that the 2 input families should not have contacts (a.k.a. ignored, if such a pair is
    /// encountered in contact detection). These 2 families can be the same (which means no contact within members of
    /// that family).
    void DisableContactBetweenFamilies(unsigned int ID1, unsigned int ID2);

    /// Prevent entites associated with this family to be outputted to files
    void DisableFamilyOutput(unsigned int ID);

    /// Mark all entities in this family to be fixed
    void SetFamilyFixed(unsigned int ID);
    ///
    void SetFamilyPrescribedLinVel(unsigned int ID,
                                   const std::string& velX,
                                   const std::string& velY,
                                   const std::string& velZ);
    ///
    void SetFamilyPrescribedAngVel(unsigned int ID,
                                   const std::string& velX,
                                   const std::string& velY,
                                   const std::string& velZ);

    /// Change all entities with family number ID_from to have a new number ID_to, when the condition defined by the
    /// string is satisfied by the entities in question. This should be called before initialization, and will be baked
    /// into the solver, so the conditions will be checked and changes applied every time step.
    void ChangeFamilyWhen(unsigned int ID_from, unsigned int ID_to, const std::string& condition);

    /// Change all entities with family number ID_from to have a new number ID_to, immediately. This is callable when kT
    /// and dT are hanging, not when they are actively working, or the behavior is not defined.
    void ChangeFamilyNow(unsigned int ID_from, unsigned int ID_to);

    ///
    void SetFamilyPrescribedPosition(unsigned int ID, const std::string& X, const std::string& Y, const std::string& Z);
    ///
    void SetFamilyPrescribedQuaternion(unsigned int ID, const std::string& q_formula);

    /// Define a custom contact force model by a string
    void DefineContactForceModel(const std::string& model);

    /// Add an (analytical or clump-represented) external object to the simulation system
    std::shared_ptr<DEMExternObj> AddExternalObject();
    std::shared_ptr<DEMExternObj> AddBCPlane(const float3 pos,
                                             const float3 normal,
                                             const std::shared_ptr<DEMMaterial>& material);

    // Add content to the flattened analytical component array
    // Note that analytical component is big different in that they each has a position in the jitified analytical
    // templates, insteads of like a clump, has an extra ComponentOffset array points it to the right jitified template
    // location.
    unsigned int AddAnalCompTemplate(const objType_t type,
                                     const std::shared_ptr<DEMMaterial>& material,
                                     const unsigned int owner,
                                     const float3 pos,
                                     const float3 rot = make_float3(0),
                                     const float d1 = 0.f,
                                     const float d2 = 0.f,
                                     const float d3 = 0.f,
                                     const objNormal_t normal = DEM_ENTITY_NORMAL_INWARD);

    /// Remove host-side cached vectors (so you can re-define them, and then re-initialize system)
    void ClearCache();

    /// Return total kinetic energy of all clumps
    float GetTotalKineticEnergy() const;
    /// Return the kinetic energy of all clumps in a set of families
    // TODO: float GetTotalKineticEnergy(std::vector<unsigned int> families) const;

    /// Write the current status of clumps to a file
    void WriteClumpFile(const std::string& outfilename) const;

    /// Intialize the simulation system
    void Initialize();

    /// Advance simulation by this amount of time, and at the end of this call, synchronize kT and dT. This is suitable
    /// for a longer call duration and without co-simulation.
    void DoStepDynamicsSync(double thisCallDuration);

    /// Advance simulation by this amount of time (but does not attempt to sync kT and dT). This can work with both long
    /// and short call durations and allows interplay with co-simulation APIs.
    void DoStepDynamics(double thisCallDuration);

    /// Copy the cached sim params to the GPU-accessible managed memory, so that they are picked up from the next ts of
    /// simulation. Usually used when you want to change simulation parameters after the system is already Intialized.
    void UpdateSimParams();

    /// Reset kT and dT back to a status like when the simulation system is constructed. In general the user does not
    /// need to call it, unless they want to run another test without re-constructing the entire DEM simulation system.
    /// Also note this call does not reset the collaboration log between kT and dT.
    void ResetWorkerThreads();

    /// Show the collaboration stats between dT and kT. This is more useful for tweaking the number of time steps that
    /// dT should be allowed to be in advance of kT.
    void ShowThreadCollaborationStats();

    /// Reset the collaboration stats between dT and kT back to the initial value (0). You should call this if you want
    /// to start over and re-inspect the stats of the new run; otherwise, it is generally not needed, you can go ahead
    /// and destroy DEMSolver.
    void ClearThreadCollaborationStats();

    /*
      protected:
        DEMSolver() : m_sys(nullptr) {}
        DEMSolver_impl* m_sys;
    */

    /// Choose between outputting particles as individual component spheres (results in larger files but less
    /// post-processing), or as owner clumps (e.g. xyz location means clump CoM locations, etc.), by
    /// DEM_OUTPUT_MODE::SPHERE and DEM_OUTPUT_MODE::CLUMP options
    void SetClumpOutputMode(DEM_OUTPUT_MODE mode) { m_clump_out_mode = mode; }
    /// Choose output format
    void SetOutputFormat(DEM_OUTPUT_FORMAT format) { m_out_format = format; }
    /// Specify the information that needs to go into the output files
    void SetOutputContent(unsigned int content) { m_out_content = content; }

  private:
    // A number of behavior-related variables
    // Verbosity
    DEM_VERBOSITY verbosity = INFO;
    // If true, kT should sort contact arrays then transfer them to dT
    bool kT_should_sort = true;
    // NOTE: compact force calculation (in the hope to use shared memory) is not implemented
    bool use_compact_sweep_force_strat = false;
    // If true, the solvers may need to do a per-step sweep to apply family number changes
    bool m_famnum_change_conditionally = false;

    // Force model, as a string
    std::string m_force_model = DEM_HERTZIAN_FORCE_MODEL();
    bool m_user_defined_force_model = false;

    // User explicitly set a bin size to use
    bool m_use_user_instructed_bin_size = false;

    // This is the cached material information.
    // It will be massaged into the managed memory upon Initialize().
    std::vector<std::shared_ptr<DEMMaterial>> m_loaded_sp_materials;
    // Materials info is processed at API level (on initialization) for generating proxy arrays
    std::vector<float> m_E_proxy;
    std::vector<float> m_nu_proxy;
    std::vector<float> m_CoR_proxy;
    std::vector<float> m_mu_proxy;
    std::vector<float> m_Crr_proxy;

    // This is the cached clump structure information. Note although not stated explicitly, those are only `clump'
    // templates, not including triangles, analytical geometries etc.
    std::vector<std::shared_ptr<DEMClumpTemplate>> m_templates;
    // Clump templates will be flatten and transferred into kernels upon Initialize()
    std::vector<float> m_template_mass;
    std::vector<float3> m_template_moi;
    std::vector<std::vector<unsigned int>> m_template_sp_mat_ids;
    std::vector<std::vector<float>> m_template_sp_radii;
    std::vector<std::vector<float3>> m_template_sp_relPos;

    // Shared pointers to external objects cached at the API system
    std::vector<std::shared_ptr<DEMExternObj>> cached_extern_objs;

    // Flattened (analytical) object component definition arrays, potentially jitifiable
    // These extra analytical entities' owners' ID will be appended to those added thru normal AddClump
    std::vector<unsigned int> m_anal_owner;
    // Material types of these analytical geometries
    std::vector<materialsOffset_t> m_anal_materials;
    // Initial locations of this obj's components relative to obj's CoM
    std::vector<float3> m_anal_comp_pos;
    // Some float3 quantity that is representitive of a component's initial orientation (such as plane normal, and its
    // meaning can vary among different types)
    std::vector<float3> m_anal_comp_rot;
    // Some float quantity that is representitive of a component's size (e.g. for a cylinder, top radius)
    std::vector<float> m_anal_size_1;
    // Some float quantity that is representitive of a component's size (e.g. for a cylinder, bottom radius)
    std::vector<float> m_anal_size_2;
    // Some float quantity that is representitive of a component's size (e.g. for a cylinder, its length)
    std::vector<float> m_anal_size_3;
    // Component object types
    std::vector<objType_t> m_anal_types;
    // Component object normal direction, defaulting to inward. If this object is topologically a plane then this param
    // is meaningless, since its normal is determined by its rotation.
    std::vector<objNormal_t> m_anal_normals;
    // Extra clumps are those loaded by adding external object. They typically consist of many spheres (~thousands).
    std::vector<inertiaOffset_t> m_extra_clump_type;
    // Extra clumps' owners' ID will be appended to those added thru normal AddClump, and are consistent with external
    // obj IDs
    std::vector<unsigned int> m_extra_clump_owner;
    std::vector<float3> m_extra_clump_xyz;

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

    // I/O related flags
    DEM_OUTPUT_MODE m_clump_out_mode = DEM_OUTPUT_MODE::SPHERE;
    DEM_OUTPUT_FORMAT m_out_format = DEM_OUTPUT_FORMAT::CHPF;
    unsigned int m_out_content = DEM_OUTPUT_CONTENT::QUAT | DEM_OUTPUT_CONTENT::ABSV;

    // ``World'' size along X dir (user-defined)
    float m_boxX = 0.f;
    // ``World'' size along Y dir (user-defined)
    float m_boxY = 0.f;
    // ``World'' size along Z dir (user-defined)
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
    // Total number of bins
    size_t m_num_bins;
    // Number of bins on each direction
    binID_t nbX;
    binID_t nbY;
    binID_t nbZ;
    // The amount at which all geometries inflate (for safer contact detection)
    float m_expand_factor = 0.f;
    // When the user suggests the expand factor without explicitly setting it, the `just right' amount of expansion is
    // multiplied by this expand_safety_param, so the geometries over-expand for CD purposes. This creates more false
    // positives, and risks leading to more bodies in a bin than a block can handle, but helps prevent contacts being
    // left undiscovered by CD.
    float m_expand_safety_param = 1.f;

    // Total number of spheres
    size_t nSpheresGM = 0;
    // Total number of triangle facets
    size_t nTriGM = 0;
    // Number of analytical entites (as components of some external objects)
    unsigned int nAnalGM = 0;
    // Total number of owner bodies
    size_t nOwnerBodies = 0;
    // Number of loaded clumps
    size_t nOwnerClumps = 0;
    // Number of loaded external objects
    unsigned int nExtObj = 0;
    // Number of loaded triangle-represented (mesh) objects
    size_t nTriEntities = 0;
    // nExtObj + nOwnerClumps + nTriEntities == nOwnerBodies

    // The number of user-estimated (max) number of owners that will be present in the simulation. If 0, then the arrays
    // will just be resized at intialization based on the input size.
    size_t m_instructed_num_owners = 0;

    unsigned int nDistinctClumpComponents;
    unsigned int nDistinctClumpBodyTopologies;
    unsigned int nDistinctMassProperties;
    unsigned int nMatTuples;
    unsigned int nDistinctFamilies;

    // This many clump template can be jitified, and the rest need to exist in global memory
    // Note all `mass' properties are jitified, it's just this many clump templates' component info will not be
    // jitified. Therefore, this quantity does not seem to be useful beyond reporting to the user.
    unsigned int nJitifiableClumpTopo;
    // Number of jitified clump components
    unsigned int nJitifiableClumpComponents;

    // Whether the number of voxels and length unit l is explicitly given by the user
    bool explicit_nv_override = false;
    // Whether the GPU-side systems have been initialized
    bool sys_initialized = false;
    // Smallest sphere radius (used to let the user know whether the expand factor is sufficient)
    float m_smallest_radius = FLT_MAX;

    // Cached state vectors such as the types and locations/velocities of the initial clumps to fill the sim domain
    // with. User managed arrays usually use unsigned int to represent integers and ensure safety; however
    // m_input_clump_types tends to be long and should use _t definition. Same goes for m_extra_clump_type, but not
    // user-input family number, because that number can be anything.
    std::vector<inertiaOffset_t> m_input_clump_types;
    // Some input clump initial profiles
    std::vector<float3> m_input_clump_xyz;
    // std::vector<float4> m_input_clump_rot;
    std::vector<float3> m_input_clump_vel;
    // Specify the ``family'' code for each clump. Then you can specify if they should go with some prescribed motion or
    // some special physics (for example, being fixed). The default behavior (without specification) for every family is
    // using ``normal'' physics.
    std::vector<unsigned int> m_input_clump_family;

    struct familyPair_t {
        unsigned int ID1;
        unsigned int ID2;
    };
    // Change family number from ID1 to ID2 when conditions are met
    std::vector<familyPair_t> m_family_change_pairs;
    // Corrsponding family number changing conditions
    std::vector<std::string> m_family_change_conditions;
    // Cached user-input no-contact family pairs
    std::vector<familyPair_t> m_input_no_contact_pairs;
    // TODO: add APIs to allow specification of prescribed motions for each family. This information is only needed by
    // dT. (Prescribed types: an added force as a function of sim time or location; prescribed velocity/angVel as a
    // function; prescribed location as a function)
    // Upper-triangular interaction `mask' matrix, which clarifies the family codes that a family can interact with.
    // This is needed by kT only.
    std::vector<notStupidBool_t> m_family_mask_matrix;
    // Host-side mapping array that maps like this: map.at(user family number) = (corresponding impl-level family
    // number)
    std::unordered_map<unsigned int, family_t> m_family_user_impl_map;
    // Host-side mapping array that maps like this: map.at(impl-level family number) = (corresponding user family
    // number)
    std::unordered_map<family_t, unsigned int> m_family_impl_user_map;
    // The familes that should not be outputted
    std::set<unsigned int> m_no_output_families;
    // TODO: fixed particles should automatically attain status indicating they don't interact with each other.

    // Unlike clumps, external objects do not have _types (each is its own type)
    std::vector<float3> m_input_ext_obj_xyz;
    // std::vector<float4> m_input_ext_obj_rot;
    std::vector<unsigned int> m_input_ext_obj_family;

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
    // User-input prescribed motion
    std::vector<familyPrescription_t> m_input_family_prescription;
    // Processed unique family prescription info
    std::vector<familyPrescription_t> m_unique_family_prescription;

    // Cached tracked objects that can be leveraged by the user to assume explicit control over some simulation objects
    std::vector<std::shared_ptr<DEMTrackedObj>> m_tracked_objs;
    // std::vector<std::shared_ptr<DEMTracker>> m_trackers;

    // The number of dT steps before it waits for a kT update. The default value 0 means every dT step will wait for a
    // newly produced contact-pair info (from kT) before proceeding.
    int m_updateFreq = 0;

    // The contact model is historyless, or not. It affects jitification.
    bool m_isHistoryless = false;

    WorkerReportChannel* kTMain_InteractionManager;
    WorkerReportChannel* dTMain_InteractionManager;
    GpuManager* dTkT_GpuManager;
    ThreadManager* dTkT_InteractionManager;
    DEMKinematicThread* kT;
    DEMDynamicThread* dT;

    /// Pre-process some user inputs so we acquire the knowledge on how to jitify the kernels
    void generateJITResources();
    /// Make sure the input represents something we can simulate, and if not, tell the reasons
    void postJITResourceGenSanityCheck();
    /// Flatten cached clump templates (from ClumpTemplate structs to float arrays)
    void preprocessClumpTemplates();
    /// Jitify GPU kernels, based on pre-processed user inputs
    void jitifyKernels();
    /// Figure out the unit length l and numbers of voxels along each direction, based on domain size X, Y, Z
    void figureOutNV();
    /// Set the default bin (for contact detection) size to be the same of the smallest sphere
    void decideBinSize();
    /// Transfer cached solver preferences/instructions to dT and kT.
    void transferSolverParams();
    /// Transfer (CPU-side) cached simulation data (about sim world) to the GPU-side. It is called automatically during
    /// system initialization.
    void transferSimParams();
    /// Transfer (CPU-side) cached clump templates info and initial clump type/position info to GPU-side arrays
    void initializeArrays();
    /// Pack array pointers to a struct so they can be easily used as kernel arguments
    void packDataPointers();
    /// Warn users if the data types defined in DEMDefines.h do not blend well with the user inputs (fist-round
    /// coarse-grain sanity check)
    void validateUserInputs();
    /// Compute the number of dT for cycles based on the amount of time the user wants to advance the simulation
    inline size_t computeDTCycles(double thisCallDuration);
    /// Prepare the material/contact proxy matrix force computation kernels
    void figureOutMaterialProxies();
    /// Figure out info about external objects/clump templates and whether they can be jitified
    void preprocessAnalyticalObjs();
    /// Report simulation stats at initialization
    inline void reportInitStats() const;
    /// Based on user input, prepare family_mask_matrix (family contact map matrix)
    void figureOutFamilyMasks();

    // Some JIT packaging helpers
    inline void equipClumpTemplates(std::unordered_map<std::string, std::string>& strMap);
    inline void equipClumpTemplateAcquisition(std::unordered_map<std::string, std::string>& strMap);
    inline void equipSimParams(std::unordered_map<std::string, std::string>& strMap);
    inline void equipClumpMassMat(std::unordered_map<std::string, std::string>& strMap);
    inline void equipAnalGeoTemplates(std::unordered_map<std::string, std::string>& strMap);
    inline void equipFamilyMasks(std::unordered_map<std::string, std::string>& strMap);
    inline void equipFamilyPrescribedMotions(std::unordered_map<std::string, std::string>& strMap);
    inline void equipFamilyOnFlyChanges(std::unordered_map<std::string, std::string>& strMap);
    inline void equipForceModel(std::unordered_map<std::string, std::string>& strMap);
};

// A struct to get or set tracked owner entities, mainly for co-simulation
class DEMTracker {
  private:
    // Its parent DEMSolver system
    const DEMSolver* sys;

  public:
    DEMTracker(DEMSolver* sim_sys) : sys(sim_sys) {}
    ~DEMTracker() {}

    // The tracked object
    std::shared_ptr<DEMTrackedObj> obj;
    // Methods to get info from this owner
    float3 Pos() { return sys->GetOwnerPosition(obj->ownerID); }
    float3 AngVel() { return sys->GetOwnerAngVel(obj->ownerID); }
    float3 Vel() { return sys->GetOwnerVelocity(obj->ownerID); }
    float4 OriQ() { return sys->GetOwnerOriQ(obj->ownerID); }
};

}  // namespace sgps
