//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef SGPS_DEM_API
#define SGPS_DEM_API

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
#include <DEM/DEMBdrsAndObjs.h>
#include <DEM/DEMModels.h>
#include <DEM/DEMAuxClasses.h>

namespace sgps {

// class DEMKinematicThread;
// class DEMDynamicThread;
// class ThreadManager;
class DEMInspector;
class DEMTracker;

//////////////////////////////////////////////////////////////
// TODO LIST: 1. Variable ts size (MAX_VEL flavor uses tracked max cp vel)
//            2. Allow ext obj init CoM setting
//            3. Instruct how many dT steps should at LEAST do before receiving kT update
//            4. Sleepers that don't participate CD or integration
//            5. Make force model a struct, and inside it...
//            6. Select whether to acquire mat, acquire what history, and whether
//               to use a/several custom float arrays to store custom config data
//            7. This custom array can be defined at clump template/anal obj/mesh obj generation
//            9. wT takes care of an extra output when it crushes
//////////////////////////////////////////////////////////////

class DEMSolver {
  public:
    DEMSolver(unsigned int nGPUs = 2);
    virtual ~DEMSolver();

    /// Set output detail level
    void SetVerbosity(DEM_VERBOSITY verbose) { verbosity = verbose; }

    /// Instruct the dimension of the `world'. On initialization, this info will be used to figure out how to assign the
    /// num of voxels in each direction. If your `useful' domain is not box-shaped, then define a box that contains your
    /// domian. O is the coordinate of the left-bottom-front point of your simulation `world'.
    void InstructBoxDomainDimension(float x, float y, float z, const std::string dir_exact = "none");

    /// Explicitly instruct the number of voxels (as 2^{x,y,z}) along each direction, as well as the smallest unit
    /// length l. This is usually for test purposes, and will overwrite other size-related definitions of the big
    /// domain.
    void InstructBoxDomainNumVoxel(unsigned char x, unsigned char y, unsigned char z, float len_unit = 1e-10f);

    /// Instruct if and how we should add boundaries to the simulation world upon initialization. Choose between `none',
    /// `all' (add 6 boundary planes) and `top_open' (add 5 boundary planes and leave the z-directon top open). Also
    /// specifies the material that should be assigned to those bounding boundaries.
    void InstructBoxDomainBoundingBC(const std::string& inst, const std::shared_ptr<DEMMaterial>& mat) {
        m_user_add_bounding_box = inst;
        m_bounding_box_material = mat;
    }

    /// Set gravity
    void SetGravitationalAcceleration(float3 g) { G = g; }
    /// Set the initial time step size. If using constant step size, then this will be used throughout; otherwise, the
    /// actual step size depends on the variable step strategy.
    void SetInitTimeStep(double ts_size) { m_ts_size = ts_size; }
    /// Return the number of clumps that are currently in the simulation
    size_t GetNumClumps() { return nOwnerClumps; }
    /// Get the current time step size in simulation
    double GetTimeStepSize();
    /// Getthe current expand factor in simulation
    float GetExpandFactor();
    /// Set the number of dT steps before it waits for a contact-pair info update from kT
    void SetCDUpdateFreq(int freq) { m_updateFreq = freq; }
    // TODO: Implement an API that allows setting ts size through a list

    /// Sets the origin of your coordinate system
    void SetCoordSysOrigin(const std::string& where) { m_user_instructed_origin = where; }
    void SetCoordSysOrigin(float3 O) {
        m_boxLBF = O;
        m_user_instructed_origin = "explicit";
    }

    /// Explicitly instruct the bin size (for contact detection) that the solver should use
    void SetInitBinSize(double bin_size) {
        use_user_defined_bin_size = true;
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

    /// Instruct the solver to rearrange and consolidate clump templates information, then jitify it into GPU kernels
    /// (if set to true), rather than using flattened sphere component configuration arrays whose entries are associated
    /// with individual spheres. Note: setting it to true gives no performance benefit known to me.
    void SetJitifyClumpTemplates(bool use = true) { jitify_clump_templates = use; }
    /// Instruct the solver to rearrange and consolidate mass property information (for all owner types), then jitify it
    /// into GPU kernels (if set to true), rather than using flattened mass property arrays whose entries are associated
    /// with individual owners. Note: setting it to true gives no performance benefit known to me.
    void SetJitifyMassProperties(bool use = true) { jitify_mass_moi = use; }

    /// Instruct the contact detection process to use one thread to process a bin (if true), instead of using a block to
    /// process a bin. This probably also requires you to manually set a smaller SGPS_DEM_MAX_SPHERES_PER_BIN. This can
    /// potentially be faster especially in a scenario where the spheres are of similar sizes.
    void SetOneBinPerThread(bool use = true) { use_one_bin_per_thread = use; }

    // NOTE: compact force calculation (in the hope to use shared memory) is not implemented
    void UseCompactForceKernel(bool use_compact);

    /// (Explicitly) set the amount by which the radii of the spheres (and the thickness of the boundaries) are expanded
    /// for the purpose of contact detection (safe, and creates false positives). If fix is set to true, then this
    /// expand factor does not change even if the user uses variable time step size.
    void SetExpandFactor(float beta, bool fix = true) {
        m_expand_factor = beta;
        use_user_defined_expand_factor = fix;
    }
    /// Input the maximum expected particle velocity. This is mainly to help the solver automatically select a expand
    /// factor and do sanity checks during the simulation.
    void SetMaxVelocity(float max_vel) { m_approx_max_vel = max_vel; }
    /// Further enlarge the safety perimeter needed by the input amount. Large number means even safer contact detection
    /// (missing no contacts), but creates more false positives, and risks leading to more bodies in a bin than a block
    /// can handle.
    void SetExpandSafetyParam(float param) { m_expand_safety_param = param; }

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
    /// An overload of LoadClumpType which loads sphere components from a file
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(float mass,
                                                    float3 moi,
                                                    const std::string filename,
                                                    const std::vector<std::shared_ptr<DEMMaterial>>& sp_materials);
    /// An overload of LoadClumpType which loads sphere components from a file and all components use the same material
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(float mass,
                                                    float3 moi,
                                                    const std::string filename,
                                                    const std::shared_ptr<DEMMaterial>& sp_material);

    /// A simplified version of LoadClumpType: it just loads a one-sphere clump template
    std::shared_ptr<DEMClumpTemplate> LoadClumpSimpleSphere(float mass,
                                                            float radius,
                                                            const std::shared_ptr<DEMMaterial>& material);

    /// Load materials properties (Young's modulus, Poisson's ratio, Coeff of Restitution...) into
    /// the API-level cache. Return the ptr of the material type just loaded.
    std::shared_ptr<DEMMaterial> LoadMaterialType(DEMMaterial& mat);
    std::shared_ptr<DEMMaterial> LoadMaterialType(float E, float nu, float CoR, float mu, float Crr);
    std::shared_ptr<DEMMaterial> LoadMaterialType(float E, float nu, float CoR) {
        return LoadMaterialType(E, nu, CoR, 0.5, 0.0);
    }

    /// Get position of a owner
    float3 GetOwnerPosition(bodyID_t ownerID) const;
    /// Get angular velocity of a owner
    float3 GetOwnerAngVel(bodyID_t ownerID) const;
    /// Get quaternion of a owner
    float4 GetOwnerOriQ(bodyID_t ownerID) const;
    /// Get velocity of a owner
    float3 GetOwnerVelocity(bodyID_t ownerID) const;
    /// Set position of a owner in user unit
    void SetOwnerPosition(bodyID_t ownerID, float3 pos);
    /// Set angular velocity of a owner
    void SetOwnerAngVel(bodyID_t ownerID, float3 angVel);
    /// Set velocity of a owner
    void SetOwnerVelocity(bodyID_t ownerID, float3 vel);
    /// Set quaternion of a owner
    void SetOwnerOriQ(bodyID_t ownerID, float4 oriQ);

    /// Load input clumps (topology types and initial locations) on a per-pair basis. Note that the initial location
    /// means the location of the clumps' CoM coordinates in the global frame.
    std::shared_ptr<DEMClumpBatch> AddClumps(DEMClumpBatch& input_batch);
    std::shared_ptr<DEMClumpBatch> AddClumps(const std::vector<std::shared_ptr<DEMClumpTemplate>>& input_types,
                                             const std::vector<float3>& input_xyz);
    std::shared_ptr<DEMClumpBatch> AddClumps(std::shared_ptr<DEMClumpTemplate>& input_type, float3 input_xyz) {
        return AddClumps(std::vector<std::shared_ptr<DEMClumpTemplate>>(1, input_type),
                         std::vector<float3>(1, input_xyz));
    }
    std::shared_ptr<DEMClumpBatch> AddClumps(std::shared_ptr<DEMClumpTemplate>& input_type,
                                             const std::vector<float3>& input_xyz) {
        return AddClumps(std::vector<std::shared_ptr<DEMClumpTemplate>>(input_xyz.size(), input_type), input_xyz);
    }

    /// Load a mesh-represented object
    std::shared_ptr<DEMMeshConnected> AddWavefrontMeshObject(const std::string& filename,
                                                             bool load_normals = true,
                                                             bool load_uv = false);
    std::shared_ptr<DEMMeshConnected> AddWavefrontMeshObject(DEMMeshConnected& mesh);

    /// Create a DEMTracker to allow direct control/modification/query to this external object
    std::shared_ptr<DEMTracker> Track(std::shared_ptr<DEMExternObj>& obj);
    /// Create a DEMTracker to allow direct control/modification/query to this batch of clumps. By default, it refers to
    /// the first clump in this batch. The user can refer to other clumps in this batch by supplying an offset when
    /// using this tracker's querying or assignment methods.
    std::shared_ptr<DEMTracker> Track(std::shared_ptr<DEMClumpBatch>& obj);

    /// Create a inspector object that can help query some statistical info of the clumps in the simulation
    std::shared_ptr<DEMInspector> CreateInspector(const std::string& quantity = "clump_max_z");
    // std::shared_ptr<DEMInspector> CreateInspector(const std::string& quantity, region);

    /// Instruct the solver that the 2 input families should not have contacts (a.k.a. ignored, if such a pair is
    /// encountered in contact detection). These 2 families can be the same (which means no contact within members of
    /// that family).
    void DisableContactBetweenFamilies(unsigned int ID1, unsigned int ID2);

    /// Re-enable contact between 2 families after the system is initialized
    void EnableContactBetweenFamilies(unsigned int ID1, unsigned int ID2);

    /// Prevent entites associated with this family to be outputted to files
    void DisableFamilyOutput(unsigned int ID);

    /// Mark all entities in this family to be fixed
    void SetFamilyFixed(unsigned int ID);
    /// Set the prescribed linear velocity to all entities in a family. If dictate is set to true, then this
    /// prescription completely dictates this family's motions.
    void SetFamilyPrescribedLinVel(unsigned int ID,
                                   const std::string& velX,
                                   const std::string& velY,
                                   const std::string& velZ,
                                   bool dictate = true);
    /// Set the prescribed angular velocity to all entities in a family. If dictate is set to true, then this
    /// prescription completely dictates this family's motions.
    void SetFamilyPrescribedAngVel(unsigned int ID,
                                   const std::string& velX,
                                   const std::string& velY,
                                   const std::string& velZ,
                                   bool dictate = true);

    /// Change all entities with family number ID_from to have a new number ID_to, when the condition defined by the
    /// string is satisfied by the entities in question. This should be called before initialization, and will be baked
    /// into the solver, so the conditions will be checked and changes applied every time step.
    void ChangeFamilyWhen(unsigned int ID_from, unsigned int ID_to, const std::string& condition);

    /// Change all entities with family number ID_from to have a new number ID_to, immediately. This is callable when kT
    /// and dT are hanging, not when they are actively working, or the behavior is not defined.
    void ChangeFamily(unsigned int ID_from, unsigned int ID_to);

    ///
    void SetFamilyPrescribedPosition(unsigned int ID, const std::string& X, const std::string& Y, const std::string& Z);
    ///
    void SetFamilyPrescribedQuaternion(unsigned int ID, const std::string& q_formula);

    /// Change the sizes of the clumps by a factor. This method directly works on the clump components spheres,
    /// therefore requiring sphere components to be store in flattened array (default behavior), not jitified templates.
    void ChangeClumpSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors);

    /// Define a custom contact force model by a string
    void DefineContactForceModel(const std::string& model);

    /// If true, each jitification string substitution will do a one-liner to one-liner replacement, so that if the
    /// kernel compilation fails, the error meessage line number will reflex the actual spot where that happens (instead
    /// of some random number)
    void EnsureKernelErrMsgLineNum(bool flag = true) { m_ensure_kernel_line_num = flag; }

    /// Add an (analytical or clump-represented) external object to the simulation system
    std::shared_ptr<DEMExternObj> AddExternalObject();
    std::shared_ptr<DEMExternObj> AddBCPlane(const float3 pos,
                                             const float3 normal,
                                             const std::shared_ptr<DEMMaterial>& material);

    /// Remove host-side cached vectors (so you can re-define them, and then re-initialize system)
    void ClearCache();

    /// Return total kinetic energy of all entities
    float GetTotalKineticEnergy() const;
    /// Return the kinetic energy of all clumps in a set of families
    //// TODO: Implement it
    float GetClumpKineticEnergy() const;
    //// TODO: float GetTotalKineticEnergy(std::vector<unsigned int> families) const;

    /// Write the current status of clumps to a file
    void WriteClumpFile(const std::string& outfilename) const;
    /// Write the current status of `clumps' to a file, but not as clumps, instead, as each individual sphere. This may
    /// make small-scale rendering easier.
    void WriteSphereFile(const std::string& outfilename) const;

    /// Read clump coordinates from a CSV file (whose format is consistent with this solver's clump output file).
    /// Returns an unordered_map which maps each unique clump type name to a vector of float3 (XYZ coordinates).
    static std::unordered_map<std::string, std::vector<float3>> ReadClumpXyzFromCsv(
        const std::string& infilename,
        const std::string& clump_header = DEM_OUTPUT_FILE_CLUMP_TYPE_NAME,
        const std::string& x_header = DEM_OUTPUT_FILE_X_COL_NAME,
        const std::string& y_header = DEM_OUTPUT_FILE_Y_COL_NAME,
        const std::string& z_header = DEM_OUTPUT_FILE_Z_COL_NAME) {
        std::unordered_map<std::string, std::vector<float3>> type_xyz_map;
        io::CSVReader<4, io::trim_chars<' ', '\t'>, io::no_quote_escape<','>, io::throw_on_overflow,
                      io::empty_line_comment>
            in(infilename);
        in.read_header(io::ignore_extra_column, clump_header, x_header, y_header, z_header);
        std::string type_name;
        float3 XYZ;
        size_t count = 0;
        while (in.read_row(type_name, XYZ.x, XYZ.y, XYZ.z)) {
            type_xyz_map[type_name].push_back(XYZ);
            count++;
        }
        return type_xyz_map;
    }
    /// Read clump quaternions from a CSV file (whose format is consistent with this solver's clump output file).
    /// Returns an unordered_map which maps each unique clump type name to a vector of float4 (4 components of the
    /// quaternion, (1, 0, 0, 0) means 0 rotation).
    static std::unordered_map<std::string, std::vector<float4>> ReadClumpQuatFromCsv(
        const std::string& infilename,
        const std::string& clump_header = DEM_OUTPUT_FILE_CLUMP_TYPE_NAME,
        const std::string& q0_header = "Q0",
        const std::string& q1_header = "Q1",
        const std::string& q2_header = "Q2",
        const std::string& q3_header = "Q3") {
        std::unordered_map<std::string, std::vector<float4>> type_Q_map;
        io::CSVReader<5, io::trim_chars<' ', '\t'>, io::no_quote_escape<','>, io::throw_on_overflow,
                      io::empty_line_comment>
            in(infilename);
        in.read_header(io::ignore_extra_column, clump_header, q0_header, q1_header, q2_header, q3_header);
        std::string type_name;
        float4 Q;
        size_t count = 0;
        while (in.read_row(type_name, Q.x, Q.y, Q.z, Q.w)) {
            type_Q_map[type_name].push_back(Q);
            count++;
        }
        return type_Q_map;
    }

    /// Intialize the simulation system
    void Initialize();

    /// Advance simulation by this amount of time, and at the end of this call, synchronize kT and dT. This is suitable
    /// for a longer call duration and without co-simulation.
    void DoDynamicsThenSync(double thisCallDuration);

    /// Advance simulation by this amount of time (but does not attempt to sync kT and dT). This can work with both long
    /// and short call durations and allows interplay with co-simulation APIs.
    void DoDynamics(double thisCallDuration);

    /// Equivalent to calling DoDynamics with the time step size as the argument
    void DoStepDynamics() { DoDynamics(m_ts_size); }

    /// Copy the cached sim params to the GPU-accessible managed memory, so that they are picked up from the next ts of
    /// simulation. Usually used when you want to change simulation parameters after the system is already Intialized.
    void UpdateSimParams();

    /// Transfer newly loaded clumps to the GPU-side in mid-simulation
    void UpdateClumps();

    /// Show the collaboration stats between dT and kT. This is more useful for tweaking the number of time steps that
    /// dT should be allowed to be in advance of kT.
    void ShowThreadCollaborationStats();

    /// Show the wall time and percentages of wall time spend on various solver tasks
    void ShowTimingStats();

    /// Reset the collaboration stats between dT and kT back to the initial value (0). You should call this if you want
    /// to start over and re-inspect the stats of the new run; otherwise, it is generally not needed, you can go ahead
    /// and destroy DEMSolver.
    void ClearThreadCollaborationStats();

    /// Reset the recordings of the wall time and percentages of wall time spend on various solver tasks
    void ClearTimingStats();

    /// Removes all entities associated with a family from the arrays (to save memory space)
    void PurgeFamily(unsigned int family_num);

    /// Release the memory for the flattened arrays (which are used for initialization pre-processing and transferring
    /// info the worker threads)
    void ReleaseFlattenedArrays();

    /*
      protected:
        DEMSolver() : m_sys(nullptr) {}
        DEMSolver_impl* m_sys;
    */

    // Choose between outputting particles as individual component spheres (results in larger files but less
    // post-processing), or as owner clumps (e.g. xyz location means clump CoM locations, etc.), by
    // DEM_OUTPUT_MODE::SPHERE and DEM_OUTPUT_MODE::CLUMP options.
    // NOTE: I did not implement this functionality; the flavor of output depends on the actual write-to-file function
    // call.
    // void SetClumpOutputMode(DEM_OUTPUT_MODE mode) { m_clump_out_mode = mode; }

    /// Choose output format
    void SetOutputFormat(DEM_OUTPUT_FORMAT format) { m_out_format = format; }
    /// Specify the information that needs to go into the output files
    void SetOutputContent(unsigned int content) { m_out_content = content; }

    /// Let dT do this call and return the reduce value of the inspected quantity
    float dTInspectReduce(const std::shared_ptr<jitify::Program>& inspection_kernel,
                          const std::string& kernel_name,
                          DEM_INSPECT_ENTITY_TYPE thing_to_insp,
                          DEM_CUB_REDUCE_FLAVOR reduce_flavor,
                          bool all_domain);

  private:
    ////////////////////////////////////////////////////////////////////////////////
    // Flag-like behavior-related variables cached on the host side
    ////////////////////////////////////////////////////////////////////////////////

    // Verbosity
    DEM_VERBOSITY verbosity = INFO;
    // If true, kT should sort contact arrays then transfer them to dT
    bool kT_should_sort = true;
    // NOTE: compact force calculation (in the hope to use shared memory) is not implemented
    bool use_compact_sweep_force_strat = false;
    // If true, the solvers may need to do a per-step sweep to apply family number changes
    bool famnum_can_change_conditionally = false;

    // Force model, as a string
    std::string m_force_model = DEM_HERTZIAN_FORCE_MODEL();
    bool use_user_defined_force_model = false;
    // Should jitify clump template into kernels
    bool jitify_clump_templates = false;
    // Should jitify mass/MOI properties into kernels
    bool jitify_mass_moi = false;
    // CD uses one thread (not one block) to process a bin
    bool use_one_bin_per_thread = false;

    // User explicitly set a bin size to use
    bool use_user_defined_bin_size = false;
    // User explicity specify a expand factor to use
    bool use_user_defined_expand_factor = false;

    // I/O related flags
    // DEM_OUTPUT_MODE m_clump_out_mode = DEM_OUTPUT_MODE::SPHERE;
    DEM_OUTPUT_FORMAT m_out_format = DEM_OUTPUT_FORMAT::CSV;
    unsigned int m_out_content = DEM_OUTPUT_CONTENT::QUAT | DEM_OUTPUT_CONTENT::ABSV;

    // User instructed simulation `world' size. Note it is an approximate of the true size and we will generate a world
    // not smaller than this.
    float3 m_user_boxSize = make_float3(-1.f);

    // Exact `World' size along X dir (determined at init time)
    float m_boxX = -1.f;
    // Exact `World' size along Y dir (determined at init time)
    float m_boxY = -1.f;
    // Exact `World' size along Z dir (determined at init time)
    float m_boxZ = -1.f;
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
    // User-instructed approximate maximum velocity (of any point on a body in the simulation)
    float m_approx_max_vel = -1.f;

    // The number of user-estimated (max) number of owners that will be present in the simulation. If 0, then the arrays
    // will just be resized at intialization based on the input size.
    size_t m_instructed_num_owners = 0;

    // Whether the number of voxels and length unit l is explicitly given by the user
    bool explicit_nv_override = false;
    // Whether the GPU-side systems have been initialized
    bool sys_initialized = false;
    // Smallest sphere radius (used to let the user know whether the expand factor is sufficient)
    float m_smallest_radius = FLT_MAX;

    // The number of dT steps before it waits for a kT update. The default value 0 means every dT step will wait for a
    // newly produced contact-pair info (from kT) before proceeding.
    int m_updateFreq = 0;

    // The contact model is historyless, or not. It affects jitification.
    bool m_isHistoryless = false;

    // Where the user wants the origin of the coordinate system to be
    std::string m_user_instructed_origin = "explicit";

    // If and how we should add boundaries to the simulation world upon initialization. Choose between none, all and
    // top_open.
    std::string m_user_add_bounding_box = "none";
    // And the material should be used for the bounding BCs
    std::shared_ptr<DEMMaterial> m_bounding_box_material;

    // If we should ensure that when kernel jitification fails, the line number reported reflexes where error happens
    bool m_ensure_kernel_line_num = false;

    ////////////////////////////////////////////////////////////////////////////////
    // No user method is provided to modify the following key quantities, even if
    // there are entites added to/removed from the simulation, in which case
    // they will just be modified. At the time these quantities should be cleared,
    // the user might as well reconstruct the simulator.
    ////////////////////////////////////////////////////////////////////////////////

    // Cached tracked objects that can be leveraged by the user to assume explicit control over some simulation objects
    std::vector<std::shared_ptr<DEMTrackedObj>> m_tracked_objs;
    // std::vector<std::shared_ptr<DEMTracker>> m_trackers;

    // Cached inspectors that can be used to query the simulation system
    std::vector<std::shared_ptr<DEMInspector>> m_inspectors;

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

    // Number of batches of clumps loaded by the user. Note this number never decreases, it just records how many times
    // the user loaded clumps into the simulation for the duration of this class.
    size_t nBatchClumpsLoad = 0;
    // Number of times when an external (analytical) object is loaded by the user. Never decreases.
    unsigned int nExtObjLoad = 0;
    // Number of times when a meshed object is loaded by the user. Never decreses.
    size_t nTriObjLoad = 0;
    // Number of clump templates loaded. Never decreases.
    size_t nClumpTemplateLoad = 0;

    // The above quantities, when they were last time initialized. Used for sanity checks at user re-initialization.
    size_t nLastTimeClumpTemplateLoad = 0;
    unsigned int nLastTimeExtObjLoad = 0;
    size_t nLastTimeBatchClumpsLoad = 0;
    size_t nLastTimeTriObjLoad = 0;
    unsigned int nLastTimeMatNum = 0;
    unsigned int nLastTimeClumpTemplateNum = 0;
    unsigned int nLastTimeFamilyPreNum = 0;

    ////////////////////////////////////////////////////////////////////////////////
    // These quantities will be reset at the time of jitification or re-jitification,
    // but not when entities are added to/removed from the simulation. No method is
    // provided to directly modify them as it is not needed.
    ////////////////////////////////////////////////////////////////////////////////

    // Num of sphere components that all clump templates have
    unsigned int nDistinctClumpComponents;

    // Num of clump templates types, basically. It's also the number of clump template mass properties.
    unsigned int nDistinctClumpBodyTopologies;

    // A design choice is that each analytical obj and meshed obj is its own mass type, so the following 2 quantities
    // are not independent, so we just won't use them Num of analytical objects loaded unsigned int
    // nExtObjMassProperties; Num of meshed objects loaded unsigned int nMeshMassProperties;

    // Sum of the above 3 items (but in fact nDistinctClumpBodyTopologies + nExtObj + nTriEntities)
    unsigned int nDistinctMassProperties;

    // Num of material types
    unsigned int nMatTuples;

    // Not used anymore
    // unsigned int nDistinctFamilies;

    // This many clump template can be jitified, and the rest need to exist in global memory
    // Note all `mass' properties are jitified, it's just this many clump templates' component info will not be
    // jitified. Therefore, this quantity does not seem to be useful beyond reporting to the user.
    unsigned int nJitifiableClumpTopo;
    // Number of jitified clump components
    unsigned int nJitifiableClumpComponents;

    ////////////////////////////////////////////////////////////////////////////////
    // Cached user's direct (raw) inputs concerning the actual physics objects
    // presented in the simulation, which need to be processed before shipment,
    // at initialization time. These items can be cleared before users add entites
    // from the simulation on-the-fly, and be loaded with new entities; but this
    // is the responsibility of the user. If the user does not clear these cached
    // data and then re-initialize using the `Overwrite' style, then it is like
    // starting the original simulation over. If the user clear these, then add
    // some new data, and then re-initialize using the `Add' style, it is like adding
    // more entities to the existing simulation system.
    ////////////////////////////////////////////////////////////////////////////////
    //// TODO: These re-initialization flavors haven't been added

    // This is the cached material information.
    // It will be massaged into the managed memory upon Initialize().
    std::vector<std::shared_ptr<DEMMaterial>> m_loaded_materials;

    // This is the cached clump structure information. Note although not stated explicitly, those are only `clump'
    // templates, not including triangles, analytical geometries etc.
    std::vector<std::shared_ptr<DEMClumpTemplate>> m_templates;

    // Shared pointers to a batch of clumps loaded into the system. Through this returned handle, the user can further
    // specify the vel, ori etc. of this batch of clumps.
    std::vector<std::shared_ptr<DEMClumpBatch>> cached_input_clump_batches;

    // Shared pointers to analytical objects cached at the API system
    std::vector<std::shared_ptr<DEMExternObj>> cached_extern_objs;

    // Shared pointers to meshed objects cached at the API system
    std::vector<std::shared_ptr<DEMMeshConnected>> cached_mesh_objs;

    // User-input prescribed motion
    std::vector<familyPrescription_t> m_input_family_prescription;
    // TODO: fixed particles should automatically attain status indicating they don't interact with each other.
    // The familes that should not be outputted
    std::set<unsigned int> m_no_output_families;
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

    ////////////////////////////////////////////////////////////////////////////////
    // Flattened and sometimes processed user inputs, ready to be transferred to
    // worker threads. Will be automatically cleared after initialization.
    ////////////////////////////////////////////////////////////////////////////////

    // Family mask that helps determine whether between 2 families there should be contacts, or not
    std::vector<notStupidBool_t> m_family_mask_matrix;
    // Map between clump templates and the user-assigned name. It is needed when the clump is outputted to files.
    std::unordered_map<unsigned int, std::string> m_template_number_name_map;

    // Unlike clumps, external objects do not have _types (each is its own type)
    std::vector<float3> m_input_ext_obj_xyz;
    std::vector<float4> m_input_ext_obj_rot;
    std::vector<unsigned int> m_input_ext_obj_family;
    // Mesh is also flattened before sending to kT and dT
    std::vector<float3> m_input_mesh_obj_xyz;
    std::vector<float4> m_input_mesh_obj_rot;
    std::vector<unsigned int> m_input_mesh_obj_family;

    // Processed unique family prescription info
    std::vector<familyPrescription_t> m_unique_family_prescription;

    // Flattened array of all family numbers the user used. This needs to be prepared each time at initialization time
    // since we need to know the range and amount of unique family numbers the user used, as we did not restrict what
    // naming scheme the user must use when defining family numbers.
    std::vector<unsigned int> m_input_clump_family;

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

    // These extra mesh facets' owners' ID will be appended to analytical entities'
    std::vector<unsigned int> m_mesh_facet_owner;
    // Material types of these mesh facets
    std::vector<materialsOffset_t> m_mesh_facet_materials;
    // Material types of these mesh facets
    std::vector<DEMTriangle> m_mesh_facets;

    // Clump templates will be flatten and transferred into kernels upon Initialize()
    std::vector<float> m_template_clump_mass;
    std::vector<float3> m_template_clump_moi;
    std::vector<std::vector<unsigned int>> m_template_sp_mat_ids;
    std::vector<std::vector<float>> m_template_sp_radii;
    std::vector<std::vector<float3>> m_template_sp_relPos;
    // Analytical objects that will be flatten and transferred into kernels upon Initialize()
    std::vector<float> m_ext_obj_mass;
    std::vector<float3> m_ext_obj_moi;
    // Meshed objects that will be flatten and transferred into kernels upon Initialize()
    std::vector<float> m_mesh_obj_mass;
    std::vector<float3> m_mesh_obj_moi;
    /*
    // Dan and Ruochun decided NOT to extract unique input values.
    // Instead, we trust users: we simply store all clump template info users give.
    // So this unique-value-extractor block is disabled and commented.

    // unique clump masses derived from m_template_clump_mass
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

    // Materials info is processed at API level (on initialization) for generating proxy arrays
    std::vector<float> m_E_proxy;
    std::vector<float> m_nu_proxy;
    std::vector<float> m_CoR_proxy;
    std::vector<float> m_mu_proxy;
    std::vector<float> m_Crr_proxy;

    ////////////////////////////////////////////////////////////////////////////////
    // DEM system's workers, helpers, friends
    ////////////////////////////////////////////////////////////////////////////////

    WorkerReportChannel* kTMain_InteractionManager;
    WorkerReportChannel* dTMain_InteractionManager;
    GpuManager* dTkT_GpuManager;
    ThreadManager* dTkT_InteractionManager;
    DEMKinematicThread* kT;
    DEMDynamicThread* dT;

    ////////////////////////////////////////////////////////////////////////////////
    // DEM system's private methods
    ////////////////////////////////////////////////////////////////////////////////

    /// Pre-process some user inputs regarding the simulation entities that fill up the world, so we acquire the
    /// knowledge on how to jitify the kernels
    void generateEntityResources();
    /// Pre-process some user inputs regarding the (sizes, features of) simulation world
    void generatePolicyResources();
    /// Must be called after generateEntityResources and generatePolicyResources to wrap the info in the previous steps
    /// up
    void postResourceGen();
    /// Make sure the input represents something we can simulate, and if not, tell the reasons
    void postResourceGenChecksAndTabKeeping();
    /// Flatten some input clump information, to figure out the size of the input, and their associated family numbers
    /// (to make jitifying family policies easier)
    void preprocessClumps();
    /// Flatten cached clump templates (from ClumpTemplate structs to float arrays)
    void preprocessClumpTemplates();
    /// Count the number of `things' that should be in the simulation now
    void updateTotalEntityNum();
    /// Jitify GPU kernels, based on pre-processed user inputs
    void jitifyKernels();
    /// Figure out the unit length l and numbers of voxels along each direction, based on domain size X, Y, Z
    void figureOutNV();
    /// Derive the origin of the coordinate system using user inputs
    void figureOutOrigin();
    /// Set the default bin (for contact detection) size to be the same of the smallest sphere
    void decideBinSize();
    /// Add boundaries to the simulation `world' based on user instructions
    void addWorldBoundingBox();
    /// Transfer cached solver preferences/instructions to dT and kT.
    void transferSolverParams();
    /// Transfer (CPU-side) cached simulation data (about sim world) to the GPU-side. It is called automatically during
    /// system initialization.
    void transferSimParams();
    /// Transfer cached clump templates info etc. to GPU-side arrays
    void initializeGPUArrays();
    /// Allocate memory space for GPU-side arrays
    void allocateGPUArrays();
    /// Pack array pointers to a struct so they can be easily used as kernel arguments
    void packDataPointers();
    /// Warn users if the data types defined in DEMDefines.h do not blend well with the user inputs (fist-round
    /// coarse-grain sanity check)
    void validateUserInputs();
    /// Prepare the material/contact proxy matrix force computation kernels
    void figureOutMaterialProxies();
    /// Figure out info about external objects and how they should be jitified
    void preprocessAnalyticalObjs();
    /// Figure out info about external meshed objects
    void preprocessTriangleObjs();
    /// Report simulation stats at initialization
    inline void reportInitStats() const;
    /// Based on user input, prepare family_mask_matrix (family contact map matrix)
    void figureOutFamilyMasks();
    /// Reset kT and dT back to a status like when the simulation system is constructed. I decided to make this a
    /// private method because it can be dangerous, as if it is called when kT is waiting at the outer loop, it will
    /// stall the siumulation. So perhaps the user should not call it without knowing what they are doing. Also note
    /// this call does not reset the collaboration log between kT and dT.
    void resetWorkerThreads();
    /// Transfer newly loaded clumps/meshed objects to the GPU-side in mid-simulation and allocate GPU memory space for
    /// them
    void updateClumpMeshArrays(size_t nOwners, size_t nClumps, size_t nSpheres, size_t nTriMesh, size_t nFacets);
    /// Add content to the flattened analytical component array.
    /// Note that analytical component is big different in that they each has a position in the jitified analytical
    /// templates, insteads of like a clump, has an extra ComponentOffset array points it to the right jitified template
    /// location.
    void addAnalCompTemplate(const objType_t type,
                             const std::shared_ptr<DEMMaterial>& material,
                             const unsigned int owner,
                             const float3 pos,
                             const float3 rot = make_float3(0),
                             const float d1 = 0.f,
                             const float d2 = 0.f,
                             const float d3 = 0.f,
                             const objNormal_t normal = DEM_ENTITY_NORMAL_INWARD);

    // Some JIT packaging helpers
    inline void equipClumpTemplates(std::unordered_map<std::string, std::string>& strMap);
    inline void equipSimParams(std::unordered_map<std::string, std::string>& strMap);
    inline void equipMassMOI(std::unordered_map<std::string, std::string>& strMap);
    inline void equipMaterials(std::unordered_map<std::string, std::string>& strMap);
    inline void equipAnalGeoTemplates(std::unordered_map<std::string, std::string>& strMap);
    // inline void equipFamilyMasks(std::unordered_map<std::string, std::string>& strMap);
    inline void equipFamilyPrescribedMotions(std::unordered_map<std::string, std::string>& strMap);
    inline void equipFamilyOnFlyChanges(std::unordered_map<std::string, std::string>& strMap);
    inline void equipForceModel(std::unordered_map<std::string, std::string>& strMap);
};

}  // namespace sgps

#endif
