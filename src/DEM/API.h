//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_API
#define DEME_API

#include <vector>
#include <set>
#include <cfloat>

#include <core/ApiVersion.h>
#include <DEM/kT.h>
#include <DEM/dT.h>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/ThreadManager.h>
#include <core/utils/GpuManager.h>
#include <core/utils/DEMEPaths.h>
#include <nvmath/helper_math.cuh>
#include <DEM/Defines.h>
#include <DEM/Structs.h>
#include <DEM/BdrsAndObjs.h>
#include <DEM/Models.h>
#include <DEM/AuxClasses.h>

/// Main namespace for the DEM-Engine package.
namespace deme {

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
//            5. Check if entities are initially in box
//            6. Force model has a/several custom owner arrays to store custom config data
//            7. This custom array can be defined at clump template/anal obj/mesh obj generation
//            8. Right now force model position is wrt LBF, not user origin...
//            9. wT takes care of an extra output when it crashes
//            10. Recover sph--mesh contact pairs in restarted sim by mesh name
//            11. A dry-run to map contact pair file with current clump batch based on cnt points location
//                  (this is done by fake an initialization with this batch)
//            12. Right now, the force model file is loaded from source, not install dir?
//////////////////////////////////////////////////////////////

/// Main DEM-Engine solver.
class DEMSolver {
  public:
    DEMSolver(unsigned int nGPUs = 2);
    ~DEMSolver();

    /// Set output detail level
    void SetVerbosity(VERBOSITY verbose) { verbosity = verbose; }

    /// Instruct the dimension of the `world'. On initialization, this info will be used to figure out how to assign the
    /// num of voxels in each direction. If your `useful' domain is not box-shaped, then define a box that contains your
    /// domian.
    void InstructBoxDomainDimension(float x, float y, float z, SPATIAL_DIR dir_exact = SPATIAL_DIR::NONE);

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

    /// Sets the origin of the coordinate system (the coordinate system that all entities in this simulation are using)
    /// using a string identifier. Choose from "center" (being at the center of the user-specifier BoxDomain), or "0",
    /// "1", ..., "13", "-1", ..., "-13", which are the balanced ternary-identifier for points in the 8 octants, see
    /// https://en.wikipedia.org/wiki/Octant_(solid_geometry).
    void SetCoordSysOrigin(const std::string& where) { m_user_instructed_origin = where; }

    /// Sets the origin of the coordinate system (the coordinate system that all entities in this simulation are using).
    /// This origin point should be reported as the vector pointing from the left-bottom-front point of your simulation
    /// `world' (box domain), to the origin of the coordinate system. For example, if the world size is [2,2,2] and the
    /// origin should be right at its center, then the argument for this call should be make_float3(1,1,1). The other
    /// alternative to achieve the same, is through calling SetBoxDomainLBFPoint.
    void SetCoordSysOrigin(float3 O) {
        m_boxLBF = -O;
        m_user_instructed_origin = "explicit";
    }

    /// Set the left-bottom-front point of the box domain (or the user-specified simulation `world'), in the user's
    /// global coordinate system. The other alternative to achieve the same, is through calling SetCoordSysOrigin.
    void SetBoxDomainLBFPoint(float3 O) {
        m_boxLBF = O;
        m_user_instructed_origin = "explicit";
    }

    /// Set the integrator for this simulator
    void SetIntegrator(TIME_INTEGRATOR intg) { m_integrator = intg; }

    /// Return whether this simulation system is initialized
    bool GetInitStatus() { return sys_initialized; }

    /// Get the jitification string substitution laundary list. It is needed by some of this simulation system's friend
    /// classes.
    std::unordered_map<std::string, std::string> GetJitStringSubs() { return m_subs; }

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

    /// Instruct the solver to use frictonal (history-based) Hertzian contact force model
    std::shared_ptr<DEMForceModel> UseFrictionalHertzianModel();
    /// Instruct the solver to use frictonless Hertzian contact force model
    std::shared_ptr<DEMForceModel> UseFrictionlessHertzianModel();
    /// Define a custom contact force model by a string. Returns a shared_ptr to the force model in use.
    std::shared_ptr<DEMForceModel> DefineContactForceModel(const std::string& model);
    /// Read user custom contact force model from a file (which by default should reside in kernel/DEMUserScripts).
    /// Returns a shared_ptr to the force model in use.
    std::shared_ptr<DEMForceModel> ReadContactForceModel(const std::string& filename);

    /// Instruct the solver if contact pair arrays should be sorted (based on the types of contacts) before usage. This
    /// can potentially make dT run faster. This is currently not implemented due to technical difficulties.
    void SetSortContactPairs(bool use_sort = true) { should_sort_contacts = use_sort; }

    /// Instruct the solver to rearrange and consolidate clump templates information, then jitify it into GPU kernels
    /// (if set to true), rather than using flattened sphere component configuration arrays whose entries are associated
    /// with individual spheres. Note: setting it to true gives no performance benefit known to me.
    void SetJitifyClumpTemplates(bool use = true) { jitify_clump_templates = use; }
    /// Instruct the solver to rearrange and consolidate mass property information (for all owner types), then jitify it
    /// into GPU kernels (if set to true), rather than using flattened mass property arrays whose entries are associated
    /// with individual owners. Note: setting it to true gives no performance benefit known to me.
    void SetJitifyMassProperties(bool use = true) { jitify_mass_moi = use; }

    /// Instruct the contact detection process to use one thread to process a bin (if true), instead of using a block to
    /// process a bin. This probably also requires you to manually set a smaller DEME_MAX_SPHERES_PER_BIN. This can
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

    /// Set the number of threads per block in force calculation (default 256)
    void SetForceCalcThreadsPerBlock(unsigned int nTh) { dT->DT_FORCE_CALC_NTHREADS_PER_BLOCK = nTh; }

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
    std::shared_ptr<DEMClumpTemplate> LoadSphereType(float mass,
                                                     float radius,
                                                     const std::shared_ptr<DEMMaterial>& material);

    /// Load materials properties (Young's modulus, Poisson's ratio, Coeff of Restitution...) into
    /// the API-level cache. Return the ptr of the material type just loaded.
    std::shared_ptr<DEMMaterial> LoadMaterial(const std::unordered_map<std::string, float>& mat_prop);

    /// Get position of a owner
    float3 GetOwnerPosition(bodyID_t ownerID) const;
    /// Get angular velocity of a owner
    float3 GetOwnerAngVel(bodyID_t ownerID) const;
    /// Get quaternion of a owner
    float4 GetOwnerOriQ(bodyID_t ownerID) const;
    /// Get velocity of a owner
    float3 GetOwnerVelocity(bodyID_t ownerID) const;
    /// Get the acceleration of a owner
    float3 GetOwnerAcc(bodyID_t ownerID) const;
    /// Get the angular acceleration of a owner
    float3 GetOwnerAngAcc(bodyID_t ownerID) const;
    /// Set position of a owner in user unit
    void SetOwnerPosition(bodyID_t ownerID, float3 pos);
    /// Set angular velocity of a owner
    void SetOwnerAngVel(bodyID_t ownerID, float3 angVel);
    /// Set velocity of a owner
    void SetOwnerVelocity(bodyID_t ownerID, float3 vel);
    /// Set quaternion of a owner
    void SetOwnerOriQ(bodyID_t ownerID, float4 oriQ);
    /// Rewrite the relative positions of the flattened triangle soup, starting from `start', using triangle nodal
    /// positions in `triangles'. If `overwrite' is true, then it is overwriting the existing nodal info; otherwise it
    /// just adds to it.
    void SetTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& triangles, bool overwrite = true);

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
                                                             const std::shared_ptr<DEMMaterial>& mat,
                                                             bool load_normals = true,
                                                             bool load_uv = false);
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
    /// Create a DEMTracker to allow direct control/modification/query to this triangle mesh object
    std::shared_ptr<DEMTracker> Track(std::shared_ptr<DEMMeshConnected>& obj);

    /// Create a inspector object that can help query some statistical info of the clumps in the simulation
    std::shared_ptr<DEMInspector> CreateInspector(const std::string& quantity = "clump_max_z");
    std::shared_ptr<DEMInspector> CreateInspector(const std::string& quantity, const std::string& region);

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
    /// Let the linear velocities of all entites in this family always keep `as is', and not fluenced by the force
    /// exerted from other simulation entites.
    void SetFamilyPrescribedLinVel(unsigned int ID);
    /// Set the prescribed angular velocity to all entities in a family. If dictate is set to true, then this
    /// prescription completely dictates this family's motions.
    void SetFamilyPrescribedAngVel(unsigned int ID,
                                   const std::string& velX,
                                   const std::string& velY,
                                   const std::string& velZ,
                                   bool dictate = true);
    /// Let the linear velocities of all entites in this family always keep `as is', and not fluenced by the force
    /// exerted from other simulation entites.
    void SetFamilyPrescribedAngVel(unsigned int ID);

    /// Keep the positions of all entites in this family to remain exactly the user-specified values
    void SetFamilyPrescribedPosition(unsigned int ID, const std::string& X, const std::string& Y, const std::string& Z);
    /// Let the positions of all entites in this family always keep `as is'
    void SetFamilyPrescribedPosition(unsigned int ID);
    /// Keep the orientation quaternions of all entites in this family to remain exactly the user-specified values
    void SetFamilyPrescribedQuaternion(unsigned int ID, const std::string& q_formula);

    /// Globally modify a owner wildcard's value
    void SetOwnerWildcardValue(const std::string& name, float val);
    /// Modify the owner wildcard values of all entities in family N
    void SetFamilyOwnerWildcardValue(unsigned int N, const std::string& name, float val);

    /// Change all entities with family number ID_from to have a new number ID_to, when the condition defined by the
    /// string is satisfied by the entities in question. This should be called before initialization, and will be baked
    /// into the solver, so the conditions will be checked and changes applied every time step.
    void ChangeFamilyWhen(unsigned int ID_from, unsigned int ID_to, const std::string& condition);

    /// Change all entities with family number ID_from to have a new number ID_to, immediately. This is callable when kT
    /// and dT are hanging, not when they are actively working, or the behavior is not defined.
    void ChangeFamily(unsigned int ID_from, unsigned int ID_to);

    /// Change the sizes of the clumps by a factor. This method directly works on the clump components spheres,
    /// therefore requiring sphere components to be store in flattened array (default behavior), not jitified templates.
    void ChangeClumpSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors);

    /// If true, each jitification string substitution will do a one-liner to one-liner replacement, so that if the
    /// kernel compilation fails, the error meessage line number will reflex the actual spot where that happens (instead
    /// of some random number)
    void EnsureKernelErrMsgLineNum(bool flag = true) { ensure_kernel_line_num = flag; }

    /// Whether the force collection (acceleration calc and reduction) process should be using CUB. If true, the
    /// acceleration array is flattened and reduced using CUB; if false, the acceleration is computed and directly
    /// applied to each body through atomic operations.
    void UseCubForceCollection(bool flag = true) { use_cub_to_reduce_force = flag; }

    /// Reduce contact forces to accelerations right after calculating them, in the same kernel. This may give some
    /// performance boost if you have only polydisperse spheres, no clumps.
    void SetCollectAccRightAfterForceCalc(bool flag = true) { collect_force_in_force_kernel = flag; }

    /// Instruct the solver that there is no need to record the contact force (and contact point location etc.) in an
    /// array. If set to true, the contact forces must be reduced to accelerations right in the force calculation kernel
    /// (meaning SetCollectAccRightAfterForceCalc is effectively called too). Calling this method could reduce some
    /// memory usage, but will disable contact pair output.
    void SetNoForceRecord(bool flag = true) {
        no_recording_contact_forces = flag;
        if (flag)
            collect_force_in_force_kernel = flag;
    }

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
    void WriteClumpFile(const std::string& outfilename, unsigned int accuracy = 10) const;
    /// Write the current status of `clumps' to a file, but not as clumps, instead, as each individual sphere. This may
    /// make small-scale rendering easier.
    void WriteSphereFile(const std::string& outfilename) const;
    /// Write all contact pairs to a file
    void WriteContactFile(const std::string& outfilename, float force_thres = DEME_TINY_FLOAT) const;
    /// Write the current status of all meshes to a file
    void WriteMeshFile(const std::string& outfilename) const;

    /// Read clump coordinates from a CSV file (whose format is consistent with this solver's clump output file).
    /// Returns an unordered_map which maps each unique clump type name to a vector of float3 (XYZ coordinates).
    static std::unordered_map<std::string, std::vector<float3>> ReadClumpXyzFromCsv(
        const std::string& infilename,
        const std::string& clump_header = OUTPUT_FILE_CLUMP_TYPE_NAME,
        const std::string& x_header = OUTPUT_FILE_X_COL_NAME,
        const std::string& y_header = OUTPUT_FILE_Y_COL_NAME,
        const std::string& z_header = OUTPUT_FILE_Z_COL_NAME) {
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
    /// quaternion, (Qx, Qy, Qz, Qw) = (0, 0, 0, 1) means 0 rotation).
    static std::unordered_map<std::string, std::vector<float4>> ReadClumpQuatFromCsv(
        const std::string& infilename,
        const std::string& clump_header = OUTPUT_FILE_CLUMP_TYPE_NAME,
        const std::string& qw_header = "Qw",
        const std::string& qx_header = "Qx",
        const std::string& qy_header = "Qy",
        const std::string& qz_header = "Qz") {
        std::unordered_map<std::string, std::vector<float4>> type_Q_map;
        io::CSVReader<5, io::trim_chars<' ', '\t'>, io::no_quote_escape<','>, io::throw_on_overflow,
                      io::empty_line_comment>
            in(infilename);
        in.read_header(io::ignore_extra_column, clump_header, qw_header, qx_header, qy_header, qz_header);
        std::string type_name;
        float4 Q;
        size_t count = 0;
        while (in.read_row(type_name, Q.w, Q.x, Q.y, Q.z)) {
            type_Q_map[type_name].push_back(Q);
            count++;
        }
        return type_Q_map;
    }

    /// Read all contact pairs (geometry ID) from a contact file
    static std::vector<std::pair<bodyID_t, bodyID_t>> ReadContactPairsFromCsv(
        const std::string& infilename,
        const std::string& cntType = OUTPUT_FILE_SPH_SPH_CONTACT_NAME,
        const std::string& cntColName = OUTPUT_FILE_CNT_TYPE_NAME,
        const std::string& first_name = OUTPUT_FILE_GEO_ID_1_NAME,
        const std::string& second_name = OUTPUT_FILE_GEO_ID_2_NAME) {
        io::CSVReader<3, io::trim_chars<' ', '\t'>, io::no_quote_escape<','>, io::throw_on_overflow,
                      io::empty_line_comment>
            in(infilename);
        in.read_header(io::ignore_extra_column, cntColName, first_name, second_name);
        bodyID_t A, B;
        std::string cnt_type_name;
        std::vector<std::pair<bodyID_t, bodyID_t>> pairs;
        size_t count = 0;
        while (in.read_row(cnt_type_name, A, B)) {
            if (cnt_type_name == cntType) {  // only the type of contact we care
                pairs.push_back(std::pair<bodyID_t, bodyID_t>(A, B));
                count++;
            }
        }
        return pairs;
    }

    /// Read all contact wildcards from a contact file
    static std::unordered_map<std::string, std::vector<float>> ReadContactWildcardsFromCsv(
        const std::string& infilename,
        const std::string& cntType = OUTPUT_FILE_SPH_SPH_CONTACT_NAME,
        const std::string& cntColName = OUTPUT_FILE_CNT_TYPE_NAME) {
        io::LineReader in_header(infilename);
        char* f_header = in_header.next_line();
        std::vector<std::string> header_names = parse_string_line(std::string(f_header));
        std::vector<std::string> wildcard_names;
        // Find those col names that are not contact file standard names: they have to be wildcard names
        for (const auto& col_name : header_names) {
            if (!check_exist(CNT_FILE_KNOWN_COL_NAMES, col_name)) {
                wildcard_names.push_back(col_name);
            }
        }
        // Now parse in the csv file
        std::unordered_map<std::string, std::vector<float>> w_vals;
        size_t count = 0;
        for (const auto& wildcard_name : wildcard_names) {
            io::CSVReader<2, io::trim_chars<' ', '\t'>, io::no_quote_escape<','>, io::throw_on_overflow,
                          io::empty_line_comment>
                in(infilename);
            in.read_header(io::ignore_extra_column, OUTPUT_FILE_CNT_TYPE_NAME, wildcard_name);
            std::string cnt_type_name;
            float w_val;

            while (in.read_row(cnt_type_name, w_val)) {
                if (cnt_type_name == cntType) {  // only the type of contact we care (SS by default)
                    w_vals[wildcard_name].push_back(w_val);
                    count++;
                }
            }
        }

        return w_vals;
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
    // OUTPUT_MODE::SPHERE and OUTPUT_MODE::CLUMP options.
    // NOTE: I did not implement this functionality; the flavor of output depends on the actual write-to-file function
    // call.
    // void SetClumpOutputMode(OUTPUT_MODE mode) { m_clump_out_mode = mode; }

    /// Choose output format
    void SetOutputFormat(OUTPUT_FORMAT format) { m_out_format = format; }
    /// Specify the information that needs to go into the clump or sphere output files
    void SetOutputContent(unsigned int content) { m_out_content = content; }
    /// Specify the file format of contact pairs
    void SetContactOutputFormat(OUTPUT_FORMAT format) { m_cnt_out_format = format; }
    /// Specify the information that needs to go into the contact pair output files
    void SetContactOutputContent(unsigned int content) { m_cnt_out_content = content; }
    /// Specify the file format of meshes
    void SetMeshOutputFormat(MESH_FORMAT format) { m_mesh_out_format = format; }

    /// Let dT do this call and return the reduce value of the inspected quantity
    float dTInspectReduce(const std::shared_ptr<jitify::Program>& inspection_kernel,
                          const std::string& kernel_name,
                          INSPECT_ENTITY_TYPE thing_to_insp,
                          CUB_REDUCE_FLAVOR reduce_flavor,
                          bool all_domain);

  private:
    ////////////////////////////////////////////////////////////////////////////////
    // Flag-like behavior-related variables cached on the host side
    ////////////////////////////////////////////////////////////////////////////////

    // Verbosity
    VERBOSITY verbosity = INFO;
    // If true, dT should sort contact arrays (based on contact type) before usage (not implemented)
    bool should_sort_contacts = false;
    // If true, the solvers may need to do a per-step sweep to apply family number changes
    bool famnum_can_change_conditionally = false;

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
    // The output file format for clumps and spheres
    // OUTPUT_MODE m_clump_out_mode = OUTPUT_MODE::SPHERE;
    OUTPUT_FORMAT m_out_format = OUTPUT_FORMAT::CSV;
    unsigned int m_out_content = OUTPUT_CONTENT::QUAT | OUTPUT_CONTENT::ABSV;
    // The output file format for contact pairs
    OUTPUT_FORMAT m_cnt_out_format = OUTPUT_FORMAT::CSV;
    // The output file content for contact pairs
    unsigned int m_cnt_out_content = CNT_OUTPUT_CONTENT::GEO_ID | CNT_OUTPUT_CONTENT::FORCE |
                                     CNT_OUTPUT_CONTENT::POINT | CNT_OUTPUT_CONTENT::WILDCARD;
    // The output file format for meshes
    MESH_FORMAT m_mesh_out_format = MESH_FORMAT::VTK;

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

    // Where the user wants the origin of the coordinate system to be
    std::string m_user_instructed_origin = "center";

    // If and how we should add boundaries to the simulation world upon initialization. Choose between none, all and
    // top_open.
    std::string m_user_add_bounding_box = "none";
    // And the material should be used for the bounding BCs
    std::shared_ptr<DEMMaterial> m_bounding_box_material;
    // Along which direction the size of the simulation world representable with our integer-based voxels needs to be
    // exactly the same as user-instructed simulation domain size?
    SPATIAL_DIR m_box_dir_length_is_exact = SPATIAL_DIR::NONE;

    // If we should ensure that when kernel jitification fails, the line number reported reflexes where error happens
    bool ensure_kernel_line_num = false;

    // If we should flatten then reduce forces (true), or use atomic operation to reduce forces (false)
    bool use_cub_to_reduce_force = false;

    // See SetNoForceRecord
    bool no_recording_contact_forces = false;
    // See SetCollectAccRightAfterForceCalc
    bool collect_force_in_force_kernel = false;

    // Integrator type
    TIME_INTEGRATOR m_integrator = TIME_INTEGRATOR::EXTENDED_TAYLOR;

    // The force model which will be used
    std::shared_ptr<DEMForceModel> m_force_model =
        std::make_shared<DEMForceModel>(std::move(DEMForceModel(FORCE_MODEL::HERTZIAN)));

    ////////////////////////////////////////////////////////////////////////////////
    // No user method is provided to modify the following key quantities, even if
    // there are entites added to/removed from the simulation, in which case
    // they will just be modified. At the time these quantities should be cleared,
    // the user might as well reconstruct the simulator.
    ////////////////////////////////////////////////////////////////////////////////

    // All material properties names
    std::set<std::string> m_material_prop_names;

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
    size_t nTriMeshes = 0;
    // nExtObj + nOwnerClumps + nTriMeshes == nOwnerBodies

    // Number of batches of clumps loaded by the user. Note this number never decreases, it just records how many times
    // the user loaded clumps into the simulation for the duration of this class.
    size_t nBatchClumpsLoad = 0;
    // Number of times when an external (analytical) object is loaded by the user. Never decreases.
    unsigned int nExtObjLoad = 0;
    // Number of times when a meshed object is loaded by the user. Never decreses.
    size_t nTriObjLoad = 0;
    // Number of clump templates loaded. Never decreases.
    size_t nClumpTemplateLoad = 0;
    // Number of materials loaded. Never decreases.
    size_t nMaterialsLoad = 0;

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

    // Sum of the above 3 items (but in fact nDistinctClumpBodyTopologies + nExtObj + nTriMeshes)
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

    // A big fat tab for all string replacement that the JIT compiler needs to consider
    std::unordered_map<std::string, std::string> m_subs;

    // A map that records the numbering for user-defined owner wildcards
    std::unordered_map<std::string, unsigned int> m_owner_wc_num;

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
    // Component object normal direction (represented by sign, 1 or -1), defaulting to inward (1). If this object is
    // topologically a plane then this param is meaningless, since its normal is determined by its rotation.
    std::vector<float> m_anal_normals;

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
    std::vector<float> m_template_clump_volume;
    // Analytical objects that will be flatten and transferred into kernels upon Initialize()
    std::vector<float> m_ext_obj_mass;
    std::vector<float3> m_ext_obj_moi;
    std::vector<unsigned int> m_ext_obj_comp_num;  // number of component of each analytical obj
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

    // Number of contact pairs that the user manually added for this initialization call. Note unlike nTriGM or
    // nOwnerBodies and such, this number is temporary, and becomes useless after an initialization call, as we don't
    // generally know the number of contacts, that's kT dT's problem.
    size_t nExtraContacts = 0;

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
    /// Warn users if the data types defined in Defines.h do not blend well with the user inputs (fist-round
    /// coarse-grain sanity check)
    void validateUserInputs();
    /// Prepare the material/contact proxy matrix force computation kernels
    void figureOutMaterialProxies();
    /// Figure out info about external objects and how they should be jitified
    void preprocessAnalyticalObjs();
    /// Figure out info about external meshed objects
    void preprocessTriangleObjs();
    /// Report simulation stats at initialization
    void reportInitStats() const;
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
                             const objNormal_t normal = ENTITY_NORMAL_INWARD);
    /// Assert that the DEM simulation system is initialized
    void assertSysInit(const std::string& method_name);
    /// Assert that the DEM simulation system is not initialized
    void assertSysNotInit(const std::string& method_name);

    // Some JIT packaging helpers
    inline void equipClumpTemplates(std::unordered_map<std::string, std::string>& strMap);
    inline void equipSimParams(std::unordered_map<std::string, std::string>& strMap);
    inline void equipMassMoiVolume(std::unordered_map<std::string, std::string>& strMap);
    inline void equipMaterials(std::unordered_map<std::string, std::string>& strMap);
    inline void equipAnalGeoTemplates(std::unordered_map<std::string, std::string>& strMap);
    // inline void equipFamilyMasks(std::unordered_map<std::string, std::string>& strMap);
    inline void equipFamilyPrescribedMotions(std::unordered_map<std::string, std::string>& strMap);
    inline void equipFamilyOnFlyChanges(std::unordered_map<std::string, std::string>& strMap);
    inline void equipForceModel(std::unordered_map<std::string, std::string>& strMap);
    inline void equipIntegrationScheme(std::unordered_map<std::string, std::string>& strMap);
};

}  // namespace deme

#endif
