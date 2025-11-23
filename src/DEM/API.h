//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_API
#define DEME_API

#include <vector>
#include <set>
#include <cfloat>
#include <functional>

#include "kT.h"
#include "dT.h"
#include "../core/utils/CudaAllocator.hpp"
#include "../core/utils/ThreadManager.h"
#include "../core/utils/GpuManager.h"
#include "../core/utils/DEMEPaths.h"
#include "../core/utils/JitHelper.h"
#include "Defines.h"
#include "Structs.h"
#include "BdrsAndObjs.h"
#include "Models.h"
#include "AuxClasses.h"

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
//            5. Update the game of life demo (it's about model ingredient usage)
//            7. Set the device numbers to use
//            9. wT takes care of an extra output when it crashes
//            10. Recover sph--mesh contact pairs in restarted sim by mesh name
//            11. A dry-run to map contact pair file with current clump batch based on cnt points location
//                  (this is done by fake an initialization with this batch)
//////////////////////////////////////////////////////////////

/// Main DEM-Engine solver.
class DEMSolver {
  public:
    DEMSolver(unsigned int nGPUs = 2);
    ~DEMSolver();

    /// Set output detail level.
    void SetVerbosity(VERBOSITY verbose) { verbosity = verbose; }

    /// Instruct the dimension of the `world'. On initialization, this info will be used to figure out how to assign the
    /// num of voxels in each direction. If your `useful' domain is not box-shaped, then define a box that contains your
    /// domian.
    void InstructBoxDomainDimension(float x, float y, float z, const std::string& dir_exact = "none");
    /// @brief Set the size of the simulation `world'.
    /// @param x Lower and upper limit for X coordinate.
    /// @param y Lower and upper limit for Y coordinate.
    /// @param z Lower and upper limit for Z coordinate.
    /// @param dir_exact The direction for which the user-instructed size must strictly agree with the actual generated
    /// size. Pick between "X", "Y", "Z" or "none".
    void InstructBoxDomainDimension(const std::pair<float, float>& x,
                                    const std::pair<float, float>& y,
                                    const std::pair<float, float>& z,
                                    const std::string& dir_exact = "none");

    /// Instruct if and how we should add boundaries to the simulation world upon initialization. Choose between `none',
    /// `all' (add 6 boundary planes) and `top_open' (add 5 boundary planes and leave the z-directon top open). Also
    /// specifies the material that should be assigned to those bounding boundaries.
    void InstructBoxDomainBoundingBC(const std::string& inst, const std::shared_ptr<DEMMaterial>& mat) {
        m_user_add_bounding_box = inst;
        m_bounding_box_material = mat;
    }

    /// Set gravitational pull.
    void SetGravitationalAcceleration(float3 g) { G = g; }
    void SetGravitationalAcceleration(const std::vector<float>& g) {
        assertThreeElements(g, "SetGravitationalAcceleration", "g");
        G = make_float3(g[0], g[1], g[2]);
    }
    /// Set the initial time step size. If using constant step size, then this will be used throughout; otherwise, the
    /// actual step size depends on the variable step strategy.
    void SetInitTimeStep(double ts_size) { m_ts_size = ts_size; }
    /// Return the number of clumps that are currently in the simulation. Must be used after initialization.
    size_t GetNumClumps() const { return nOwnerClumps; }
    /// Return the total number of owners (clumps + meshes + analytical objects) that are currently in the simulation.
    /// Must be used after initialization.
    size_t GetNumOwners() const { return nOwnerBodies; }
    /// @brief Get the number of kT-reported potential contact pairs.
    /// @return Number of potential contact pairs.
    size_t GetNumContacts() const { return dT->getNumContacts(); }
    /// Get the current time step size in simulation.
    double GetTimeStepSize() const { return m_ts_size; }
    /// Get the current expand factor in simulation.
    float GetExpandFactor() const;
    /// Set the number of dT steps before it waits for a contact-pair info update from kT.
    void SetCDUpdateFreq(int freq) {
        m_updateFreq = freq;
        m_suggestedFutureDrift = 2 * freq;
        if (freq < 0) {
            DisableAdaptiveUpdateFreq();
        }
    }
    /// Get the simulation time passed since the start of simulation.
    double GetSimTime() const;
    /// Set the simulation time manually.
    void SetSimTime(double time);
    /// @brief Set the strategy for auto-adapting time step size (NOT implemented, no effect yet).
    /// @param type "none" or "max_vel" or "int_diff".
    void SetAdaptiveTimeStepType(const std::string& type);

    /// @brief Set the time integrator for this simulator.
    /// @param intg "forward_euler" or "extended_taylor" or "centered_difference".
    void SetIntegrator(const std::string& intg);
    /// @brief Set the time integrator for this simulator.
    void SetIntegrator(TIME_INTEGRATOR intg) { m_integrator = intg; }

    /// Return whether this simulation system is initialized
    bool GetInitStatus() const { return sys_initialized; }

    /// Get the jitification string substitution laundary list. It is needed by some of this simulation system's friend
    /// classes.
    std::unordered_map<std::string, std::string> GetJitStringSubs() const { return m_subs; }
    /// Get current jitification options. It is needed by some of this simulation system's friend classes.
    std::vector<std::string> GetJitifyOptions() const { return m_jitify_options; }
    /// Set the jitification options. It is only needed by advanced users.
    void SetJitifyOptions(const std::vector<std::string>& options) { m_jitify_options = options; }

    /// Explicitly instruct the bin size (for contact detection) that the solver should use.
    void SetInitBinSize(double bin_size) {
        use_user_defined_bin_size = INIT_BIN_SIZE_TYPE::EXPLICIT;
        m_binSize = bin_size;
    }
    /// Explicitly instruct the bin size (for contact detection) that the solver should use, as a multiple of the radius
    /// of the smallest sphere in simulation.
    void SetInitBinSizeAsMultipleOfSmallestSphere(float bin_size) {
        use_user_defined_bin_size = INIT_BIN_SIZE_TYPE::MULTI_MIN_SPH;
        m_binSize_as_multiple = bin_size;
    }
    /// @brief Set the target number of bins (for contact detection) at the start of the simulation upon initialization.
    void SetInitBinNumTarget(size_t num) {
        use_user_defined_bin_size = INIT_BIN_SIZE_TYPE::TARGET_NUM;
        m_target_init_bin_num = num;
    }

    /// Explicitly instruct the sizes for the arrays at initialization time. This is useful when the number of owners
    /// tends to change (especially gradually increase) frequently in the simulation, by reducing the need for
    /// reallocation. Note however, whatever instruction the user gives here it won't affect the correctness of the
    /// simulation, since if the arrays are not long enough they will always be auto-resized. This is not implemented
    /// yet :/
    void InstructNumOwners(size_t numOwners) { m_instructed_num_owners = numOwners; }

    /// Instruct the solver to use frictonal (history-based) Hertzian contact force model.
    std::shared_ptr<DEMForceModel> UseFrictionalHertzianModel();
    /// Instruct the solver to use frictonless Hertzian contact force model.
    std::shared_ptr<DEMForceModel> UseFrictionlessHertzianModel();
    /// Define a custom contact force model by a string. Returns a shared_ptr to the force model in use.
    std::shared_ptr<DEMForceModel> DefineContactForceModel(const std::string& model);
    /// Read user custom contact force model from a file (which by default should reside in kernel/DEMUserScripts).
    /// Returns a shared_ptr to the force model in use.
    std::shared_ptr<DEMForceModel> ReadContactForceModel(const std::string& filename);
    /// Get the current force model.
    std::shared_ptr<DEMForceModel> GetContactForceModel() { return m_force_model[DEFAULT_FORCE_MODEL_NAME]; }

    /// Instruct the solver if contact pair arrays should be sorted (based on the types of contacts) before usage.
    void SetSortContactPairs(bool use_sort) { should_sort_contacts = use_sort; }

    /// Instruct the solver to rearrange and consolidate clump templates information, then jitify it into GPU kernels
    /// (if set to true), rather than using flattened sphere component configuration arrays whose entries are associated
    /// with individual spheres.
    void SetJitifyClumpTemplates(bool use = true) { jitify_clump_templates = use; }
    /// Use flattened sphere component configuration arrays whose entries are associated with individual spheres, rather
    /// than jitifying them it into GPU kernels.
    void DisableJitifyClumpTemplates() { jitify_clump_templates = false; }
    /// Instruct the solver to rearrange and consolidate mass property information (for all owner types), then jitify it
    /// into GPU kernels (if set to true), rather than using flattened mass property arrays whose entries are associated
    /// with individual owners.
    void SetJitifyMassProperties(bool use = true) { jitify_mass_moi = use; }
    /// Use flattened mass property arrays whose entries are associated with individual spheres, rather than jitifying
    /// them it into GPU kernels.
    void DisableJitifyMassProperties() { jitify_mass_moi = false; }

    // NOTE: compact force calculation (in the hope to use shared memory) is not implemented
    void UseCompactForceKernel(bool use_compact);

    /// (Explicitly) set the amount by which the radii of the spheres (and the thickness of the boundaries) are expanded
    /// for the purpose of contact detection (safe, and creates false positives). If fix is set to true, then this
    /// expand factor does not change even if the user uses variable time step size.
    void SetExpandFactor(float beta, bool fix = true) {
        m_expand_factor = beta;
        use_user_defined_expand_factor = fix;
    }
    /// Input the maximum expected particle velocity. If `force' is set to false, the solver will not use a velocity
    /// larger than max_vel for determining the margin thickness; if `force' is set to true, the solver will not
    /// calculate maximum system velocity and will always use max_vel to calculate the margin thickness.

    /// @brief Set the maximum expected particle velocity. The solver will not use a velocity larger than this for
    /// determining the margin thickness, and velocity larger than this will be considered a system anomaly.
    /// @param max_vel Expected max velocity.
    void SetMaxVelocity(float max_vel);
    /// @brief Set the method this solver uses to derive current system velocity (for safety purposes in contact
    /// detection).
    /// @param insp_type A string. If "auto": the solver automatically derives.
    void SetExpandSafetyType(const std::string& insp_type);
    // void SetExpandSafetyType(const std::shared_ptr<DEMInspector>& insp) {
    //     m_max_v_finder_type = MARGIN_FINDER_TYPE::DEM_INSPECTOR;
    //     m_approx_max_vel_func = insp;
    // }

    /// Assign a multiplier to our estimated maximum system velocity, when deriving the thinckness of the contact
    /// `safety' margin. This can be greater than one if the simulation velocity can increase significantly in one kT
    /// update cycle, but this is not common and should be close to 1 in general.
    void SetExpandSafetyMultiplier(float param) { m_expand_safety_multi = param; }
    /// Set a `base' velocity, which we will always add to our estimated maximum system velocity, when deriving the
    /// thinckness of the contact `safety' margin. This need not to be large unless the simulation velocity can increase
    /// significantly in one kT update cycle.
    void SetExpandSafetyAdder(float vel) { m_expand_base_vel = vel; }

    /// @brief Used to force the solver to error out when there are too many spheres in a bin. A huge number can be used
    /// to discourage this error type.
    /// @param max_sph Max number of spheres in a bin.
    void SetMaxSphereInBin(unsigned int max_sph) { threshold_too_many_spheres_in_bin = max_sph; }

    /// @brief Used to force the solver to error out when there are too many spheres in a bin. A huge number can be used
    /// to discourage this error type.
    /// @param max_tri Max number of triangles in a bin.
    void SetMaxTriangleInBin(unsigned int max_tri) { threshold_too_many_tri_in_bin = max_tri; }

    /// @brief Set the velocity which when exceeded, the solver errors out. A huge number can be used to discourage this
    /// error type. Defaulted to 5e4.
    /// @param vel Error-out velocity.
    void SetErrorOutVelocity(float vel) { threshold_error_out_vel = vel; }

    /// @brief Set the average number of contacts a sphere has, before the solver errors out. A huge number can be used
    /// to discourage this error type. Defaulted to 100.
    /// @param num_cnts Error-out contact number.
    void SetErrorOutAvgContacts(float num_cnts) { threshold_error_out_num_cnts = num_cnts; }

    /// @brief Get the current number of contacts each sphere has.
    /// @return Number of contacts.
    float GetAvgSphContacts() const { return kT->stateParams.avgCntsPerSphere; }

    /// @brief Enable or disable the use of adaptive bin size (by default it is on).
    /// @param use Enable or disable.
    void UseAdaptiveBinSize(bool use = true) { auto_adjust_bin_size = use; }
    /// @brief Disable the use of adaptive bin size (always use initial size).
    void DisableAdaptiveBinSize() { auto_adjust_bin_size = false; }
    /// @brief Enable or disable the use of adaptive max update step count (by default it is on).
    /// @param use Enable or disable.
    void UseAdaptiveUpdateFreq(bool use = true) { auto_adjust_update_freq = use; }
    /// @brief Disable the use of adaptive max update step count (always use initial update frequency).
    void DisableAdaptiveUpdateFreq() { auto_adjust_update_freq = false; }
    /// @brief Adjust how frequent kT updates the bin size.
    /// @param n Number of contact detections before kT makes one adjustment to bin size.
    void SetAdaptiveBinSizeDelaySteps(unsigned int n) {
        if (n < NUM_STEPS_RESERVED_AFTER_CHANGING_BIN_SIZE)
            DEME_WARNING(
                "SetAdaptiveBinSizeDelaySteps is called with argument %u.\nThis is probably sub-optimal and may cause "
                "kT to try adjusting bin size before enough knowledge is gained.",
                n);
        auto_adjust_observe_steps = (n >= 1) ? n : 1;
    }
    /// @brief Set the max rate that the bin size can change in one adjustment.
    /// @param rate 0: never changes; 1: can double or halve size in one go; suggest using default.
    void SetAdaptiveBinSizeMaxRate(float rate) { auto_adjust_max_rate = (rate > 0) ? rate : 0; }
    /// @brief Set how fast kT changes the direction of bin size adjustmemt when there's a more beneficial direction.
    /// @param acc 0.01: slowly change direction; 1: quickly change direction
    void SetAdaptiveBinSizeAcc(float acc) { auto_adjust_acc = clampBetween(acc, 0.01, 1.0); }
    /// @brief Set how proactive the solver is in avoiding the bin being too big (leading to too many geometries in a
    /// bin).
    /// @param ratio 0: not proavtive; 1: very proactive.
    void SetAdaptiveBinSizeUpperProactivity(float ratio) {
        auto_adjust_upper_proactive_ratio = clampBetween(ratio, 0.0, 1.0);
    }
    /// @brief Set how proactive the solver is in avoiding the bin being too small (leading to too many bins in domain).
    /// @param ratio 0: not proavtive; 1: very proactive.
    void SetAdaptiveBinSizeLowerProactivity(float ratio) {
        auto_adjust_lower_proactive_ratio = clampBetween(ratio, 0.0, 1.0);
    }
    /// @brief Get the current bin (for contact detection) size. Must be called from synchronized stance.
    /// @return Bin size.
    double GetBinSize() { return kT->simParams->binSize; }
    // NOTE: No need to get binSize from the device, as binSize is only changed on the host

    /// @brief Get the current number of bins (for contact detection). Must be called from synchronized stance.
    /// @return Number of bins.
    size_t GetBinNum() { return kT->stateParams.numBins; }

    /// @brief Set the upper bound of kT update frequency (when it is adjusted automatically).
    /// @details This only affects when the update freq is updated automatically. To manually control the freq, use
    /// SetCDUpdateFreq then call DisableAdaptiveUpdateFreq.
    /// @param max_freq dT will not receive updates less frequently than 1 update per max_freq steps.
    void SetCDMaxUpdateFreq(unsigned int max_freq) { upper_bound_future_drift = 2 * max_freq; }
    /// @brief Set the number of steps dT configures its max drift more than average drift steps.
    /// @param n Number of steps. Suggest using default.
    void SetCDNumStepsMaxDriftAheadOfAvg(float n) { max_drift_ahead_of_avg_drift = n; }
    /// @brief Set the multiplier which dT configures its max drift to be w.r.t. the average drift steps.
    /// @param m The multiplier. Suggest using default.
    void SetCDNumStepsMaxDriftMultipleOfAvg(float m) { max_drift_multiple_of_avg_drift = m; }
    /// @brief Set the number of past kT updates that dT will use to calibrate the max future drift limit.
    /// @param n Number of kT updates. Suggest using default.
    void SetCDNumStepsMaxDriftHistorySize(unsigned int n);
    /// @brief Get the current update frequency used by the solver.
    /// @return The current update frequency.
    float GetUpdateFreq() const;

    /// Set the number of threads per block in force calculation (default 256).
    void SetForceCalcThreadsPerBlock(unsigned int nTh) { dT->DT_FORCE_CALC_NTHREADS_PER_BLOCK = nTh; }

    /// @brief Load a clump type into the API-level cache.
    /// @return the shared ptr to the clump type just loaded.
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(float mass,
                                                    float3 moi,
                                                    const std::vector<float>& sp_radii,
                                                    const std::vector<float3>& sp_locations_xyz,
                                                    const std::vector<std::shared_ptr<DEMMaterial>>& sp_materials);
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(float mass,
                                                    const std::vector<float>& moi,
                                                    const std::vector<float>& sp_radii,
                                                    const std::vector<std::vector<float>>& sp_locations_xyz,
                                                    const std::vector<std::shared_ptr<DEMMaterial>>& sp_materials) {
        assertThreeElements(moi, "LoadClumpType", "moi");
        assertThreeElementsVector(sp_locations_xyz, "LoadClumpType", "sp_locations_xyz");
        std::vector<float3> loc_xyz(sp_locations_xyz.size());
        for (size_t i = 0; i < sp_locations_xyz.size(); i++) {
            loc_xyz[i] = make_float3(sp_locations_xyz[i][0], sp_locations_xyz[i][1], sp_locations_xyz[i][2]);
        }
        return LoadClumpType(mass, make_float3(moi[0], moi[1], moi[2]), sp_radii, loc_xyz, sp_materials);
    }
    /// An overload of LoadClumpType where all components use the same material
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(float mass,
                                                    float3 moi,
                                                    const std::vector<float>& sp_radii,
                                                    const std::vector<float3>& sp_locations_xyz,
                                                    const std::shared_ptr<DEMMaterial>& sp_material);
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(float mass,
                                                    const std::vector<float>& moi,
                                                    const std::vector<float>& sp_radii,
                                                    const std::vector<std::vector<float>>& sp_locations_xyz,
                                                    const std::shared_ptr<DEMMaterial>& sp_material) {
        assertThreeElements(moi, "LoadClumpType", "moi");
        assertThreeElementsVector(sp_locations_xyz, "LoadClumpType", "sp_locations_xyz");
        std::vector<float3> loc_xyz(sp_locations_xyz.size());
        for (size_t i = 0; i < sp_locations_xyz.size(); i++) {
            loc_xyz[i] = make_float3(sp_locations_xyz[i][0], sp_locations_xyz[i][1], sp_locations_xyz[i][2]);
        }
        return LoadClumpType(mass, make_float3(moi[0], moi[1], moi[2]), sp_radii, loc_xyz, sp_material);
    }
    /// An overload of LoadClumpType where the user builds the DEMClumpTemplate struct themselves then supply it
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(DEMClumpTemplate& clump);
    /// An overload of LoadClumpType which loads sphere components from a file
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(float mass,
                                                    float3 moi,
                                                    const std::string filename,
                                                    const std::vector<std::shared_ptr<DEMMaterial>>& sp_materials);
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(float mass,
                                                    const std::vector<float>& moi,
                                                    const std::string filename,
                                                    const std::vector<std::shared_ptr<DEMMaterial>>& sp_materials) {
        assertThreeElements(moi, "LoadClumpType", "moi");
        return LoadClumpType(mass, make_float3(moi[0], moi[1], moi[2]), filename, sp_materials);
    }
    /// An overload of LoadClumpType which loads sphere components from a file and all components use the same material
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(float mass,
                                                    float3 moi,
                                                    const std::string filename,
                                                    const std::shared_ptr<DEMMaterial>& sp_material);
    std::shared_ptr<DEMClumpTemplate> LoadClumpType(float mass,
                                                    const std::vector<float>& moi,
                                                    const std::string filename,
                                                    const std::shared_ptr<DEMMaterial>& sp_material) {
        assertThreeElements(moi, "LoadClumpType", "moi");
        return LoadClumpType(mass, make_float3(moi[0], moi[1], moi[2]), filename, sp_material);
    }
    /// A simplified version of LoadClumpType: it just loads a one-sphere clump template
    std::shared_ptr<DEMClumpTemplate> LoadSphereType(float mass,
                                                     float radius,
                                                     const std::shared_ptr<DEMMaterial>& material);

    /// @brief Load materials properties (Young's modulus, Poisson's ratio...) into the system.
    /// @param mat_prop Property name--value pairs, as an unordered_map.
    /// @return A shared pointer for this material.
    std::shared_ptr<DEMMaterial> LoadMaterial(const std::unordered_map<std::string, float>& mat_prop);
    /// @brief Load materials properties into the system.
    /// @param a_material A DEMMaterial object.
    /// @return A shared pointer for this material.
    std::shared_ptr<DEMMaterial> LoadMaterial(DEMMaterial& a_material);

    /// @brief Duplicate a material that is loaded into the system.
    /// @param ptr Shared pointer for the object to duplicate.
    /// @return A duplicate of the object (with effectively a deep copy).
    std::shared_ptr<DEMMaterial> Duplicate(const std::shared_ptr<DEMMaterial>& ptr);
    /// @brief Duplicate a clump template that is loaded into the system.
    /// @param ptr Shared pointer for the object to duplicate.
    /// @return A duplicate of the object (with effectively a deep copy).
    std::shared_ptr<DEMClumpTemplate> Duplicate(const std::shared_ptr<DEMClumpTemplate>& ptr);
    /// @brief Duplicate a batch of clumps that is loaded into the system.
    /// @param ptr Shared pointer for the object to duplicate.
    /// @return A duplicate of the object (with effectively a deep copy).
    std::shared_ptr<DEMClumpBatch> Duplicate(const std::shared_ptr<DEMClumpBatch>& ptr);

    /// @brief Set the value for a material property that by nature involves a pair of a materials (e.g. friction
    /// coefficient).
    /// @param name The name of this property (which should have already been referred to in a previous LoadMaterial
    /// call).
    /// @param mat1 Material 1 that is involved in this pair.
    /// @param mat2 Material 2 that is involved in this pair.
    /// @param val The value.
    void SetMaterialPropertyPair(const std::string& name,
                                 const std::shared_ptr<DEMMaterial>& mat1,
                                 const std::shared_ptr<DEMMaterial>& mat2,
                                 float val);

    /// @brief Get the clumps that are in contact with this owner as a vector.
    /// @details No multi-owner bulk version. This is due to efficiency concerns. If getting multiple owners' contacting
    /// clumps is needed, use family-based GetContacts method, then the owner ID list-based c method if you further
    /// need the contact forces information.
    /// @param ownerID The ID of the owner that is being queried.
    /// @return Clump owner IDs in contact with this owner.
    std::vector<bodyID_t> GetOwnerContactClumps(bodyID_t ownerID) const;
    /// Get position of n consecutive owners.
    std::vector<float3> GetOwnerPosition(bodyID_t ownerID, bodyID_t n = 1) const;
    /// Get angular velocity of n consecutive owners.
    std::vector<float3> GetOwnerAngVel(bodyID_t ownerID, bodyID_t n = 1) const;
    /// Get quaternion of n consecutive owners.
    std::vector<float4> GetOwnerOriQ(bodyID_t ownerID, bodyID_t n = 1) const;
    /// Get velocity of n consecutive owners.
    std::vector<float3> GetOwnerVelocity(bodyID_t ownerID, bodyID_t n = 1) const;
    /// Get the acceleration of n consecutive owners.
    std::vector<float3> GetOwnerAcc(bodyID_t ownerID, bodyID_t n = 1) const;
    /// Get the angular acceleration of n consecutive owners.
    std::vector<float3> GetOwnerAngAcc(bodyID_t ownerID, bodyID_t n = 1) const;
    /// @brief Get the family number of n consecutive owners.
    /// @param ownerID First owner's ID.
    /// @param n The number of consecutive owners.
    /// @return The family number.
    std::vector<unsigned int> GetOwnerFamily(bodyID_t ownerID, bodyID_t n = 1) const;
    /// @brief Get the mass of n consecutive owners.
    /// @param ownerID First owner's ID.
    /// @param n The number of consecutive owners.
    /// @return The mass.
    std::vector<float> GetOwnerMass(bodyID_t ownerID, bodyID_t n = 1) const;
    /// @brief Get the moment of inertia (in principal axis frame) of n consecutive owners.
    /// @param ownerID First owner's ID.
    /// @param n The number of consecutive owners.
    /// @return The moment of inertia (in principal axis frame).
    std::vector<float3> GetOwnerMOI(bodyID_t ownerID, bodyID_t n = 1) const;

    /// @brief Set position of consecutive owners starting from ownerID, based on input position vector. N (the size of
    /// the input vector) elements will be modified.
    void SetOwnerPosition(bodyID_t ownerID, const std::vector<float3>& pos);
    /// Set angular velocity of consecutive owners starting from ownerID, based on input angular velocity vector. N (the
    /// size of the input vector) elements will be modified.
    void SetOwnerAngVel(bodyID_t ownerID, const std::vector<float3>& angVel);
    /// Set velocity of consecutive owners starting from ownerID, based on input velocity vector. N (the size of the
    /// input vector) elements will be modified.
    void SetOwnerVelocity(bodyID_t ownerID, const std::vector<float3>& vel);
    /// Set quaternion of consecutive owners starting from ownerID, based on input quaternion vector. N (the size of the
    /// input vector) elements will be modified.
    void SetOwnerOriQ(bodyID_t ownerID, const std::vector<float4>& oriQ);
    /// @brief Set the family number of consecutive owners.
    /// @param ownerID The ID of the owner.
    /// @param fam Family number.
    /// @param n Number of consecutive owners.
    void SetOwnerFamily(bodyID_t ownerID, unsigned int fam, bodyID_t n = 1);

    /// @brief Add an extra accelerations to consecutive owners for the next time step.
    /// @param ownerID The number of the starting owner.
    /// @param acc The extra acceleration to add. N (the size of this vector) elements will be modified based on its
    /// values.
    void AddOwnerNextStepAcc(bodyID_t ownerID, const std::vector<float3>& acc);
    /// @brief Add an extra angular accelerations to consecutive owners for the next time step.
    /// @param ownerID The number of the starting owner.
    /// @param acc The extra angular acceleration to add. N (the size of this vector) elements will be modified based on
    /// its values.
    void AddOwnerNextStepAngAcc(bodyID_t ownerID, const std::vector<float3>& angAcc);

    /// @brief Rewrite the relative positions of the flattened triangle soup.
    void SetTriNodeRelPos(size_t owner, size_t triID, const std::vector<float3>& new_nodes);
    /// @brief Update the relative positions of the flattened triangle soup.
    void UpdateTriNodeRelPos(size_t owner, size_t triID, const std::vector<float3>& updates);
    /// @brief Get a handle for the mesh this tracker is tracking.
    /// @return Pointer to the mesh.
    std::shared_ptr<DEMMeshConnected>& GetCachedMesh(bodyID_t ownerID);
    /// @brief Get the current locations of all the nodes in the mesh being tracked.
    /// @param ownerID The ownerID of the mesh.
    /// @return A vector of float3 representing the global coordinates of the mesh nodes.
    std::vector<float3> GetMeshNodesGlobal(bodyID_t ownerID);

    /// @brief Get all clump--clump contact ID pairs in the simulation system. Note all GetContact-like methods reports
    /// potential contacts (not necessarily confirmed contacts), meaning they are similar to what
    /// WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does.
    /// @details Do not call this method with high frequency, as it is not efficient.
    /// @return A sorted (based on contact body A's owner ID) vector of contact pairs. First is the owner ID of contact
    /// body A, and Second is that of contact body B.
    std::vector<std::pair<bodyID_t, bodyID_t>> GetClumpContacts() const;
    /// @brief Get all clump--clump contact ID pairs in the simulation system. Note all GetContact-like methods reports
    /// potential contacts (not necessarily confirmed contacts), meaning they are similar to what
    /// WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does.
    /// @details Do not call this method with high frequency, as it is not efficient.
    /// @param family_to_include Contacts that involve a body in a family not listed in this argument are ignored.
    /// @return A sorted (based on contact body A's owner ID) vector of contact pairs. First is the owner ID of contact
    /// body A, and Second is that of contact body B.
    std::vector<std::pair<bodyID_t, bodyID_t>> GetClumpContacts(const std::set<family_t>& family_to_include) const;
    /// @brief Get all clump--clump contact ID pairs in the simulation system. Note all GetContact-like methods reports
    /// potential contacts (not necessarily confirmed contacts), meaning they are similar to what
    /// WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does.
    /// @details Do not call this method with high frequency, as it is not efficient.
    /// @param family_pair Functions returns a vector of contact body family number pairs. First is the family number of
    /// contact body A, and Second is that of contact body B.
    /// @return A sorted (based on contact body A's owner ID) vector of contact pairs. First is the owner ID of contact
    /// body A, and Second is that of contact body B.
    std::vector<std::pair<bodyID_t, bodyID_t>> GetClumpContacts(
        std::vector<std::pair<family_t, family_t>>& family_pair) const;

    /// @brief Get all contact ID pairs in the simulation system. Note all GetContact-like methods reports potential
    /// contacts (not necessarily confirmed contacts), meaning they are similar to what
    /// WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does.
    /// @details Do not call this method with high frequency, as it is not efficient.
    /// @return A sorted (based on contact body A's owner ID) vector of contact pairs. First is the owner ID of contact
    /// body A, and Second is that of contact body B.
    std::vector<std::pair<bodyID_t, bodyID_t>> GetContacts() const;
    /// @brief Get all contact ID pairs in the simulation system. Note all GetContact-like methods reports potential
    /// contacts (not necessarily confirmed contacts), meaning they are similar to what
    /// WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does.
    /// @details Do not call this method with high frequency, as it is not efficient.
    /// @param family_to_include Contacts that involve a body in a family not listed in this argument are ignored.
    /// @return A sorted (based on contact body A's owner ID) vector of contact pairs. First is the owner ID of contact
    /// body A, and Second is that of contact body B.
    std::vector<std::pair<bodyID_t, bodyID_t>> GetContacts(const std::set<family_t>& family_to_include) const;
    /// @brief Get all contact ID pairs in the simulation system. Note all GetContact-like methods reports potential
    /// contacts (not necessarily confirmed contacts), meaning they are similar to what
    /// WriteContactFileIncludingPotentialPairs does, not what WriteContactFile does.
    /// @details Do not call this method with high frequency, as it is not efficient.
    /// @param family_pair Functions returns a vector of contact body family number pairs. First is the family number of
    /// contact body A, and Second is that of contact body B.
    /// @return A sorted (based on contact body A's owner ID) vector of contact pairs. First is the owner ID of contact
    /// body A, and Second is that of contact body B.
    std::vector<std::pair<bodyID_t, bodyID_t>> GetContacts(
        std::vector<std::pair<family_t, family_t>>& family_pair) const;

    /// @brief Get all contact pairs' detailed information (actual content based on the setting with
    /// SetContactOutputContent; default are owner IDs, contact point location, contact force, and associated wildcard
    /// values) in the simulation system. Note all GetContact-like methods reports potential contacts (not necessarily
    /// confirmed contacts), meaning they are similar to what WriteContactFileIncludingPotentialPairs does, not what
    /// WriteContactFile does.
    /// @details Do not call this method with high frequency, as it is not efficient.
    /// @param force_thres Only contacts with force larger than this value are returned. Setting it to a small positive
    /// number to, instead of getting all potential contacts, only get the ones that are currently confirmed to generate
    /// force.
    /// @return A map that may have the following keys: "ContactType", "Point", "AOwner", "BOwner", "AOwnerFamily",
    /// "BOwnerFamily", "Force", "Torque", "Normal" and wildcard names, each corresponding to a vector of values. The
    /// "ContactType" is a vector of strings, each indicating the type of contact (e.g., "sphere-sphere",
    /// "sphere-triangle", etc.). The "Point" is a vector of float3s, each indicating the contact point location in
    /// global coordinates. The "AOwner" and "BOwner" are vectors of body IDs for the two bodies in contact. The "Force"
    /// is a vector of float3s, each indicating the contact force at the contact point. The "Torque" is a vector of
    /// float3s, each indicating the torque at the contact point. The "Normal" is a vector of float3s, each indicating
    /// the normal direction at the contact point.
    std::shared_ptr<ContactInfoContainer> GetContactDetailedInfo(float force_thres = -1.0) const;

    /// @brief Get the host memory usage (in bytes) on dT.
    /// @return Number of bytes.
    size_t GetHostMemUsageDynamic() const { return dT->estimateHostMemUsage(); }
    /// @brief Get the device memory usage (in bytes) on dT.
    /// @return Number of bytes.
    size_t GetDeviceMemUsageDynamic() const { return dT->estimateDeviceMemUsage(); }
    /// @brief Get the host memory usage (in bytes) on kT.
    /// @return Number of bytes.
    size_t GetHostMemUsageKinematic() const { return kT->estimateHostMemUsage(); }
    /// @brief Get the device memory usage (in bytes) on kT.
    /// @return Number of bytes.
    size_t GetDeviceMemUsageKinematic() const { return kT->estimateDeviceMemUsage(); }
    /// @brief Print the current memory usage in pretty format.
    void ShowMemStats() const;

    /// Load input clumps (topology types and initial locations) on a per-pair basis. Note that the initial location
    /// means the location of the clumps' CoM coordinates in the global frame.
    std::shared_ptr<DEMClumpBatch> AddClumps(DEMClumpBatch& input_batch);
    /// @brief Load clumps into the simulation.
    /// @param input_types Vector of the types of the clumps (vector of shared pointers).
    /// @param input_xyz Vector of the initial locations of the clumps.
    /// @return Handle to the loaded batch of clumps.
    std::shared_ptr<DEMClumpBatch> AddClumps(const std::vector<std::shared_ptr<DEMClumpTemplate>>& input_types,
                                             const std::vector<float3>& input_xyz);
    std::shared_ptr<DEMClumpBatch> AddClumps(const std::vector<std::shared_ptr<DEMClumpTemplate>>& input_types,
                                             const std::vector<std::vector<float>>& input_xyz) {
        assertThreeElementsVector(input_xyz, "AddClumps", "input_xyz");
        std::vector<float3> loc_xyz(input_xyz.size());
        for (size_t i = 0; i < input_xyz.size(); i++) {
            loc_xyz[i] = make_float3(input_xyz[i][0], input_xyz[i][1], input_xyz[i][2]);
        }
        return AddClumps(input_types, loc_xyz);
    }

    /// @brief Load a clump into the simulation.
    /// @param input_type The type (shared pointer pointing to the clump type handle).
    /// @param input_xyz Initial location of the clump.
    /// @return Handle to the clump.
    std::shared_ptr<DEMClumpBatch> AddClumps(std::shared_ptr<DEMClumpTemplate>& input_type, float3 input_xyz) {
        return AddClumps(std::vector<std::shared_ptr<DEMClumpTemplate>>(1, input_type),
                         std::vector<float3>(1, input_xyz));
    }
    std::shared_ptr<DEMClumpBatch> AddClumps(std::shared_ptr<DEMClumpTemplate>& input_type,
                                             const std::vector<float>& input_xyz) {
        assertThreeElements(input_xyz, "AddClumps", "input_xyz");
        return AddClumps(input_type, make_float3(input_xyz[0], input_xyz[1], input_xyz[2]));
    }

    /// @brief Load clumps (of the same template) into the simulation.
    /// @param input_types The type (shared pointer pointing to the clump type handle).
    /// @param input_xyz Vector of the initial locations of the clumps.
    /// @return Handle to the loaded batch of clumps.
    std::shared_ptr<DEMClumpBatch> AddClumps(std::shared_ptr<DEMClumpTemplate>& input_type,
                                             const std::vector<float3>& input_xyz) {
        return AddClumps(std::vector<std::shared_ptr<DEMClumpTemplate>>(input_xyz.size(), input_type), input_xyz);
    }
    std::shared_ptr<DEMClumpBatch> AddClumps(std::shared_ptr<DEMClumpTemplate>& input_type,
                                             const std::vector<std::vector<float>>& input_xyz) {
        assertThreeElementsVector(input_xyz, "AddClumps", "input_xyz");
        std::vector<float3> loc_xyz(input_xyz.size());
        for (size_t i = 0; i < input_xyz.size(); i++) {
            loc_xyz[i] = make_float3(input_xyz[i][0], input_xyz[i][1], input_xyz[i][2]);
        }
        return AddClumps(input_type, loc_xyz);
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

    /// @brief Create a DEMTracker to allow direct control/modification/query to this external object/batch of
    /// clumps/triangle mesh object.
    /// @details By default, it refers to the first clump in this batch. The user can refer to other clumps in this
    /// batch by supplying an offset when using this tracker's querying or assignment methods.
    template <typename T>
    std::shared_ptr<DEMTracker> Track(const std::shared_ptr<T>& obj) {
        // Create a middle man: DEMTrackedObj. The reason we use it is because a simple struct should be used to
        // transfer to dT for owner-number processing. If we cut the middle man and use things such as DEMExtObj, there
        // will not be a universal treatment that dT can apply, besides we may have some include-related issues.
        DEMTrackedObj tracked_obj;
        tracked_obj.load_order = obj->load_order;
        tracked_obj.obj_type = obj->obj_type;
        m_tracked_objs.push_back(std::make_shared<DEMTrackedObj>(std::move(tracked_obj)));

        // Create a Tracker for this tracked object
        DEMTracker tracker(this);
        tracker.obj = m_tracked_objs.back();
        return std::make_shared<DEMTracker>(std::move(tracker));
    }

    /// @brief Create a DEMTracker to allow direct control/modification/query to this external object/batch of
    /// clumps/triangle mesh object.
    /// @details C++ users do not have to use this method. Using Track is enough. This method is for Python wrapper.
    std::shared_ptr<DEMTracker> PythonTrack(const std::shared_ptr<DEMInitializer>& obj) {
        return Track<DEMInitializer>(obj);
    }

    /// Create a inspector object that can help query some statistical info of the clumps in the simulation
    std::shared_ptr<DEMInspector> CreateInspector(const std::string& quantity = "clump_max_z");
    std::shared_ptr<DEMInspector> CreateInspector(const std::string& quantity, const std::string& region);

    /// Instruct the solver that the 2 input families should not have contacts (a.k.a. ignored, if such a pair is
    /// encountered in contact detection). These 2 families can be the same (which means no contact within members of
    /// that family).
    void DisableContactBetweenFamilies(unsigned int ID1, unsigned int ID2);

    /// Re-enable contact between 2 families after the system is initialized.
    void EnableContactBetweenFamilies(unsigned int ID1, unsigned int ID2);

    /// Prevent entites associated with this family to be outputted to files.
    void DisableFamilyOutput(unsigned int ID);

    /// Mark all entities in this family to be fixed.
    void SetFamilyFixed(unsigned int ID);

    /// If dictate is set to true, then

    /// @brief Set the prescribed linear velocity to all entities in a family.
    /// @param ID Family number.
    /// @param velX X component of velocity.
    /// @param velY Y component of velocity.
    /// @param velZ Z component of velocity.
    /// @param dictate If true, this family will not be influenced by the force exerted from other simulation entites
    /// (both linear and rotational motions); if false, only specified components (that is, not specified with "none")
    /// will not be influenced by the force exerted from other simulation entites.
    /// @param pre Prerequisite code. For example, you can generate a float3 with this prerequisite code, then assign
    /// XYZ components based on this float3.
    void SetFamilyPrescribedLinVel(unsigned int ID,
                                   const std::string& velX,
                                   const std::string& velY,
                                   const std::string& velZ,
                                   bool dictate = true,
                                   const std::string& pre = "none");
    /// Let the linear velocities of all entites in this family always keep `as is', and not influenced by the force
    /// exerted from other simulation entites.
    void SetFamilyPrescribedLinVel(unsigned int ID);
    /// Let the X component of the linear velocities of all entites in this family always keep `as is', and not
    /// influenced by the force exerted from other simulation entites.
    void SetFamilyPrescribedLinVelX(unsigned int ID);
    /// Let the Y component of the linear velocities of all entites in this family always keep `as is', and not
    /// influenced by the force exerted from other simulation entites.
    void SetFamilyPrescribedLinVelY(unsigned int ID);
    /// Let the Z component of the linear velocities of all entites in this family always keep `as is', and not
    /// influenced by the force exerted from other simulation entites.
    void SetFamilyPrescribedLinVelZ(unsigned int ID);

    /// @brief Set the prescribed angular velocity to all entities in a family.
    /// @param ID Family number.
    /// @param velX X component of angular velocity.
    /// @param velY Y component of angular velocity.
    /// @param velZ Z component of angular velocity.
    /// @param dictate If true, this family will not be influenced by the force exerted from other simulation entites
    /// (both linear and rotational motions); if false, only specified components (that is, not specified with "none")
    /// will not be influenced by the force exerted from other simulation entites.
    /// @param pre Prerequisite code. For example, you can generate a float3 with this prerequisite code, then assign
    /// XYZ components based on this float3.
    void SetFamilyPrescribedAngVel(unsigned int ID,
                                   const std::string& velX,
                                   const std::string& velY,
                                   const std::string& velZ,
                                   bool dictate = true,
                                   const std::string& pre = "none");
    /// Let the linear velocities of all entites in this family always keep `as is', and not influenced by the force
    /// exerted from other simulation entites.
    void SetFamilyPrescribedAngVel(unsigned int ID);
    /// Let the X component of the angular velocities of all entites in this family always keep `as is', and not
    /// influenced by the force exerted from other simulation entites.
    void SetFamilyPrescribedAngVelX(unsigned int ID);
    /// Let the Y component of the angular velocities of all entites in this family always keep `as is', and not
    /// influenced by the force exerted from other simulation entites.
    void SetFamilyPrescribedAngVelY(unsigned int ID);
    /// Let the Z component of the angular velocities of all entites in this family always keep `as is', and not
    /// influenced by the force exerted from other simulation entites.
    void SetFamilyPrescribedAngVelZ(unsigned int ID);

    /// @brief Keep the positions of all entites in this family to remain exactly the user-specified values.
    /// @param ID Family number.
    /// @param X X coordinate (can be an expression).
    /// @param Y Y coordinate (can be an expression).
    /// @param Z Z coordinate (can be an expression).
    /// @param dictate If true, prevent entities in this family to have (both linear and rotational) positional updates
    /// resulted from the `simulation physics'; if false, only specified components (that is, not specified with "none")
    /// will not be influenced by the force exerted from other simulation entites.
    /// @param pre Prerequisite code. For example, you can generate a float3 with this prerequisite code, then assign
    /// XYZ components based on this float3.
    void SetFamilyPrescribedPosition(unsigned int ID,
                                     const std::string& X,
                                     const std::string& Y,
                                     const std::string& Z,
                                     bool dictate = true,
                                     const std::string& pre = "none");
    /// @brief Let the linear positions of all entites in this family always keep `as is'.
    void SetFamilyPrescribedPosition(unsigned int ID);
    /// @brief Let the X component of the linear positions of all entites in this family always keep `as is'.
    void SetFamilyPrescribedPositionX(unsigned int ID);
    /// @brief Let the Y component of the linear positions of all entites in this family always keep `as is'.
    void SetFamilyPrescribedPositionY(unsigned int ID);
    /// @brief Let the Z component of the linear positions of all entites in this family always keep `as is'.
    void SetFamilyPrescribedPositionZ(unsigned int ID);

    /// @brief Keep the orientation quaternions of all entites in this family to remain exactly the user-specified
    /// values.
    /// @param ID Family number.
    /// @param q_formula The code from which the quaternion should be calculated. Must `return' a float4. For example,
    /// "float tmp=make_float4(1,1,1,1); return tmp;".
    /// @param dictate If true, prevent entities in this family to have (both linear and rotational) positional updates
    /// resulted from the `simulation physics'; otherwise, the `simulation physics' still takes effect.
    void SetFamilyPrescribedQuaternion(unsigned int ID, const std::string& q_formula, bool dictate = true);
    /// @brief Let the orientation quaternions of all entites in this family always keep `as is'.
    void SetFamilyPrescribedQuaternion(unsigned int ID);

    /// @brief The entities in this family will always experience an extra acceleration defined using this method.
    /// @param pre Prerequisite code. For example, you can generate a float3 with this prerequisite code, then assign
    /// XYZ components based on this float3.
    void AddFamilyPrescribedAcc(unsigned int ID,
                                const std::string& X,
                                const std::string& Y,
                                const std::string& Z,
                                const std::string& pre = "none");
    /// @brief The entities in this family will always experience an extra angular acceleration defined using this
    /// method.
    /// @param pre Prerequisite code. For example, you can generate a float3 with this prerequisite code, then assign
    /// XYZ components based on this float3.
    void AddFamilyPrescribedAngAcc(unsigned int ID,
                                   const std::string& X,
                                   const std::string& Y,
                                   const std::string& Z,
                                   const std::string& pre = "none");

    /// @brief The entities in this family will always experience an added linear-velocity correction defined using this
    /// method. At the same time, they are still subject to the `simulation physics'.
    /// @param pre Prerequisite code. For example, you can generate a float3 with this prerequisite code, then assign
    /// XYZ components based on this float3.
    void CorrectFamilyLinVel(unsigned int ID,
                             const std::string& X,
                             const std::string& Y,
                             const std::string& Z,
                             const std::string& pre = "none");
    /// @brief The entities in this family will always experience an added angular-velocity correction defined using
    /// this method. At the same time, they are still subject to the `simulation physics'.
    /// @param pre Prerequisite code. For example, you can generate a float3 with this prerequisite code, then assign
    /// XYZ components based on this float3.
    void CorrectFamilyAngVel(unsigned int ID,
                             const std::string& X,
                             const std::string& Y,
                             const std::string& Z,
                             const std::string& pre = "none");

    /// @brief The entities in this family will always experience an added positional correction defined using this
    /// method. At the same time, they are still subject to the `simulation physics'.
    /// @param pre Prerequisite code. For example, you can generate a float3 with this prerequisite code, then assign
    /// XYZ components based on this float3.
    void CorrectFamilyPosition(unsigned int ID,
                               const std::string& X,
                               const std::string& Y,
                               const std::string& Z,
                               const std::string& pre = "none");
    /// @brief The entities in this family will always experience an added quaternion correction defined using this
    /// method. At the same time, they are still subject to the `simulation physics'.
    /// @param q_formula The code from which the quaternion should be calculated. Must `return' a float4. For example,
    /// "float tmp=make_float4(1,1,1,1); return tmp;".
    void CorrectFamilyQuaternion(unsigned int ID, const std::string& q_formula);

    /// @brief Set the names for the extra quantities that will be associated with each contact pair.
    void SetContactWildcards(const std::set<std::string>& wildcards);
    /// @brief Set the names for the extra quantities that will be associated with each owner.
    void SetOwnerWildcards(const std::set<std::string>& wildcards);
    /// @brief Set the names for the extra quantities that will be associated with each geometry entity (such as sphere,
    /// triangle).
    void SetGeometryWildcards(const std::set<std::string>& wildcards);

    /// @brief Change the value of contact wildcards to val if either of the contact geometries is in family N.
    /// @param N Family number. If one contact geometry is in N, this contact wildcard is modified.
    /// @param name Name of the contact wildcard to modify.
    /// @param val The value to change to.
    void SetFamilyContactWildcardValueEither(unsigned int N, const std::string& name, float val);
    /// @brief Change the value of contact wildcards to val if both of the contact geometries are in family N.
    /// @param N Family number. Only if both contact geometries are in N, this contact wildcard is modified.
    /// @param name Name of the contact wildcard to modify.
    /// @param val The value to change to.
    void SetFamilyContactWildcardValueBoth(unsigned int N, const std::string& name, float val);
    /// @brief Change the value of contact wildcards to val if one of the contact geometry is in family N1, and the
    /// other is in N2.
    /// @param N1 First family number.
    /// @param N2 Second family number.
    /// @param name Name of the contact wildcard to modify.
    /// @param val The value to change to.
    void SetFamilyContactWildcardValue(unsigned int N1, unsigned int N2, const std::string& name, float val);
    /// @brief Change the value of contact wildcards to val. Apply to all simulation bodies that are present.
    /// @param name Name of the contact wildcard to modify.
    /// @param val The value to change to.
    void SetContactWildcardValue(const std::string& name, float val);

    /// @brief Make it so that for any currently-existing contact, if one of its contact geometries is in family N, then
    /// this contact will never be removed.
    /// @details This contact might be created through contact detection, or being a contact the user manually loaded at
    /// the start of simulation. Note this is a one-time assignment and will not continuously mark future emerging
    /// contact to be persistent.
    /// @param N Family number.
    void MarkFamilyPersistentContactEither(unsigned int N);
    /// @brief Make it so that for any currently-existing contact, if both of its contact geometries are in family N,
    /// then this contact will never be removed.
    /// @details This contact might be created through contact detection, or being a contact the user manually loaded at
    /// the start of simulation. Note this is a one-time assignment and will not continuously mark future emerging
    /// contact to be persistent.
    /// @param N Family number.
    void MarkFamilyPersistentContactBoth(unsigned int N);
    /// @brief Make it so that if for any currently-existing contact, if its two contact geometries are in family N1 and
    /// N2 respectively, this contact will never be removed.
    /// @details This contact might be created through contact detection, or being a contact the user manually loaded at
    /// the start of simulation. Note this is a one-time assignment and will not continuously mark future emerging
    /// contact to be persistent.
    /// @param N1 Family number 1.
    /// @param N2 Family number 2.
    void MarkFamilyPersistentContact(unsigned int N1, unsigned int N2);
    /// @brief Make it so that all currently-existing contacts in this simulation will never be removed.
    /// @details The contacts might be created through contact detection, or being manually loaded at the start of
    /// simulation. Note this is a one-time assignment and will not continuously mark future emerging contact to be
    /// persistent.
    void MarkPersistentContact();

    /// @brief Cancel contact persistence qualification. Work like the inverse of MarkFamilyPersistentContactEither.
    void RemoveFamilyPersistentContactEither(unsigned int N);
    /// @brief Cancel contact persistence qualification. Work like the inverse of MarkFamilyPersistentContactBoth.
    void RemoveFamilyPersistentContactBoth(unsigned int N);
    /// @brief Cancel contact persistence qualification. Work like the inverse of MarkFamilyPersistentContact.
    void RemoveFamilyPersistentContact(unsigned int N1, unsigned int N2);
    /// @brief Cancel contact persistence qualification. Work like the inverse of MarkPersistentContact.
    void RemovePersistentContact();

    /// @brief Get all contact forces that concern a list of owners.
    /// @param ownerIDs The IDs of the owners.
    /// @param points Fill this vector of float3 with the XYZ components of the contact points.
    /// @param forces Fill this vector of float3 with the XYZ components of the forces.
    /// @return Number of force pairs.
    size_t GetOwnerContactForces(const std::vector<bodyID_t>& ownerIDs,
                                 std::vector<float3>& points,
                                 std::vector<float3>& forces);

    /// @brief Get all contact forces that concern a list of owners.
    /// @details If a contact involves at least one of the owner IDs provided as the first arg this method, it will be
    /// outputted. Note if a contact involves two IDs of the user-provided list, then the force for that contact will be
    /// given as the force experienced by whichever owner that appears earlier in the ID list.
    /// @param ownerIDs The IDs of the owners.
    /// @param points Fill this vector of float3 with the XYZ components of the contact points.
    /// @param forces Fill this vector of float3 with the XYZ components of the forces.
    /// @param torques Fill this vector of float3 with the XYZ components of the torques (in local frame).
    /// @param torque_in_local If true, output torque in this body's local ref frame.
    /// @return Number of force pairs.
    size_t GetOwnerContactForces(const std::vector<bodyID_t>& ownerIDs,
                                 std::vector<float3>& points,
                                 std::vector<float3>& forces,
                                 std::vector<float3>& torques,
                                 bool torque_in_local = false);

    /// @brief Set the wildcard values of some triangles.
    /// @param geoID The ID of the starting (first) triangle that needs to be modified.
    /// @param name The name of the wildcard.
    /// @param vals A vector of values that will be assigned to the triangles starting from geoID.
    void SetTriWildcardValue(bodyID_t geoID, const std::string& name, const std::vector<float>& vals);
    /// @brief Set the wildcard values of some spheres.
    /// @param geoID The ID of the starting (first) sphere that needs to be modified.
    /// @param name The name of the wildcard.
    /// @param vals A vector of values that will be assigned to the spheres starting from geoID.
    void SetSphereWildcardValue(bodyID_t geoID, const std::string& name, const std::vector<float>& vals);
    /// @brief Set the wildcard values of some analytical components.
    /// @param geoID The ID of the starting (first) analytical component that needs to be modified.
    /// @param name The name of the wildcard.
    /// @param vals A vector of values that will be assigned to the analytical components starting from geoID.
    void SetAnalWildcardValue(bodyID_t geoID, const std::string& name, const std::vector<float>& vals);

    /// @brief Set the wildcard values of some owners.
    /// @param ownerID The ID of the starting (first) owner that needs to be modified.
    /// @param name The name of the wildcard.
    /// @param vals A vector of values that will be assigned to the owners starting from ownerID.
    void SetOwnerWildcardValue(bodyID_t ownerID, const std::string& name, const std::vector<float>& vals);
    /// @brief Set the wildcard values of some owners.
    /// @param ownerID The ID of the starting (first) owner that needs to be modified.
    /// @param name The name of the wildcard.
    /// @param val The value to set.
    /// @param n The number of owners starting from the first that will be modified by this call. Default is 1.
    void SetOwnerWildcardValue(bodyID_t ownerID, const std::string& name, float val, size_t n = 1) {
        SetOwnerWildcardValue(ownerID, name, std::vector<float>(n, val));
    }

    /// Modify the owner wildcard's values of all entities in family N.
    void SetFamilyOwnerWildcardValue(unsigned int N, const std::string& name, const std::vector<float>& vals);
    /// Modify the owner wildcard's values of all entities in family N.
    void SetFamilyOwnerWildcardValue(unsigned int N, const std::string& name, float val) {
        SetFamilyOwnerWildcardValue(N, name, std::vector<float>(1, val));
    }
    /// @brief Set all clumps in this family to have this material.
    /// @param N Family number.
    /// @param mat Material type.
    void SetFamilyClumpMaterial(unsigned int N, const std::shared_ptr<DEMMaterial>& mat);
    /// @brief Set all meshes in this family to have this material.
    /// @param N Family number.
    /// @param mat Material type.
    void SetFamilyMeshMaterial(unsigned int N, const std::shared_ptr<DEMMaterial>& mat);

    /// @brief Add an extra contact margin to entities in a family so they are registered as potential contact pairs
    /// earlier.
    /// @details You typically need this method when the custom force model contains non-contact forces. The solver
    /// needs this extra margin to preemptively registered contact pairs. If the extra margin is not added, then the
    /// contact pair will not be computed until entities are in physical contact. Note this margin should be as small as
    /// needed, since it potentially increases the total number of contact pairs greatly.
    /// @param N Family number.
    /// @param extra_size The thickness of the extra contact margin.
    void SetFamilyExtraMargin(unsigned int N, float extra_size);

    /// @brief Get the owner wildcard's values of some owners.
    /// @param ownerID Starting owner's ID.
    /// @param name Wildcard's name.
    /// @param n Total number of owners to query, starting from ownerID.
    /// @return Value of this wildcard.
    std::vector<float> GetOwnerWildcardValue(bodyID_t ownerID, const std::string& name, bodyID_t n = 1);
    /// @brief Get the owner wildcard's values of all entities.
    std::vector<float> GetAllOwnerWildcardValue(const std::string& name);
    /// @brief Get the owner wildcard's values of all entities in family N.
    std::vector<float> GetFamilyOwnerWildcardValue(unsigned int N, const std::string& name);

    /// @brief Get the geometry wildcard's values of a series of triangles.
    /// @param geoID The ID of the first triangle.
    /// @param name Wildcard's name.
    /// @param n The number of triangles to query following the ID of the first one.
    /// @return Vector of values of the wildcards.
    std::vector<float> GetTriWildcardValue(bodyID_t geoID, const std::string& name, size_t n);
    /// @brief Get the geometry wildcard's values of a series of spheres.
    /// @param geoID The ID of the first sphere.
    /// @param name Wildcard's name.
    /// @param n The number of spheres to query following the ID of the first one.
    /// @return Vector of values of the wildcards.
    std::vector<float> GetSphereWildcardValue(bodyID_t geoID, const std::string& name, size_t n);
    /// @brief Get the geometry wildcard's values of a series of analytical entities.
    /// @param geoID The ID of the first analytical entity.
    /// @param name Wildcard's name.
    /// @param n The number of analytical entities to query following the ID of the first one.
    /// @return Vector of values of the wildcards.
    std::vector<float> GetAnalWildcardValue(bodyID_t geoID, const std::string& name, size_t n);

    /// @brief If the user used async-ed version of a tracker's get/set methods (to get a speed boost in many piecemeal
    /// accesses of a long array), this method should be called to mark the end of to-host transactions. But usually,
    /// the user would use sync-ed version of the methods by default and this call is not needed in that case.
    void SyncMemoryTransfer();

    /// Change all entities with family number ID_from to have a new number ID_to, when the condition defined by the
    /// string is satisfied by the entities in question. This should be called before initialization, and will be baked
    /// into the solver, so the conditions will be checked and changes applied every time step.
    void ChangeFamilyWhen(unsigned int ID_from, unsigned int ID_to, const std::string& condition);

    /// Change all entities with family number ID_from to have a new number ID_to, immediately. This is callable when kT
    /// and dT are hanging, not when they are actively working, or the behavior is not defined.
    void ChangeFamily(unsigned int ID_from, unsigned int ID_to);

    /// @brief Change the family number for the clumps in a box region to the specified value.
    /// @param fam_num The family number to change into.
    /// @param X {L, U} that discribes the lower and upper bound of the X coord of the box region.
    /// @param Y The lower and upper bound of the Y coord of the box region.
    /// @param Z The lower and upper bound of the Z coord of the box region.
    /// @param orig_fam Only clumps that originally have these family numbers will be modified. Leave empty to apply
    /// changes regardless of original family numbers.
    /// @return The number of owners that get changed by this call.
    size_t ChangeClumpFamily(
        unsigned int fam_num,
        const std::pair<double, double>& X = std::pair<double, double>(-DEME_HUGE_FLOAT, DEME_HUGE_FLOAT),
        const std::pair<double, double>& Y = std::pair<double, double>(-DEME_HUGE_FLOAT, DEME_HUGE_FLOAT),
        const std::pair<double, double>& Z = std::pair<double, double>(-DEME_HUGE_FLOAT, DEME_HUGE_FLOAT),
        const std::set<unsigned int>& orig_fam = std::set<unsigned int>());

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

    /// Add an analytical object to the simulation system.
    std::shared_ptr<DEMExternObj> AddExternalObject();
    /// @brief Add an analytical plane to the simulation.
    /// @param pos A point on the plane.
    /// @param normal The normal direction of the plane. Note entities are always considered in-contact with the plane
    /// from the positive normal direction.
    /// @param material Material of the plane.
    /// @return A handle to the added plane object.
    std::shared_ptr<DEMExternObj> AddBCPlane(const float3 pos,
                                             const float3 normal,
                                             const std::shared_ptr<DEMMaterial>& material);
    std::shared_ptr<DEMExternObj> AddBCPlane(const std::vector<float>& pos,
                                             const std::vector<float>& normal,
                                             const std::shared_ptr<DEMMaterial>& material) {
        assertThreeElements(pos, "AddBCPlane", "pos");
        assertThreeElements(normal, "AddBCPlane", "normal");
        return AddBCPlane(make_float3(pos[0], pos[1], pos[2]), make_float3(normal[0], normal[1], normal[2]), material);
    }

    /// Remove host-side cached vectors (so you can re-define them, and then re-initialize system).
    void ClearCache();

    /// Write the current status of clumps to a file
    void WriteClumpFile(const std::string& outfilename, unsigned int accuracy = 10) const;
    void WriteClumpFile(const std::filesystem::path& outfilename, unsigned int accuracy = 10) const {
        WriteClumpFile(outfilename.string(), accuracy);
    }
    /// Write the current status of `clumps' to a file, but not as clumps, instead, as each individual sphere. This may
    /// make small-scale rendering easier.
    void WriteSphereFile(const std::string& outfilename) const;
    void WriteSphereFile(const std::filesystem::path& outfilename) const { WriteSphereFile(outfilename.string()); }
    /// @brief Write all contact pairs to a file.
    /// @details The outputted torque using this method is in global, rather than each object's local coordinate system.
    /// @param outfilename Output filename.
    /// @param force_thres Forces with magnitude smaller than this amount will not be outputted.
    void WriteContactFile(const std::string& outfilename, float force_thres = DEME_TINY_FLOAT) const;
    void WriteContactFile(const std::filesystem::path& outfilename) const { WriteContactFile(outfilename.string()); }
    /// @brief Write all contact pairs kT-supplied to a file, thus including the potential ones (those are not yet in
    /// contact, or recently used to be in contact).
    /// @details The outputted torque using this method is in global, rather than each object's local coordinate system.
    /// @param outfilename Output filename.
    void WriteContactFileIncludingPotentialPairs(const std::string& outfilename) const {
        WriteContactFile(outfilename, -1.0);
    }
    void WriteContactFileIncludingPotentialPairs(const std::filesystem::path& outfilename) const {
        WriteContactFileIncludingPotentialPairs(outfilename.string());
    }
    /// Write the current status of all meshes to a file.
    void WriteMeshFile(const std::string& outfilename) const;
    void WriteMeshFile(const std::filesystem::path& outfilename) const { WriteMeshFile(outfilename.string()); }

    /// @brief Read 3 columns of your choice from a CSV filem and group them by clump_header.
    /// @param infilename CSV filename.
    /// @param x_header CSV header for the first col.
    /// @param y_header CSV header for the second col.
    /// @param z_header CSV header for the third col.
    /// @param clump_header The identifier column to separate types of clumps.
    /// @return Unordered_map which maps types of clumps to a respective vector of float3s.
    static std::unordered_map<std::string, std::vector<float3>> ReadClumpFloat3FromCsv(
        const std::string& infilename,
        const std::string& x_header,
        const std::string& y_header,
        const std::string& z_header,
        const std::string& clump_header) {
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
    /// Read clump coordinates from a CSV file (whose format is consistent with this solver's clump output file).
    /// Returns an unordered_map which maps each unique clump type name to a vector of float3 (XYZ coordinates).
    static std::unordered_map<std::string, std::vector<float3>> ReadClumpXyzFromCsv(const std::string& infilename) {
        return ReadClumpFloat3FromCsv(infilename, OUTPUT_FILE_X_COL_NAME, OUTPUT_FILE_Y_COL_NAME,
                                      OUTPUT_FILE_Z_COL_NAME, OUTPUT_FILE_CLUMP_TYPE_NAME);
    }
    /// Read clump velocity from a CSV file (whose format is consistent with this solver's clump output file).
    /// Returns an unordered_map which maps each unique clump type name to a vector of float3 (velocity).
    static std::unordered_map<std::string, std::vector<float3>> ReadClumpVelFromCsv(const std::string& infilename) {
        return ReadClumpFloat3FromCsv(infilename, OUTPUT_FILE_VEL_X_COL_NAME, OUTPUT_FILE_VEL_Y_COL_NAME,
                                      OUTPUT_FILE_VEL_Z_COL_NAME, OUTPUT_FILE_CLUMP_TYPE_NAME);
    }
    /// Read clump angular velocity from a CSV file (whose format is consistent with this solver's clump output file).
    /// Returns an unordered_map which maps each unique clump type name to a vector of float3 (angular velocity).
    static std::unordered_map<std::string, std::vector<float3>> ReadClumpAngVelFromCsv(const std::string& infilename) {
        return ReadClumpFloat3FromCsv(infilename, OUTPUT_FILE_ANGVEL_X_COL_NAME, OUTPUT_FILE_ANGVEL_Y_COL_NAME,
                                      OUTPUT_FILE_ANGVEL_Z_COL_NAME, OUTPUT_FILE_CLUMP_TYPE_NAME);
    }

    /// Read clump quaternions from a CSV file (whose format is consistent with this solver's clump output file).
    /// Returns an unordered_map which maps each unique clump type name to a vector of float4 (4 components of the
    /// quaternion, (Qx, Qy, Qz, Qw) = (0, 0, 0, 1) means 0 rotation).
    static std::unordered_map<std::string, std::vector<float4>> ReadClumpQuatFromCsv(const std::string& infilename) {
        std::unordered_map<std::string, std::vector<float4>> type_Q_map;
        io::CSVReader<5, io::trim_chars<' ', '\t'>, io::no_quote_escape<','>, io::throw_on_overflow,
                      io::empty_line_comment>
            in(infilename);
        in.read_header(io::ignore_extra_column, OUTPUT_FILE_CLUMP_TYPE_NAME, OUTPUT_FILE_QW_COL_NAME,
                       OUTPUT_FILE_QX_COL_NAME, OUTPUT_FILE_QY_COL_NAME, OUTPUT_FILE_QZ_COL_NAME);
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

    /// Intialize the simulation system.
    void Initialize(bool dry_run = false);

    /// Advance simulation by this amount of time, and at the end of this call, synchronize kT and dT. This is suitable
    /// for a longer call duration and without co-simulation.
    void DoDynamicsThenSync(double thisCallDuration);

    /// Advance simulation by this amount of time (but does not attempt to sync kT and dT). This can work with both long
    /// and short call durations and allows interplay with co-simulation APIs.
    void DoDynamics(double thisCallDuration);

    /// Equivalent to calling DoDynamics with the time step size as the argument.
    void DoStepDynamics() { DoDynamics(m_ts_size); }

    /// @brief Transfer the cached sim params to the workers. Used for sim environment modification after system
    /// initialization.
    void UpdateSimParams();

    /// @brief Transfer newly loaded clumps to the GPU-side in mid-simulation.
    void UpdateClumps();

    /// @brief Update the time step size. Used after system initialization.
    /// @param ts Time step size.
    void UpdateStepSize(double ts);

    /// Show the collaboration stats between dT and kT. This is more useful for tweaking the number of time steps that
    /// dT should be allowed to be in advance of kT.
    void ShowThreadCollaborationStats();

    /// Show the wall time and percentages of wall time spend on various solver tasks.
    void ShowTimingStats();

    /// Show potential anomalies that may have been there in the simulation, then clear the anomaly log.
    void ShowAnomalies();

    /// Reset the collaboration stats between dT and kT back to the initial value (0). You should call this if you want
    /// to start over and re-inspect the stats of the new run; otherwise, it is generally not needed, you can go ahead
    /// and destroy DEMSolver.
    void ClearThreadCollaborationStats();

    /// Reset the recordings of the wall time and percentages of wall time spend on various solver tasks.
    void ClearTimingStats();

    /// Removes all entities associated with a family from the arrays (to save memory space).
    void PurgeFamily(unsigned int family_num);

    /// Release the memory for the flattened arrays (which are used for initialization pre-processing and transferring
    /// info the worker threads).
    void ReleaseFlattenedArrays();

    /// @brief Return whether the solver is currently reducing force in the force calculation kernel.
    bool GetWhetherForceCollectInKernel() { return collect_force_in_force_kernel; }

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

    /// Choose output format.
    void SetOutputFormat(OUTPUT_FORMAT format) { m_out_format = format; }
    /// Specify the information that needs to go into the clump or sphere output files.
    void SetOutputContent(unsigned int content) { m_out_content = content; }
    /// Specify the file format of contact pairs.
    void SetContactOutputFormat(OUTPUT_FORMAT format) { m_cnt_out_format = format; }
    /// Specify the information that needs to go into the contact pair output files.
    void SetContactOutputContent(unsigned int content) { m_cnt_out_content = content; }
    /// Specify the file format of meshes.
    void SetMeshOutputFormat(MESH_FORMAT format) { m_mesh_out_format = format; }
    /// Enable/disable outputting owner wildcard values to file.
    void EnableOwnerWildcardOutput(bool enable = true) { m_is_out_owner_wildcards = enable; }
    /// Enable/disable outputting contact wildcard values to the contact file.
    void EnableContactWildcardOutput(bool enable = true) { m_is_out_cnt_wildcards = enable; }
    /// Enable/disable outputting geometry wildcard values to the contact file.
    void EnableGeometryWildcardOutput(bool enable = true) { m_is_out_geo_wildcards = enable; }

    /// @brief Set the verbosity level of the solver.
    /// @param verbose "QUIET", "ERROR", "WARNING", "INFO", "STEP_ANOMALY", "STEP_METRIC", "DEBUG" or "STEP_DEBUG".
    /// Recommend "INFO".
    void SetVerbosity(const std::string& verbose);
    /// @brief Choose sphere and clump output file format.
    /// @param format Choice among "CSV", "BINARY".
    void SetOutputFormat(const std::string& format);
    /// @brief Specify the information that needs to go into the clump or sphere output files.
    /// @param content A list of "XYZ", "QUAT", "ABSV", "VEL", "ANG_VEL", "ABS_ACC", "ACC", "ANG_ACC", "FAMILY", "MAT",
    /// "OWNER_WILDCARD" and/or "GEO_WILDCARD".
    void SetOutputContent(const std::vector<std::string>& content);
    /// @brief Specify the file format of contact pairs.
    /// @param format Choice among "CSV", "BINARY".
    void SetContactOutputFormat(const std::string& format);
    /// @brief Specify the information that needs to go into the contact pair output files.
    /// @param content A list of "CNT_TYPE", "FORCE", "POINT", "COMPONENT", "NORMAL", "TORQUE", "CNT_WILDCARD", "OWNER",
    /// "GEO_ID" and/or "NICKNAME".
    void SetContactOutputContent(const std::vector<std::string>& content);
    /// @brief Specify the output file format of meshes.
    /// @param format A choice between "VTK", "OBJ".
    void SetMeshOutputFormat(const std::string& format);

    // void SetOutputContent(const std::string& content) { SetOutputContent({content}); }
    // void SetContactOutputContent(const std::string& content) { SetContactOutputContent({content}); }

    /// @brief Add a library that the kernels will be compiled with (so that the user can use the provided methods in
    /// their customized code, like force model).
    /// @param lib_name The lib to include. For example, "math_functions.h".
    void AddKernelInclude(const std::string& lib_name);
    /// @brief Set the kernels' headers' extra include lines. Useful for customization.
    /// @param includes The extra headers, as a string.
    void SetKernelInclude(const std::string& includes) { kernel_includes = includes; }
    /// @brief Remove all extra libraries that the kernels `include' in their headers.
    void RemoveKernelInclude() { kernel_includes = " "; }

    /// @brief Print kT's scratch space usage. This is a debug method.
    void PrintKinematicScratchSpaceUsage() const { kT->printScratchSpaceUsage(); }

    /// Let dT do this call and return the reduce value of the inspected quantity.
    float dTInspectReduce(const std::shared_ptr<jitify::Program>& inspection_kernel,
                          const std::string& kernel_name,
                          INSPECT_ENTITY_TYPE thing_to_insp,
                          CUB_REDUCE_FLAVOR reduce_flavor,
                          bool all_domain);
    float* dTInspectNoReduce(const std::shared_ptr<jitify::Program>& inspection_kernel,
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
    // If true, dT should sort contact arrays (based on contact type) before usage
    bool should_sort_contacts = true;
    // If true, the solvers may need to do a per-step sweep to apply family number changes
    bool famnum_can_change_conditionally = false;

    // Should jitify clump template into kernels
    bool jitify_clump_templates = true;
    // Should jitify mass/MOI properties into kernels
    bool jitify_mass_moi = true;

    enum class INIT_BIN_SIZE_TYPE { EXPLICIT, MULTI_MIN_SPH, TARGET_NUM };
    // User explicitly set a bin size to use
    INIT_BIN_SIZE_TYPE use_user_defined_bin_size = INIT_BIN_SIZE_TYPE::TARGET_NUM;
    // User explicity specify a expand factor to use
    bool use_user_defined_expand_factor = false;
    // Whether to auto-adjust the bin size and the max update frequency
    bool auto_adjust_bin_size = true;
    bool auto_adjust_update_freq = true;
    // User-instructed initial bin size as a multiple of smallest sphere radius
    float m_binSize_as_multiple = 8.0;
    // Target initial bin number
    size_t m_target_init_bin_num = 1e6;

    // I/O related flags
    // The output file format for clumps and spheres
    // OUTPUT_MODE m_clump_out_mode = OUTPUT_MODE::SPHERE;
    OUTPUT_FORMAT m_out_format = OUTPUT_FORMAT::CSV;
    unsigned int m_out_content = OUTPUT_CONTENT::QUAT | OUTPUT_CONTENT::ABSV;
    // The output file format for contact pairs
    OUTPUT_FORMAT m_cnt_out_format = OUTPUT_FORMAT::CSV;
    // The output file content for contact pairs
    unsigned int m_cnt_out_content = CNT_OUTPUT_CONTENT::OWNER | CNT_OUTPUT_CONTENT::GEO_ID |
                                     CNT_OUTPUT_CONTENT::FORCE | CNT_OUTPUT_CONTENT::CNT_POINT |
                                     CNT_OUTPUT_CONTENT::CNT_WILDCARD;
    // The output file format for meshes
    MESH_FORMAT m_mesh_out_format = MESH_FORMAT::VTK;
    // If the solver should output wildcards to file
    bool m_is_out_owner_wildcards = false;
    bool m_is_out_cnt_wildcards = false;
    bool m_is_out_geo_wildcards = false;

    // User-instructed simulation `world' size. Note it is an approximate of the true size and we will generate a world
    // not smaller than this. This is useful if the user want to automatically add BCs enclosing this user-defined
    // domain.
    float3 m_user_box_min = make_float3(-DEFAULT_BOX_DOMAIN_SIZE / 2.);
    float3 m_user_box_max = make_float3(DEFAULT_BOX_DOMAIN_SIZE / 2.);

    // The enlarged user-instructed box.. We do this because we don't want the box domain to have boundaries exactly on
    // the edge of the world.
    float3 m_target_box_min = make_float3(-DEFAULT_BOX_DOMAIN_SIZE * (1. + DEFAULT_BOX_DOMAIN_ENLARGE_RATIO) / 2.);
    float3 m_target_box_max = make_float3(DEFAULT_BOX_DOMAIN_SIZE * (1. + DEFAULT_BOX_DOMAIN_ENLARGE_RATIO) / 2.);

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
    double m_ts_size = 1e-5;
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
    float m_expand_safety_multi = 1.f;
    // The `base' velocity we always consider entities to have, when determining the thickness of the margin to add for
    // contact detection.
    float m_expand_base_vel = 3.f;

    // The method of determining the thickness of the margin added to CD
    // Default is using a max_vel inspector of the clumps to decide it
    enum class MARGIN_FINDER_TYPE { DEM_INSPECTOR, DEFAULT };
    MARGIN_FINDER_TYPE m_max_v_finder_type = MARGIN_FINDER_TYPE::DEFAULT;
    // User-instructed approximate maximum velocity (of any point on a body in the simulation)
    float m_approx_max_vel = DEME_HUGE_FLOAT;
    // The inspector that will be used for querying system max velocity
    std::shared_ptr<DEMInspector> m_approx_max_vel_func;

    // The number of user-estimated (max) number of owners that will be present in the simulation. If 0, then the arrays
    // will just be resized at intialization based on the input size.
    size_t m_instructed_num_owners = 0;

    // Whether the GPU-side systems have been initialized
    bool sys_initialized = false;
    // Smallest sphere radius (used to let the user know whether the expand factor is sufficient)
    float m_smallest_radius = FLT_MAX;

    // The number of dT steps before it waits for a kT update. The default value means every dT step will wait for a
    // newly produced contact-pair info (from kT) before proceeding.
    int m_suggestedFutureDrift = 40;

    // This is an unused variable which is supposed to be related to m_suggestedFutureDrift...
    int m_updateFreq = 20;

    // The extra libs that the kernels need to include.
    std::string kernel_includes = "#include <curand_kernel.h>\n";

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

    // If the solver sees there are more spheres in a bin than a this `maximum', it errors out
    unsigned int threshold_too_many_spheres_in_bin = 32768;
    // If the solver sees there are more triangles in a bin than a this `maximum', it errors out
    unsigned int threshold_too_many_tri_in_bin = 32768;
    // The max velocity at which the simulation should error out
    float threshold_error_out_vel = 1e3;
    // Num of steps that kT takes average before making a conclusion on the performance of this bin size
    unsigned int auto_adjust_observe_steps = 25;
    // See corresponding method for those...
    float auto_adjust_max_rate = 0.03;
    float auto_adjust_acc = 0.2;
    float auto_adjust_upper_proactive_ratio = 0.75;
    float auto_adjust_lower_proactive_ratio = 0.5;
    unsigned int upper_bound_future_drift = 200;
    float max_drift_ahead_of_avg_drift = 4.;
    float max_drift_multiple_of_avg_drift = 1.05;
    unsigned int max_drift_gauge_history_size = 200;

    // See SetNoForceRecord
    bool no_recording_contact_forces = false;
    // See SetCollectAccRightAfterForceCalc
    bool collect_force_in_force_kernel = false;

    // Error-out avg num contacts
    float threshold_error_out_num_cnts = 100.;

    // Integrator type
    TIME_INTEGRATOR m_integrator = TIME_INTEGRATOR::EXTENDED_TAYLOR;

    // The force model which will be used
    std::unordered_map<std::string, std::shared_ptr<DEMForceModel>> m_force_model;

    // Strategy for auto-adapting time steps size
    ADAPT_TS_TYPE adapt_ts_type = ADAPT_TS_TYPE::NONE;

    ////////////////////////////////////////////////////////////////////////////////
    // No user method is provided to modify the following key quantities, even if
    // there are entites added to/removed from the simulation, in which case
    // they will just be modified. At the time these quantities should be cleared,
    // the user might as well reconstruct the simulator.
    ////////////////////////////////////////////////////////////////////////////////

    // All material properties names
    std::set<std::string> m_material_prop_names;

    // The material properties that are pair-wise (example: friction coeff)
    std::set<std::string> m_pairwise_material_prop_names;

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
    // jitify's compilation options
    std::vector<std::string> m_jitify_options;

    // A map that records the numbering for user-defined owner wildcards
    std::unordered_map<std::string, unsigned int> m_owner_wc_num;
    // A map that records the numbering for user-defined geometry wildcards
    std::unordered_map<std::string, unsigned int> m_geo_wc_num;
    // A map that records the numbering for user-defined per-contact wildcards
    std::unordered_map<std::string, unsigned int> m_cnt_wc_num;

    // Meshes cached on dT side that has corresponding owner number associated. Useful for modifying meshes.
    std::vector<std::shared_ptr<DEMMeshConnected>> m_meshes;
    // A map between the owner of mesh, and the offset this mesh lives in m_meshes array.
    std::unordered_map<bodyID_t, unsigned int> m_owner_mesh_map;

    ////////////////////////////////////////////////////////////////////////////////
    // Cached user's direct (raw) inputs concerning the actual physics objects
    // presented in the simulation, which need to be processed before shipment,
    // at initialization time. These items will be cleared after initialization,
    // and before users add entites from the simulation on-the-fly.
    ////////////////////////////////////////////////////////////////////////////////
    //// TODO: These re-initialization flavors haven't been added

    // This is the cached material information.
    // It will be massaged into the GPU memory upon Initialize().
    std::vector<std::shared_ptr<DEMMaterial>> m_loaded_materials;

    // Pair-wise material properties
    std::unordered_map<std::string, std::vector<std::pair<std::pair<unsigned int, unsigned int>, float>>>
        m_pairwise_matprop;

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
    /// Set the default bin (for contact detection) size to be the same of the smallest sphere
    void decideBinSize();
    /// The method of deciding the thickness of contact margin (user-specified max vel; or a custom inspector)
    void decideCDMarginStrat();
    /// Add boundaries to the simulation `world' based on user instructions
    void addWorldBoundingBox();
    /// Transfer cached solver preferences/instructions to dT and kT.
    void setSolverParams();
    /// Transfer (CPU-side) cached simulation data (about sim world) to the GPU-side. It is called automatically during
    /// system initialization.
    void setSimParams();
    /// Transfer cached clump templates info etc. to GPU-side arrays.
    void initializeGPUArrays();
    /// Allocate memory space for GPU-side arrays.
    void allocateGPUArrays();
    /// Pack array pointers to a struct so they can be easily used as kernel arguments.
    void packDataPointers();
    /// @brief Move host-prepared simulation parameter data to device.
    void migrateSimParamsToDevice();
    /// @brief Move host-prepared array data to device.
    void migrateArrayDataToDevice();
    /// @brief Move device-modified array data to host. This is important when the simulation already started, but some
    /// data need to be re-cooked on host. For small updates to the host, we don't need to do this, just directly modify
    /// the device array.
    void migrateArrayDataToHost();
    /// Warn users if the data types defined in Defines.h do not blend well with the user inputs (fist-round
    /// coarse-grain sanity check).
    void validateUserInputs();
    /// Prepare the material/contact proxy matrix force computation kernels.
    void figureOutMaterialProxies();
    /// Figure out info about external objects and how they should be jitified.
    void preprocessAnalyticalObjs();
    /// Figure out info about external meshed objects.
    void preprocessTriangleObjs();
    /// Report simulation stats at initialization.
    void reportInitStats() const;
    /// Based on user input, prepare family_mask_matrix (family contact map matrix).
    void figureOutFamilyMasks();
    /// Reset kT and dT back to a status like when the simulation system is constructed. I decided to make this a
    /// private method because it can be dangerous, as if it is called when kT is waiting at the outer loop, it will
    /// stall the siumulation. So perhaps the user should not call it without knowing what they are doing. Also note
    /// this call does not reset the collaboration log between kT and dT.
    void resetWorkerThreads();
    /// Transfer newly loaded clumps/meshed objects to the GPU-side in mid-simulation and allocate GPU memory space for
    /// them.
    void updateClumpMeshArrays(size_t nOwners,
                               size_t nClumps,
                               size_t nSpheres,
                               size_t nTriMesh,
                               size_t nFacets,
                               unsigned int nExtObj_old,
                               unsigned int nAnalGM_old);
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
    /// Print due information on worker threads reported anomalies
    bool goThroughWorkerAnomalies();
    /// @brief Implementation of getting (unsorted) contact pairs from dT.
    /// @param type_func Exclude certain contact types from being outputted if this evaluates to false.
    void getContacts_impl(std::vector<bodyID_t>& idA,
                          std::vector<bodyID_t>& idB,
                          std::vector<contact_t>& cnt_type,
                          std::vector<family_t>& famA,
                          std::vector<family_t>& famB,
                          std::function<bool(contact_t)> type_func) const;
    /// The implimentation of persistency assignment
    void assignFamilyPersistentContact_impl(
        unsigned int N1,
        unsigned int N2,
        notStupidBool_t is_or_not,
        const std::function<bool(family_t, family_t, unsigned int, unsigned int)>& condition);

    // Persistent contact implementations
    void assignFamilyPersistentContactEither(unsigned int N, notStupidBool_t is_or_not);
    void assignFamilyPersistentContactBoth(unsigned int N, notStupidBool_t is_or_not);
    void assignFamilyPersistentContact(unsigned int N1, unsigned int N2, notStupidBool_t is_or_not);
    void assignPersistentContact(notStupidBool_t is_or_not);

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
    inline void equipKernelIncludes(std::unordered_map<std::string, std::string>& strMap);

    // Default solver params at construction time
    void setDefaultSolverParams();
};

}  // namespace deme

#endif
