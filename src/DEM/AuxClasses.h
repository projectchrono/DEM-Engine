//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_INSPECTOR_HPP
#define DEME_INSPECTOR_HPP

#include <unordered_map>
#include <core/utils/JitHelper.h>
#include <DEM/Defines.h>
// #include <cuda.h>
// #include <cuda_runtime_api.h>

// Forward declare jitify::Program to avoid downstream dependency
namespace jitify {
class Program;
}

namespace deme {

class DEMSolver;
class DEMDynamicThread;

/// A class that the user can construct to inspect a certain property (such as void ratio, maximum Z coordinate...) of
/// their simulation entites, in a given region.
class DEMInspector {
  private:
    std::shared_ptr<jitify::Program> inspection_kernel;

    std::string inspection_code;
    std::string in_region_code;
    // The name for the index of that quantity that is being inspected
    std::string index_name;
    CUB_REDUCE_FLAVOR reduce_flavor = CUB_REDUCE_FLAVOR::NONE;
    INSPECT_ENTITY_TYPE thing_to_insp;
    std::string kernel_name;
    bool all_domain = true;
    bool isVoidRatio = false;

    bool initialized = false;

    // Its parent DEMSolver and dT system
    DEMSolver* sys;
    DEMDynamicThread* dT;

    // Based on user input...
    void switch_quantity_type(const std::string& quantity);

    void assertInit();

  public:
    friend class DEMSolver;
    friend class DEMDynamicThread;

    DEMInspector(DEMSolver* sim_sys, DEMDynamicThread* dT_sys, const std::string& quantity) : sys(sim_sys), dT(dT_sys) {
        switch_quantity_type(quantity);
        // In default constructor, all entities are considered `in-region' by default
        in_region_code = " ";
    }
    DEMInspector(DEMSolver* sim_sys, DEMDynamicThread* dT_sys, const std::string& quantity, const std::string& region)
        : sys(sim_sys), dT(dT_sys) {
        switch_quantity_type(quantity);
        in_region_code = region;
        all_domain = false;
    }
    //// TODO: Another overload for region definition
    ~DEMInspector() {}

    void SetInspectionCode(const std::string& code) { inspection_code = code; }

    // Initialize with the DEM simulation system (user should not call this)
    void Initialize(const std::unordered_map<std::string, std::string>& Subs, bool force = false);

    /// Get the reduce value of the quantity that you wish to inspect.
    float GetValue();

    /// Get the values of the quantity that you wish to inspect (for non-reduced inspections).
    float* GetValues();

    /// Get the value (as a vector) of the quantity that you wish to inspect
    // std::vector<float> GetVector();

    /// Get value directly within dT
    float* dT_GetValue();
};

// A struct to get or set tracked owner entities, mainly for co-simulation
class DEMTracker {
  private:
    void assertMesh(const std::string& name);
    void assertGeoSize(size_t input_length, const std::string& func_name, const std::string& geo_type);
    void assertOwnerSize(size_t input_length, const std::string& name);
    void assertThereIsForcePairs(const std::string& name);
    // Its parent DEMSolver system
    DEMSolver* sys;

  public:
    DEMTracker(DEMSolver* sim_sys) : sys(sim_sys) {}
    ~DEMTracker() {}

    // The tracked object
    std::shared_ptr<DEMTrackedObj> obj;

    /// Get the owner ID of the tracked obj.
    bodyID_t GetOwnerID(size_t offset = 0);

    /// Get the position of this tracked object.
    float3 Pos(size_t offset = 0);
    std::vector<float> GetPos(size_t offset = 0);
    /// Get the angular velocity of this tracked object in its own local coordinate system. Applying OriQ to it would
    /// give you the ang vel in global frame.
    float3 AngVelLocal(size_t offset = 0);
    std::vector<float> GetAngVelLocal(size_t offset = 0);
    /// Get the angular velocity of this tracked object in global coordinate system.
    float3 AngVelGlobal(size_t offset = 0);
    std::vector<float> GetAngVelGlobal(size_t offset = 0);
    /// Get the velocity of this tracked object in global frame.
    float3 Vel(size_t offset = 0);
    std::vector<float> GetVel(size_t offset = 0);
    /// Get the quaternion that represents the orientation of this tracked object's own coordinate system.
    float4 OriQ(size_t offset = 0);
    /// @brief Get the quaternion that represents the orientation of this tracked object's own coordinate system.
    /// @return A vector of 4 floats. The order is (x, y, z, w). If using Chrono naming convention, then it is (e1, e2,
    /// e3, e0).
    std::vector<float> GetOriQ(size_t offset = 0);
    /// @brief Get the family number of the tracked object.
    /// @param offset The offset of the entites to get family number out of.
    /// @return The family number.
    unsigned int GetFamily(size_t offset = 0);

    /// @brief Get the clumps that are in contact with this tracked owner as a vector.
    /// @param offset Offset to the first item this tracker is tracking. Default is 0.
    /// @return Clump owner IDs in contact with this owner.
    std::vector<bodyID_t> GetContactClumps(size_t offset = 0);

    /// Get the a portion of the acceleration of this tracked object, that is the result of its contact with other
    /// simulation entities. In most cases, this means excluding the gravitational acceleration. The acceleration is in
    /// global frame.
    float3 ContactAcc(size_t offset = 0);
    std::vector<float> GetContactAcc(size_t offset = 0);
    /// Get the a portion of the angular acceleration of this tracked object, that is the result of its contact with
    /// other simulation entities. The acceleration is in this object's local frame.
    float3 ContactAngAccLocal(size_t offset = 0);
    std::vector<float> GetContactAngAccLocal(size_t offset = 0);
    /// Get the a portion of the angular acceleration of this tracked object, that is the result of its contact with
    /// other simulation entities. The acceleration is in this object's global frame.
    float3 ContactAngAccGlobal(size_t offset = 0);
    std::vector<float> GetContactAngAccGlobal(size_t offset = 0);

    /// Get the owner's wildcard value.
    float GetOwnerWildcardValue(const std::string& name, size_t offset = 0);
    /// @brief Get the geometry wildcard values for all the geometry entities tracked by this tracker.
    /// @param name Name of the wildcard.
    /// @return All the values.
    std::vector<float> GetGeometryWildcardValues(const std::string& name);
    /// @brief Get the geometry wildcard values for a geometry entity tracked by this tracker.
    /// @param name Name of the wildcard.
    /// @param offset The offset to this entity (where to start the modification). If first entity, input 0.
    /// @return The value.
    float GetGeometryWildcardValue(const std::string& name, size_t offset);

    /// @brief Set the position of this tracked object.
    void SetPos(float3 pos, size_t offset = 0);
    void SetPos(const std::vector<float>& pos, size_t offset = 0);
    /// @brief Set the angular velocity of this tracked object in its own local coordinate system.
    void SetAngVel(float3 angVel, size_t offset = 0);
    void SetAngVel(const std::vector<float>& angVel, size_t offset = 0);
    /// @brief Set the velocity of this tracked object in global frame.
    void SetVel(float3 vel, size_t offset = 0);
    void SetVel(const std::vector<float>& vel, size_t offset = 0);
    /// @brief Set the quaternion which represents the orientation of this tracked object's coordinate system.
    void SetOriQ(float4 oriQ, size_t offset = 0);
    void SetOriQ(const std::vector<float>& oriQ, size_t offset = 0);
    /// Add an extra acc to the tracked body, for the next time step. Note if the user intends to add a persistent
    /// external force, then using family prescription is the better method.
    void AddAcc(float3 acc, size_t offset = 0);
    void AddAcc(const std::vector<float>& acc, size_t offset = 0);
    /// Add an extra angular acceleration to the tracked body, for the next time step. Note if the user intends to add a
    /// persistent external torque, then using family prescription is the better method.
    void AddAngAcc(float3 angAcc, size_t offset = 0);
    void AddAngAcc(const std::vector<float>& angAcc, size_t offset = 0);
    /// Change the size of clump entities
    void ChangeClumpSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors);
    /// @brief Change the family numbers of all the entities tracked by this tracker.
    /// @param fam_num Family number to change to.
    void SetFamily(unsigned int fam_num);
    /// @brief Change the family numbers of all the entities tracked by this tracker.
    /// @param fam_nums The family numbers to change to for each entity.
    void SetFamily(const std::vector<unsigned int>& fam_nums);
    /// @brief Change the family number of one entities tracked by this tracker.
    /// @param fam_num Family number to change to.
    /// @param offset The offset to this entites. If first entites, input 0.
    void SetFamily(unsigned int fam_num, size_t offset);
    /// @brief Get the mass of the tracked object.
    /// @param offset The offset to this entites. If first entites, input 0.
    /// @return Mass.
    float Mass(size_t offset = 0);
    /// @brief Get the moment of inertia (in principal axis frame) of the tracked object.
    /// @param offset The offset to this entites. If first entites, input 0.
    /// @return The moment of inertia (in principal axis frame).
    float3 MOI(size_t offset = 0);
    std::vector<float> GetMOI(size_t offset = 0);

    /// @brief Apply the new mesh node positions such that the tracked mesh is replaced by the new_nodes.
    /// @details This affects triangle facets' relative positions wrt the mesh center (CoM) only; mesh's overall
    /// position/rotation in simulation is not affected. So if provided input is the new mesh location with
    /// consideration of its CoM's motion, then you should not use tracker to further modify the mesh's CoM; if the
    /// provided input is the new mesh location without considering the displacement of mesh's CoM (aka only the mesh
    /// deformation), then you should then use tracker to further update the mesh's CoM.
    /// @param new_nodes New locations of mesh nodes. The length of the argument vector must agree with the number of
    /// nodes in the tracked mesh.
    void UpdateMesh(const std::vector<float3>& new_nodes);
    void UpdateMesh(const std::vector<std::vector<float>>& new_nodes);
    /// @brief Change the coordinates of each mesh node by the given amount.
    /// @details This affects triangle facets' relative positions wrt the mesh center (CoM) only; mesh's overall
    /// position/rotation in simulation is not affected. So if provided input is the mesh deformation with
    /// consideration of its CoM's motion, then you should not use tracker to further modify the mesh's CoM; if the
    /// provided input is the mesh deformation without considering the displacement of mesh's CoM, then you should then
    /// use tracker to further update the mesh's CoM.
    /// @param deformation Deformation of mesh nodes. The length of the argument vector must agree with the number of
    /// nodes in the tracked mesh.
    void UpdateMeshByIncrement(const std::vector<float3>& deformation);
    void UpdateMeshByIncrement(const std::vector<std::vector<float>>& deformation);
    /// @brief Get a handle for the mesh this tracker is tracking.
    /// @return Pointer to the mesh.
    std::shared_ptr<DEMMeshConnected>& GetMesh();
    /// @brief Get the current locations of all the nodes in the mesh being tracked.
    /// @return A vector of float3 representing the global coordinates of the mesh nodes.
    std::vector<float3> GetMeshNodesGlobal();
    /// @brief Get the current locations of all the nodes in the mesh being tracked.
    /// @details C++ users do not have to use this method. This is mainly for python wrapper.
    std::vector<std::vector<float>> GetMeshNodesGlobalAsVectorOfVector();

    /// @brief Set a wildcard value of the owner this tracker is tracking.
    /// @param name Name of the wildcard.
    /// @param wc Wildcard value.
    /// @param offset The offset to this owner (where to start the modification). If first owner, input 0.
    void SetOwnerWildcardValue(const std::string& name, float wc, size_t offset = 0);
    /// @brief Set a wildcard value of the owner this tracker is tracking.
    /// @param name Name of the wildcard.
    /// @param wc Wildcard values as a vector (must have same length as the number of tracked owners).
    void SetOwnerWildcardValues(const std::string& name, const std::vector<float>& wc);

    /// @brief Set a wildcard value of the geometry entity this tracker is tracking.
    /// @param name Name of the wildcard.
    /// @param wc Wildcard value.
    /// @param offset The offset to this entity (where to start the modification). If first entity, input 0.
    void SetGeometryWildcardValue(const std::string& name, float wc, size_t offset = 0);
    /// @brief Set a wildcard value of the geometry entities this tracker is tracking.
    /// @param name Name of the wildcard.
    /// @param wc Wildcard values as a vector (must have same length as the number of tracked geometry entities).
    void SetGeometryWildcardValues(const std::string& name, const std::vector<float>& wc);

    /// @brief Get all contact forces that concern this track object, as a vector.
    /// @details Every force pair will be queried using this function, instead of a reduced total force that this object
    /// experiences.
    /// @param points The contact point XYZ as float3 vector.
    /// @param forces The force in XYZ as float3 vector. The force in global frame.
    /// @param offset The offset to this owner (where to start querying). If first entity, input 0.
    /// @return Number of force pairs.
    size_t GetContactForces(std::vector<float3>& points, std::vector<float3>& forces, size_t offset = 0);
    std::vector<std::vector<std::vector<float>>> GetContactForces(size_t offset = 0);

    /// @brief Get all contact forces and global torques that concern this track object, as a vector.
    /// @details Every force pair will be queried using this function, instead of a reduced total force that this object
    /// experiences. Since we are getting all force pairs, the torque should be considered as `extra torque', since you
    /// should be able to derive the normal and tangential force-induced torques based on all the force pairs. The extra
    /// torques emerge depending on your force model. For example, in the default force model, rolling friction could
    /// contribute to the torque. But if you do not have rolling friction, then you do not have torque here. The torques
    /// are given in the global frame of this object that is being tracked.
    /// @param points The contact point XYZ as float3 vector.
    /// @param forces The force in XYZ as float3 vector. The force in global frame.
    /// @param torques The contact torque. The torque is in global frame.
    /// @param offset The offset to this owner (where to start querying). If first entity, input 0.
    /// @return Number of force pairs.
    size_t GetContactForcesAndGlobalTorque(std::vector<float3>& points,
                                           std::vector<float3>& forces,
                                           std::vector<float3>& torques,
                                           size_t offset = 0);
    std::vector<std::vector<std::vector<float>>> GetContactForcesAndGlobalTorque(size_t offset = 0);

    /// @brief Get all contact forces and local torques that concern this track object, as a vector.
    /// @details Every force pair will be queried using this function, instead of a reduced total force that this object
    /// experiences. Since we are getting all force pairs, the torque should be considered as `extra torque', since you
    /// should be able to derive the normal and tangential force-induced torques based on all the force pairs. The extra
    /// torques emerge depending on your force model. For example, in the default force model, rolling friction could
    /// contribute to the torque. But if you do not have rolling friction, then you do not have torque here. The torques
    /// are given in the local frame of this object that is being tracked.
    /// @param points The contact point XYZ as float3 vector.
    /// @param forces The force in XYZ as float3 vector. The force in global frame.
    /// @param torques The contact torque. The torque is in local frame.
    /// @param offset The offset to this owner (where to start querying). If first entity, input 0.
    /// @return Number of force pairs.
    size_t GetContactForcesAndLocalTorque(std::vector<float3>& points,
                                          std::vector<float3>& forces,
                                          std::vector<float3>& torques,
                                          size_t offset = 0);
    std::vector<std::vector<std::vector<float>>> GetContactForcesAndLocalTorque(size_t offset = 0);
};

// class DEMForceModel : public std::enable_shared_from_this<DEMForceModel> {
class DEMForceModel {
  protected:
    // Those material property names that the user must set. This is non-empty usually when the user uses our on-shelf
    // force models. If they use their custom models, then this will be empty.
    std::set<std::string> m_must_have_mat_props;
    // These material properties are `pair-wise', meaning they should also be defined as the interaction between 2
    // materials. An example is friction coeff. Young's modulus on the other hand, is not pair-wise.
    std::set<std::string> m_pairwise_mat_props;
    // Custom or on-shelf
    FORCE_MODEL type = FORCE_MODEL::CUSTOM;
    // The model
    std::string m_force_model = " ";
    // Quatity names that we want to associate each contact pair with. An array will be allocated for storing this, and
    // it lives and die with contact pairs.
    std::set<std::string> m_contact_wildcards;
    // Quatity names that we want to associate each owner with. An array will be allocated for storing this, and it
    // lives and die with its associated owner.
    std::set<std::string> m_owner_wildcards;
    // Quatity names that we want to associate each owner with. An array will be allocated for storing this, and it
    // lives and die with its associated geometry representation (most typically a sphere).
    std::set<std::string> m_geo_wildcards;

  public:
    friend class DEMSolver;

    DEMForceModel(FORCE_MODEL model_type) { SetForceModelType(model_type); }
    DEMForceModel() {}
    ~DEMForceModel() {}

    /// Set the contact force model type
    void SetForceModelType(FORCE_MODEL model_type);
    /// Define user-custom force model with a string which is your force calculation code
    void DefineCustomModel(const std::string& model);
    /// Read user-custom force model from a file (which by default should reside in kernel/DEMUserScripts), which
    /// contains your force calculation code. Returns 0 if read successfully, otherwise 1.
    int ReadCustomModelFile(const std::filesystem::path& sourcefile);

    /// @brief Specifiy the material properties that this force model will use.
    /// @param props Material property names.
    void SetMustHaveMatProp(const std::set<std::string>& props) { m_must_have_mat_props = props; }
    /// @brief Specifiy the material properties that are pair-wise (instead of being associated with each individual
    /// material).
    /// @param props Material property names.
    void SetMustPairwiseMatProp(const std::set<std::string>& props) { m_pairwise_mat_props = props; }

    /// Set the names for the extra quantities that will be associated with each contact pair. For example,
    /// history-based models should have 3 float arrays to store contact history. Only float is supported. Note the
    /// initial value of all contact wildcard arrays is automatically 0.
    //// TODO: Maybe allow non-0 initialization?
    void SetPerContactWildcards(const std::set<std::string>& wildcards);
    /// Set the names for the extra quantities that will be associated with each owner. For example, you can use this to
    /// associate a cohesion parameter to each particle. Only float is supported.
    void SetPerOwnerWildcards(const std::set<std::string>& wildcards);
    /// Set the names for the extra quantities that will be associated with each geometry. For example, you can use this
    /// to associate certain electric charges to each particle's each component which represents a distribution of the
    /// charges. Only float is supported.
    void SetPerGeometryWildcards(const std::set<std::string>& wildcards);
};

}  // END namespace deme

#endif