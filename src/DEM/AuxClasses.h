//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_INSPECTOR_HPP
#define DEME_INSPECTOR_HPP

#include <unordered_map>
#include <core/utils/JitHelper.h>
#include <DEM/Defines.h>

// Forward declare jitify::Program to avoid downstream dependency
namespace jitify {
class Program;
}

namespace deme {

class DEMSolver;

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

    // Its parent DEMSolver system
    DEMSolver* sys;

    // Based on user input...
    void switch_quantity_type(const std::string& quantity);

    void assertInit();

  public:
    friend class DEMSolver;

    DEMInspector(DEMSolver* sim_sys, const std::string& quantity) : sys(sim_sys) {
        switch_quantity_type(quantity);
        // In default constructor, all entities are considered `in-region'
        in_region_code = " ";
        all_domain = true;
    }
    DEMInspector(DEMSolver* sim_sys, const std::string& quantity, const std::string& region) : sys(sim_sys) {
        switch_quantity_type(quantity);
        in_region_code = region;
        all_domain = false;
    }
    //// TODO: Another overload for region definition
    ~DEMInspector() {}

    void SetInspectionCode(const std::string& code) { inspection_code = code; }

    // Initialize with the DEM simulation system (user should not call this)
    void Initialize(const std::unordered_map<std::string, std::string>& Subs);

    /// Get the reduce value of the quantity that you wish to inspect
    float GetValue();

    /// Get the value (as a vector) of the quantity that you wish to inspect
    // std::vector<float> GetVector();
};

// A struct to get or set tracked owner entities, mainly for co-simulation
class DEMTracker {
  private:
    void assertMesh(const std::string& name);
    void assertMeshSize(size_t input_length, const std::string& name);
    // Its parent DEMSolver system
    DEMSolver* sys;

  public:
    DEMTracker(DEMSolver* sim_sys) : sys(sim_sys) {}
    ~DEMTracker() {}

    // The tracked object
    std::shared_ptr<DEMTrackedObj> obj;
    /// Get the position of this tracked object
    float3 Pos(size_t offset = 0);
    /// Get the angular velocity of this tracked object in its own local coordinate system. Applying OriQ to it would
    /// give you the ang vel in global frame.
    float3 AngVelLocal(size_t offset = 0);
    /// Get the velocity of this tracked object in global frame
    float3 Vel(size_t offset = 0);
    /// Get the quaternion that represents the orientation of this tracked object's own coordinate system
    float4 OriQ(size_t offset = 0);

    // float3 Acc(size_t offset = 0);
    // float3 AngAcc(size_t offset = 0);

    /// Get the a portion of the acceleration of this tracked object, that is the result of its contact with other
    /// simulation entities. In most cases, this means excluding the gravitational acceleration. The acceleration is in
    /// global frame.
    float3 ContactAcc(size_t offset = 0);
    /// Get the a portion of the angular acceleration of this tracked object, that is the result of its contact with
    /// other simulation entities. The acceleration is in this object's local frame.
    float3 ContactAngAccLocal(size_t offset = 0);
    /// Set the position of this tracked object
    void SetPos(float3 pos, size_t offset = 0);
    /// Set the angular velocity of this tracked object in its own local coordinate system
    void SetAngVel(float3 angVel, size_t offset = 0);
    /// Set the velocity of this tracked object in global frame
    void SetVel(float3 vel, size_t offset = 0);
    /// Set the quaternion which represents the orientation of this tracked object's coordinate system
    void SetOriQ(float4 oriQ, size_t offset = 0);
    /// Add an extra acc to the tracked body, for the next time step. Note if the user intends to add a persistent
    /// external force, then using family prescription is the better method.
    void AddAcc(float3 force, size_t offset = 0);
    /// Change the size of clump entities
    void ChangeClumpSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors);

    /// Apply the mesh deformation such that the tracked mesh is replaced by the new_mesh. This affects triangle facets'
    /// relative positions wrt the mesh center (CoM) only; mesh's overall position/rotation in simulation is not
    /// affected.
    void UpdateMesh(std::shared_ptr<DEMMeshConnected>& new_mesh);
    /// Change the coordinates of each mesh node by the given amount. This is also for mesh deformation, but unlike
    /// UpdateMesh, it adds to the existing node coordinate XYZs. The length of the argument vector must agree with the
    /// number of nodes in the tracked mesh.
    void UpdateMeshByIncrement(const std::vector<float3>& deformation);
};

class DEMForceModel {
  protected:
    // Those material property names that the user must set. This is non-empty usually when the user uses our on-shelf
    // force models. If they use their custom models, then this will be empty.
    std::set<std::string> m_must_have_mat_props;
    // Custom or on-shelf
    FORCE_MODEL type;
    // The model
    std::string m_force_model = " ";
    // Quatity names that we want to associate each contact pair with. An array will be allocated for storing this, and
    // it lives and die with contact pairs.
    std::set<std::string> m_contact_wildcards;
    // Quatity names that we want to associate each owner with. An array will be allocated for storing this, and it
    // lives and die with its associated owner.
    std::set<std::string> m_owner_wildcards;

  public:
    friend class DEMSolver;

    DEMForceModel(FORCE_MODEL model_type = FORCE_MODEL::CUSTOM) { SetForceModelType(model_type); }
    ~DEMForceModel() {}

    /// Set the contact force model type
    void SetForceModelType(FORCE_MODEL model_type);
    /// Define user-custom force model with a string which is your force calculation code
    void DefineCustomModel(const std::string& model);
    /// Read user-custom force model from a file (which by default should reside in kernel/DEMUserScripts), which
    /// contains your force calculation code. Returns 0 if read successfully, otherwise 1.
    int ReadCustomModelFile(const std::filesystem::path& sourcefile);
    /// Set the names for the extra quantities that will be associated with each contact pair. For example,
    /// history-based models should have 3 float arrays to store contact history. Only float is supported. Note the
    /// initial value of all contact wildcard arrays is automatically 0.
    //// TODO: Maybe allow non-0 initialization?
    void SetPerContactWildcards(const std::set<std::string>& wildcards);
    /// Set the names for the extra quantities that will be associated with each owner. For example, you can use this to
    /// associate electric charge to each particle. Only float is supported.
    void SetPerOwnerWildcards(const std::set<std::string>& wildcards);
};

}  // END namespace deme

#endif