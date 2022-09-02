//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef SGPS_DEM_INSPECTOR_HPP
#define SGPS_DEM_INSPECTOR_HPP

#include <unordered_map>
#include <core/utils/JitHelper.h>
#include <DEM/DEMDefines.h>

// Forward declare jitify::Program to avoid downstream dependency
namespace jitify {
class Program;
}

namespace sgps {

class DEMSolver;

/// A class that the user can construct to inspect a certain property (such as void ratio, maximum Z coordinate...) of
/// their simulation entites, in a given region.
class DEMInspector {
  private:
    std::shared_ptr<jitify::Program> inspection_kernel;

    std::string inspection_code;
    std::string in_region_code;
    DEM_CUB_REDUCE_FLAVOR reduce_flavor = DEM_CUB_REDUCE_FLAVOR::NONE;
    DEM_INSPECT_ENTITY_TYPE thing_to_insp;
    std::string kernel_name;
    bool all_domain = true;

    bool initialized = false;

    // Its parent DEMSolver system
    DEMSolver* sys;

    // Based on user input...
    void switch_quantity_type(const std::string& quantity);

    void assertInit() {
        if (!initialized) {
            std::stringstream ss;
            ss << "Inspector should only be used for querying after system initialization!" << std::endl;
            throw std::runtime_error(ss.str());
        }
    }

  public:
    DEMInspector(DEMSolver* sim_sys, const std::string& quantity) : sys(sim_sys) {
        switch_quantity_type(quantity);
        // In default constructor, all entities are considered `in-region'
        in_region_code = " ";
        all_domain = true;
    }
    DEMInspector(DEMSolver* sim_sys, const std::string& quantity, const std::string& region) : sys(sim_sys) {}
    ~DEMInspector() {}

    void SetInspectionCode(const std::string& code) { inspection_code = code; }

    // Initialize with the DEM simulation system (user should not call this)
    void initializeInspector(const std::unordered_map<std::string, std::string>& Subs);

    /// Get the reduce value of the quantity that you wish to inspect
    float GetValue();

    /// Get the value (as a vector) of the quantity that you wish to inspect
    // std::vector<float> GetVector();
};

// A struct to get or set tracked owner entities, mainly for co-simulation
class DEMTracker {
  private:
    // Its parent DEMSolver system
    DEMSolver* sys;

  public:
    DEMTracker(DEMSolver* sim_sys) : sys(sim_sys) {}
    ~DEMTracker() {}

    // The tracked object
    std::shared_ptr<DEMTrackedObj> obj;
    // Methods to get info from this owner
    float3 Pos(size_t offset = 0);
    float3 AngVel(size_t offset = 0);
    float3 Vel(size_t offset = 0);
    float4 OriQ(size_t offset = 0);
    // float3 Acc(size_t offset = 0);
    // float3 AngAcc(size_t offset = 0);
    float3 ContactAcc(size_t offset = 0);
    float3 ContactAngAcc(size_t offset = 0);
    // Methods to set motions to this owner
    void SetPos(float3 pos, size_t offset = 0);
    void SetAngVel(float3 angVel, size_t offset = 0);
    void SetVel(float3 vel, size_t offset = 0);
    void SetOriQ(float4 oriQ, size_t offset = 0);
    /// Add an extra acc to the tracked body, for the next time step. Note if the user intends to add a persistent
    /// external force, then using family prescription is the better method.
    void AddAcc(float3 force, size_t offset = 0);
    /// Change the size of clump entities
    void ChangeClumpSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors);
};

}  // END namespace sgps

#endif