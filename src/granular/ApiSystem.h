//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <vector>
#include <set>

#include <core/ApiVersion.h>
#include <granular/PhysicsSystem.h>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/ThreadManager.h>
#include <core/utils/GpuManager.h>
#include <core/utils/Macros.h>
#include <granular/GranularDefines.h>

namespace sgps {

// class SGPS_impl;
// class kinematicThread;
// class dynamicThread;
// class ThreadManager;

class SGPS {
  public:
    SGPS(float rad);
    virtual ~SGPS();

    // load possible clump types into the API-level cache
    // return the index of the clump type just loaded
    clumpBodyInertiaOffset_t LoadClumpType(float mass,
                                           float moiX,
                                           float moiY,
                                           float moiZ,
                                           const std::vector<float>& sp_radii,
                                           const std::vector<float>& sp_locations_x,
                                           const std::vector<float>& sp_locations_y,
                                           const std::vector<float>& sp_locations_z,
                                           const std::vector<materialsOffset_t>& sp_material_ids);
    // TODO: need to overload with (vec_float radii, vec_float3 location, vec_sp_material_ids)
    // TODO: need to overload with (vec_distinctSphereRadiiOffset_t spheres_component_type, vec_float3 location). If
    // this method is called then corresponding sphere_types must have been defined via LoadSphereType.

    // a simplified version of LoadClumpType: it's just a one-sphere clump
    clumpBodyInertiaOffset_t LoadClumpSimpleSphere(float mass, float radius, materialsOffset_t material_id);

    // load possible materials into the API-level cache
    // return the index of the material type just loaded
    materialsOffset_t LoadMaterialType(float density, float E);

    // return the voxel ID of a clump by its numbering
    voxelID_t GetClumpVoxelID(unsigned int i) const;

    int Initialize();

    int LaunchThreads();

    /*
      protected:
        SGPS() : m_sys(nullptr) {}
        SGPS_impl* m_sys;
    */

  private:
    // This is the cached material information.
    // It will be massaged into kernels upon Initialize.
    struct Material {
        float density;
        float E;
    };
    std::vector<Material> m_sp_materials;

    // This is the cached clump structure information.
    // It will be massaged into kernels upon Initialize.
    std::vector<float> m_clumps_mass;
    std::vector<float> m_clumps_moiX;
    std::vector<float> m_clumps_moiY;
    std::vector<float> m_clumps_moiZ;
    std::vector<std::vector<float>> m_clumps_sp_radii;
    std::vector<std::vector<float>> m_clumps_sp_location_x;
    std::vector<std::vector<float>> m_clumps_sp_location_y;
    std::vector<std::vector<float>> m_clumps_sp_location_z;
    std::vector<std::vector<materialsOffset_t>> m_clumps_sp_material_ids;

    // unique clump masses derived from m_clumps_mass
    std::set<float> m_clumps_mass_types;
    std::vector<clumpBodyInertiaOffset_t> m_clumps_mass_type_offset;
    // unique sphere radii types derived from m_clumps_sp_radii
    std::set<float> m_clumps_sp_radii_types;
    std::vector<std::vector<distinctSphereRadiiOffset_t>> m_clumps_sp_radii_type_offset;

    float sphereUU;

    unsigned int nDistinctSphereRadii_computed;
    unsigned int nDistinctSphereRelativePositions_computed;
    unsigned int nDistinctClumpBodyTopologies_computed;

    int updateFreq = 1;
    int timeDynamicSide = 1;
    int timeKinematicSide = 1;
    int nDynamicCycles = 5;

    GpuManager* dTkT_GpuManager;
    ThreadManager* dTkT_InteractionManager;
    kinematicThread* kT;
    dynamicThread* dT;

    int generateJITResources();
};

}  // namespace sgps
