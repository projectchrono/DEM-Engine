//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <vector>
#include <set>
#include <cfloat>

#include <core/ApiVersion.h>
#include <granular/PhysicsSystem.h>
#include <core/utils/ManagedAllocator.hpp>
#include <core/utils/ThreadManager.h>
#include <core/utils/GpuManager.h>
#include <core/utils/Macros.h>
#include <helper_math.cuh>
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

    // Instruct the dimension of the ``world'', as well as the origin point of this ``world''. On initialization, this
    // info will be used to figure out how to assign the num of voxels in each direction. If your ``useful'' domain is
    // not box-shaped, then define a box that contains your domian.
    void InstructBoxDomainDimension(float x, float y, float z, float3 O = make_float3(0));

    // Explicitly instruct the number of voxels (as 2^{x,y,z}) along each direction, as well as the smallest unit length
    // l. This is usually for test purposes, and will overwrite other size-related definitions of the big domain.
    void InstructBoxDomainNumVoxel(unsigned char x,
                                   unsigned char y,
                                   unsigned char z,
                                   float len_unit = 1e-10f,
                                   float3 O = make_float3(0));

    // Load possible clump types into the API-level cache.
    // Return the index of the clump type just loaded.
    clumpBodyInertiaOffset_default_t LoadClumpType(float mass,
                                                   float3 moi,
                                                   const std::vector<float>& sp_radii,
                                                   const std::vector<float3>& sp_locations_xyz,
                                                   const std::vector<materialsOffset_default_t>& sp_material_ids);
    // TODO: need to overload with (vec_distinctSphereRadiiOffset_default_t spheres_component_type, vec_float3
    // location). If this method is called then corresponding sphere_types must have been defined via LoadSphereType.

    // a simplified version of LoadClumpType: it's just a one-sphere clump
    clumpBodyInertiaOffset_default_t LoadClumpSimpleSphere(float mass,
                                                           float radius,
                                                           materialsOffset_default_t material_id);

    // Load possible materials into the API-level cache
    // Return the index of the material type just loaded
    materialsOffset_default_t LoadMaterialType(float density, float E);

    // Load input clumps (topology types and initial locations) on a per-pair basis
    void SetClumps(const std::vector<clumpBodyInertiaOffset_default_t>& types, const std::vector<float3>& xyz);

    // Return the voxel ID of a clump by its numbering
    voxelID_default_t GetClumpVoxelID(unsigned int i) const;

    // Write current simulation status to a file
    // Write overlapping spheres, not clumps. It makes the file larger, but less trouble to visualize. Use for test
    // purposes only.
    void WriteFileAsSpheres(const std::string& outfilename) const;

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
    std::vector<float3> m_clumps_moi;
    std::vector<std::vector<float>> m_clumps_sp_radii;
    std::vector<std::vector<float3>> m_clumps_sp_location_xyz;
    std::vector<std::vector<materialsOffset_default_t>> m_clumps_sp_material_ids;

    // unique clump masses derived from m_clumps_mass
    std::set<float> m_clumps_mass_types;
    std::vector<clumpBodyInertiaOffset_default_t> m_clumps_mass_type_offset;
    // unique sphere radii types derived from m_clumps_sp_radii
    std::set<float> m_clumps_sp_radii_types;
    std::vector<std::vector<distinctSphereRadiiOffset_default_t>> m_clumps_sp_radii_type_offset;
    // unique sphere (local) location types derived from m_clumps_sp_location_xyz
    // std::set<float3, float3_less_than> m_clumps_sp_location_types;
    std::set<float3> m_clumps_sp_location_types;
    std::vector<std::vector<distinctSphereRelativePositions_default_t>> m_clumps_sp_location_type_offset;

    // ``World'' size along X dir
    float m_boxX = 0.f;
    // ``World'' size along Y dir
    float m_boxY = 0.f;
    // ``World'' size along Z dir
    float m_boxZ = 0.f;
    // Origin of the ``world''
    float3 m_boxO = make_float3(0);
    // Number of voxels in the X direction, expressed as a power of 2
    unsigned char nvXp2;
    // Number of voxels in the Y direction, expressed as a power of 2
    unsigned char nvYp2;
    // Number of voxels in the Z direction, expressed as a power of 2
    unsigned char nvZp2;

    float sphereUU;

    // The length unit. Any XYZ we report to the user, is under the hood a multiple of this l.
    float l = FLT_MAX;
    // Whether the number of voxels and length unit l is explicitly given by the user.
    bool explicit_nv_override = false;

    unsigned int nDistinctSphereRadii_computed;
    unsigned int nDistinctSphereRelativePositions_computed;
    unsigned int nDistinctClumpBodyTopologies_computed;
    unsigned int nMatTuples_computed;

    // cached state vectors such as the types and locations of the initial clumps to fill the sim domain with
    std::vector<clumpBodyInertiaOffset_default_t> m_input_clump_types;
    std::vector<float3> m_input_clump_xyz;

    int updateFreq = 1;
    int timeDynamicSide = 1;
    int timeKinematicSide = 1;
    int nDynamicCycles = 5;

    GpuManager* dTkT_GpuManager;
    ThreadManager* dTkT_InteractionManager;
    kinematicThread* kT;
    dynamicThread* dT;

    int generateJITResources();
    int figureOutNV();
};

}  // namespace sgps
