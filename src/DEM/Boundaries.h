//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <vector>
#include <limits>

#include <helper_math.cuh>
#include <DEM/DEMDefines.h>
#include <core/utils/ManagedAllocator.hpp>

extern sgps::DEM_VERBOSITY verbosity;

namespace sgps {

/// External object type
/// Note all of them are `shell', not solid objects. If you need a solid cylinder for example, then use one CYLINDER as
/// the side plus 2 CIRCLE as the ends to emulate it. Please be sure to set OUTWARD CYLINDER normal in this case.
enum class DEM_OBJ_COMPONENT { CLUMP, PLANE, SPHERE, PLATE, CIRCLE, CYLINDER, CYL_INF, CONE, CONE_INF, TRIANGLE };
/// Normal type: inward or outward?
enum class DEM_OBJ_NORMAL { INWARD, OUTWARD };

/// Sphere
template <typename T, typename T3>
struct DEMSphereParams_t {
    float radius;
    T normal_sign;
};

/// Cone (pos is tip pos)
template <typename T, typename T3>
struct DEMConeParams_t {
    T3 cone_tip;
    float slope;
    T hmax;
    T hmin;
    T normal_sign;
};

/// Infinite Plane defined by point in plane and normal
struct DEMPlaneParams_t {
    float3 normal;
    float3 position;
};

/// Customized finite Plate defined by center of the plate, normal and y dim
struct DEMPlateParams_t {
    float3 normal;
    float3 center;
    float h_dim_x;
    float h_dim_y;
};

/// Infinite Z-aligned cylinder
template <typename T, typename T3>
struct DEMCylinderParams_t {
    T3 center;
    T radius;
    T normal_sign;
};

/// GPU-side struct that holds external object component info. Only component, not their parents, so this is the
/// equivalent of clump templates. External objects themselves (and their position, velocity etc.) are not stored with
/// this struct; instead, they are considered a general owner (or clump)
class DEMObjComponent {
  public:
    // float3* pSomething;

    //
    // std::vector<scratch_t, ManagedAllocator<scratch_t>> something;

    union {
        ManagedAllocator<DEMPlateParams_t> plate;
        ManagedAllocator<DEMPlaneParams_t> plane;
    };

    DEMObjComponent() {
        // cudaMallocManaged(&pSomething, sizeof(float3));
    }
    ~DEMObjComponent() {
        // cudaFree(pSomething);
    }
};

/// API-(Host-)side struct that holds cached user-input external objects
struct DEMExternObj {
    // Component object types
    std::vector<DEM_OBJ_COMPONENT> types;
    // Component object materials
    std::vector<unsigned int> materials;
    // Family code (used in prescribing its motions etc.)
    unsigned int family_code = DEM_RESERVED_FAMILY_NUM;  ///< Means it is default to the `fixed' family
    // Obj's CoM initial position
    float3 init_pos = make_float3(0);
    // Obj's initial orientation quaternion
    float4 init_oriQ = make_float4(1.f, 0.f, 0.f, 0.f);
    // Obj's mass (huge by default)
    float mass = SGPS_DEM_HUGE_FLOAT;
    // Obj's MOI (huge by default)
    float3 MOI = make_float3(SGPS_DEM_HUGE_FLOAT);
    // The (big) clump types that are a part of this extern obj. Note an external object should at most have one clump
    // as its component.
    unsigned int clump_type;
    // Flag for having loaded more than one clump
    bool clump_defined = false;

    union DEMAnalEntParams {
        DEMPlateParams_t plate;
        DEMPlaneParams_t plane;
    };
    std::vector<DEMAnalEntParams> entity_params;

    /// Add a clump (using loaded clump template number) as a component to this external object
    void AddClumpType(unsigned int type) {
        if (clump_defined) {
            SGPS_DEM_WARNING(
                "Each external object can contain no more than one clump. When you load the second one, the first one "
                "is overwritten.");
        } else {
            types.push_back(DEM_OBJ_COMPONENT::CLUMP);
        }
        clump_type = type;
        clump_defined = true;
    }

    /// Define object contact family number
    void SetFamily(const unsigned int code) { family_code = code; }

    /// Define object mass and MOI
    void SetMassMOI(const float input_mass, const float3 input_MOI) {
        mass = input_mass;
        MOI = input_MOI;
    }

    /// Add a plane with infinite size
    void AddPlane(const float3 pos, const float3 normal, const unsigned int material) {
        types.push_back(DEM_OBJ_COMPONENT::PLANE);
        materials.push_back(material);
        DEMAnalEntParams params;
        params.plane.position = pos;
        float3 unit_normal = normalize(normal);
        params.plane.normal = unit_normal;
        entity_params.push_back(params);
    }
    /// Assuming the normal you specified is the z-direction and that normal vector originates from the pos point you
    /// input. Then specify the dimensions along x- and y-axes to define the plate's area.
    void AddPlate(const float3 pos,
                  const float3 normal,
                  const float xdim,
                  const float ydim,
                  const unsigned int material) {
        types.push_back(DEM_OBJ_COMPONENT::PLATE);
        materials.push_back(material);
        DEMAnalEntParams params;
        params.plate.center = pos;
        float3 unit_normal = normalize(normal);
        params.plate.normal = unit_normal;
        params.plate.h_dim_x = xdim / 2.0;
        params.plate.h_dim_y = ydim / 2.0;
        entity_params.push_back(params);
    }
};

}  // namespace sgps
