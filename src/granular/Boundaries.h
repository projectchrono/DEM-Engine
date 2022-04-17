//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <vector>
#include <limits>

#include <helper_math.cuh>
#include <granular/GranularDefines.h>
#include <core/utils/ManagedAllocator.hpp>

namespace sgps {

/// External object type
/// Note all of them are `shell', not solid objects. If you need a solid cylinder for example, then use one CYLINDER as
/// the side plus 2 CIRCLE as the ends to emulate it. Please be sure to set OUTWARD CYLINDER normal in this case.
enum class DEM_EXTERN_OBJ { CLUMP, PLANE, SPHERE, PLATE, CIRCLE, CYLINDER, CYL_INF, CONE, CONE_INF };
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
    float3 plate_center;
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
        ManagedAllocator<DEMPlateParams_t> plate_params;
        ManagedAllocator<DEMPlaneParams_t> plane_params;
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
    std::vector<DEM_EXTERN_OBJ> types;
    // Component object normal direction, defaulting to inward. If this object is topologically a plane then this param
    // is meaningless, since its normal is determined by its rotation.
    std::vector<DEM_OBJ_NORMAL> normals;
    // Family code (used in prescribing its motions etc.)
    family_t family_code = std::numeric_limits<family_t>::max();  ///< Means it is default to the `fixed' family
    // Initial locations
    std::vector<float3> init_pos;
    // Some float3 quantity that is representitive of an obj's initial orientation (such as plane normal)
    std::vector<float3> init_rot;
    // Some float quantity that is representitive of an obj's size (e.g. for a cylinder, top radius)
    std::vector<float> size_1;
    // Some float quantity that is representitive of an obj's size (e.g. for a cylinder, bottom radius)
    std::vector<float> size_2;
    // Some float quantity that is representitive of an obj's size (e.g. for a cylinder, its length)
    std::vector<float> size_3;
    // The (big) clump types that are a part of this extern obj. Note these types of big clumps have components whose
    // offset IDs are managed by objComponentOffset, not clumpComponentOffset.
    std::vector<unsigned int> clump_types;

    /// Define object contact family number
    void SetFamily(const unsigned int code) { family_code = code; }

    /// Define primitive components that define this external object. Other Add????? methods are its wrappers. Returns
    /// the offset to this component just added in this external object.
    size_t AddComponent(const DEM_EXTERN_OBJ type,
                        const float3 pos,
                        const float3 rot = make_float3(0),
                        const float d1 = 0.f,
                        const float d2 = 0.f,
                        const float d3 = 0.f,
                        const DEM_OBJ_NORMAL normal = DEM_OBJ_NORMAL::INWARD) {
        types.push_back(type);
        normals.push_back(normal);
        init_pos.push_back(pos);
        init_rot.push_back(rot);
        size_1.push_back(d1);
        size_2.push_back(d2);
        size_3.push_back(d3);
        return types.size() - 1;
    }
    /// Add a plane with infinite size
    size_t AddPlane(const float3 pos, const float3 normal) { return AddComponent(DEM_EXTERN_OBJ::PLANE, pos, normal); }
    /// Assuming the normal you specified is the z-direction and that normal vector originates from the pos point you
    /// input. Then specify the dimensions along x- and y-axes to define the plate's area.
    size_t AddPlate(const float3 pos, const float3 normal, const float xdim, const float ydim) {
        return AddComponent(DEM_EXTERN_OBJ::PLATE, pos, normal, xdim, ydim);
    }
};

}  // namespace sgps
