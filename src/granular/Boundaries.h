//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <vector>
#include <limits>

#include <helper_math.cuh>
#include <granular/GranularDefines.h>

namespace sgps {

/// Sphere
template <typename T, typename T3>
struct SphereBCParams_t {
    float radius;
    T normal_sign;
};

/// Cone (pos is tip pos)
template <typename T, typename T3>
struct ConeBCParams_t {
    T3 cone_tip;
    float slope;
    T hmax;
    T hmin;
    T normal_sign;
};

/// Infinite Plane defined by point in plane and normal
template <typename T3>
struct PlaneBCParams_t {
    float3 normal;
    T3 position;
    T3 rotation_center;
    float3 angular_acc;
};

/// Customized finite Plate defined by center of the plate, normal and y dim
template <typename T3>
struct PlateBCParams_t {
    float3 normal;
    T3 plate_center;
    float h_dim_y;
};

/// Infinite Z-aligned cylinder
template <typename T, typename T3>
struct Z_Cylinder_BC_params_t {
    T3 center;
    T radius;
    T normal_sign;
};

/// big enum to handle all possible boundary conditions
template <typename T, typename T3>
struct BC_params_t {
    /// Is this boundary condition active?
    bool active;
    /// Is the boundary condition fixed in space?
    bool fixed;
    /// Whether or not to track reaction forces
    bool track_forces;
    float3 reaction_forces;
    /// velocity in SU if the motion is being prescribed
    float3 vel_SU;
    union {
        SphereBCParams_t<T, T3> sphere_params;
        ConeBCParams_t<T, T3> cone_params;
        PlateBCParams_t<T3> plate_params;
        PlaneBCParams_t<T3> plane_params;          // plane only needs one template arg
        Z_Cylinder_BC_params_t<T, T3> cyl_params;  // plane only needs one arg
    };
};

//
struct DEMExternObj {
    // Component object types
    std::vector<DEM_EXTERN_OBJ> types;
    // Family code (used in prescribing its motions etc.)
    family_t family_code = std::numeric_limits<family_t>::max();  ///< Means it is default to the `fixed' family
    // Initial locations
    std::vector<float3> init_pos;
    // Some float3 quantities that is representitive of an obj's initial orientation
    std::vector<float3> init_rot;
    // The (big) clump types that are a part of this extern obj. Note these types of big clumps have components whose
    // offset IDs are managed by objComponentOffset, not clumpComponentOffset.
    std::vector<unsigned int> clump_types;

    // TODO: add value-appending methods
};

}  // namespace sgps
