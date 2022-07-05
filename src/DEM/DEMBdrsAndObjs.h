//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#ifndef SGPS_DEM_BOUNDARIES
#define SGPS_DEM_BOUNDARIES

#include <vector>
#include <string>
#include <limits>
#include <iostream>
#include <sstream>

#include <nvmath/helper_math.cuh>
#include <DEM/DEMDefines.h>
#include <DEM/DEMStructs.h>
#include <core/utils/ManagedAllocator.hpp>

extern sgps::DEM_VERBOSITY verbosity;

namespace sgps {

/// External object type
/// Note all of them are `shell', not solid objects. If you need a solid cylinder for example, then use one CYLINDER as
/// the side plus 2 CIRCLE as the ends to emulate it. Please be sure to set OUTWARD CYLINDER normal in this case.
enum class DEM_OBJ_COMPONENT { PLANE, SPHERE, PLATE, CIRCLE, CYLINDER, CYL_INF, CONE, CONE_INF, TRIANGLE };
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
    std::vector<std::shared_ptr<DEMMaterial>> materials;
    // Family code (used in prescribing its motions etc.)
    unsigned int family_code = DEM_RESERVED_FAMILY_NUM;  ///< Means it is default to the `fixed' family
    // The coordinate of the CoM of this external object, in the frame where all its components' properties are
    // reported. This is usually all-0 (meaning you should define the object's components in its CoM frame to begin
    // with), but it can be user-specified.
    float3 CoM = make_float3(0);
    // CoM frame's orientation quaternion in the frame which is used to report all its components' properties. Usually
    // unit quaternion.
    float4 CoM_oriQ = make_float4(1.f, 0.f, 0.f, 0.f);
    // Obj's CoM initial position
    float3 init_pos = make_float3(0);
    // Obj's initial orientation quaternion
    float4 init_oriQ = make_float4(1.f, 0.f, 0.f, 0.f);
    // Obj's mass (huge by default)
    float mass = SGPS_DEM_HUGE_FLOAT;
    // Obj's MOI (huge by default)
    float3 MOI = make_float3(SGPS_DEM_HUGE_FLOAT);
    // Its offset when this obj got loaded into the API-level user raw-input array
    unsigned int load_order;

    union DEMAnalEntParams {
        DEMPlateParams_t plate;
        DEMPlaneParams_t plane;
    };
    std::vector<DEMAnalEntParams> entity_params;

    /// Define object contact family number
    void SetFamily(const unsigned int code) { family_code = code; }

    /// Define object mass and MOI
    void SetMassMOI(const float input_mass, const float3 input_MOI) {
        mass = input_mass;
        MOI = input_MOI;
    }

    /// Add a plane with infinite size
    void AddPlane(const float3 pos, const float3 normal, const std::shared_ptr<DEMMaterial>& material) {
        types.push_back(DEM_OBJ_COMPONENT::PLANE);
        materials.push_back(material);
        DEMAnalEntParams params;
        params.plane.position = pos;
        float3 unit_normal = normalize(normal);
        params.plane.normal = unit_normal;
        entity_params.push_back(params);
    }
    /// Add a plate with finite size.
    /// Assuming the normal you specified is the z-direction and that normal vector originates from the pos point you
    /// input. Then specify the dimensions along x- and y-axes to define the plate's area.
    void AddPlate(const float3 pos,
                  const float3 normal,
                  const float xdim,
                  const float ydim,
                  const std::shared_ptr<DEMMaterial>& material) {
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

/// API-(Host-)side struct that holds cached user-input batches of clumps
class DEMClumpBatch {
  private:
    const size_t nClumps;
    void assertLength(size_t len, const std::string name) {
        if (len != nClumps) {
            std::stringstream ss;
            ss << name << " input argument must have length " << nClumps << " (not " << len
               << "), same as the number of clumps you originally added via AddClumps." << std::endl;
            throw std::runtime_error(ss.str());
        }
    }

  public:
    bool families_isSpecified = false;
    std::vector<std::shared_ptr<DEMClumpTemplate>> types;
    std::vector<unsigned int> families;
    std::vector<float3> vel;
    std::vector<float3> angVel;
    std::vector<float3> xyz;
    std::vector<float4> oriQ;
    // Its offset when this obj got loaded into the API-level user raw-input array
    size_t load_order;
    DEMClumpBatch(size_t num) : nClumps(num) {
        types.resize(num);
        families.resize(num, DEM_DEFAULT_CLUMP_FAMILY_NUM);
        vel.resize(num, make_float3(0));
        angVel.resize(num, make_float3(0));
        xyz.resize(num);
        oriQ.resize(num, make_float4(1.f, 0.f, 0.f, 0.f));
    }
    ~DEMClumpBatch() {}
    size_t GetNumClumps() const { return nClumps; }
    void SetTypes(const std::vector<std::shared_ptr<DEMClumpTemplate>>& input) {
        assertLength(input.size(), "SetTypes");
        types = input;
    }
    void SetTypes(const std::shared_ptr<DEMClumpTemplate>& input) {
        SetTypes(std::vector<std::shared_ptr<DEMClumpTemplate>>(nClumps, input));
    }
    void SetPos(const std::vector<float3>& input) {
        assertLength(input.size(), "SetPos");
        xyz = input;
    }
    void SetPos(float3 input) { SetPos(std::vector<float3>(nClumps, input)); }
    void SetVel(const std::vector<float3>& input) {
        assertLength(input.size(), "SetVel");
        vel = input;
    }
    void SetVel(float3 input) { SetVel(std::vector<float3>(nClumps, input)); }
    void SetAngVel(const std::vector<float3>& input) {
        assertLength(input.size(), "SetAngVel");
        angVel = input;
    }
    void SetAngVel(float3 input) { SetAngVel(std::vector<float3>(nClumps, input)); }
    void SetOriQ(const std::vector<float4>& input) {
        assertLength(input.size(), "SetOriQ");
        oriQ = input;
    }
    void SetOriQ(float4 input) { SetOriQ(std::vector<float4>(nClumps, input)); }
    /// Specify the `family' code for each clump. Then you can specify if they should go with some prescribed motion or
    /// some special physics (for example, being fixed). The default behavior (without specification) for every family
    /// is using `normal' physics.
    void SetFamilies(const std::vector<unsigned int>& input) {
        assertLength(input.size(), "SetFamilies");
        families = input;
        families_isSpecified = true;
    }
    void SetFamilies(unsigned int input) { SetFamilies(std::vector<unsigned int>(nClumps, input)); }
};

}  // namespace sgps

#endif
