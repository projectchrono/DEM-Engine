//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef SMUG_DEM_BOUNDARIES
#define SMUG_DEM_BOUNDARIES

#include <vector>
#include <string>
#include <limits>
#include <iostream>
#include <sstream>
#include <array>
#include <cmath>

#include <nvmath/helper_math.cuh>
#include <DEM/DEMDefines.h>
#include <DEM/DEMStructs.h>
#include <core/utils/ManagedAllocator.hpp>
#include <DEM/HostSideHelpers.hpp>

namespace smug {

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
    float4 CoM_oriQ = host_make_float4(0, 0, 0, 1);
    // Obj's CoM initial position
    float3 init_pos = make_float3(0);
    // Obj's initial orientation quaternion
    float4 init_oriQ = host_make_float4(0, 0, 0, 1);
    // Obj's mass (huge by default)
    float mass = SMUG_DEM_HUGE_FLOAT;
    // Obj's MOI (huge by default)
    float3 MOI = make_float3(SMUG_DEM_HUGE_FLOAT);
    // Its offset when this obj got loaded into the API-level user raw-input array
    unsigned int load_order;

    union DEMAnalEntParams {
        DEMPlateParams_t plate;
        DEMPlaneParams_t plane;
    };
    std::vector<DEMAnalEntParams> entity_params;

    /// Define object contact family number
    void SetFamily(const unsigned int code) {
        if (code > std::numeric_limits<family_t>::max()) {
            std::stringstream ss;
            ss << "An external object is instructed to have family number " << code
               << ", which is larger than the max allowance " << std::numeric_limits<family_t>::max() << std::endl;
            throw std::runtime_error(ss.str());
        }
        family_code = code;
    }

    /// Set mass
    void SetMass(float mass) { this->mass = mass; }
    /// Set MOI (in principal frame)
    void SetMOI(float3 MOI) { this->MOI = MOI; }

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

// DEM mesh object
class DEMMeshConnected {
  private:
    void assertLength(size_t len, const std::string name) {
        if (nTri == 0) {
            std::cerr << "The settings at the " << name << " call were applied to 0 mesh facet.\nPlease consider using "
                      << name
                      << " only after loading the mesh file, because mesh utilities are supposed to provide per-facet "
                         "control of your mesh, so we need to know the mesh first."
                      << std::endl;
        }
        if (len != nTri) {
            std::stringstream ss;
            ss << name << " input argument must have length " << nTri << " (not " << len
               << "), same as the number of triangle facets in the mesh." << std::endl;
            throw std::runtime_error(ss.str());
        }
    }

  public:
    // Number of triangle facets in the mesh
    size_t nTri = 0;

    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<float3> UV;
    std::vector<float3> colors;

    std::vector<int3> face_v_indices;
    std::vector<int3> face_n_indices;
    std::vector<int3> face_uv_indices;
    std::vector<int3> face_col_indices;

    // Material types for each mesh facet
    std::vector<std::shared_ptr<DEMMaterial>> materials;
    bool isMaterialSet = false;
    // Family code (used in prescribing its motions etc.)
    unsigned int family_code = DEM_RESERVED_FAMILY_NUM;  ///< Means it is default to the `fixed' family
    // The coordinate of the CoM of this meshed object, in the frame where all the mesh's node coordinates are
    // reported. This is usually all-0 (meaning you should define the object's components in its CoM frame to begin
    // with), but it can be user-specified.
    float3 CoM = make_float3(0);
    // CoM frame's orientation quaternion in the frame which is used to report all the mesh's node coordinates.
    // It is usually unit quaternion.
    float4 CoM_oriQ = host_make_float4(0, 0, 0, 1);
    // Obj's CoM initial position
    float3 init_pos = make_float3(0);
    // Obj's initial orientation quaternion
    float4 init_oriQ = host_make_float4(0, 0, 0, 1);
    // Obj's mass (huge by default)
    float mass = 1.f;
    // Obj's MOI (huge by default)
    float3 MOI = make_float3(1.f);
    // Its offset when this obj got loaded into the API-level user raw-input array
    unsigned int load_order;

    std::string filename;  ///< file string if loading an obj file

    // If true, when the mesh is initialized into the system, it will re-order the nodes of each triangle so that the
    // normals derived from right-hand-rule are the same as the normals in the mesh file
    bool use_mesh_normals = false;

    DEMMeshConnected() {}
    DEMMeshConnected(std::string input_file) { LoadWavefrontMesh(input_file); }
    DEMMeshConnected(std::string input_file, const std::shared_ptr<DEMMaterial>& mat) {
        LoadWavefrontMesh(input_file);
        SetMaterial(mat);
    }
    ~DEMMeshConnected() {}

    /// Load a triangle mesh saved as a Wavefront .obj file
    bool LoadWavefrontMesh(std::string input_file, bool load_normals = true, bool load_uv = false);

    /// Write the specified meshes in a Wavefront .obj file
    static void WriteWavefront(const std::string& filename, std::vector<DEMMeshConnected>& meshes);

    /// Utility function for merging multiple meshes.
    static DEMMeshConnected Merge(std::vector<DEMMeshConnected>& meshes);

    /// Get the number of triangles already added to this mesh
    size_t GetNumTriangles() const { return nTri; }

    /// Instruct that when the mesh is initialized into the system, it will re-order the nodes of each triangle so that
    /// the normals derived from right-hand-rule are the same as the normals in the mesh file
    void UseNormals(bool use = true) { use_mesh_normals = use; }

    /// Access the n-th triangle in mesh
    DEMTriangle GetTriangle(size_t index) const {
        return DEMTriangle(vertices[face_v_indices[index].x], vertices[face_v_indices[index].y],
                           vertices[face_v_indices[index].z]);
    }

    /// Clear all data
    void Clear() {
        this->vertices.clear();
        this->normals.clear();
        this->UV.clear();
        this->colors.clear();
        this->face_v_indices.clear();
        this->face_n_indices.clear();
        this->face_uv_indices.clear();
        this->face_col_indices.clear();
    }

    /// Set mass
    void SetMass(float mass) { this->mass = mass; }
    /// Set MOI (in principal frame)
    void SetMOI(float3 MOI) { this->MOI = MOI; }

    /// Set material types for the mesh. Technically, you can set that for each individual mesh facet.
    void SetMaterial(const std::vector<std::shared_ptr<DEMMaterial>>& input) {
        assertLength(input.size(), "SetMaterial");
        materials = input;
        isMaterialSet = true;
    }
    void SetMaterial(const std::shared_ptr<DEMMaterial>& input) {
        SetMaterial(std::vector<std::shared_ptr<DEMMaterial>>(nTri, input));
    }

    /// Compute barycenter, mass and MOI in CoM frame
    void ComputeMassProperties(double& mass, float3& center, float3& inertia);

    /// Transforme the meshed object so it gets to its initial position, before the simulation starts
    void Rotate(const float4 rotQ);
    void Translate(const float3 displ);

    /// Create a map of neighboring triangles, vector of:
    /// [Ti TieA TieB TieC]
    /// (the free sides have triangle id = -1).
    /// Return false if some edge has more than 2 neighboring triangles
    bool ComputeNeighbouringTriangleMap(std::vector<std::array<int, 4>>& tri_map) const;
};

}  // namespace smug

#endif
