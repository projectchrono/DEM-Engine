//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_BOUNDARIES
#define DEME_BOUNDARIES

#include <vector>
#include <string>
#include <limits>
#include <iostream>
#include <sstream>
#include <array>
#include <cmath>

#include <nvmath/helper_math.cuh>
#include <DEM/Defines.h>
#include <DEM/Structs.h>
#include <core/utils/ManagedAllocator.hpp>
#include <DEM/HostSideHelpers.hpp>

namespace deme {

/// External object type
/// Note all of them are `shell', not solid objects. If you need a solid cylinder for example, then use one CYLINDER as
/// the side plus 2 CIRCLE as the ends to emulate it. Please be sure to set OUTWARD CYLINDER normal in this case.
enum class OBJ_COMPONENT { PLANE, SPHERE, PLATE, CIRCLE, CYL, CYL_INF, CONE, CONE_INF, TRIANGLE };

/// Sphere
struct DEMSphereParams_t {
    float radius;
    objNormal_t normal;
};

/// Cone (pos is tip pos)
struct DEMConeParams_t {
    float3 cone_tip;
    float slope;
    float hmax;
    float hmin;
    objNormal_t normal;
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
struct DEMCylinderParams_t {
    float3 center;
    float3 dir;
    float radius;
    objNormal_t normal;
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
    std::vector<OBJ_COMPONENT> types;
    // Component object materials
    std::vector<std::shared_ptr<DEMMaterial>> materials;
    // Family code (used in prescribing its motions etc.)
    unsigned int family_code = RESERVED_FAMILY_NUM;  ///< Means it is default to the `fixed' family
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
    float mass = DEME_HUGE_FLOAT;
    // Obj's MOI (huge by default)
    float3 MOI = make_float3(DEME_HUGE_FLOAT);
    // Its offset when this obj got loaded into the API-level user raw-input array
    unsigned int load_order;

    union DEMAnalEntParams {
        DEMPlateParams_t plate;
        DEMPlaneParams_t plane;
        DEMCylinderParams_t cyl;
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
        types.push_back(OBJ_COMPONENT::PLANE);
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
        types.push_back(OBJ_COMPONENT::PLATE);
        materials.push_back(material);
        DEMAnalEntParams params;
        params.plate.center = pos;
        float3 unit_normal = normalize(normal);
        params.plate.normal = unit_normal;
        params.plate.h_dim_x = xdim / 2.0;
        params.plate.h_dim_y = ydim / 2.0;
        entity_params.push_back(params);
    }
    /// Add a z-axis-aligned cylinder of infinite length
    void AddZCylinder(const float3 pos,
                      const float rad,
                      const std::shared_ptr<DEMMaterial>& material,
                      const objNormal_t normal = ENTITY_NORMAL_INWARD) {
        types.push_back(OBJ_COMPONENT::CYL_INF);
        materials.push_back(material);
        DEMAnalEntParams params;
        params.cyl.center = pos;
        params.cyl.radius = rad;
        params.cyl.dir = host_make_float3(0, 0, 1);
        params.cyl.normal = normal;
        entity_params.push_back(params);
    }
    /// Add a cylinder of infinite length, which is along a user-specific axis
    void AddCylinder(const float3 pos,
                     const float3 axis,
                     const float rad,
                     const std::shared_ptr<DEMMaterial>& material,
                     const objNormal_t normal = ENTITY_NORMAL_INWARD) {
        types.push_back(OBJ_COMPONENT::CYL_INF);
        materials.push_back(material);
        DEMAnalEntParams params;
        params.cyl.center = pos;
        params.cyl.radius = rad;
        params.cyl.dir = normalize(axis);
        params.cyl.normal = normal;
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

    // Owner number in DEM simulation
    bodyID_t owner = NULL_BODYID;

    std::vector<float3> m_vertices;
    std::vector<float3> m_normals;
    std::vector<float3> m_UV;
    std::vector<float3> m_colors;

    std::vector<int3> m_face_v_indices;
    std::vector<int3> m_face_n_indices;
    std::vector<int3> m_face_uv_indices;
    std::vector<int3> m_face_col_indices;

    bodyID_t getOwner() { return owner; }

    std::vector<float3>& getCoordsVertices() { return m_vertices; }
    std::vector<float3>& getCoordsNormals() { return m_normals; }
    std::vector<float3>& getCoordsUV() { return m_UV; }
    std::vector<float3>& getCoordsColors() { return m_colors; }

    std::vector<int3>& getIndicesVertexes() { return m_face_v_indices; }
    std::vector<int3>& getIndicesNormals() { return m_face_n_indices; }
    std::vector<int3>& getIndicesUV() { return m_face_uv_indices; }
    std::vector<int3>& getIndicesColors() { return m_face_col_indices; }

    // Material types for each mesh facet
    std::vector<std::shared_ptr<DEMMaterial>> materials;
    bool isMaterialSet = false;
    // Family code (used in prescribing its motions etc.)
    unsigned int family_code = RESERVED_FAMILY_NUM;  ///< Means it is default to the `fixed' family
    // The coordinate of the CoM of this meshed object, in the frame where all the mesh's node coordinates are
    // reported. This is usually all-0 (meaning you should define the object's components in its CoM frame to begin
    // with), but it can be user-specified.
    float3 CoM = make_float3(0);
    // CoM frame's orientation quaternion in the frame which is used to report all the mesh's node coordinates.
    // It is usually unit quaternion.
    float4 CoM_oriQ = host_make_float4(0, 0, 0, 1);
    // Mesh's CoM initial position
    float3 init_pos = make_float3(0);
    // Mesh's initial orientation quaternion
    float4 init_oriQ = host_make_float4(0, 0, 0, 1);
    // Mesh's mass
    float mass = 1.f;
    // Mesh's MOI
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
        return DEMTriangle(m_vertices[m_face_v_indices[index].x], m_vertices[m_face_v_indices[index].y],
                           m_vertices[m_face_v_indices[index].z]);
    }

    /// Clear all data
    void Clear() {
        this->m_vertices.clear();
        this->m_normals.clear();
        this->m_UV.clear();
        this->m_colors.clear();
        this->m_face_v_indices.clear();
        this->m_face_n_indices.clear();
        this->m_face_uv_indices.clear();
        this->m_face_col_indices.clear();
        this->owner = NULL_BODYID;
    }

    /// Set mass
    void SetMass(float mass) { this->mass = mass; }
    /// Set MOI (in principal frame)
    void SetMOI(float3 MOI) { this->MOI = MOI; }
    /// Set mesh family number
    void SetFamily(unsigned int num) { this->family_code = num; }

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

    /// Transform the meshed object so it gets to its initial position, before the simulation starts
    void SetInitQuat(const float4 rotQ) { init_oriQ = rotQ; }
    void SetInitPos(const float3 displ) { init_pos = displ; }

    /// If this mesh's component sphere relPos is not reported by the user in its CoM frame, then the user needs to
    /// call this method immediately to report this mesh's Volume Centroid and Principal Axes, and relPos will be
    /// adjusted by this call.
    void InformCentroidPrincipal(float3 center, float4 prin_Q) {
        // Getting to Centroid and Principal is a translation then a rotation (local), so the undo order to undo
        // rotation then translation
        float4 g_to_loc_prin_Q = prin_Q;
        g_to_loc_prin_Q.x = -g_to_loc_prin_Q.x;
        g_to_loc_prin_Q.y = -g_to_loc_prin_Q.y;
        g_to_loc_prin_Q.z = -g_to_loc_prin_Q.z;
        for (auto& node : m_vertices) {
            hostApplyFrameTransform(node, -center, g_to_loc_prin_Q);
        }
    }
    /// The opposite of InformCentroidPrincipal, and it is another way to align this mesh's coordinate system with its
    /// centroid and principal system: rotate then move this clump, so that at the end of this operation, the original
    /// `origin' point should hit the CoM of this mesh.
    void Move(float3 vec, float4 rot_Q) {
        for (auto& node : m_vertices) {
            hostApplyFrameTransform(node, vec, rot_Q);
        }
    }
    /// Mirror all points in the mesh about a plane. If this changes the mass properties of this mesh, it is the user's
    /// responsibility to reset them.
    void Mirror(float3 plane_point, float3 plane_normal) {
        plane_normal = normalize(plane_normal);
        for (auto& node : m_vertices) {
            float3 node2plane = plane_point - node;
            float proj = dot(node2plane, plane_normal);
            // If proj is negative, we need to go along the neg dir of plane normal anyway; if proj is positive, we need
            // to go along the positive dir of the plane anyway
            node += 2 * proj * plane_normal;
        }
    }
    /// Scale all geometry component of this mesh
    void Scale(float s) {
        for (auto& node : m_vertices) {
            node *= s;
        }
        mass *= (double)s * (double)s * (double)s;
        MOI *= (double)s * (double)s * (double)s * (double)s * (double)s;
    }
    void Scale(float3 s) {
        for (auto& node : m_vertices) {
            node = node * s;
        }
        // Really just an estimate. The user should reset mass properties manually afterwards.
        double prod = (double)s.x * (double)s.y * (double)s.z;
        mass *= prod;
        MOI.x *= prod * s.x * s.x;
        MOI.y *= prod * s.y * s.y;
        MOI.z *= prod * s.z * s.z;
    }

    /// Create a map of neighboring triangles, vector of:
    /// [Ti TieA TieB TieC]
    /// (the free sides have triangle id = -1).
    /// Return false if some edge has more than 2 neighboring triangles
    bool ComputeNeighbouringTriangleMap(std::vector<std::array<int, 4>>& tri_map) const;
};

}  // namespace deme

#endif
