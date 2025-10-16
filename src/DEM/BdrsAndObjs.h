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

#include "../kernel/DEMHelperKernels.cuh"
#include "Defines.h"
#include "Structs.h"
#include "../core/utils/CudaAllocator.hpp"
#include "HostSideHelpers.hpp"

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

/// API-(Host-)side struct that holds cached user-input external objects
class DEMExternObj : public DEMInitializer {
  public:
    DEMExternObj() { obj_type = OWNER_TYPE::ANALYTICAL; }
    ~DEMExternObj() {}

    // Component object types. This means the types of each component, and is different from obj_type.
    std::vector<OBJ_COMPONENT> types;
    // Component object materials
    std::vector<std::shared_ptr<DEMMaterial>> materials;
    // Family code (used in prescribing its motions etc.)
    unsigned int family_code = RESERVED_FAMILY_NUM;  ///< Means it is default to the `fixed' family

    // // The coordinate of the CoM of this external object, in the frame where all its components' properties are
    // // reported. This is usually all-0 (meaning you should define the object's components in its CoM frame to begin
    // // with), but it can be user-specified.
    // float3 CoM = make_float3(0);
    // // CoM frame's orientation quaternion in the frame which is used to report all its components' properties.
    // Usually
    // // unit quaternion.
    // float4 CoM_oriQ = make_float4(0, 0, 0, 1);

    // Obj's CoM initial position
    float3 init_pos = make_float3(0);
    // Obj's initial orientation quaternion
    float4 init_oriQ = make_float4(0, 0, 0, 1);
    // Obj's mass (huge by default)
    float mass = 1e6;
    // Obj's MOI (huge by default)
    float3 MOI = make_float3(1e6);

    float GetMass() { return mass; }
    std::vector<float> GetMOI() {
        std::vector<float> res = {MOI.x, MOI.y, MOI.z};
        return res;
    }

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
    void SetMOI(const std::vector<float>& MOI) {
        assertThreeElements(MOI, "SetMOI", "MOI");
        SetMOI(make_float3(MOI[0], MOI[1], MOI[2]));
    }

    /// @brief Set the initial quaternion for this object (before simulation initializes).
    /// @param rotQ Initial quaternion.
    void SetInitQuat(const float4 rotQ) { init_oriQ = rotQ; }
    void SetInitQuat(const std::vector<float>& rotQ) {
        assertFourElements(rotQ, "SetInitQuat", "rotQ");
        SetInitQuat(make_float4(rotQ[0], rotQ[1], rotQ[2], rotQ[3]));
    }

    /// @brief Set the initial position for this object (before simulation initializes).
    /// @param displ Initial position.
    void SetInitPos(const float3 displ) { init_pos = displ; }
    void SetInitPos(const std::vector<float>& displ) {
        assertThreeElements(displ, "SetInitPos", "displ");
        SetInitPos(make_float3(displ[0], displ[1], displ[2]));
    }

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
    void AddPlane(const std::vector<float>& pos,
                  const std::vector<float>& normal,
                  const std::shared_ptr<DEMMaterial>& material) {
        assertThreeElements(pos, "AddPlane", "pos");
        assertThreeElements(normal, "AddPlane", "normal");
        AddPlane(make_float3(pos[0], pos[1], pos[2]), make_float3(normal[0], normal[1], normal[2]), material);
    }

    /*
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
    */

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
        params.cyl.dir = make_float3(0, 0, 1);
        params.cyl.normal = normal;
        entity_params.push_back(params);
    }
    void AddZCylinder(const std::vector<float>& pos,
                      const float rad,
                      const std::shared_ptr<DEMMaterial>& material,
                      const objNormal_t normal = ENTITY_NORMAL_INWARD) {
        assertThreeElements(pos, "AddZCylinder", "pos");
        AddZCylinder(make_float3(pos[0], pos[1], pos[2]), rad, material, normal);
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
    void AddCylinder(const std::vector<float>& pos,
                     const std::vector<float>& axis,
                     const float rad,
                     const std::shared_ptr<DEMMaterial>& material,
                     const objNormal_t normal = ENTITY_NORMAL_INWARD) {
        assertThreeElements(pos, "AddCylinder", "pos");
        assertThreeElements(axis, "AddCylinder", "axis");
        AddCylinder(make_float3(pos[0], pos[1], pos[2]), make_float3(axis[0], axis[1], axis[2]), rad, material, normal);
    }
};

// DEM mesh object
class DEMMeshConnected : public DEMInitializer {
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

    // Position in the m_meshes array
    unsigned int cache_offset = 0;

    std::vector<float3> m_vertices;
    std::vector<float3> m_normals;
    std::vector<float3> m_UV;
    std::vector<float3> m_colors;

    std::vector<int3> m_face_v_indices;
    std::vector<int3> m_face_n_indices;
    std::vector<int3> m_face_uv_indices;
    std::vector<int3> m_face_col_indices;

    /// @brief Get the coordinates of the vertices of this mesh.
    /// @return A reference to the vertices data vector (of float3) of the mesh.
    std::vector<float3>& GetCoordsVertices() { return m_vertices; }
    /// @brief Get the coordinates of the vertices of this mesh.
    /// @return N (number of vertices) by 3 matrix.
    std::vector<std::vector<float>> GetCoordsVerticesAsVectorOfVectors();

    std::vector<float3>& GetCoordsNormals() { return m_normals; }

    // std::vector<std::vector<float>>& GetCoordsNormalsPython() {
    //     std::vector<std::vector<float>> return_vec;

    //     for (int i = 0; i < m_normals.size(); i++) {
    //         std::vector<float> tmp;
    //         tmp.push_back(m_normals[i].x);
    //         tmp.push_back(m_normals[i].y);
    //         tmp.push_back(m_normals[i].z);

    //         return_vec.push_back(tmp);
    //     }

    //     return return_vec;
    // }

    std::vector<float3>& GetCoordsUV() { return m_UV; }

    // std::vector<std::vector<float>>& GetCoordsUVPython() {
    //     std::vector<std::vector<float>> return_vec;

    //     for (int i = 0; i < m_UV.size(); i++) {
    //         std::vector<float> tmp;
    //         tmp.push_back(m_UV[i].x);
    //         tmp.push_back(m_UV[i].y);
    //         tmp.push_back(m_UV[i].z);

    //         return_vec.push_back(tmp);
    //     }

    //     return return_vec;
    // }

    std::vector<float3>& GetCoordsColors() { return m_colors; }

    /// @brief Get the vertices number of all the triangles of this mesh.
    /// @return A reference to the vertices number data vector (of int3) of the mesh.
    std::vector<int3>& GetIndicesVertexes() { return m_face_v_indices; }
    /// @brief Get the vertices number of all the triangles of this mesh.
    /// @return N (number of vertices) by 3 matrix.
    std::vector<std::vector<int>> GetIndicesVertexesAsVectorOfVectors();

    std::vector<int3>& GetIndicesNormals() { return m_face_n_indices; }

    // std::vector<std::vector<int>>& GetIndicesNormalsPython() {
    //     std::vector<std::vector<int>> return_vec;
    //     for (int i = 0; i < m_face_n_indices.size(); i++) {
    //         std::vector<int> tmp;

    //         tmp.push_back(m_face_n_indices[i].x);
    //         tmp.push_back(m_face_n_indices[i].y);
    //         tmp.push_back(m_face_n_indices[i].z);
    //     }

    //     return return_vec;
    // }

    std::vector<int3>& GetIndicesUV() { return m_face_uv_indices; }

    // std::vector<std::vector<int>>& GetIndicesUVPython() {
    //     std::vector<std::vector<int>> return_vec;
    //     for (int i = 0; i < m_face_uv_indices.size(); i++) {
    //         std::vector<int> tmp;

    //         tmp.push_back(m_face_uv_indices[i].x);
    //         tmp.push_back(m_face_uv_indices[i].y);
    //         tmp.push_back(m_face_uv_indices[i].z);
    //     }

    //     return return_vec;
    // }

    std::vector<int3>& GetIndicesColors() { return m_face_col_indices; }

    // std::vector<std::vector<int>>& GetIndicesColorsPython() {
    //     std::vector<std::vector<int>> return_vec;

    //     for (int i = 0; i < m_face_col_indices.size(); i++) {
    //         std::vector<int> tmp;

    //         tmp.push_back(m_face_col_indices[i].x);
    //         tmp.push_back(m_face_col_indices[i].y);
    //         tmp.push_back(m_face_col_indices[i].z);
    //     }

    //     return return_vec;
    // }

    // Material types for each mesh facet
    std::vector<std::shared_ptr<DEMMaterial>> materials;
    bool isMaterialSet = false;
    // Family code (used in prescribing its motions etc.)
    unsigned int family_code = RESERVED_FAMILY_NUM;  ///< Means it is default to the `fixed' family

    // // The coordinate of the CoM of this meshed object, in the frame where all the mesh's node coordinates are
    // // reported. This is usually all-0 (meaning you should define the object's components in its CoM frame to begin
    // // with), but it can be user-specified.
    // float3 CoM = make_float3(0);
    // // CoM frame's orientation quaternion in the frame which is used to report all the mesh's node coordinates.
    // // It is usually unit quaternion.
    // float4 CoM_oriQ = make_float4(0, 0, 0, 1);

    // Mesh's CoM initial position
    float3 init_pos = make_float3(0);
    // Mesh's initial orientation quaternion
    float4 init_oriQ = make_float4(0, 0, 0, 1);
    // Mesh's mass
    float mass = 1.f;
    // Mesh's MOI
    float3 MOI = make_float3(1.f);

    float GetMass() { return mass; }
    std::vector<float> GetMOI() {
        std::vector<float> res = {MOI.x, MOI.y, MOI.z};
        return res;
    }

    std::string filename;  ///< file string if loading an obj file

    // If true, when the mesh is initialized into the system, it will re-order the nodes of each triangle so that the
    // normals derived from right-hand-rule are the same as the normals in the mesh file
    bool use_mesh_normals = false;

    DEMMeshConnected() { obj_type = OWNER_TYPE::MESH; }
    DEMMeshConnected(std::string input_file) {
        LoadWavefrontMesh(input_file);
        obj_type = OWNER_TYPE::MESH;
    }
    DEMMeshConnected(std::string input_file, const std::shared_ptr<DEMMaterial>& mat) {
        LoadWavefrontMesh(input_file);
        SetMaterial(mat);
        obj_type = OWNER_TYPE::MESH;
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

    /// Get the number of nodes in the mesh
    size_t GetNumNodes() const { return m_vertices.size(); }

    /// Instruct that when the mesh is initialized into the system, it will re-order the nodes of each triangle so that
    /// the normals derived from right-hand-rule are the same as the normals in the mesh file
    void UseNormals(bool use = true) { use_mesh_normals = use; }

    /// Access the n-th triangle in mesh
    DEMTriangle GetTriangle(size_t index) const {  // No need to wrap (for Shlok)
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

    /// Set mass.
    void SetMass(float mass) { this->mass = mass; }
    /// Set MOI (in principal frame).
    void SetMOI(float3 MOI) { this->MOI = MOI; }
    /// Set MOI (in principal frame).
    void SetMOI(const std::vector<float>& MOI) {
        assertThreeElements(MOI, "SetMOI", "MOI");
        SetMOI(make_float3(MOI[0], MOI[1], MOI[2]));
    }
    /// Set mesh family number.
    void SetFamily(unsigned int num) { this->family_code = num; }

    /// Set material types for the mesh. Technically, you can set that for each individual mesh facet.
    void SetMaterial(const std::vector<std::shared_ptr<DEMMaterial>>& input) {
        assertLength(input.size(), "SetMaterial");
        materials = input;
        isMaterialSet = true;
    }
    /// Set material types for the mesh. Technically, you can set that for each individual mesh facet.
    void SetMaterial(const std::shared_ptr<DEMMaterial>& input) {
        SetMaterial(std::vector<std::shared_ptr<DEMMaterial>>(nTri, input));
    }

    /*
    /// Compute barycenter, mass and MOI in CoM frame
    void ComputeMassProperties(double& mass, float3& center, float3& inertia);

    /// Create a map of neighboring triangles, vector of:
    /// [Ti TieA TieB TieC]
    /// (the free sides have triangle id = -1).
    /// Return false if some edge has more than 2 neighboring triangles
    bool ComputeNeighbouringTriangleMap(std::vector<std::array<int, 4>>& tri_map) const;
    */

    /// @brief Give the meshed object an initial rotation, before the simulation starts.
    void SetInitQuat(const float4 rotQ) { init_oriQ = rotQ; }
    void SetInitQuat(const std::vector<float>& rotQ) {
        assertFourElements(rotQ, "SetInitQuat", "rotQ");
        SetInitQuat(make_float4(rotQ[0], rotQ[1], rotQ[2], rotQ[3]));
    }

    /// @brief Transform the meshed object so it gets to its initial position, before the simulation starts.
    void SetInitPos(const float3 displ) { init_pos = displ; }
    void SetInitPos(const std::vector<float>& displ) {
        assertThreeElements(displ, "SetInitPos", "displ");
        SetInitPos(make_float3(displ[0], displ[1], displ[2]));
    }

    /// If this mesh's component triangles are not reported by the user in its centroid and principal system,
    /// then the user needs to call this method immediately to report this mesh's volume centroid and principal axes,
    /// and nodes will be adjusted by this call so that the mesh's frame is its centroid and principal system.
    void InformCentroidPrincipal(float3 center, float4 prin_Q) {
        // Getting to Centroid and Principal is a translation then a rotation (local), so the undo order to undo
        // translation then rotation
        for (auto& node : m_vertices) {
            applyFrameTransformGlobalToLocal(node, center, prin_Q);
        }
    }
    void InformCentroidPrincipal(const std::vector<float>& center, const std::vector<float>& prin_Q) {
        assertThreeElements(center, "InformCentroidPrincipal", "center");
        assertFourElements(prin_Q, "InformCentroidPrincipal", "prin_Q");
        InformCentroidPrincipal(make_float3(center[0], center[1], center[2]),
                                make_float4(prin_Q[0], prin_Q[1], prin_Q[2], prin_Q[3]));
    }

    /// The opposite of InformCentroidPrincipal, and it is another way to align this mesh's coordinate system with its
    /// centroid and principal system: rotate then move this mesh, so that at the end of this operation, the mesh's
    /// frame is its centroid and principal system.
    void Move(float3 vec, float4 rot_Q) {
        for (auto& node : m_vertices) {
            applyFrameTransformLocalToGlobal(node, vec, rot_Q);
        }
    }
    void Move(const std::vector<float>& vec, const std::vector<float>& rot_Q) {
        assertThreeElements(vec, "Move", "vec");
        assertFourElements(rot_Q, "Move", "rot_Q");
        Move(make_float3(vec[0], vec[1], vec[2]), make_float4(rot_Q[0], rot_Q[1], rot_Q[2], rot_Q[3]));
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
        // The nodal normal also changes. Although, we don't need it in general.
        for (auto& normal : m_normals) {
            float proj = dot(normal, plane_normal);
            // Different from mirroring nodes
            normal -= 2 * proj * plane_normal;
        }
        // Mirroring will change the order of the facet nodes, so RHR becomes LHR. We have to account for that.
        for (auto& face_v_indices : m_face_v_indices) {
            auto tmp = face_v_indices.y;
            face_v_indices.y = face_v_indices.z;
            face_v_indices.z = tmp;
        }
        for (auto& face_n_indices : m_face_n_indices) {
            auto tmp = face_n_indices.y;
            face_n_indices.y = face_n_indices.z;
            face_n_indices.z = tmp;
        }
        for (auto& face_uv_indices : m_face_uv_indices) {
            auto tmp = face_uv_indices.y;
            face_uv_indices.y = face_uv_indices.z;
            face_uv_indices.z = tmp;
        }
        for (auto& face_col_indices : m_face_col_indices) {
            auto tmp = face_col_indices.y;
            face_col_indices.y = face_col_indices.z;
            face_col_indices.z = tmp;
        }
    }
    void Mirror(const std::vector<float>& plane_point, const std::vector<float>& plane_normal) {
        assertThreeElements(plane_point, "Mirror", "plane_point");
        assertThreeElements(plane_normal, "Mirror", "plane_normal");
        Mirror(make_float3(plane_point[0], plane_point[1], plane_point[2]),
               make_float3(plane_normal[0], plane_normal[1], plane_normal[2]));
    }

    /// @brief Scale all geometry component of this mesh.
    void Scale(float s) {
        // Never let mass become negative.
        assertPositive(s, "Scale", "s");
        for (auto& node : m_vertices) {
            node *= s;
        }
        double double_s = (double)std::abs(s);
        mass *= double_s * double_s * double_s;
        MOI *= double_s * double_s * double_s * double_s * double_s;
    }
    /// @brief Scale all geometry component of this mesh. Specify x, y, z respectively.
    void Scale(float3 s) {
        // Never let mass become negative.
        assertPositive(s.x, "Scale", "s");
        assertPositive(s.y, "Scale", "s");
        assertPositive(s.z, "Scale", "s");
        for (auto& node : m_vertices) {
            node = node * s;
        }
        // Really just an estimate. The user should reset mass properties manually afterwards.
        double prod = (double)s.x * (double)s.y * (double)s.z;
        mass *= prod;
        MOI.x *= prod * s.x * s.x;  // Square, so always positive
        MOI.y *= prod * s.y * s.y;
        MOI.z *= prod * s.z * s.z;
    }
    void Scale(const std::vector<float>& s) {
        assertThreeElements(s, "Scale", "s");
        Scale(make_float3(s[0], s[1], s[2]));
    }

    ////////////////////////////////////////////////////////
    // Some geo wildcard-related stuff
    ////////////////////////////////////////////////////////
    // Initial geometry wildcard that all triangles should have
    std::unordered_map<std::string, std::vector<float>> geo_wildcards;
    // Can be used to save mem after initialization
    void ClearWildcards() { deallocate_array(geo_wildcards); }
    void SetGeometryWildcards(const std::unordered_map<std::string, std::vector<float>>& wildcards) {
        if (wildcards.begin()->second.size() != nTri) {
            std::stringstream ss;
            ss << "Input gemometry wildcard arrays in a SetGeometryWildcards call must all have the same size as the "
                  "number of triangles in this mesh.\nHere, the input array has length "
               << wildcards.begin()->second.size() << " but this mesh has " << nTri << " triangles." << std::endl;
            throw std::runtime_error(ss.str());
        }
        geo_wildcards = wildcards;
    }
    void AddGeometryWildcard(const std::string& name, const std::vector<float>& vals) {
        if (vals.size() != nTri) {
            std::stringstream ss;
            ss << "Input gemometry wildcard array in a AddGeometryWildcard call must have the same size as the number "
                  "of triangles in this mesh.\nHere, the input array has length "
               << vals.size() << " but this mesh has " << nTri << " triangles." << std::endl;
            throw std::runtime_error(ss.str());
        }
        geo_wildcards[name] = vals;
    }
    void AddGeometryWildcard(const std::string& name, float val) {
        AddGeometryWildcard(name, std::vector<float>(nTri, val));
    }
};

/*
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
*/

}  // namespace deme

#endif
