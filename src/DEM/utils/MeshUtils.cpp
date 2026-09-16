//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// This file contains modifications of the code by Alessandro Tasora and Radu Serban
// Below is the BSD license

// Copyright (c) 2016, Project Chrono Development Team
// All Rghts Reserved.

// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
// following conditions are met:

//  - Redistributions of source code must retain the above copyright notice, this list of conditions and the following
//  disclaimer.
//  - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
//  following disclaimer in the documentation and/or other materials provided with the distribution.
//  - Neither the name of the nor the names of its contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cstdint>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "DEM/BdrsAndObjs.h"
#include "DEM/utils/HostSideHelpers.hpp"
#include "DEM/utils/MeshUtils.hpp"
#include "core/utils/Logger.hpp"
#include "core/utils/WavefrontMeshLoader.hpp"
#include "kernel/DEMHelperKernels.cuh"

namespace deme {

using namespace WAVEFRONT;

std::vector<std::vector<float>> DEMMesh::GetCoordsVerticesAsVectorOfVectors() {
    auto vec = GetCoordsVertices();
    std::vector<std::vector<float>> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        res[i] = {vec[i].x, vec[i].y, vec[i].z};
    }
    return res;
}

std::vector<std::vector<int>> DEMMesh::GetIndicesVertexesAsVectorOfVectors() {
    auto vec = GetIndicesVertexes();
    std::vector<std::vector<int>> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        res[i] = {vec[i].x, vec[i].y, vec[i].z};
    }
    return res;
}

bool DEMMesh::LoadSTLMesh(std::string input_file, bool load_normals) {
    Clear();
    filename = input_file;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        DEME_ERROR_NOTHROW("Error loading STL file %s", filename.c_str());
        return false;
    }
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (buffer.size() < 84) {
        DEME_ERROR_NOTHROW("STL file %s is too small to contain any triangles.", filename.c_str());
        return false;
    }

    auto set_default_patch_info = [this]() {
        this->nTri = m_face_v_indices.size();
        this->m_patch_ids.clear();
        this->m_patch_ids.resize(this->nTri, 0);
        this->nPatches = 1;
        this->patches_explicitly_set = false;
    };

    auto load_binary = [&](uint32_t tri_count) -> bool {
        size_t expected_size = 84 + static_cast<size_t>(tri_count) * 50;
        if (buffer.size() < expected_size) {
            DEME_ERROR_NOTHROW("Binary STL file %s ended unexpectedly.", filename.c_str());
            return false;
        }
        const unsigned char* data = reinterpret_cast<const unsigned char*>(buffer.data());
        size_t offset = 84;
        for (uint32_t i = 0; i < tri_count; i++) {
            float floats[12];
            std::memcpy(floats, data + offset, sizeof(float) * 12);
            float3 v0 = make_float3(floats[3], floats[4], floats[5]);
            float3 v1 = make_float3(floats[6], floats[7], floats[8]);
            float3 v2 = make_float3(floats[9], floats[10], floats[11]);
            size_t base = m_vertices.size();
            m_vertices.push_back(v0);
            m_vertices.push_back(v1);
            m_vertices.push_back(v2);
            m_face_v_indices.push_back(make_int3((int)base, (int)base + 1, (int)base + 2));
            offset += 50;
        }
        return true;
    };

    // Heuristics to decide if STL is binary
    uint32_t tri_count = 0;
    std::memcpy(&tri_count, buffer.data() + 80, sizeof(uint32_t));
    size_t expected_size = 84 + static_cast<size_t>(tri_count) * 50;
    bool looks_binary = expected_size == buffer.size();
    bool looks_ascii = false;
    if (!looks_binary) {
        std::string header(buffer.data(), buffer.data() + std::min<size_t>(buffer.size(), 5));
        if (header == "solid") {
            looks_ascii = true;
        }
    }

    bool parsed = false;
    if (looks_binary) {
        parsed = load_binary(tri_count);
    }
    if (!parsed) {
        // Fallback to ASCII parsing
        std::istringstream iss(std::string(buffer.begin(), buffer.end()));
        std::string line;
        std::vector<float3> facet_vertices;
        facet_vertices.reserve(3);
        while (std::getline(iss, line)) {
            std::istringstream ls(line);
            std::string token;
            ls >> token;
            if (token == "facet") {
                continue;
            }
            if (token == "vertex") {
                float3 v{};
                ls >> v.x >> v.y >> v.z;
                facet_vertices.push_back(v);
                if (facet_vertices.size() == 3) {
                    size_t base = m_vertices.size();
                    m_vertices.push_back(facet_vertices[0]);
                    m_vertices.push_back(facet_vertices[1]);
                    m_vertices.push_back(facet_vertices[2]);
                    m_face_v_indices.push_back(make_int3((int)base, (int)base + 1, (int)base + 2));
                    facet_vertices.clear();
                }
            }
        }
    }

    if (m_face_v_indices.empty()) {
        DEME_ERROR_NOTHROW("Failed to parse STL file %s.", filename.c_str());
        return false;
    }

    set_default_patch_info();

    // Compute one geometric normal per facet from triangle winding.
    // This restores the previous DEM contact behavior and avoids relying on
    // inconsistent/inverted normals embedded in STL files.
    if (load_normals) {
        m_normals.clear();
        m_face_n_indices.clear();
        m_normals.reserve(m_face_v_indices.size());
        m_face_n_indices.reserve(m_face_v_indices.size());
        for (size_t i = 0; i < m_face_v_indices.size(); ++i) {
            const int3& f = m_face_v_indices[i];
            const float3& v0 = m_vertices[f.x];
            const float3& v1 = m_vertices[f.y];
            const float3& v2 = m_vertices[f.z];
            const float3 n = face_normal(v0, v1, v2);
            m_normals.push_back(n);
            m_face_n_indices.push_back(make_int3((int)i, (int)i, (int)i));
        }
    } else {
        m_normals.clear();
        m_face_n_indices.clear();
    }
    // STL has no UV by design; clear to mirror OBJ loader when UVs are absent.
    m_UV.clear();
    m_face_uv_indices.clear();

    {
        size_t boundary_edges = 0;
        size_t nonmanifold_edges = 0;
        if (!IsWatertight(&boundary_edges, &nonmanifold_edges)) {
            DEME_WARNING(
                "Mesh %s is not watertight (boundary edges: %zu, non-manifold edges: %zu). Auto Volume/MOI may be "
                "inaccurate.",
                filename.c_str(), boundary_edges, nonmanifold_edges);
        }
    }
    return true;
}

bool DEMMesh::LoadPLYMesh(std::string input_file, bool load_normals) {
    Clear();
    filename = input_file;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        DEME_ERROR_NOTHROW("Error loading PLY file %s", filename.c_str());
        return false;
    }

    std::string line;
    if (!std::getline(file, line) || line != "ply") {
        DEME_ERROR_NOTHROW("PLY file %s is missing magic header.", filename.c_str());
        return false;
    }

    enum class PLYFormat { ASCII, BINARY_LE, BINARY_BE };
    PLYFormat format = PLYFormat::ASCII;
    size_t num_vertices = 0;
    size_t num_faces = 0;
    // Track vertex property order to find position/normal fields
    std::vector<std::string> vertex_props;
    std::vector<std::string> vertex_prop_types;
    bool in_vertex = false;
    // Face list types
    std::string face_count_type;
    std::string face_index_type;

    while (std::getline(file, line)) {
        if (line == "end_header") {
            break;
        }
        std::istringstream ls(line);
        std::string token;
        ls >> token;
        if (token == "format") {
            std::string fmt;
            ls >> fmt;
            if (fmt.find("ascii") == 0) {
                format = PLYFormat::ASCII;
            } else if (fmt.find("binary_little_endian") == 0) {
                format = PLYFormat::BINARY_LE;
            } else if (fmt.find("binary_big_endian") == 0) {
                format = PLYFormat::BINARY_BE;
            }
        } else if (token == "element") {
            std::string elem;
            ls >> elem;
            if (elem == "vertex") {
                ls >> num_vertices;
                in_vertex = true;
            } else if (elem == "face") {
                ls >> num_faces;
                in_vertex = false;
            } else {
                in_vertex = false;
            }
        } else if (token == "property" && in_vertex) {
            std::string type, name;
            ls >> type >> name;
            if (!name.empty()) {
                vertex_props.push_back(name);
                vertex_prop_types.push_back(type);
            }
        } else if (token == "property" && !in_vertex) {
            std::string maybe_list;
            ls >> maybe_list;
            if (maybe_list == "list") {
                ls >> face_count_type >> face_index_type;
                // ignore name
            }
        }
    }

    if (format == PLYFormat::BINARY_BE) {
        DEME_ERROR_NOTHROW("PLY file %s uses big-endian binary, which is not supported.", filename.c_str());
        return false;
    }
    if (num_vertices == 0 || num_faces == 0) {
        DEME_ERROR_NOTHROW("PLY file %s does not contain vertices or faces.", filename.c_str());
        return false;
    }

    auto find_prop = [&](const std::string& name) -> int {
        for (int i = 0; i < static_cast<int>(vertex_props.size()); ++i) {
            if (vertex_props[i] == name)
                return i;
        }
        return -1;
    };
    const int idx_x = find_prop("x");
    const int idx_y = find_prop("y");
    const int idx_z = find_prop("z");
    const int idx_nx = find_prop("nx");
    const int idx_ny = find_prop("ny");
    const int idx_nz = find_prop("nz");
    const bool has_vertex_normals = idx_nx >= 0 && idx_ny >= 0 && idx_nz >= 0;

    m_vertices.reserve(num_vertices);
    m_face_v_indices.reserve(num_faces);

    auto read_scalar_le = [&](std::istream& is, const std::string& type, double& out) -> bool {
        if (type == "float" || type == "float32") {
            float v;
            if (!is.read(reinterpret_cast<char*>(&v), sizeof(float)))
                return false;
            out = static_cast<double>(v);
            return true;
        }
        if (type == "double" || type == "float64") {
            double v;
            if (!is.read(reinterpret_cast<char*>(&v), sizeof(double)))
                return false;
            out = v;
            return true;
        }
        if (type == "uchar" || type == "uint8") {
            std::uint8_t v;
            if (!is.read(reinterpret_cast<char*>(&v), sizeof(std::uint8_t)))
                return false;
            out = static_cast<double>(v);
            return true;
        }
        if (type == "char" || type == "int8") {
            std::int8_t v;
            if (!is.read(reinterpret_cast<char*>(&v), sizeof(std::int8_t)))
                return false;
            out = static_cast<double>(v);
            return true;
        }
        if (type == "int" || type == "int32") {
            std::int32_t v;
            if (!is.read(reinterpret_cast<char*>(&v), sizeof(std::int32_t)))
                return false;
            out = static_cast<double>(v);
            return true;
        }
        if (type == "uint" || type == "uint32") {
            std::uint32_t v;
            if (!is.read(reinterpret_cast<char*>(&v), sizeof(std::uint32_t)))
                return false;
            out = static_cast<double>(v);
            return true;
        }
        return false;
    };

    // Read vertices
    for (size_t i = 0; i < num_vertices; ++i) {
        if (format == PLYFormat::ASCII) {
            if (!std::getline(file, line)) {
                DEME_ERROR_NOTHROW("Unexpected EOF while reading vertices in %s.", filename.c_str());
                return false;
            }
            std::istringstream ls(line);
            std::vector<double> vals;
            double v;
            while (ls >> v) {
                vals.push_back(v);
            }
            if (idx_x < 0 || idx_y < 0 || idx_z < 0 || vals.size() <= std::max({idx_x, idx_y, idx_z})) {
                DEME_ERROR_NOTHROW("Vertex position data missing in %s.", filename.c_str());
                return false;
            }
            float3 p = make_float3(static_cast<float>(vals[idx_x]), static_cast<float>(vals[idx_y]),
                                   static_cast<float>(vals[idx_z]));
            m_vertices.push_back(p);
            if (has_vertex_normals && vals.size() > static_cast<size_t>(std::max({idx_nx, idx_ny, idx_nz}))) {
                float3 n = make_float3(static_cast<float>(vals[idx_nx]), static_cast<float>(vals[idx_ny]),
                                       static_cast<float>(vals[idx_nz]));
                m_normals.push_back(n);
            }
        } else {
            // Binary little-endian
            std::vector<double> vals(vertex_props.size(), 0.0);
            for (size_t p = 0; p < vertex_props.size(); ++p) {
                if (!read_scalar_le(file, vertex_prop_types[p], vals[p])) {
                    DEME_ERROR_NOTHROW("Failed to read vertex data in binary PLY %s.", filename.c_str());
                    return false;
                }
            }
            if (idx_x < 0 || idx_y < 0 || idx_z < 0) {
                DEME_ERROR_NOTHROW("Vertex position data missing in %s.", filename.c_str());
                return false;
            }
            float3 p = make_float3(static_cast<float>(vals[idx_x]), static_cast<float>(vals[idx_y]),
                                   static_cast<float>(vals[idx_z]));
            m_vertices.push_back(p);
            if (has_vertex_normals && vals.size() > static_cast<size_t>(std::max({idx_nx, idx_ny, idx_nz}))) {
                float3 n = make_float3(static_cast<float>(vals[idx_nx]), static_cast<float>(vals[idx_ny]),
                                       static_cast<float>(vals[idx_nz]));
                m_normals.push_back(n);
            }
        }
    }

    // Read faces
    std::vector<int3> faces;
    faces.reserve(num_faces);
    for (size_t i = 0; i < num_faces; ++i) {
        if (format == PLYFormat::ASCII) {
            if (!std::getline(file, line)) {
                DEME_ERROR_NOTHROW("Unexpected EOF while reading faces in %s.", filename.c_str());
                return false;
            }
            std::istringstream ls(line);
            int verts_in_face = 0;
            ls >> verts_in_face;
            if (verts_in_face < 3) {
                continue;  // ignore degenerate
            }
            std::vector<int> idx(verts_in_face);
            for (int j = 0; j < verts_in_face; ++j) {
                ls >> idx[j];
            }
            for (int t = 1; t < verts_in_face - 1; ++t) {
                faces.push_back(make_int3(idx[0], idx[t], idx[t + 1]));
            }
        } else {
            // Binary little-endian faces: expect list uchar count, int indices
            double count_d = 0.0;
            if (!read_scalar_le(file, face_count_type.empty() ? "uchar" : face_count_type, count_d)) {
                DEME_ERROR_NOTHROW("Failed to read face count in binary PLY %s.", filename.c_str());
                return false;
            }
            int verts_in_face = static_cast<int>(count_d);
            if (verts_in_face < 3) {
                // Skip indices
                for (int j = 0; j < verts_in_face; ++j) {
                    double throwaway;
                    if (!read_scalar_le(file, face_index_type.empty() ? "int" : face_index_type, throwaway)) {
                        DEME_ERROR_NOTHROW("Failed to skip face indices in binary PLY %s.", filename.c_str());
                        return false;
                    }
                }
                continue;
            }
            std::vector<int> idx(verts_in_face);
            for (int j = 0; j < verts_in_face; ++j) {
                double v = 0.0;
                if (!read_scalar_le(file, face_index_type.empty() ? "int" : face_index_type, v)) {
                    DEME_ERROR_NOTHROW("Failed to read face indices in binary PLY %s.", filename.c_str());
                    return false;
                }
                idx[j] = static_cast<int>(v);
            }
            for (int t = 1; t < verts_in_face - 1; ++t) {
                faces.push_back(make_int3(idx[0], idx[t], idx[t + 1]));
            }
        }
    }

    if (faces.empty()) {
        DEME_ERROR_NOTHROW("No faces parsed from PLY file %s.", filename.c_str());
        return false;
    }

    m_face_v_indices = std::move(faces);
    nTri = m_face_v_indices.size();

    if (load_normals) {
        m_normals.clear();
        m_face_n_indices.clear();
        m_normals.reserve(nTri);
        m_face_n_indices.reserve(nTri);
        for (size_t i = 0; i < nTri; ++i) {
            const int3& f = m_face_v_indices[i];
            const float3& v0 = m_vertices[f.x];
            const float3& v1 = m_vertices[f.y];
            const float3& v2 = m_vertices[f.z];
            float3 n = face_normal(v0, v1, v2);
            m_normals.push_back(n);
            m_face_n_indices.push_back(make_int3((int)i, (int)i, (int)i));
        }
    } else {
        m_normals.clear();
        m_face_n_indices.clear();
    }
    m_UV.clear();
    m_face_uv_indices.clear();

    // Default patch info: one patch
    m_patch_ids.assign(nTri, 0);
    nPatches = 1;
    patches_explicitly_set = false;
    return true;
}

bool DEMMesh::LoadWavefrontMesh(std::string input_file, bool load_normals, bool load_uv) {
    this->m_vertices.clear();
    this->m_normals.clear();
    this->m_UV.clear();
    this->m_face_v_indices.clear();
    this->m_face_n_indices.clear();
    this->m_face_uv_indices.clear();

    GeometryInterface emptybm;  // BuildMesh bm;

    filename = input_file;

    OBJ obj;

    int ret = obj.LoadMesh(filename.c_str(), &emptybm, true);
    if (ret == -1) {
        DEME_ERROR_NOTHROW("Error loading OBJ file %s", filename.c_str());
        return false;
    }

    float3 tmp_f3;
    int3 tmp_i3;
    for (unsigned int iv = 0; iv < obj.mVerts.size(); iv += 3) {
        tmp_f3.x = obj.mVerts[iv];
        tmp_f3.y = obj.mVerts[iv + 1];
        tmp_f3.z = obj.mVerts[iv + 2];
        this->m_vertices.push_back(tmp_f3);
    }
    for (unsigned int in = 0; in < obj.mNormals.size(); in += 3) {
        tmp_f3.x = obj.mNormals[in];
        tmp_f3.y = obj.mNormals[in + 1];
        tmp_f3.z = obj.mNormals[in + 2];
        this->m_normals.push_back(tmp_f3);
    }
    for (unsigned int it = 0; it < obj.mTexels.size(); it += 2)  // +2 because only u,v each texel
    {
        tmp_f3.x = obj.mTexels[it];
        tmp_f3.y = obj.mTexels[it + 1];
        tmp_f3.z = 0;
        this->m_UV.push_back(tmp_f3);
    }
    for (unsigned int iiv = 0; iiv < obj.mIndexesVerts.size(); iiv += 3) {
        tmp_i3.x = obj.mIndexesVerts[iiv];
        tmp_i3.y = obj.mIndexesVerts[iiv + 1];
        tmp_i3.z = obj.mIndexesVerts[iiv + 2];
        this->m_face_v_indices.push_back(tmp_i3);
    }
    for (unsigned int iin = 0; iin < obj.mIndexesNormals.size(); iin += 3) {
        tmp_i3.x = obj.mIndexesNormals[iin];
        tmp_i3.y = obj.mIndexesNormals[iin + 1];
        tmp_i3.z = obj.mIndexesNormals[iin + 2];
        this->m_face_n_indices.push_back(tmp_i3);
    }
    for (unsigned int iit = 0; iit < obj.mIndexesTexels.size(); iit += 3) {
        tmp_i3.x = obj.mIndexesTexels[iit];
        tmp_i3.y = obj.mIndexesTexels[iit + 1];
        tmp_i3.z = obj.mIndexesTexels[iit + 2];
        this->m_face_uv_indices.push_back(tmp_i3);
    }

    if (!load_normals) {
        this->m_normals.clear();
        this->m_face_n_indices.clear();
    }
    if (!load_uv) {
        this->m_UV.clear();
        this->m_face_uv_indices.clear();
    }

    this->nTri = m_face_v_indices.size();

    // Initialize default patch info: all triangles in patch 0 (assuming convex mesh)
    this->m_patch_ids.clear();
    this->m_patch_ids.resize(this->nTri, 0);
    this->nPatches = 1;
    this->patches_explicitly_set = false;

    {
        size_t boundary_edges = 0;
        size_t nonmanifold_edges = 0;
        if (!IsWatertight(&boundary_edges, &nonmanifold_edges)) {
            DEME_WARNING(
                "Mesh %s is not watertight (boundary edges: %zu, non-manifold edges: %zu). Volume/MOI may be "
                "inaccurate.",
                filename.c_str(), boundary_edges, nonmanifold_edges);
        }
    }

    return true;
}

// Write the specified meshes in a Wavefront .obj file
void DEMMesh::WriteWavefront(const std::string& filename, std::vector<DEMMesh>& meshes) {
    std::ofstream mf(filename);

    //// TODO: include normal information if available

    // Create a single object mesh
    std::vector<int> v_offsets;
    int v_off = 1;
    for (auto& m : meshes) {
        for (auto& v : m.m_vertices) {
            mf << "v " << v.x << " " << v.y << " " << v.z << std::endl;
        }
        v_offsets.push_back(v_off);
        v_off += static_cast<int>(m.m_vertices.size());
    }

    std::vector<bool> has_normals;
    std::vector<int> vn_offsets;
    int vn_off = 1;
    for (auto& m : meshes) {
        has_normals.push_back(m.m_normals.size() > 0);
        for (auto& v : m.m_normals) {
            mf << "vn " << v.x << " " << v.y << " " << v.z << std::endl;
        }
        vn_offsets.push_back(vn_off);
        vn_off += static_cast<int>(m.m_normals.size());
    }

    for (size_t i = 0; i < meshes.size(); i++) {
        v_off = v_offsets[i];
        if (has_normals[i]) {
            auto& idxV = meshes[i].m_face_v_indices;
            auto& idxN = meshes[i].m_face_n_indices;
            assert(idxV.size() == idxN.size());
            vn_off = vn_offsets[i];
            for (int j = 0; j < idxV.size(); j++) {
                mf << "f " << idxV[j].x + v_off << "//" << idxN[j].x + vn_off << " " << idxV[j].y + v_off << "//"
                   << idxN[j].y + vn_off << " " << idxV[j].z + v_off << "//" << idxN[j].z + vn_off << std::endl;
            }
        } else {
            for (auto& f : meshes[i].m_face_v_indices) {
                mf << "f " << f.x + v_off << " " << f.y + v_off << " " << f.z + v_off << std::endl;
            }
        }
    }

    mf.close();
}

// Helper function to compute face normal for a triangle
static float3 computeFaceNormal(const float3& v0, const float3& v1, const float3& v2) {
    float3 edge1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 edge2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

    // Cross product
    float3 normal = make_float3(edge1.y * edge2.z - edge1.z * edge2.y, edge1.z * edge2.x - edge1.x * edge2.z,
                                edge1.x * edge2.y - edge1.y * edge2.x);

    // Normalize
    float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (length > DEME_TINY_FLOAT) {
        normal.x /= length;
        normal.y /= length;
        normal.z /= length;
    }

    return normal;
}

// Helper function to compute angle between two normals (in degrees)
static float computeAngleBetweenNormals(const float3& n1, const float3& n2) {
    // Compute dot product
    float dot_product = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;

    // Clamp to [-1, 1] to avoid numerical issues
    dot_product = std::max(-1.0f, std::min(1.0f, dot_product));

    // Compute angle in radians and convert to degrees
    float angle_rad = std::acos(dot_product);
    return angle_rad * 180.0f / deme::PI;
}

// ------------------------------------------------------------
// Helpers for advanced patching
// ------------------------------------------------------------
struct EdgeAdjInfo {
    size_t nbr = 0;
    int va = -1;               // oriented edge vertex A (as appears in the current triangle)
    int vb = -1;               // oriented edge vertex B (as appears in the current triangle)
    bool oriented_ok = false;  // true if the neighbor sees the shared edge reversed (good sign for oriented manifold)
};

static inline float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline float3 cross3(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float norm3(const float3& v) {
    return std::sqrt(dot3(v, v));
}
static inline float3 normalize3(const float3& v) {
    float n = norm3(v);
    if (n > DEME_TINY_FLOAT)
        return make_float3(v.x / n, v.y / n, v.z / n);
    return make_float3(0, 0, 0);
}
static inline float3 add3(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static inline float3 mul3(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}
static inline float clamp11(float x) {
    return std::max(-1.0f, std::min(1.0f, x));
}
static inline float deg2rad(float deg) {
    return deg * (deme::PI / 180.0f);
}
static inline float rad2deg(float rad) {
    return rad * (180.0f / deme::PI);
}

static float computeTriangleArea(const float3& v0, const float3& v1, const float3& v2) {
    float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 c = cross3(e1, e2);
    return 0.5f * norm3(c);
}

// Signed dihedral angle (deg) around oriented edge va->vb of the current triangle.
// Sign is meaningful only when edge orientation is reliable (oriented_ok == true).
static float signedDihedralDeg(const float3& n_cur, const float3& n_nbr, const float3& vA, const float3& vB) {
    float3 e = normalize3(make_float3(vB.x - vA.x, vB.y - vA.y, vB.z - vA.z));
    float s = dot3(e, cross3(n_cur, n_nbr));
    float c = clamp11(dot3(n_cur, n_nbr));
    float theta = std::atan2(s, c);  // [-pi, pi]
    return rad2deg(theta);
}

// Build triangle adjacency WITH oriented shared-edge info.
// Patch splitting must use geometric adjacency, not raw OBJ/STL vertex-index adjacency: those formats commonly
// duplicate vertices per face to carry separate normals/UVs. Canonical vertex IDs let such coincident vertices share
// edges for grouping while preserving the mesh's stored topology. Non-manifold edges (shared by != 2 faces) are treated
// as boundaries.
static std::vector<std::vector<EdgeAdjInfo>> buildAdjacencyWithEdgeInfo(const std::vector<int3>& face_v_indices,
                                                                        const std::vector<float3>& vertices) {
    struct EdgeRec {
        size_t f;
        int a_raw;
        int b_raw;
        size_t a_canon;
        size_t b_canon;
    };

    const size_t num_faces = face_v_indices.size();
    std::vector<std::vector<EdgeAdjInfo>> adj(num_faces);

    std::vector<size_t> canon;
    if (!vertices.empty()) {
        const double eps = computeVertexQuantEps(vertices);
        canon = buildCanonicalVertexMap(vertices, eps);
    }

    std::map<std::pair<size_t, size_t>, std::vector<EdgeRec>> edge_map;

    auto add_edge = [&](size_t f, int a, int b) {
        if (a < 0 || b < 0) {
            return;
        }
        size_t ca = static_cast<size_t>(a);
        size_t cb = static_cast<size_t>(b);
        if (!canon.empty()) {
            if (ca >= canon.size() || cb >= canon.size()) {
                return;
            }
            ca = canon[ca];
            cb = canon[cb];
        }
        if (ca == cb) {
            return;
        }
        size_t lo = std::min(ca, cb);
        size_t hi = std::max(ca, cb);
        edge_map[{lo, hi}].push_back(EdgeRec{f, a, b, ca, cb});
    };

    for (size_t i = 0; i < num_faces; ++i) {
        const int3& tri = face_v_indices[i];
        add_edge(i, tri.x, tri.y);
        add_edge(i, tri.y, tri.z);
        add_edge(i, tri.z, tri.x);
    }

    for (const auto& kv : edge_map) {
        const auto& recs = kv.second;
        if (recs.size() != 2) {
            continue;  // boundary or non-manifold
        }
        const EdgeRec& r0 = recs[0];
        const EdgeRec& r1 = recs[1];

        bool oriented_ok_0 = (r0.a_canon == r1.b_canon && r0.b_canon == r1.a_canon);
        bool oriented_ok_1 = oriented_ok_0;

        adj[r0.f].push_back(EdgeAdjInfo{r1.f, r0.a_raw, r0.b_raw, oriented_ok_0});
        adj[r1.f].push_back(EdgeAdjInfo{r0.f, r1.a_raw, r1.b_raw, oriented_ok_1});
    }

    return adj;
}

// Split mesh into connected patches using region-growing. The default option set preserves the original behavior:
// merge adjacent triangles when their local face-normal angle is below hard_angle_deg.
unsigned int DEMMesh::SplitIntoConvexPatches(float hard_angle_deg,
                                             const PatchSplitOptions& opt,
                                             PatchQualityReport* out_report,
                                             const PatchQualityOptions& qopt) {
    auto set_empty_report = [&]() {
        if (!out_report) {
            return;
        }
        *out_report = PatchQualityReport();
        out_report->requested_min = opt.patch_min;
        out_report->requested_max = opt.patch_max;
    };

    if (nTri == 0) {
        patches_explicitly_set = false;
        nPatches = 1;
        set_empty_report();
        return 0;
    }

    m_patch_ids.clear();
    m_patch_ids.resize(nTri, -1);

    std::vector<float3> face_normals(nTri);
    std::vector<float> face_areas(nTri, 0.0f);
    for (size_t i = 0; i < nTri; ++i) {
        const int3& face = m_face_v_indices[i];
        const float3& v0 = m_vertices[face.x];
        const float3& v1 = m_vertices[face.y];
        const float3& v2 = m_vertices[face.z];
        face_normals[i] = computeFaceNormal(v0, v1, v2);
        face_areas[i] = std::max(computeTriangleArea(v0, v1, v2), static_cast<float>(DEME_TINY_FLOAT));
    }

    const std::vector<std::vector<EdgeAdjInfo>> adjacency = buildAdjacencyWithEdgeInfo(m_face_v_indices, m_vertices);
    std::vector<size_t> seed_order(nTri);
    for (size_t i = 0; i < nTri; ++i) {
        seed_order[i] = i;
    }
    if (opt.seed_largest_first) {
        std::sort(seed_order.begin(), seed_order.end(),
                  [&](size_t a, size_t b) { return face_areas[a] > face_areas[b]; });
    }

    int current_patch_id = 0;
    std::vector<size_t> queue;
    const float hard_angle = std::max(0.0f, hard_angle_deg);
    const float patch_normal_max = opt.patch_normal_max_deg;
    const float concave_allow = std::max(0.0f, opt.concave_allow_deg);

    for (size_t seed : seed_order) {
        if (m_patch_ids[seed] != -1) {
            continue;
        }

        queue.clear();
        queue.push_back(seed);
        m_patch_ids[seed] = current_patch_id;
        float3 patch_normal_sum = mul3(face_normals[seed], face_areas[seed]);
        float patch_area_sum = face_areas[seed];

        size_t queue_idx = 0;
        while (queue_idx < queue.size()) {
            size_t current = queue[queue_idx++];

            for (const EdgeAdjInfo& edge_info : adjacency[current]) {
                const size_t neighbor = edge_info.nbr;
                if (m_patch_ids[neighbor] != -1) {
                    continue;
                }

                const float local_angle = computeAngleBetweenNormals(face_normals[current], face_normals[neighbor]);
                if (local_angle > hard_angle) {
                    continue;
                }

                if (opt.block_concave_edges && edge_info.oriented_ok) {
                    const float dihedral = signedDihedralDeg(face_normals[current], face_normals[neighbor],
                                                             m_vertices[edge_info.va], m_vertices[edge_info.vb]);
                    if (dihedral < -concave_allow) {
                        continue;
                    }
                }

                if (patch_normal_max >= 0.0f && patch_area_sum > DEME_TINY_FLOAT) {
                    const float3 mean_normal = normalize3(patch_normal_sum);
                    const float patch_angle = computeAngleBetweenNormals(mean_normal, face_normals[neighbor]);
                    if (patch_angle > patch_normal_max) {
                        continue;
                    }
                }

                m_patch_ids[neighbor] = current_patch_id;
                queue.push_back(neighbor);
                patch_normal_sum = add3(patch_normal_sum, mul3(face_normals[neighbor], face_areas[neighbor]));
                patch_area_sum += face_areas[neighbor];
            }
        }

        current_patch_id++;
    }

    nPatches = current_patch_id;
    patches_explicitly_set = true;

    if (isMaterialSet) {
        if (materials.size() == 1) {
            materials = std::vector<std::shared_ptr<DEMMaterial>>(nPatches, materials[0]);
        } else if (materials.size() != nPatches) {
            DEME_ERROR(
                "The number of materials set (%zu) does not match the number of patches (%u). Please set the "
                "material (SetMaterial) for each patch or use a single material for all patches.",
                materials.size(), nPatches);
        }
    }

    if (out_report) {
        *out_report = PatchQualityReport();
        out_report->achieved_patches = nPatches;
        out_report->requested_min = opt.patch_min;
        out_report->requested_max = opt.patch_max;
        out_report->per_patch.resize(nPatches);
        if (nPatches > opt.patch_max) {
            out_report->constraint_status = PatchConstraintStatus::TOO_MANY_UNMERGEABLE;
        } else if (nPatches < opt.patch_min) {
            out_report->constraint_status = PatchConstraintStatus::TOO_FEW_UNSPLITTABLE;
        }

        std::vector<float3> normal_sums(nPatches, make_float3(0, 0, 0));
        std::vector<float> area_sums(nPatches, 0.0f);
        for (size_t i = 0; i < nTri; ++i) {
            const patchID_t patch_id = m_patch_ids[i];
            if (patch_id < 0 || patch_id >= static_cast<patchID_t>(nPatches)) {
                continue;
            }
            normal_sums[patch_id] = add3(normal_sums[patch_id], mul3(face_normals[i], face_areas[i]));
            area_sums[patch_id] += face_areas[i];
            out_report->per_patch[patch_id].n_tris++;
        }

        for (unsigned int p = 0; p < nPatches; ++p) {
            PatchQualityPatch& patch_report = out_report->per_patch[p];
            const float sum_norm = norm3(normal_sums[p]);
            patch_report.coherence_r = (area_sums[p] > DEME_TINY_FLOAT) ? (sum_norm / area_sums[p]) : 1.0f;
            const float3 mean_normal =
                (sum_norm > DEME_TINY_FLOAT) ? mul3(normal_sums[p], 1.0f / sum_norm) : make_float3(0, 0, 0);
            for (size_t i = 0; i < nTri; ++i) {
                if (m_patch_ids[i] != static_cast<patchID_t>(p)) {
                    continue;
                }
                patch_report.worst_angle_deg =
                    std::max(patch_report.worst_angle_deg, computeAngleBetweenNormals(mean_normal, face_normals[i]));
                for (const EdgeAdjInfo& edge_info : adjacency[i]) {
                    if (edge_info.nbr <= i || m_patch_ids[edge_info.nbr] != static_cast<patchID_t>(p)) {
                        continue;
                    }
                    const float local_angle = computeAngleBetweenNormals(face_normals[i], face_normals[edge_info.nbr]);
                    if (local_angle > hard_angle) {
                        patch_report.hard_crossings++;
                    }
                    if (opt.block_concave_edges) {
                        if (!edge_info.oriented_ok) {
                            patch_report.unoriented_edges++;
                        } else {
                            const float dihedral =
                                signedDihedralDeg(face_normals[i], face_normals[edge_info.nbr],
                                                  m_vertices[edge_info.va], m_vertices[edge_info.vb]);
                            if (dihedral < -concave_allow) {
                                patch_report.concave_crossings++;
                            }
                        }
                    }
                }
            }

            const float reference_angle = (patch_normal_max >= 0.0f) ? patch_normal_max : hard_angle;
            bool critical = false;
            bool warn = false;
            critical |= qopt.hard_crossings_are_critical && patch_report.hard_crossings > 0;
            critical |= qopt.concave_crossings_are_critical && patch_report.concave_crossings > 0;
            warn |= patch_report.hard_crossings > 0 || patch_report.concave_crossings > 0;
            warn |= opt.block_concave_edges && patch_report.unoriented_edges > qopt.unoriented_warn_threshold;
            warn |= patch_report.coherence_r < qopt.safe_r;
            warn |= patch_report.worst_angle_deg > reference_angle + qopt.warn_worst_angle_margin_deg;
            if (patch_report.coherence_r < qopt.warn_r) {
                warn = true;
            }

            patch_report.level =
                critical ? PatchQualityLevel::CRITICAL : (warn ? PatchQualityLevel::WARN : PatchQualityLevel::SAFE);
            if (static_cast<int>(patch_report.level) > static_cast<int>(out_report->overall)) {
                out_report->overall = patch_report.level;
            }
        }
        if (out_report->constraint_status != PatchConstraintStatus::SATISFIED &&
            out_report->overall == PatchQualityLevel::SAFE) {
            out_report->overall = PatchQualityLevel::WARN;
        }
    }

    return nPatches;
}

// Manually set patch IDs for each triangle.
void DEMMesh::SetPatchIDs(const std::vector<patchID_t>& patch_ids) {
    assertTriLength(patch_ids.size(), "SetPatchIDs");

    auto [compressed_ids, changed] = rank_transform<patchID_t>(patch_ids);

    if (changed) {
        DEME_WARNING(
            std::string("Patch IDs you supplied for a mesh were not contiguous or did not start from 0.\nThey have "
                        "been transformed to be contiguous and start from 0."));
    }

    m_patch_ids = compressed_ids;

    if (!compressed_ids.empty()) {
        int max_patch_id = *std::max_element(compressed_ids.begin(), compressed_ids.end());
        nPatches = max_patch_id + 1;
    } else {
        nPatches = 1;
    }

    patches_explicitly_set = true;

    if (isMaterialSet) {
        if (materials.size() == 1) {
            materials = std::vector<std::shared_ptr<DEMMaterial>>(nPatches, materials[0]);
        } else if (materials.size() != nPatches) {
            DEME_ERROR(
                "The number of materials set (%zu) does not match the number of patches (%u). Please set the "
                "material (SetMaterial) for each patch or use a single material for all patches.",
                materials.size(), nPatches);
        }
    }
}

// Assign consecutive patch IDs so no two triangles are combined into the same mesh patch.
void DEMMesh::SetEachTriangleAsPatch() {
    constexpr size_t max_patch_count = static_cast<size_t>(std::numeric_limits<patchID_t>::max()) + 1;
    if (nTri > max_patch_count) {
        DEME_ERROR(
            "SetEachTriangleAsPatch cannot assign %zu triangles to unique patches because patchID_t supports at most "
            "%zu non-negative patch IDs.",
            nTri, max_patch_count);
    }

    // Preserve material assignment when the source mesh already uses one material per multi-triangle patch.
    if (isMaterialSet && materials.size() > 1 && materials.size() != nTri) {
        if (materials.size() != nPatches || m_patch_ids.size() != nTri) {
            DEME_ERROR(
                "SetEachTriangleAsPatch cannot expand %zu mesh materials over %zu triangles because the existing "
                "patch assignment is inconsistent.",
                materials.size(), nTri);
        }
        std::vector<std::shared_ptr<DEMMaterial>> triangle_materials(nTri);
        for (size_t triangle = 0; triangle < nTri; triangle++) {
            const patchID_t old_patch = m_patch_ids[triangle];
            if (old_patch < 0 || static_cast<size_t>(old_patch) >= materials.size()) {
                DEME_ERROR("SetEachTriangleAsPatch found invalid existing patch ID %d at triangle %zu.", old_patch,
                           triangle);
            }
            triangle_materials[triangle] = materials[old_patch];
        }
        materials = std::move(triangle_materials);
    }

    std::vector<patchID_t> patch_ids(nTri);
    for (size_t triangle = 0; triangle < nTri; triangle++) {
        patch_ids[triangle] = static_cast<patchID_t>(triangle);
    }
    SetPatchIDs(patch_ids);
}

// Compute patch locations (relative to CoM, which is implicitly at 0,0,0)
// If not explicitly set, calculates as:
// - Single patch: (0,0,0)
// - Multiple patches: average of triangle centroids per patch
std::vector<float3> DEMMesh::ComputePatchLocations() const {
    std::vector<float3> patch_locations(nPatches, make_float3(0, 0, 0));

    if (nPatches == 1) {
        // Single patch: location is just CoM (0,0,0)
        return patch_locations;
    }

    // Multiple patches: compute average of triangle centroids per patch
    std::vector<int> patch_triangle_counts(nPatches, 0);

    for (size_t i = 0; i < nTri; ++i) {
        const int3& face = m_face_v_indices[i];
        const float3& v0 = m_vertices[face.x];
        const float3& v1 = m_vertices[face.y];
        const float3& v2 = m_vertices[face.z];

        float3 centroid = triangleCentroid<float3>(v0, v1, v2);
        patchID_t patch_id = (i < m_patch_ids.size()) ? m_patch_ids[i] : 0;

        // Validate patch_id is within bounds
        if (patch_id >= 0 && patch_id < static_cast<patchID_t>(nPatches)) {
            patch_locations[patch_id].x += centroid.x;
            patch_locations[patch_id].y += centroid.y;
            patch_locations[patch_id].z += centroid.z;
            patch_triangle_counts[patch_id]++;
        }
    }

    // Average the accumulated centroids
    for (unsigned int p = 0; p < nPatches; ++p) {
        if (patch_triangle_counts[p] > 0) {
            patch_locations[p].x /= patch_triangle_counts[p];
            patch_locations[p].y /= patch_triangle_counts[p];
            patch_locations[p].z /= patch_triangle_counts[p];
        }
    }

    return patch_locations;
}

// Compute volume, centroid and MOI in CoM frame (unit density).
// ATTENTION: For solid meshes, this assumes a watertight mesh with fine and non-degenerate triangles.
void DEMMesh::ComputeMassProperties(double& volume, float3& center, float3& inertia) const {
    float3 inertia_products = make_float3(0, 0, 0);
    ComputeMassProperties(volume, center, inertia, inertia_products);
}

// Compute volume, centroid and full inertia tensor in CoM frame (unit density).
// ATTENTION: Only correct for "watertight" meshes with fine and non-degenerated triangles.
void DEMMesh::ComputeMassProperties(double& volume, float3& center, float3& inertia, float3& inertia_products) const {
    if (IsShell() && GetShellThickness() > DEME_TINY_FLOAT) {
        const double thickness = static_cast<double>(GetShellThickness());
        double area_total = 0.0;
        double mx = 0.0;
        double my = 0.0;
        double mz = 0.0;

        for (const auto& face : m_face_v_indices) {
            const float3& a = m_vertices[face.x];
            const float3& b = m_vertices[face.y];
            const float3& c = m_vertices[face.z];
            const float3 n = cross(b - a, c - a);
            const double area = 0.5 * static_cast<double>(length(n));
            if (area <= DEME_TINY_FLOAT) {
                continue;
            }
            area_total += area;
            mx += area * static_cast<double>(a.x + b.x + c.x) / 3.0;
            my += area * static_cast<double>(a.y + b.y + c.y) / 3.0;
            mz += area * static_cast<double>(a.z + b.z + c.z) / 3.0;
        }

        if (!(area_total > 0.0) || !std::isfinite(area_total)) {
            volume = 0.0;
            center = make_float3(0, 0, 0);
            inertia = make_float3(0, 0, 0);
            inertia_products = make_float3(0, 0, 0);
            return;
        }

        const double cx = mx / area_total;
        const double cy = my / area_total;
        const double cz = mz / area_total;
        const double vol = area_total * thickness;

        double Ixx = 0.0;
        double Iyy = 0.0;
        double Izz = 0.0;
        double Ixy = 0.0;
        double Iyz = 0.0;
        double Izx = 0.0;

        for (const auto& face : m_face_v_indices) {
            const float3& a = m_vertices[face.x];
            const float3& b = m_vertices[face.y];
            const float3& c = m_vertices[face.z];

            const float3 n_vec = cross(b - a, c - a);
            const double n_len = static_cast<double>(length(n_vec));
            const double area = 0.5 * n_len;
            if (area <= DEME_TINY_FLOAT) {
                continue;
            }

            const double ax = static_cast<double>(a.x), ay = static_cast<double>(a.y), az = static_cast<double>(a.z);
            const double bx = static_cast<double>(b.x), by = static_cast<double>(b.y), bz = static_cast<double>(b.z);
            const double cxv = static_cast<double>(c.x), cyv = static_cast<double>(c.y), czv = static_cast<double>(c.z);

            const double int_xx = area * ((ax * ax + bx * bx + cxv * cxv) + (ax * bx + ax * cxv + bx * cxv)) / 6.0;
            const double int_yy = area * ((ay * ay + by * by + cyv * cyv) + (ay * by + ay * cyv + by * cyv)) / 6.0;
            const double int_zz = area * ((az * az + bz * bz + czv * czv) + (az * bz + az * czv + bz * czv)) / 6.0;
            const double int_xy = area * ((ax * ay + bx * by + cxv * cyv) / 6.0 +
                                          (ax * by + bx * ay + ax * cyv + cxv * ay + bx * cyv + cxv * by) / 12.0);
            const double int_yz = area * ((ay * az + by * bz + cyv * czv) / 6.0 +
                                          (ay * bz + by * az + ay * czv + cyv * az + by * czv + cyv * bz) / 12.0);
            const double int_zx = area * ((az * ax + bz * bx + czv * cxv) / 6.0 +
                                          (az * bx + bz * ax + az * cxv + czv * ax + bz * cxv + czv * bx) / 12.0);

            // Mid-surface lamina contribution with areal density = thickness (unit volumetric density).
            Ixx += thickness * (int_yy + int_zz);
            Iyy += thickness * (int_xx + int_zz);
            Izz += thickness * (int_xx + int_yy);
            Ixy -= thickness * int_xy;
            Iyz -= thickness * int_yz;
            Izx -= thickness * int_zx;

            // Through-thickness contribution for centered extrusion along facet normal.
            const double nx = static_cast<double>(n_vec.x) / n_len;
            const double ny = static_cast<double>(n_vec.y) / n_len;
            const double nz = static_cast<double>(n_vec.z) / n_len;
            const double coeff = area * thickness * thickness * thickness / 12.0;
            Ixx += coeff * (1.0 - nx * nx);
            Iyy += coeff * (1.0 - ny * ny);
            Izz += coeff * (1.0 - nz * nz);
            Ixy += coeff * (-nx * ny);
            Iyz += coeff * (-ny * nz);
            Izx += coeff * (-nz * nx);
        }

        // Shift inertia tensor from origin to center of mass.
        Ixx -= vol * (cy * cy + cz * cz);
        Iyy -= vol * (cx * cx + cz * cz);
        Izz -= vol * (cx * cx + cy * cy);
        Ixy += vol * cx * cy;
        Iyz += vol * cy * cz;
        Izx += vol * cz * cx;

        volume = vol;
        center = make_float3(static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(cz));
        inertia = make_float3(static_cast<float>(Ixx), static_cast<float>(Iyy), static_cast<float>(Izz));
        inertia_products = make_float3(static_cast<float>(Ixy), static_cast<float>(Iyz), static_cast<float>(Izx));
        return;
    }

    double vol = 0.0;
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    double ix2 = 0.0;
    double iy2 = 0.0;
    double iz2 = 0.0;
    double ixy = 0.0;
    double iyz = 0.0;
    double izx = 0.0;

    for (const auto& face : m_face_v_indices) {
        const float3& a = m_vertices[face.x];
        const float3& b = m_vertices[face.y];
        const float3& c = m_vertices[face.z];

        const float3 bcross = cross(b, c);
        const double v = static_cast<double>(dot(a, bcross)) / 6.0;

        vol += v;
        mx += v * (static_cast<double>(a.x) + b.x + c.x) / 4.0;
        my += v * (static_cast<double>(a.y) + b.y + c.y) / 4.0;
        mz += v * (static_cast<double>(a.z) + b.z + c.z) / 4.0;

        const double ax = a.x, ay = a.y, az = a.z;
        const double bx = b.x, by = b.y, bz = b.z;
        const double cx = c.x, cy = c.y, cz = c.z;

        const double f1x = ax * ax + bx * bx + cx * cx + ax * bx + bx * cx + cx * ax;
        const double f1y = ay * ay + by * by + cy * cy + ay * by + by * cy + cy * ay;
        const double f1z = az * az + bz * bz + cz * cz + az * bz + bz * cz + cz * az;

        ix2 += v * f1x / 10.0;
        iy2 += v * f1y / 10.0;
        iz2 += v * f1z / 10.0;

        const double fxy =
            2.0 * (ax * ay + bx * by + cx * cy) + (ax * by + ay * bx + bx * cy + by * cx + cx * ay + cy * ax);
        const double fyz =
            2.0 * (ay * az + by * bz + cy * cz) + (ay * bz + az * by + by * cz + bz * cy + cy * az + cz * ay);
        const double fzx =
            2.0 * (az * ax + bz * bx + cz * cx) + (az * bx + ax * bz + bz * cx + bx * cz + cz * ax + cx * az);

        ixy += v * fxy / 20.0;
        iyz += v * fyz / 20.0;
        izx += v * fzx / 20.0;
    }

    if (vol == 0.0) {
        volume = 0.0;
        center = make_float3(0, 0, 0);
        inertia = make_float3(0, 0, 0);
        inertia_products = make_float3(0, 0, 0);
        return;
    }

    if (vol < 0.0) {
        vol = -vol;
        mx = -mx;
        my = -my;
        mz = -mz;
        ix2 = -ix2;
        iy2 = -iy2;
        iz2 = -iz2;
        ixy = -ixy;
        iyz = -iyz;
        izx = -izx;
    }

    const double cx = mx / vol;
    const double cy = my / vol;
    const double cz = mz / vol;

    double Ixx = iy2 + iz2;
    double Iyy = ix2 + iz2;
    double Izz = ix2 + iy2;
    double Ixy = -ixy;
    double Iyz = -iyz;
    double Izx = -izx;

    // Shift to center of mass.
    Ixx -= vol * (cy * cy + cz * cz);
    Iyy -= vol * (cx * cx + cz * cz);
    Izz -= vol * (cx * cx + cy * cy);
    Ixy += vol * cx * cy;
    Iyz += vol * cy * cz;
    Izx += vol * cz * cx;

    volume = vol;
    center = make_float3(static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(cz));
    inertia = make_float3(static_cast<float>(Ixx), static_cast<float>(Iyy), static_cast<float>(Izz));
    inertia_products = make_float3(static_cast<float>(Ixy), static_cast<float>(Iyz), static_cast<float>(Izx));
}

// Section for Watertight test, false if not

bool DEMMesh::IsWatertight(size_t* boundary_edges, size_t* nonmanifold_edges) const {
    if (boundary_edges)
        *boundary_edges = 0;
    if (nonmanifold_edges)
        *nonmanifold_edges = 0;
    if (m_face_v_indices.empty())
        return true;

    auto count_edges_by_index = [&](size_t& boundary, size_t& nonmanifold) {
        std::map<std::pair<size_t, size_t>, size_t> edge_counts;

        for (const auto& face : m_face_v_indices) {
            const int fx = face.x, fy = face.y, fz = face.z;
            if (fx < 0 || fy < 0 || fz < 0)
                continue;

            const size_t a = static_cast<size_t>(fx);
            const size_t b = static_cast<size_t>(fy);
            const size_t c = static_cast<size_t>(fz);
            if (a == b || b == c || c == a)
                continue;

            std::pair<size_t, size_t> edges[3] = {
                {std::min(a, b), std::max(a, b)}, {std::min(b, c), std::max(b, c)}, {std::min(c, a), std::max(c, a)}};
            edge_counts[edges[0]]++;
            edge_counts[edges[1]]++;
            edge_counts[edges[2]]++;
        }

        boundary = 0;
        nonmanifold = 0;
        for (const auto& kv : edge_counts) {
            if (kv.second == 1)
                boundary++;
            else if (kv.second > 2)
                nonmanifold++;
        }
    };

    size_t boundary1 = 0, nonmanifold1 = 0;
    count_edges_by_index(boundary1, nonmanifold1);

    if (boundary1 == 0 && nonmanifold1 == 0) {
        if (boundary_edges)
            *boundary_edges = 0;
        if (nonmanifold_edges)
            *nonmanifold_edges = 0;
        return true;
    }

    if (m_vertices.empty()) {
        if (boundary_edges)
            *boundary_edges = boundary1;
        if (nonmanifold_edges)
            *nonmanifold_edges = nonmanifold1;
        return false;
    }

    const double eps = computeVertexQuantEps(m_vertices);
    const auto canon = buildCanonicalVertexMap(m_vertices, eps);

    std::map<std::pair<size_t, size_t>, size_t> edge_counts2;
    for (const auto& face : m_face_v_indices) {
        const int fx = face.x, fy = face.y, fz = face.z;
        if (fx < 0 || fy < 0 || fz < 0)
            continue;

        const size_t a0 = static_cast<size_t>(fx);
        const size_t b0 = static_cast<size_t>(fy);
        const size_t c0 = static_cast<size_t>(fz);
        if (a0 >= canon.size() || b0 >= canon.size() || c0 >= canon.size())
            continue;

        const size_t a = canon[a0];
        const size_t b = canon[b0];
        const size_t c = canon[c0];
        if (a == b || b == c || c == a)
            continue;

        std::pair<size_t, size_t> edges[3] = {
            {std::min(a, b), std::max(a, b)}, {std::min(b, c), std::max(b, c)}, {std::min(c, a), std::max(c, a)}};
        edge_counts2[edges[0]]++;
        edge_counts2[edges[1]]++;
        edge_counts2[edges[2]]++;
    }

    size_t boundary2 = 0, nonmanifold2 = 0;
    for (const auto& kv : edge_counts2) {
        if (kv.second == 1)
            boundary2++;
        else if (kv.second > 2)
            nonmanifold2++;
    }

    if (boundary_edges)
        *boundary_edges = boundary2;
    if (nonmanifold_edges)
        *nonmanifold_edges = nonmanifold2;
    return boundary2 == 0 && nonmanifold2 == 0;
}

}  // end namespace deme

namespace deme {

bool loadMeshByExtension(DEMMesh& mesh, const std::string& filename, bool load_normals, bool load_uv) {
    std::string ext = std::filesystem::path(filename).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    if (ext == ".stl") {
        return mesh.LoadSTLMesh(filename, load_normals);
    }
    if (ext == ".ply") {
        return mesh.LoadPLYMesh(filename, load_normals);
    }
    // Default to OBJ/Wavefront path
    return mesh.LoadWavefrontMesh(filename, load_normals, load_uv);
}

namespace {

struct EdgeInfo {
    size_t tri = 0;
    int edge = 0;
};

inline uint64_t makeEdgeKey(int a, int b) {
    const uint32_t lo = static_cast<uint32_t>(std::min(a, b));
    const uint32_t hi = static_cast<uint32_t>(std::max(a, b));
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
}

}  // namespace

std::vector<std::array<bodyID_t, 3>> buildTriangleEdgeNeighbors(const std::vector<int3>& face_v_indices,
                                                                const std::vector<float3>& vertices) {
    const size_t n_faces = face_v_indices.size();
    std::vector<std::array<bodyID_t, 3>> neighbors(n_faces, {NULL_BODYID, NULL_BODYID, NULL_BODYID});
    if (n_faces == 0) {
        return neighbors;
    }

    std::vector<size_t> canon;
    if (!vertices.empty()) {
        const double eps = computeVertexQuantEps(vertices);
        canon = buildCanonicalVertexMap(vertices, eps);
    }

    std::unordered_map<uint64_t, std::vector<EdgeInfo>> edge_map;
    edge_map.reserve(n_faces * 3);

    for (size_t i = 0; i < n_faces; ++i) {
        const int3& face = face_v_indices[i];
        const int v0_raw = face.x;
        const int v1_raw = face.y;
        const int v2_raw = face.z;
        if (v0_raw < 0 || v1_raw < 0 || v2_raw < 0) {
            continue;
        }
        int v0 = v0_raw;
        int v1 = v1_raw;
        int v2 = v2_raw;
        if (!canon.empty()) {
            if (static_cast<size_t>(v0_raw) >= canon.size() || static_cast<size_t>(v1_raw) >= canon.size() ||
                static_cast<size_t>(v2_raw) >= canon.size()) {
                continue;
            }
            v0 = static_cast<int>(canon[static_cast<size_t>(v0_raw)]);
            v1 = static_cast<int>(canon[static_cast<size_t>(v1_raw)]);
            v2 = static_cast<int>(canon[static_cast<size_t>(v2_raw)]);
        }
        if (v0 == v1 || v1 == v2 || v2 == v0) {
            continue;
        }
        const uint64_t e0 = makeEdgeKey(v0, v1);
        const uint64_t e1 = makeEdgeKey(v1, v2);
        const uint64_t e2 = makeEdgeKey(v2, v0);
        edge_map[e0].push_back(EdgeInfo{i, 0});
        edge_map[e1].push_back(EdgeInfo{i, 1});
        edge_map[e2].push_back(EdgeInfo{i, 2});
    }

    for (const auto& entry : edge_map) {
        const auto& info = entry.second;
        if (info.size() == 2) {
            const EdgeInfo& a = info[0];
            const EdgeInfo& b = info[1];
            neighbors[a.tri][a.edge] = static_cast<bodyID_t>(b.tri);
            neighbors[b.tri][b.edge] = static_cast<bodyID_t>(a.tri);
        }
    }

    return neighbors;
}

namespace {

inline float4 normalizeQuatSafe(const float4& q) {
    const double n2 = static_cast<double>(q.x) * q.x + static_cast<double>(q.y) * q.y + static_cast<double>(q.z) * q.z +
                      static_cast<double>(q.w) * q.w;
    if (!(n2 > 0.0) || !std::isfinite(n2)) {
        return make_float4(0, 0, 0, 1);
    }
    const float inv_n = static_cast<float>(1.0 / std::sqrt(n2));
    return make_float4(q.x * inv_n, q.y * inv_n, q.z * inv_n, q.w * inv_n);
}

inline bool inferUniformScale(const float3& target, const float3& source, double& out_scale) {
    constexpr double tiny = 1e-20;
    const double src[3] = {source.x, source.y, source.z};
    const double dst[3] = {target.x, target.y, target.z};
    double ratios[3] = {0.0, 0.0, 0.0};
    int n = 0;
    for (int i = 0; i < 3; i++) {
        if (!std::isfinite(src[i]) || !std::isfinite(dst[i])) {
            return false;
        }
        if (std::abs(src[i]) <= tiny) {
            return false;
        }
        const double r = dst[i] / src[i];
        if (!(r > tiny) || !std::isfinite(r)) {
            return false;
        }
        ratios[n++] = r;
    }
    if (n != 3) {
        return false;
    }
    const double mean = (ratios[0] + ratios[1] + ratios[2]) / 3.0;
    const double denom = std::max(std::abs(mean), tiny);
    double max_rel_dev = 0.0;
    for (int i = 0; i < 3; i++) {
        max_rel_dev = std::max(max_rel_dev, std::abs(ratios[i] - mean) / denom);
    }
    if (max_rel_dev > 2e-2) {
        return false;
    }
    out_scale = mean;
    return true;
}

inline bool jacobiEigenSymmetric3(const double in_A[3][3], double eigvals[3], double eigvecs[3][3]) {
    double A[3][3];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            A[r][c] = in_A[r][c];
            eigvecs[r][c] = (r == c) ? 1.0 : 0.0;
        }
    }

    constexpr int max_iters = 32;
    constexpr double eps = 1e-18;
    for (int it = 0; it < max_iters; it++) {
        int p = 0;
        int q = 1;
        double max_off = std::abs(A[0][1]);
        const double off_02 = std::abs(A[0][2]);
        const double off_12 = std::abs(A[1][2]);
        if (off_02 > max_off) {
            p = 0;
            q = 2;
            max_off = off_02;
        }
        if (off_12 > max_off) {
            p = 1;
            q = 2;
            max_off = off_12;
        }

        const double diag_scale = std::abs(A[0][0]) + std::abs(A[1][1]) + std::abs(A[2][2]) + eps;
        if (max_off <= diag_scale * 1e-14) {
            break;
        }

        const double app = A[p][p];
        const double aqq = A[q][q];
        const double apq = A[p][q];
        if (std::abs(apq) <= eps) {
            continue;
        }

        const double tau = (aqq - app) / (2.0 * apq);
        const double t =
            (tau >= 0.0) ? (1.0 / (tau + std::sqrt(1.0 + tau * tau))) : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        for (int k = 0; k < 3; k++) {
            if (k == p || k == q) {
                continue;
            }
            const double aik = A[k][p];
            const double akq = A[k][q];
            A[k][p] = c * aik - s * akq;
            A[p][k] = A[k][p];
            A[k][q] = c * akq + s * aik;
            A[q][k] = A[k][q];
        }

        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p][q] = 0.0;
        A[q][p] = 0.0;

        for (int k = 0; k < 3; k++) {
            const double vkp = eigvecs[k][p];
            const double vkq = eigvecs[k][q];
            eigvecs[k][p] = c * vkp - s * vkq;
            eigvecs[k][q] = s * vkp + c * vkq;
        }
    }

    eigvals[0] = A[0][0];
    eigvals[1] = A[1][1];
    eigvals[2] = A[2][2];
    return std::isfinite(eigvals[0]) && std::isfinite(eigvals[1]) && std::isfinite(eigvals[2]);
}

inline float4 quatFromRotationMatrix(const double R[3][3]) {
    float4 q = make_float4(0, 0, 0, 1);
    const double trace = R[0][0] + R[1][1] + R[2][2];
    if (trace > 0.0) {
        const double s = std::sqrt(trace + 1.0) * 2.0;
        q.w = static_cast<float>(0.25 * s);
        q.x = static_cast<float>((R[2][1] - R[1][2]) / s);
        q.y = static_cast<float>((R[0][2] - R[2][0]) / s);
        q.z = static_cast<float>((R[1][0] - R[0][1]) / s);
    } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
        const double s = std::sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0;
        q.w = static_cast<float>((R[2][1] - R[1][2]) / s);
        q.x = static_cast<float>(0.25 * s);
        q.y = static_cast<float>((R[0][1] + R[1][0]) / s);
        q.z = static_cast<float>((R[0][2] + R[2][0]) / s);
    } else if (R[1][1] > R[2][2]) {
        const double s = std::sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0;
        q.w = static_cast<float>((R[0][2] - R[2][0]) / s);
        q.x = static_cast<float>((R[0][1] + R[1][0]) / s);
        q.y = static_cast<float>(0.25 * s);
        q.z = static_cast<float>((R[1][2] + R[2][1]) / s);
    } else {
        const double s = std::sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0;
        q.w = static_cast<float>((R[1][0] - R[0][1]) / s);
        q.x = static_cast<float>((R[0][2] + R[2][0]) / s);
        q.y = static_cast<float>((R[1][2] + R[2][1]) / s);
        q.z = static_cast<float>(0.25 * s);
    }
    return normalizeQuatSafe(q);
}

}  // namespace

MeshCanonicalizationResult canonicalizeMeshOwnerFrame(DEMMesh& mesh) {
    MeshCanonicalizationResult result;

    double volume = 0.0;
    float3 center = make_float3(0, 0, 0);
    float3 inertia = make_float3(0, 0, 0);
    float3 inertia_products = make_float3(0, 0, 0);
    mesh.ComputeMassProperties(volume, center, inertia, inertia_products);
    if (!(volume > 0.0) || !std::isfinite(volume)) {
        return result;
    }

    result.center_norm = std::sqrt(center.x * center.x + center.y * center.y + center.z * center.z);
    const double diag_norm = std::abs(inertia.x) + std::abs(inertia.y) + std::abs(inertia.z);
    const double offdiag_norm = std::sqrt(static_cast<double>(inertia_products.x) * inertia_products.x +
                                          static_cast<double>(inertia_products.y) * inertia_products.y +
                                          static_cast<double>(inertia_products.z) * inertia_products.z);
    result.offdiag_rel = offdiag_norm / std::max(diag_norm, 1e-20);

    float bound_r = 0.f;
    for (const auto& v : mesh.m_vertices) {
        bound_r = std::max(bound_r, length(v));
    }
    const double center_tol = std::max(1e-7, 1e-5 * std::max(1e-3, static_cast<double>(bound_r)));
    constexpr double offdiag_tol = 1e-6;
    if (result.center_norm <= center_tol && result.offdiag_rel <= offdiag_tol) {
        return result;
    }

    double A[3][3] = {
        {inertia.x, inertia_products.x, inertia_products.z},
        {inertia_products.x, inertia.y, inertia_products.y},
        {inertia_products.z, inertia_products.y, inertia.z},
    };
    double eigvals[3] = {0.0, 0.0, 0.0};
    double eigvecs[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    if (!jacobiEigenSymmetric3(A, eigvals, eigvecs)) {
        return result;
    }

    std::array<int, 3> order = {0, 1, 2};
    std::sort(order.begin(), order.end(), [&](int l, int r) { return eigvals[l] < eigvals[r]; });

    double R[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    for (int col = 0; col < 3; col++) {
        const int src_col = order[col];
        double nrm = 0.0;
        for (int row = 0; row < 3; row++) {
            R[row][col] = eigvecs[row][src_col];
            nrm += R[row][col] * R[row][col];
        }
        nrm = std::sqrt(std::max(nrm, 1e-30));
        for (int row = 0; row < 3; row++) {
            R[row][col] /= nrm;
        }
    }

    const double det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
                       R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
                       R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
    if (det < 0.0) {
        for (int row = 0; row < 3; row++) {
            R[row][2] = -R[row][2];
        }
    }

    const float4 principal_q = quatFromRotationMatrix(R);
    float3 principal_inertia = make_float3(static_cast<float>(std::max(0.0, eigvals[order[0]])),
                                           static_cast<float>(std::max(0.0, eigvals[order[1]])),
                                           static_cast<float>(std::max(0.0, eigvals[order[2]])));

    const float3 old_init_pos = mesh.init_pos;
    const float4 old_init_ori = mesh.init_oriQ;

    mesh.InformCentroidPrincipal(center, principal_q);
    if (mesh.patch_locations_explicitly_set) {
        for (auto& p_loc : mesh.m_patch_locations) {
            applyFrameTransformGlobalToLocal(p_loc, center, principal_q);
        }
    }

    float3 new_init_pos = center;
    applyFrameTransformLocalToGlobal(new_init_pos, old_init_pos, old_init_ori);
    mesh.init_pos = new_init_pos;
    mesh.init_oriQ = normalizeQuatSafe(hostHamiltonProduct(old_init_ori, principal_q));

    double scale_from_old = 0.0;
    double scale_from_principal = 0.0;
    const bool moi_from_old = inferUniformScale(mesh.MOI, inertia, scale_from_old);
    const bool moi_from_principal = inferUniformScale(mesh.MOI, principal_inertia, scale_from_principal);
    if (moi_from_old && !moi_from_principal) {
        mesh.MOI = principal_inertia * static_cast<float>(scale_from_old);
        result.moi_rescaled = true;
    } else if (!moi_from_principal && result.offdiag_rel > 1e-4) {
        result.moi_inconsistent = true;
    }

    result.transformed = true;
    return result;
}

double overlapDuration(double a0, double a1, double b0, double b1) {
    const double lo = std::max(a0, b0);
    const double hi = std::min(a1, b1);
    return (hi > lo) ? (hi - lo) : 0.0;
}

bool hasPendingWear(const std::vector<float>& pending_depth) {
    for (float d : pending_depth) {
        if (d > 0.f && std::isfinite(d)) {
            return true;
        }
    }
    return false;
}

}  // namespace deme
