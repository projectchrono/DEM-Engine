//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <DEM/utils/MeshLoadUtils.hpp>

#include <DEM/BdrsAndObjs.h>

#include <algorithm>
#include <cctype>
#include <filesystem>

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

}  // namespace deme
