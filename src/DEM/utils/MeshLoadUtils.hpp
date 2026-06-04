//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

namespace deme {

class DEMMesh;

bool loadMeshByExtension(DEMMesh& mesh, const std::string& filename, bool load_normals, bool load_uv);

}  // namespace deme
