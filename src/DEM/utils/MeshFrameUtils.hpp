//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace deme {

class DEMMesh;

struct MeshCanonicalizationResult {
    bool transformed = false;
    bool moi_rescaled = false;
    bool moi_inconsistent = false;
    double center_norm = 0.0;
    double offdiag_rel = 0.0;
};

MeshCanonicalizationResult canonicalizeMeshOwnerFrame(DEMMesh& mesh);

}  // namespace deme
