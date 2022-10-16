//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <filesystem>
#include <cstring>
#include "DEMEPaths.h"

namespace deme {

// -----------------------------------------------------------------------------
// Functions for manipulating the DEME data directory
// -----------------------------------------------------------------------------

// Set the path to the DEME data directory (ATTENTION: not thread safe)
void SetDEMEDataPath(const std::string& path) {
    DEME_data_path = path;
}

// Obtain the current path to the DEME data directory (thread safe)
const std::filesystem::path& GetDEMEDataPath() {
    return DEME_data_path;
}

// Obtain the complete path to the specified filename, given relative to the
// DEME data directory (thread safe)
std::string GetDEMEDataFile(const std::string& filename) {
    return (DEME_data_path / filename).string();
}

}  // namespace deme
