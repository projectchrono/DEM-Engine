//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <filesystem>
#include <cstring>

#include "DEMEPaths.h"
#include "JitHelper.h"

namespace deme {

// -----------------------------------------------------------------------------
// Functions for manipulating the DEME data directory
// -----------------------------------------------------------------------------

// Set the path to the DEME data directory (usually reserved for the solver's own usage)
void SetDEMEDataPath() {
    DEME_data_path = DEMERuntimeDataHelper::data_path / "data";
}

// Obtain the current path to the DEME data directory (thread safe)
std::filesystem::path& GetDEMEDataPath() {
    return DEME_data_path;
}

// Obtain the complete path to the specified filename, given relative to the
// DEME data directory (thread safe)
std::string GetDEMEDataFile(const std::string& filename) {
    return (DEME_data_path / filename).string();
}

// Set the path to the DEME data kernel directory (usually reserved for the solver's own usage)
void SetDEMEKernelPath() {
    JitHelper::KERNEL_DIR = DEMERuntimeDataHelper::data_path / "kernel";
}

// Set the path to the DEME include directory (usually reserved for the solver's own usage)
void SetDEMEIncludePath() {
    JitHelper::KERNEL_INCLUDE_DIR = DEMERuntimeDataHelper::include_path;
}

}  // namespace deme
