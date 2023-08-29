//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <filesystem>
#include <cstring>
#include "DEMEPaths.h"
#include <iostream>
#include <core/utils/JitHelper.h>

namespace deme {

// -----------------------------------------------------------------------------
// Functions for manipulating the DEME data directory
// -----------------------------------------------------------------------------

// Set the path to the DEME data directory (ATTENTION: not thread safe)
void SetDEMEDataPath(const std::string& path) {
    DEME_data_path = path;
}

// Obtain the current path to the DEME data directory (thread safe)
std::filesystem::path& GetDEMEDataPath() {
    return DEME_data_path;
}

// Obtain the complete path to the specified filename, given relative to the
// DEME data directory (thread safe)
std::string GetDEMEDataFile(const std::string& filename) {
    return (RuntimeDataHelper::data_path / "data" / filename).string();
}

// Set the path to the DEME data kernel directory
void SetDEMEKernelPath(const std::filesystem::path& kernel_path) {
    JitHelper::KERNEL_DIR = RuntimeDataHelper::data_path / "kernel";
}

// Set the path to the DEME include directory
void SetDEMEIncludePath(const std::filesystem::path& include_path) {
    JitHelper::KERNEL_INCLUDE_DIR = RuntimeDataHelper::include_path;
}

}  // namespace deme
