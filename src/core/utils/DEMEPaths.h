//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_PATHS_HPP
#define DEME_PATHS_HPP

#include <filesystem>
#include <cstring>

#include "RuntimeData.h"

namespace deme {

// -----------------------------------------------------------------------------
// Functions for manipulating the DEME data directory
// -----------------------------------------------------------------------------

// Build or installation data path
const std::filesystem::path BUILD_DATA_PATH = DEMERuntimeDataHelper::data_path / "data";
static std::filesystem::path DEME_data_path(BUILD_DATA_PATH);

// Set the path to the DEME data directory (ATTENTION: not thread safe)
void SetDEMEDataPath();
// Obtain the current path to the DEME data directory (thread safe)
std::filesystem::path& GetDEMEDataPath();

// Set the path to the DEME data kernel directory
void SetDEMEKernelPath();

// Set the path to the DEME include directory
void SetDEMEIncludePath();

// Obtain the complete path to the specified filename, given relative to the
// DEME data directory (thread safe)
std::string GetDEMEDataFile(const std::string& filename);

}  // namespace deme

#endif
