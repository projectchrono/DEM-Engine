//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_PATHS_HPP
#define DEME_PATHS_HPP

#include <filesystem>
#include <cstring>

#include <core/ApiVersion.h>
#include <core/utils/RuntimeData.h>

namespace deme {

// -----------------------------------------------------------------------------
// Functions for manipulating the DEME data directory
// -----------------------------------------------------------------------------

// Build or installation data path
static std::filesystem::path DEME_data_path(RuntimeDataHelper::data_path / "data");

// Set the path to the DEME data directory (ATTENTION: not thread safe)
void SetDEMEDataPath(const std::string& path);

// Set the path to the DEME data kernel directory
void SetDEMEKernelPath(const std::filesystem::path& kernel_path);

// Set the path to the DEME include directory
void SetDEMEIncludePath(const std::filesystem::path& include_path);

// Obtain the current path to the DEME data directory (thread safe)
std::filesystem::path& GetDEMEDataPath();

// Obtain the complete path to the specified filename, given relative to the
// DEME data directory (thread safe)
std::string GetDEMEDataFile(const std::string& filename);

}  // namespace deme

#endif
