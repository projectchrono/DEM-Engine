//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_PATHS_HPP
#define DEME_PATHS_HPP

#include <filesystem>
#include <cstring>

namespace deme {

// -----------------------------------------------------------------------------
// Functions for manipulating the DEME data directory
// -----------------------------------------------------------------------------

static std::filesystem::path DEME_data_path("../data/");

// Set the path to the DEME data directory (ATTENTION: not thread safe)
void SetDEMEDataPath(const std::string& path);

// Obtain the current path to the DEME data directory (thread safe)
const std::filesystem::path& GetDEMEDataPath();

// Obtain the complete path to the specified filename, given relative to the
// DEME data directory (thread safe)
std::string GetDEMEDataFile(const std::string& filename);

}  // namespace deme

#endif
