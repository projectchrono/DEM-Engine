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

#endif
