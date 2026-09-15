// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_PYTHON_CUDA_INCLUDES_HPP
#define DEME_PYTHON_CUDA_INCLUDES_HPP

#include <filesystem>
#include <vector>

namespace deme {
namespace python {

// Import configures this extension-local state before any solver workers start.
// Only the Python core variant includes this helper; native C++ uses its original toolkit discovery.
inline std::vector<std::filesystem::path>& CudaIncludePaths() {
    static std::vector<std::filesystem::path> paths;
    return paths;
}

}  // namespace python
}  // namespace deme

#endif
