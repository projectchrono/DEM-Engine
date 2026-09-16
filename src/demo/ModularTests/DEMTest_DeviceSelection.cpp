// Copyright (c) 2021, SBEL GPU Development Team
// Copyright (c) 2021, University of Wisconsin - Madison
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime_api.h>

#include <iostream>
#include <vector>

#include "DEM/API.h"
#include "core/utils/Logger.hpp"

using namespace deme;

// Verify that count-based and explicit device selection produce the documented, deterministic worker placement.
int main() {
    int visible_devices = 0;
    DEME_GPU_CALL(cudaGetDeviceCount(&visible_devices));
    if (visible_devices < 1) {
        std::cout << "SKIP: DEMTest_DeviceSelection requires a visible CUDA device." << std::endl;
        return 0;
    }

    {
        DEMSolver solver(1);
        const std::vector<int> devices = solver.GetGPUDeviceIDs();
        if (devices != std::vector<int>({0, 0})) {
            std::cerr << "FAIL: DEMSolver(1) did not place both workers on device 0." << std::endl;
            return 1;
        }
    }

    const int selected_device = visible_devices - 1;
    {
        DEMSolver solver(std::vector<int>{selected_device});
        const std::vector<int> devices = solver.GetGPUDeviceIDs();
        if (devices != std::vector<int>({selected_device, selected_device})) {
            std::cerr << "FAIL: single explicit device selection did not place both workers on the selected device."
                      << std::endl;
            return 1;
        }
    }

    if (visible_devices >= 2) {
        DEMSolver solver(std::vector<int>{1, 0});
        const std::vector<int> devices = solver.GetGPUDeviceIDs();
        if (devices != std::vector<int>({1, 0})) {
            std::cerr << "FAIL: two-device selection did not preserve dT/kT assignment order." << std::endl;
            return 1;
        }
    }

    std::cout << "PASS: DEMSolver GPU device selection is deterministic." << std::endl;
    return 0;
}
