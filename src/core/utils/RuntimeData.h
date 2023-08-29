//  Copyright (c) 2022, SBEL GPU Development Team
//  Copyright (c) 2022, University of Wisconsin - Madison
//
//  SPDX-License-Identifier: BSD-3-Clause

#ifndef RUNTIMEDATA_H
#define RUNTIMEDATA_H

#include <filesystem>

class RuntimeDataHelper {
  public:
    static std::filesystem::path data_path;
    static std::filesystem::path include_path;

    // or std::string
    static void SetPathPrefix(const std::filesystem::path& p);
};

#endif
