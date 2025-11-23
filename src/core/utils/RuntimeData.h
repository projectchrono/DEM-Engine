//  Copyright (c) 2022, SBEL GPU Development Team
//  Copyright (c) 2022, University of Wisconsin - Madison
//
//  SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_RUNTIMEDATA_H
#define DEME_RUNTIMEDATA_H

#include <filesystem>

#if defined(_WIN32) || defined(_WIN64)
    #ifdef DEMERuntimeDataHelper_EXPORTS
        #define DEMERuntimeDataHelper_API __declspec(dllexport)
    #else
        #define DEMERuntimeDataHelper_API __declspec(dllimport)
    #endif
#else
    #define DEMERuntimeDataHelper_API
#endif

class DEMERuntimeDataHelper_API DEMERuntimeDataHelper {
  public:
    static std::filesystem::path data_path;
    static std::filesystem::path include_path;

    // or std::string
    static void SetPathPrefix(const std::filesystem::path& p);
};

#endif
