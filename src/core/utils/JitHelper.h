//	Copyright (c) 2021, SBEL GPU Development Team
//	Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_JIT_HELPER_H
#define DEME_JIT_HELPER_H

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

#include "JitKernel.h"

#if defined(_WIN32) || defined(_WIN64)
    #undef max
    #undef min
    #undef strtok_r
#endif

class JitHelper {
  public:
    class Header {
      public:
        Header(const std::filesystem::path& sourcefile);
        const std::string& getSource();
        void substitute(const std::string& symbol, const std::string& value);

      private:
        std::string _source;
    };

    static deme::jit::Program buildProgram(
        const std::string& name,
        const std::filesystem::path& source,
        std::unordered_map<std::string, std::string> substitutions = std::unordered_map<std::string, std::string>(),
        std::vector<std::string> flags = std::vector<std::string>());

    static std::filesystem::path KERNEL_DIR;
    static std::filesystem::path KERNEL_INCLUDE_DIR;

  private:
    static deme::jit::ProgramCache* kcache;

    inline static std::string loadSourceFile(const std::filesystem::path& sourcefile) {
        std::string code;
        // If the file exists, read in the entire thing.
        if (std::filesystem::exists(sourcefile)) {
            std::ifstream input(sourcefile);
            std::getline(input, code, std::string::traits_type::to_char_type(std::string::traits_type::eof()));
        }
        return code;
    };
};

#endif
