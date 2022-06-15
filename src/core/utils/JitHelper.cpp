//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <fstream>
#include <filesystem>
#include <string>

#include <jitify/jitify.hpp>

#include <core/ApiVersion.h>
#include <core/utils/JitHelper.h>

jitify::JitCache JitHelper::kcache;

const std::filesystem::path JitHelper::KERNEL_DIR = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel";

JitHelper::Header::Header(const std::filesystem::path& sourcefile) {
    this->_source = JitHelper::loadSourceFile(sourcefile);
}

const std::string& JitHelper::Header::getSource() {
    return _source;
}

void JitHelper::Header::substitute(const std::string& symbol, const std::string& value) {
    // find occurrences of `symbol` until there are none left
    for (size_t p = this->_source.find(symbol); p != std::string::npos; p = this->_source.find(symbol)) {
        // Replace this occurrence with the new value
        this->_source.replace(p, symbol.length(), value);
    }
}

jitify::Program JitHelper::buildProgram(
    const std::string& name,
    const std::filesystem::path& source,
    std::unordered_map<std::string, std::string> substitutions,
    // std::vector<JitHelper::Header> headers, // THIS PARAMETER PROBABLY WON'T EVER BE USED
    std::vector<std::string> flags) {
    std::string code = name + "\n";

    // Generate block of substitution macros
    code.append("#ifndef SGPS_KERNEL_SUBSTITUTIONS\n");
    code.append("#define SGPS_KERNEL_SUBSTITUTIONS\n");
    for (auto& subst : substitutions) {
        code.append("#define " + subst.first + " " + subst.second + "\n");
    }
    code.append("#endif\n");

    code.append(JitHelper::loadSourceFile(source));

    std::vector<std::string> header_code;
    // THIS BLOCK IS ONLY NEEDED IF THE headers PARAMETER IS USED
    /*
    for (auto it = headers.begin(); it != headers.end(); it++) {
        header_code.push_back(it->getSource());
    }
    */

    return kcache.program(code, header_code, flags);
}
