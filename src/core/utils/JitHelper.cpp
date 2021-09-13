//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <fstream>
#include <filesystem>
#include <string>

#include <jitify.hpp>

#include <core/ApiVersion.h>
#include <core/utils/JitHelper.h>

jitify::JitCache JitHelper::kcache;

const std::filesystem::path JitHelper::KERNEL_DIR = std::filesystem::path(PROJECT_SOURCE_DIRECTORY) / "src" / "kernel";

JitHelper::Header::Header(std::filesystem::path& sourcefile) {
    this->_source = JitHelper::loadSourceFile(sourcefile);
}

const std::string& JitHelper::Header::getSource() {
    return _source;
}

// TODO: This should substitute constant values for certain symbols known at runtime
void JitHelper::Header::substitute(const std::string& symbol, const std::string& value) {
    // ...
}

jitify::Program JitHelper::buildProgram(const std::string& name,
                                        const std::filesystem::path& source,
                                        std::vector<JitHelper::Header> headers,
                                        std::vector<std::string> flags) {
    std::string code = name + "\n";

    std::vector<std::string> header_code;
    for (auto it = headers.begin(); it != headers.end(); it++) {
        header_code.push_back(it->getSource());
    }

    code.append(JitHelper::loadSourceFile(source));

    return kcache.program(code, header_code, flags);
}
