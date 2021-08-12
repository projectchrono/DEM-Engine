//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <fstream>
#include <filesystem>
#include <string>

#include "JitHelper.h"

JitHelper::Header::Header(std::filesystem::path& sourcepath) {
	
	// If the file exists, read in the entire thing.
	if (std::filesystem::exists(sourcepath)) {
		std::ifstream input(sourcepath);
		std::getline(input, this->_source, std::string::traits_type::to_char_type(
			std::string::traits_type::eof()
		));
	}

}

const std::string& JitHelper::Header::getSource() {
	return _source;
}

// TODO: This should substitute constant values for certain symbols known at runtime
void JitHelper::Header::substitute(const std::string& symbol, const std::string& value) {
	// ...
}
