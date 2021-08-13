//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#ifndef SGPS_JIT_HELPER_H
#define SGPS_JIT_HELPER_H

#include <filesystem>
#include <string>
#include <vector>

#include <jitify.hpp>

class JitHelper {
public:

	class Header {
	public:
		Header(std::filesystem::path& sourcefile);
		const std::string& getSource();
		void substitute(const std::string& symbol, const std::string& value);

	private:
		std::string _source;
	};

	static jitify::Program buildProgram(
		const std::string& name, const std::filesystem::path& source,
		std::vector<Header> headers = std::vector<Header>(),
		std::vector<std::string> flags = std::vector<std::string>()
	);

	//// I'm pretty sure C++17 auto-converts this
	// static jitify::Program buildProgram(
	// 	const std::string& name, const std::string& code,
	// 	std::vector<Header> headers = 0,
	// 	std::vector<std::string> flags = 0
	// );

private:
	static jitify::JitCache kcache;

	inline static std::string loadSourceFile(const std::filesystem::path& sourcefile) {
		std::string code;
		// If the file exists, read in the entire thing.
		if (std::filesystem::exists(sourcefile)) {
			std::ifstream input(sourcefile);
			std::getline(input, code, std::string::traits_type::to_char_type(
				std::string::traits_type::eof()
			));
		}
		return code;
	};

};


#endif
