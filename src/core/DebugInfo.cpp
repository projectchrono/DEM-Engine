//	Copyright (c) 2021, SBEL GPU Development Team
//	Copyright (c) 2021, University of Wisconsin - Madison
//	
//	SPDX-License-Identifier: BSD-3-Clause

#include <iostream>

#include <core/ApiVersion.h>

namespace deme {

void versionInfo() {
	// Project Info
	std::cout << "SBEL Multi-GPU DEM Solver (c) 2021" << std::endl;
	std::cout << "Project Version: " << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << std::endl;
	
	// C++ Info
	std::cout << "C++ Standard Revision: " << __cplusplus << std::endl;
}

} // namespace deme
