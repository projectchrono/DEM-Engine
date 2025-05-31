# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# 
#	SPDX-License-Identifier: BSD-3-Clause

message(STATUS "==== Configuring GPU Algorithm Components ====")

set (algorithms_message "[algorithms]")

add_library(algorithms OBJECT)

target_include_directories(
	algorithms	
	PUBLIC ${ProjectIncludeSource}
	PUBLIC ${ProjectIncludeGenerated}
)

if(NOT PYTHON_BUILD)
	target_link_libraries(
		algorithms
		PUBLIC CUB::CUB
	)
else()
	target_link_libraries(
		algorithms
		PUBLIC DEMERuntimeDataHelper_python
		PUBLIC CUB::CUB
	)
endif()
### HOST HEADERS ONLY (.h, .hpp) ###
set(algorithms_interface
	${CMAKE_CURRENT_SOURCE_DIR}/DEMStaticDeviceSubroutines.h
)

### INTERNAL HEADERS ONLY (.h, .hpp, or .cuh) ###
set(algorithms_headers
	${CMAKE_CURRENT_SOURCE_DIR}/DEMStaticDeviceUtilities.cuh
)

set(algorithms_sources
	${CMAKE_CURRENT_SOURCE_DIR}/DEMCubWrappers.cu 
	${CMAKE_CURRENT_SOURCE_DIR}/DEMCubForceCollection.cu
	${CMAKE_CURRENT_SOURCE_DIR}/DEMCubInstantiations.cu
	${CMAKE_CURRENT_SOURCE_DIR}/DEMCubContactDetection.cu
	${CMAKE_CURRENT_SOURCE_DIR}/DEMDynamicMisc.cu
)

target_sources(
	algorithms
	PUBLIC ${algorithms_interface}
	INTERFACE ${algorithms_headers}
	PRIVATE ${algorithms_sources}
)

# Get a list of supported architectures
include(${CMAKE_SOURCE_DIR}/cmake/CudaSupportedArchitectures.cmake)
cuda_supported_architectures()

# Convert the cuda supported architectures into their respective CMake properties
set(virtual_architectures "")
foreach(arch ${CUDASUP_ARCHITECTURES})
	list(APPEND virtual_architectures "${arch}-virtual")
endforeach()

set_target_properties(
	algorithms PROPERTIES 
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${CXXSTD_SUPPORTED}
	PUBLIC_HEADER "${algorithms_interface}"
	CUDA_ARCHITECTURES "${virtual_architectures}"
)

install(
	TARGETS algorithms
	PUBLIC_HEADER
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/algorithms
)
