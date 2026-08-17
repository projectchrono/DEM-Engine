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

if(USE_HIP)
	target_link_libraries(
		algorithms
		PUBLIC hip::hipcub
	)
else()
	target_link_libraries(
		algorithms
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

if(USE_HIP)
	# Mark .cu sources as HIP language
	set_source_files_properties(${algorithms_sources} PROPERTIES LANGUAGE HIP)

	set_target_properties(
		algorithms PROPERTIES
		POSITION_INDEPENDENT_CODE True
		CXX_STANDARD ${CXXSTD_SUPPORTED}
		PUBLIC_HEADER "${algorithms_interface}"
		HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
	)
else()
	# Get a list of supported architectures
	include(../../cmake/CudaSupportedArchitectures.cmake)
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
endif()

install(
	TARGETS algorithms
	PUBLIC_HEADER
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/algorithms
)
