# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

message(STATUS "==== Configuring GPU Algorithm Components ====")

set (algos_message "[algos]")



add_library(algos OBJECT)

target_include_directories(
	algos	
	PUBLIC ${ProjectIncludeRoot}
	PUBLIC ${NVIDIAJitifyPath}
	PUBLIC ${NVIDIAMathDir}
)

target_link_libraries(
	algos
	PUBLIC CUB::CUB
)

set(algos_headers
	# ${CMAKE_CURRENT_SOURCE_DIR}/example.cuh
	${NVIDIAMathDir}/helper_math.h
)

set(algos_sources
	${CMAKE_CURRENT_SOURCE_DIR}/example.cu # REPLACE ME WITH A REAL FILE
)

target_sources(
	algos	
	INTERFACE ${algos_headers}
	PRIVATE ${algos_sources}
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
	algos PROPERTIES 
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${CXXSTD_SUPPORTED}
	CUDA_ARCHITECTURES "${virtual_architectures}"
)
