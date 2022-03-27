# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

message(STATUS "==== Configuring GPU Algorithm Components ====")

set (algorithms_message "[algorithms]")



add_library(algorithms OBJECT)

target_include_directories(
	algorithms	
	PUBLIC ${ProjectIncludeRoot}
	PUBLIC ${NVIDIAJitifyPath}
	PUBLIC ${NVIDIAMathDir}
)

target_link_libraries(
	algorithms
	PUBLIC CUB::CUB
)

### HOST HEADERS ONLY (.h, .hpp) ###
set(algorithms_interface
	${CMAKE_CURRENT_SOURCE_DIR}/example.h # REPLACE ME WITH A REAL FILE
	${CMAKE_CURRENT_SOURCE_DIR}/SPHCubHelperFunctions.h
	${CMAKE_CURRENT_SOURCE_DIR}/DEMCubHelperFunctions.h
)

### INTERNAL HEADERS ONLY (.h, .hpp, or .cuh) ###
set(algorithms_headers
	${NVIDIAMathDir}/helper_math.cuh
)

set(algorithms_sources
	${CMAKE_CURRENT_SOURCE_DIR}/example.cu # REPLACE ME WITH A REAL FILE
	${CMAKE_CURRENT_SOURCE_DIR}/DEMCubContactDetection.cu 
	${CMAKE_CURRENT_SOURCE_DIR}/SPHCubHelperFunctions.cu 
	${CMAKE_CURRENT_SOURCE_DIR}/DEMCubForceCollection.cu
)

target_sources(
	algorithms
	INTERFACE ${algorithms_interface}
	PRIVATE ${algorithms_headers}
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
	CUDA_ARCHITECTURES "${virtual_architectures}"
)
