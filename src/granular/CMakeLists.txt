# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

message(STATUS "==== Building Granular Components ====")

add_library(granular OBJECT)

target_include_directories(
	granular
	PUBLIC ${NVIDIAJitifyPath}
	PUBLIC ${ProjectIncludeRoot}
	PUBLIC ${NVIDIAMathDir}
)

target_link_libraries(
	granular
	PUBLIC CUB::CUB
)

set(granular_headers
	${CMAKE_CURRENT_SOURCE_DIR}/kT.h
	${CMAKE_CURRENT_SOURCE_DIR}/dT.h
	${CMAKE_CURRENT_SOURCE_DIR}/ApiSystem.h
	${CMAKE_CURRENT_SOURCE_DIR}/GranularDefines.h
	${CMAKE_CURRENT_SOURCE_DIR}/DataStructs.h
	${CMAKE_CURRENT_SOURCE_DIR}/GranularStructs.h
	${CMAKE_CURRENT_SOURCE_DIR}/Boundaries.h
	${CMAKE_CURRENT_SOURCE_DIR}/HostSideHelpers.cpp
	${NVIDIAMathDir}/helper_math.cuh
)

set(granular_sources
	${CMAKE_CURRENT_SOURCE_DIR}/kT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/dT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ApiSystem.cpp
)

target_sources(
	granular
	INTERFACE ${granular_headers}
	PRIVATE ${granular_sources}
)

set_target_properties(
	granular PROPERTIES
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${CXXSTD_SUPPORTED}
)

