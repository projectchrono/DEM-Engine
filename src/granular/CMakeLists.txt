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
	${CMAKE_CURRENT_SOURCE_DIR}/PhysicsSystem.h
	${CMAKE_CURRENT_SOURCE_DIR}/ApiSystem.h
	${CMAKE_CURRENT_SOURCE_DIR}/GranularDefines.h
	${NVIDIAMathDir}/helper_math.h
)

set(granular_sources
	${CMAKE_CURRENT_SOURCE_DIR}/PhysicsSystem.cpp
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
	CXX_STANDARD ${GLOBAL_cxxver}
)

