# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

message(STATUS "==== Building DEM Components ====")

add_library(DEM OBJECT)

target_include_directories(
	DEM
	PUBLIC ${NVIDIAJitifyPath}
	PUBLIC ${ProjectIncludeRoot}
	PUBLIC ${NVIDIAMathDir}
)

target_link_libraries(
	DEM
	PUBLIC CUB::CUB
)

set(DEM_headers
	${CMAKE_CURRENT_SOURCE_DIR}/kT.h
	${CMAKE_CURRENT_SOURCE_DIR}/dT.h
	${CMAKE_CURRENT_SOURCE_DIR}/ApiSystem.h
	${CMAKE_CURRENT_SOURCE_DIR}/DEMDefines.h
	${CMAKE_CURRENT_SOURCE_DIR}/DEMStructs.h
	${CMAKE_CURRENT_SOURCE_DIR}/DEMModels.h
	${CMAKE_CURRENT_SOURCE_DIR}/VariableTypes.h
	${CMAKE_CURRENT_SOURCE_DIR}/Boundaries.h
	${CMAKE_CURRENT_SOURCE_DIR}/HostSideHelpers.cpp
	${NVIDIAMathDir}/helper_math.cuh
)

set(DEM_sources
	${CMAKE_CURRENT_SOURCE_DIR}/kT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/dT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ApiSystem.cpp
)

target_sources(
	DEM
	INTERFACE ${DEM_headers}
	PRIVATE ${DEM_sources}
)

set_target_properties(
	DEM PROPERTIES
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${CXXSTD_SUPPORTED}
)

