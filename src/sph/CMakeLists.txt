# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

message(STATUS "==== Building DEM Components ====")

add_library(sph OBJECT)

target_include_directories(
	sph
	PUBLIC ${NVIDIAJitifyPath}
	PUBLIC ${ProjectIncludeSource}
	PUBLIC ${ProjectIncludeGenerated}
	PUBLIC ${NVIDIAMathDir}
)

target_link_libraries(
	sph
	PUBLIC CUB::CUB
)

# .h, .hpp
set(sph_headers
	${CMAKE_CURRENT_SOURCE_DIR}/SPHSystem.h
	${CMAKE_CURRENT_SOURCE_DIR}/datastruct.h
)

# .c, .cpp, .cu
set(sph_sources
	${CMAKE_CURRENT_SOURCE_DIR}/SPHSystem.cpp
)

target_sources(
	sph
	PUBLIC ${sph_headers}
	PRIVATE ${sph_sources}
)

set_target_properties(
	sph PROPERTIES
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${CXXSTD_SUPPORTED}
)

