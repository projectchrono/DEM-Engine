# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

message(STATUS "==== Building Granular Components ====")

add_library(granular OBJECT)

target_include_directories(
	granular
	PUBLIC ${ProjectIncludeRoot}
)

target_link_libraries(
	granular
	PUBLIC CUB::CUB
)

set(granular_headers
	${CMAKE_CURRENT_SOURCE_DIR}/PhysicsSystem.h
	${CMAKE_CURRENT_SOURCE_DIR}/ApiSystem.h
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

