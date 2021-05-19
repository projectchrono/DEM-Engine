# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

message(STATUS "==== Configuring Core Components ====")

set (core_message "[core]")
message(STATUS "${core_message} Generating API version header...")

configure_file(
	ApiVersion.h.in ApiVersion.h
	USE_SOURCE_PERMISSIONS
	@ONLY
)

add_library(core OBJECT)

target_include_directories(
	core
	PUBLIC ${ProjectIncludeRoot}
)

target_link_libraries(
	core
	PUBLIC CUB::CUB
)

set(core_headers
	${CMAKE_CURRENT_SOURCE_DIR}/ApiVersion.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ManagedAllocator.hpp
)

set(core_sources
	${CMAKE_CURRENT_SOURCE_DIR}/DebugInfo.cpp
)

target_sources(
	core
	INTERFACE ${core_headers}
	PRIVATE ${core_sources}
)

