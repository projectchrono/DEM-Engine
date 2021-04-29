# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

message(STATUS "==== Configuring Core Components ====")

set (core_message "[core]")
message(STATUS "${core_message} Generating API version header...")

configure_file(
	ApiVer.h.in ApiVer.h
	USE_SOURCE_PERMISSIONS
	@ONLY
)

add_library(core OBJECT)

target_include_directories(
	core
	PUBLIC ${ProjectIncludeRoot}
)

set(core_headers
	${CMAKE_CURRENT_SOURCE_DIR}/ApiVer.h
)

set(core_sources
	# Add source files here
)

target_sources(
	core
	INTERFACE ${core_headers}
	PRIVATE ${core_sources}
)

