# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

message(STATUS "==== Configuring Core Components ====")

set (core_message "[core]")
message(STATUS "${core_message} Generating API version header...")
configure_file(
	ApiVersion.h.in ApiVersion.h
	#	USE_SOURCE_PERMISSIONS ## This doesn't work in CMake <3.20
	@ONLY
)

add_library(core OBJECT)


target_include_directories(
	core
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeSource}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeGenerated}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	PUBLIC $<BUILD_INTERFACE:${NVIDIAJitifyPath}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	PUBLIC $<BUILD_INTERFACE:${NVIDIAMathDir}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
	core
	PUBLIC CUB::CUB
)

set(core_headers
	${CMAKE_BINARY_DIR}/src/core/ApiVersion.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ManagedAllocator.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ManagedMemory.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/JitHelper.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ThreadManager.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuError.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuManager.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/chpf/particle_writer.hpp
)

set(core_sources
	${CMAKE_CURRENT_SOURCE_DIR}/DebugInfo.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuManager.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/JitHelper.cpp
)

target_sources(
	core
	PUBLIC ${core_headers} ${core_utils_headers}
	PRIVATE ${core_sources}
)

set_target_properties(
	core PROPERTIES 
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${CXXSTD_SUPPORTED}
	PUBLIC_HEADER "${core_headers}"
)

install(
	DIRECTORY ${ProjectIncludeSource}/core
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
		FILES_MATCHING 
			PATTERN "*.h"
			PATTERN "*.hpp"
			PATTERN "chpf/docs" EXCLUDE
)

install(
	FILES 
		"${CMAKE_BINARY_DIR}/src/core/ApiVersion.h"
	DESTINATION
		${CMAKE_INSTALL_INCLUDEDIR}/core
)

