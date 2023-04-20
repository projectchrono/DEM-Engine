# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# 
#	SPDX-License-Identifier: BSD-3-Clause

message(STATUS "==== Configuring Core Components ====")

set (core_message "[core]")
message(STATUS "${core_message} Generating API version header...")
configure_file(
	ApiVersion.h.in ApiVersion.h
	#	USE_SOURCE_PERMISSIONS ## This doesn't work in CMake <3.20
	@ONLY
)



message(STATUS "${core_message} Extracting NVIDIA Jitify header...") 
configure_file(
	${NVIDIAJitifyPath}/jitify.hpp
	${CMAKE_BINARY_DIR}/src/jitify/jitify.hpp
	COPYONLY
)

add_library(core OBJECT)


target_include_directories(
	core
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeSource}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeGenerated}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	PUBLIC $<BUILD_INTERFACE:${NVIDIAMathDir}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
	core
	PUBLIC CUB::CUB
	INTERFACE ${ChPF_IMPORTED_NAME}
)

set(core_headers
	${CMAKE_BINARY_DIR}/src/core/ApiVersion.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ManagedAllocator.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ManagedMemory.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/JitHelper.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ThreadManager.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuError.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuManager.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/WavefrontMeshLoader.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/csv.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/Timer.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/DEMEPaths.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/RuntimeData.h
)

set(core_sources
	${CMAKE_CURRENT_SOURCE_DIR}/DebugInfo.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuManager.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/JitHelper.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/DEMEPaths.cpp
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

# Install Core Headers
install(
	DIRECTORY ${ProjectIncludeSource}/core
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
		FILES_MATCHING 
			PATTERN "*.h"
			PATTERN "*.hpp"
)

# Install Generated Headers 
install(
	FILES 
		"${CMAKE_BINARY_DIR}/src/core/ApiVersion.h"
	DESTINATION
		${CMAKE_INSTALL_INCLUDEDIR}/core
)

# Install Third-party Headers 
install(
	DIRECTORY ${NVIDIAMathDir}/nvmath
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
	FILES 
		"${NVIDIAJitifyPath}/jitify.hpp"
	DESTINATION 
		${CMAKE_INSTALL_INCLUDEDIR}/jitify
)


# --------------------------------------------------------- #
# Runtime data path helper library 
# --------------------------------------------------------- #


message(STATUS "${core_message} Generating Runtime Data helper (build)...")
set(RUNTIME_DATA_DIRECTORY "${CMAKE_BINARY_DIR}")
set(RUNTIME_INCLUDE_DIRECTORY "${CMAKE_BINARY_DIR}")
configure_file(
	utils/RuntimeData.cpp.in utils/RuntimeDataBuild.cpp
	@ONLY
)

message(STATUS "${core_message} Generating Runtime Data helper (install)...")
set(RUNTIME_DATA_DIRECTORY "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/DEME")
set(RUNTIME_INCLUDE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
configure_file(
	utils/RuntimeData.cpp.in utils/RuntimeDataInstall.cpp
	@ONLY
)


# Build tree version

add_library(DEMERuntimeDataHelper SHARED)

target_sources(DEMERuntimeDataHelper
	PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils/RuntimeData.h> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/core/utils/RuntimeData.h>
	PRIVATE "${CMAKE_BINARY_DIR}/src/core/utils/RuntimeDataBuild.cpp" 
)

target_include_directories(
	DEMERuntimeDataHelper
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeSource}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(
	DEMERuntimeDataHelper PROPERTIES
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD "${CXXSTD_SUPPORTED}"
)

# Install tree version

##
## This MUST be installed after the Build tree version as it overwrites the .so file 
##

add_library(DEMERuntimeDataHelper_install SHARED)

target_sources(DEMERuntimeDataHelper_install
	PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils/RuntimeData.h> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/core/utils/RuntimeData.h>
	PRIVATE "${CMAKE_BINARY_DIR}/src/core/utils/RuntimeDataInstall.cpp" 
)


target_link_options(DEMERuntimeDataHelper_install
	PUBLIC "LINKER:-soname,libDEMERuntimeDataHelper.so"
)

target_include_directories(
	DEMERuntimeDataHelper_install
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeSource}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(
	DEMERuntimeDataHelper_install PROPERTIES
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${CXXSTD_SUPPORTED}
	LIBRARY_OUTPUT_NAME "DEMERuntimeDataHelper"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib_install"
)

# Make sure this target is generated **AFTER** the build-tree version 
add_dependencies(DEMERuntimeDataHelper_install DEMERuntimeDataHelper)

