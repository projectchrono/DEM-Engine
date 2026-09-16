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

# Prefer DEME's Jitify variant, which adds the optional persistent CUDA-header cache. Keep the submodule path as a
# fallback so source layouts that intentionally omit the vendored variant still configure with upstream Jitify.
set(DEMEJitifyHeader "${ProjectIncludeSource}/jitify/jitify.hpp")
if(NOT EXISTS "${DEMEJitifyHeader}")
	set(DEMEJitifyHeader "${NVIDIAJitifyPath}/jitify.hpp")
endif()

message(STATUS "${core_message} Extracting Jitify header...")
configure_file(
	${DEMEJitifyHeader}
	${CMAKE_BINARY_DIR}/src/jitify/jitify.hpp
	COPYONLY
)

add_library(core OBJECT)

# Capture CUDA include dirs from whichever variable is populated
set(_DEME_CUDA_INC_DIRS "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
if(NOT _DEME_CUDA_INC_DIRS AND DEFINED CUDAToolkit_INCLUDE_DIRS)
    set(_DEME_CUDA_INC_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
endif()
if(NOT _DEME_CUDA_INC_DIRS AND DEFINED CUDA_INCLUDE_DIRS)
    set(_DEME_CUDA_INC_DIRS "${CUDA_INCLUDE_DIRS}")
endif()

target_compile_definitions(core PRIVATE
    DEME_CUDA_TOOLKIT_INCLUDE_DIRS="$<JOIN:${_DEME_CUDA_INC_DIRS},;>"
)


target_include_directories(
	core
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeSource}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeGenerated}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


target_link_libraries(
	core
	PUBLIC CUB::CUB
)


set(core_headers
	${CMAKE_BINARY_DIR}/src/core/ApiVersion.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/BaseClasses.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/CudaAllocator.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/CudaDebugSync.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/JitHelper.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ThreadManager.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/Logger.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuManager.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/WavefrontMeshLoader.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/csv.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/Timer.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/DataMigrationHelper.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/DeviceDataTransfer.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/DEMEPaths.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/RuntimeData.h
)

set(core_sources
	${CMAKE_CURRENT_SOURCE_DIR}/DebugInfo.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuManager.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/DeviceDataTransfer.cpp
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


# The Python extension needs import-time CUDA header configuration. Compile its core separately so
# this hook and its state are absent from native C++ libraries, even when both targets are built.
if(DEME_BUILD_PYTHON)
    add_library(core_python OBJECT)
    target_sources(core_python PRIVATE ${core_sources})
    target_include_directories(core_python PUBLIC
        $<BUILD_INTERFACE:${ProjectIncludeSource}>
        $<BUILD_INTERFACE:${ProjectIncludeGenerated}>
    )
    target_link_libraries(core_python PUBLIC CUB::CUB)
    target_compile_definitions(core_python PRIVATE
        DEME_PYTHON_JIT
        DEME_CUDA_TOOLKIT_INCLUDE_DIRS="$<JOIN:${_DEME_CUDA_INC_DIRS},;>"
    )
    set_target_properties(core_python PROPERTIES
        POSITION_INDEPENDENT_CODE True
        CXX_STANDARD ${CXXSTD_SUPPORTED}
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN True
    )
endif()

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
	FILES 
		"${DEMEJitifyHeader}"
	DESTINATION 
		${CMAKE_INSTALL_INCLUDEDIR}/jitify
)

# --------------------------------------------------------- #
# Runtime data path helper library 
# --------------------------------------------------------- #


message(STATUS "${core_message} Generating Runtime Data helper (build)...")
set(DEME_RUNTIME_DATA_DIRECTORY "${CMAKE_BINARY_DIR}")
set(DEME_RUNTIME_INCLUDE_DIRECTORY "${CMAKE_BINARY_DIR}")
configure_file(
	utils/RuntimeData.cpp.in utils/RuntimeDataBuild.cpp
	@ONLY
)

message(STATUS "${core_message} Generating Runtime Data helper (install)...")
set(DEME_RUNTIME_DATA_DIRECTORY "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/DEME")
set(DEME_RUNTIME_INCLUDE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
configure_file(
	utils/RuntimeData.cpp.in utils/RuntimeDataInstall.cpp
	@ONLY
)


# Build tree version

add_library(DEMERuntimeDataHelper SHARED)

# Windows specific: Define DLL export macros
if(WIN32)
    target_compile_definitions(DEMERuntimeDataHelper PRIVATE DEMERuntimeDataHelper_EXPORTS)
endif()


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

if(WIN32)
    target_compile_definitions(DEMERuntimeDataHelper_install PRIVATE DEMERuntimeDataHelper_EXPORTS)
endif()

target_sources(DEMERuntimeDataHelper_install
	PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils/RuntimeData.h> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/core/utils/RuntimeData.h>
	PRIVATE "${CMAKE_BINARY_DIR}/src/core/utils/RuntimeDataInstall.cpp" 
)


if(WIN32)
    # Windows-specific linker options
    # You can add options here if needed. Often, you don't need to specify the DLL name as in Unix.
    target_link_libraries(DEMERuntimeDataHelper_install PRIVATE DEMERuntimeDataHelper)
else()
    # Unix-like OS specific linker options
    target_link_options(DEMERuntimeDataHelper_install
        PUBLIC "LINKER:-soname,libDEMERuntimeDataHelper.so"
    )
endif()



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

# Python version (for Python bindings)
if(DEME_BUILD_PYTHON)
	add_library(DEMERuntimeDataHelper_python STATIC)
	# The Python module calls SetPathPrefix during import, but retain an install-layout fallback if that discovery fails.
	set(DEME_RUNTIME_DATA_DIRECTORY "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/DEME")
	set(DEME_RUNTIME_INCLUDE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
	configure_file(utils/RuntimeData.cpp.in utils/RuntimeDataPython.cpp)

	target_sources(DEMERuntimeDataHelper_python
		PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils/RuntimeData.h> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/core/utils/RuntimeData.h>
		PRIVATE "${CMAKE_BINARY_DIR}/src/core/utils/RuntimeDataPython.cpp"
	)

	target_include_directories(
		DEMERuntimeDataHelper_python
		PUBLIC $<BUILD_INTERFACE:${ProjectIncludeSource}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	)

	set_target_properties(
		DEMERuntimeDataHelper_python PROPERTIES
		POSITION_INDEPENDENT_CODE True
		CXX_STANDARD "${CXXSTD_SUPPORTED}"
	)
endif()
