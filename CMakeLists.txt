# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# 
#	SPDX-License-Identifier: BSD-3-Clause

# ---------------------------------------------------------------------------- #
# CMake Project Settings
# ---------------------------------------------------------------------------- #

cmake_minimum_required(VERSION 3.18)

# VERSION is the authoritative release version for CMake and Python packaging.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" DEME_VERSION LIMIT_COUNT 1)
string(STRIP "${DEME_VERSION}" DEME_VERSION)
if(NOT DEME_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
	message(FATAL_ERROR "VERSION must contain a three-component semantic version; found '${DEME_VERSION}'")
endif()

project(
	Chrono-DEM-Engine
	VERSION ${DEME_VERSION}
	LANGUAGES CXX CUDA
)

# Preserve the established names consumed by generated API-version headers.
set(DEME_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(DEME_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(DEME_VERSION_PATCH ${PROJECT_VERSION_PATCH})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting CMAKE_BUILD_TYPE to 'Release' (default)")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# Mark compile time (not runtime), affects how some kernel files include
add_compile_definitions(DEME_BEING_CMAKE_COMPILED)

# ---------------------------------------------------------------------------- #
# Additional Packages
# ---------------------------------------------------------------------------- #

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

include(cmake/FixNinjaColors.cmake)
fix_ninja_colors()


find_package(CUDAToolkit REQUIRED)

# Some CUDA manylinux images expose the versioned NVRTC shared object without the unversioned development symlink.
# FindCUDAToolkit then reports the toolkit as found but omits CUDA::nvrtc. Recover the standard target from the
# versioned library so source and wheel builds use the same link interface.
if(NOT TARGET CUDA::nvrtc)
	find_file(
		DEME_NVRTC_LIBRARY
		NAMES libnvrtc.so libnvrtc.so.${CUDAToolkit_VERSION_MAJOR}
		HINTS
			${CUDAToolkit_LIBRARY_DIR}
			${CUDAToolkit_TARGET_DIR}/lib
			${CUDAToolkit_TARGET_DIR}/lib64
	)
	if(NOT DEME_NVRTC_LIBRARY)
		message(FATAL_ERROR
			"CUDA Toolkit ${CUDAToolkit_VERSION} was found at ${CUDAToolkit_TARGET_DIR}, but its NVRTC shared "
			"library was not found. Install the CUDA NVRTC runtime and development packages."
		)
	endif()
	add_library(CUDA::nvrtc SHARED IMPORTED)
	set_target_properties(CUDA::nvrtc PROPERTIES
		IMPORTED_LOCATION "${DEME_NVRTC_LIBRARY}"
		INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}"
	)
	message(STATUS "Recovered CUDA::nvrtc from ${DEME_NVRTC_LIBRARY}")
endif()

# Option to build Python bindings
option(DEME_BUILD_PYTHON "Build Python bindings for DEME" OFF)
option(DEME_BUILD_DEMOS "Build DEME demos and modular tests" ON)
option(DEME_BUILD_DOCS "Add the Doxygen and Sphinx documentation target" OFF)
option(DEME_BUILD_VISUALIZER "Build the native OpenGL/ImGui visualizer" ON)

# Find Python and pybind11 if building Python bindings
if(DEME_BUILD_PYTHON)
	# Extension modules need the target interpreter and its module headers, but not the embedding library. Manylinux
	# deliberately omits Development.Embed, so requesting the aggregate Development component breaks wheel builds.
	find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
	add_subdirectory(thirdparty/pybind11)
	message(STATUS "Python bindings enabled")
	message(STATUS "Python version: ${Python_VERSION}")
	message(STATUS "Python executable: ${Python_EXECUTABLE}")
endif()

# Find CUB library (this might need to be done in source-level config)
find_package(
	CUB REQUIRED
	HINTS ${CUDAToolkit_ROOT}/lib64/cmake/cub
)

# Find NVIDIA's Jitify library 
find_path(
	NVIDIAJitifyPath
	NAMES jitify.hpp
	PATHS "${CMAKE_CURRENT_LIST_DIR}/thirdparty/jitify"
)

# ---------------------------------------------------------------------------- #
# Global Configuration
# ---------------------------------------------------------------------------- #

include(cmake/CxxStdAutodetect.cmake)

# The compiler will build against this C++ standard, if available.
set(TargetCXXStandard "STD_AUTODETECT" CACHE STRING "The C++ standard used by the compiler")
set_property(
	CACHE TargetCXXStandard
	PROPERTY
	STRINGS STD_AUTODETECT STD_CXX11 STD_CXX14 STD_CXX17 STD_CXX20
)

# Convert the standard into something CMake will understand
if(TargetCXXStandard STREQUAL STD_CXX11)
	set(CXXSTD_MAX 11)
elseif(TargetCXXStandard STREQUAL STD_CXX14)
	set(CXXSTD_MAX 14)
elseif(TargetCXXStandard STREQUAL STD_CXX17)
	set(CXXSTD_MAX 17)
elseif(TargetCXXStandard STREQUAL STD_CXX20)
	set(CXXSTD_MAX 20)
else()
	set(CXXSTD_MAX 17)
endif()
cxx_std_autodetect()

# Allow the use of #include <> for project headers (allowing cleaner better relative pathing)
set(ProjectIncludeSource "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(ProjectIncludeGenerated "${CMAKE_BINARY_DIR}/src")

# Global fix for CUDA language bug
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


#------------------------------------------------------------
# Install destinations for data and demo programs
#------------------------------------------------------------

set(DEME_INSTALL_DEMO "bin")

# ---------------------------------------------------------------------------- #
# Source-level configuration
# ---------------------------------------------------------------------------- #

add_subdirectory(src/core)
add_subdirectory(src/algorithms)
if(DEME_BUILD_VISUALIZER)
	# Build only raylib's static graphics library. Audio and upstream examples are unrelated to DEME visualization.
	set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
	set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
	set(CUSTOMIZE_BUILD ON CACHE BOOL "" FORCE)
	set(USE_AUDIO OFF CACHE BOOL "" FORCE)
	set(SUPPORT_MODULE_RAUDIO OFF CACHE BOOL "" FORCE)
	set(WITH_PIC ON CACHE BOOL "" FORCE)
	add_subdirectory(thirdparty/raylib EXCLUDE_FROM_ALL)
	add_subdirectory(src/visualization)
endif()
add_subdirectory(src/DEM)

# ---------------------------------------------------------------------------- #
# Final Library Generation
# ---------------------------------------------------------------------------- #

add_library(
	simulator_multi_gpu 
	STATIC
	$<TARGET_OBJECTS:core>
	$<TARGET_OBJECTS:DEM>
	$<TARGET_OBJECTS:algorithms>
)

# Extract include directories from the object bundles
get_target_property(CORE_INTERFACE core INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(DEM_INTERFACE DEM INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(ALGORITHMS_INTERFACE algorithms INTERFACE_INCLUDE_DIRECTORIES)

# All targets use the same include directories, so we can cheat and just export one of them
target_include_directories(simulator_multi_gpu 
	PUBLIC ${CORE_INTERFACE}
)

# Native targets run against the synchronized data and kernels in the build tree, even when the Python extension is
# enabled in the same configuration. The extension links its Python-specific helper directly in src/DEM/CMakeLists.txt.
set(DEME_LINK_LIBS
	CUDA::cudart
	CUDA::nvrtc
	CUDA::cuda_driver
	DEMERuntimeDataHelper
)

target_link_libraries(simulator_multi_gpu
	PUBLIC ${DEME_LINK_LIBS}
)
# Specific to Windows...
if(WIN32)
	target_link_libraries(simulator_multi_gpu 
		PUBLIC dbghelp)
endif()

# Attach include directories to the top-level library target
set_target_properties(simulator_multi_gpu
	PROPERTIES
	LINKER_LANGUAGE CUDA
)

if(DEME_BUILD_VISUALIZER)
	# Keep graphics dependencies out of the solver library. Only its tiny callback bridge belongs in the archive.
	# Linking the static solver into both the executable and this shared facade duplicates JIT global destructors.
	target_sources(simulator_multi_gpu PRIVATE $<TARGET_OBJECTS:visualization_bridge>)
	add_library(DEMEVisualizer SHARED $<TARGET_OBJECTS:visualization>)
	target_link_libraries(DEMEVisualizer PRIVATE raylib)
	target_link_libraries(DEMEVisualizer INTERFACE simulator_multi_gpu)
	target_include_directories(DEMEVisualizer PUBLIC
		$<BUILD_INTERFACE:${ProjectIncludeSource}>
		$<BUILD_INTERFACE:${ProjectIncludeGenerated}>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	)
	set_target_properties(DEMEVisualizer PROPERTIES CXX_STANDARD ${CXXSTD_SUPPORTED} WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

# ---------------------------------------------------------------------------- #
# Export and Install The Generated Targets 
# ---------------------------------------------------------------------------- #

set(config_directory ${CMAKE_INSTALL_LIBDIR}/cmake/DEME)

# Install the library, creating an export target for it as well 
install(
	TARGETS simulator_multi_gpu DEMERuntimeDataHelper
	EXPORT DEMETargets
	LIBRARY
		DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Python wheels embed the visualizer object code in ``_deme`` and must not also install the standalone C++ facade.
# That facade links the source/install runtime helper used by native CMake consumers, while Python uses its own static
# runtime-path helper. Keeping the native facade out of the wheel prevents an invalid build-tree helper dependency.
set(DEME_INSTALL_VISUALIZER ${DEME_BUILD_VISUALIZER})
if(SKBUILD)
	set(DEME_INSTALL_VISUALIZER OFF)
endif()

if(DEME_INSTALL_VISUALIZER)
	install(
		TARGETS DEMEVisualizer
		EXPORT DEMETargets
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)
endif()

install(CODE "message(\"Removing build-tree libraries from installation target...\")")
install(CODE "file(REMOVE \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libDEMERuntimeDataHelper.so\")")

install(
	TARGETS DEMERuntimeDataHelper_install
	EXPORT DEMETargets
	LIBRARY
		DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Export the generated library target
export(EXPORT DEMETargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/DEME/DEMETargets.cmake"
	NAMESPACE DEME::
)

# Set the install location for the exported target file 
install(
	EXPORT DEMETargets
	DESTINATION 
		${config_directory}
	NAMESPACE DEME::
)


# These macros aid in the generation of the installable config files 


# Generate the **Build Tree** version of DEMEConfig.cmake from the base file
set(CONF_INCLUDE_DIRS "${ProjectIncludeSource}" "${ProjectIncludeGenerated}")

configure_file(cmake/DEMEConfig.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/DEME/DEMEConfig.cmake"
	@ONLY
)


# Generate the _Install Tree_ version of DEMEConfig.cmake from the base file
file(RELATIVE_PATH REL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/${config_directory} ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
set(CONF_INCLUDE_DIRS "\${DEMECMakeDir}/${REL_INCLUDE_DIR}")
set(CONF_SHARE_DATA_DIRS "\${DEMECMakeDir}/${REL_INCLUDE_DIR}/../share/DEME/data")

# NOTE: This is placed in CMAKE_FILES_DIRECTORY to keep it out of the build tree
configure_file(cmake/DEMEConfig.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DEMEConfig.cmake"
	@ONLY
)


# Automatically generate a DEMEConfigVersion.cmake file

write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/DEME/DEMEConfigVersion.cmake
	VERSION ${DEME_VERSION_MAJOR}.${DEME_VERSION_MINOR}.${DEME_VERSION_PATCH}
	COMPATIBILITY AnyNewerVersion
)

# Install the generated config files
install(
	FILES
		"${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DEMEConfig.cmake"
		"${CMAKE_CURRENT_BINARY_DIR}/DEME/DEMEConfigVersion.cmake"
	DESTINATION
		${config_directory}
)

# ---------------------------------------------------------------------------- #
# Copy data directory over
# ---------------------------------------------------------------------------- #
file(COPY ${CMAKE_CURRENT_LIST_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/data/)
# Keep build-tree data in sync (so newly added meshes like STL assets are always available without re-running CMake)
add_custom_target(sync_data ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_LIST_DIR}/data
            ${CMAKE_BINARY_DIR}/data
    COMMENT "Syncing data directory into build tree")
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/data/ DESTINATION ${CMAKE_INSTALL_DATADIR}/DEME/data/)

# ---------------------------------------------------------------------------- #
# Copy kernels over
# ---------------------------------------------------------------------------- #
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/kernel/ DESTINATION ${CMAKE_BINARY_DIR}/kernel/)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/DEM/Defines.h DESTINATION ${CMAKE_BINARY_DIR}/DEM)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/DEM/VariableTypes.h DESTINATION ${CMAKE_BINARY_DIR}/DEM)
add_custom_target(sync_kernels ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_LIST_DIR}/src/kernel
            ${CMAKE_BINARY_DIR}/kernel
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_LIST_DIR}/src/DEM/Defines.h
            ${CMAKE_BINARY_DIR}/DEM/Defines.h
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_LIST_DIR}/src/DEM/VariableTypes.h
            ${CMAKE_BINARY_DIR}/DEM/VariableTypes.h
    COMMENT "Syncing kernel sources into build tree")
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/kernel/ DESTINATION ${CMAKE_INSTALL_DATADIR}/DEME/kernel/)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/kernel/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kernel/
    FILES_MATCHING
        PATTERN "*.cuh"
        PATTERN "*.h"
)

# ---------------------------------------------------------------------------- #
# Build embedded demos
# ---------------------------------------------------------------------------- #
if(DEME_BUILD_DEMOS)
	add_subdirectory(src/demo)
endif()

if(DEME_BUILD_DOCS)
	add_subdirectory(docs)
endif()
