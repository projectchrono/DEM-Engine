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

string(REPLACE "." ";" DEME_VERSION_COMPONENTS "${DEME_VERSION}")
list(GET DEME_VERSION_COMPONENTS 0 DEME_VERSION_MAJOR)
list(GET DEME_VERSION_COMPONENTS 1 DEME_VERSION_MINOR)
list(GET DEME_VERSION_COMPONENTS 2 DEME_VERSION_PATCH)

# HIP/ROCm support option (must be set before project() to select the right language)
option(USE_HIP "Build with HIP for AMD GPUs" OFF)

if(USE_HIP)
	project(
		Chrono-DEM-Engine
		VERSION ${DEME_VERSION_MAJOR}.${DEME_VERSION_MINOR}.${DEME_VERSION_PATCH}
		LANGUAGES CXX HIP
	)
	# The LANGUAGES ... HIP enable above auto-detects the host GPU arch (and
	# errors on a no-GPU host); pass -DCMAKE_HIP_ARCHITECTURES to override.
	add_compile_definitions(USE_HIP)
	# CXX files that include hip_runtime.h need this defined (HIP compiler defines it automatically)
	add_compile_definitions(__HIP_PLATFORM_AMD__)
else()
	project(
		Chrono-DEM-Engine
		VERSION ${DEME_VERSION_MAJOR}.${DEME_VERSION_MINOR}.${DEME_VERSION_PATCH}
		LANGUAGES CXX CUDA
	)
endif()

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
include(FetchContent)

include(cmake/FixNinjaColors.cmake)
fix_ninja_colors()

option(DEME_BUILD_PYTHON "Build Python bindings for DEME" OFF)
option(DEME_BUILD_DEMOS "Build DEME demos and modular tests" ON)

if(DEME_BUILD_PYTHON)
	if(USE_HIP)
		message(FATAL_ERROR "Python wheels currently support the CUDA backend only")
	endif()
	find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
	if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pybind11/CMakeLists.txt")
		add_subdirectory(thirdparty/pybind11 EXCLUDE_FROM_ALL)
	else()
		find_package(pybind11 CONFIG REQUIRED)
	endif()
	message(STATUS "Python bindings enabled for Python ${Python_VERSION}")
endif()


if(USE_HIP)
	# Find hipCUB for HIP builds
	find_package(hipcub REQUIRED)
	find_package(hip REQUIRED)
else()
	find_package(CUDAToolkit REQUIRED)
	# Find CUB library (this might need to be done in source-level config)
	find_package(
		CUB REQUIRED
		HINTS ${CUDAToolkit_ROOT}/lib64/cmake/cub
	)
endif()

# Find Jitify library (NVIDIA version only; HIP uses hiprtc directly via JitKernel abstraction)
if(NOT USE_HIP)
	find_path(
		JitifyPath
		NAMES jitify.hpp
		PATHS "${CMAKE_CURRENT_LIST_DIR}/thirdparty/jitify"
	)
	# Keep backward compat alias
	set(NVIDIAJitifyPath ${JitifyPath})
endif()

# Let the user decide if they want to use ChPF
option(USE_CHPF "Toggle the use of ChPF for outputting" OFF)

if(USE_CHPF)
	# Find ChPF, else fetch it 
    find_package(ChPF 3.0 QUIET)
    if (NOT ChPF_FOUND)         
        FetchContent_Declare(
            ChPF
            GIT_REPOSITORY  https://gitlab.com/uwsbel/chpf.git
            GIT_TAG         0699f01e720cecf58719a1651ab797c6a62a1897
        )
        FetchContent_MakeAvailable(ChPF)
    endif()

	# Based on whether ChPF was found or fetched, set the target name CMake can use to find it 
    if (ChPF_FOUND) 
        set(ChPF_IMPORTED_NAME "ChPF::ChPF")
    elseif (chpf_POPULATED)
        set(ChPF_IMPORTED_NAME "ChPF")
    endif()
endif()

# Let the user decide if they want to use managed arrays, rather than default cudaMalloc and cudaMallocHost memory.
# Note that turning this on gives no performance benefits, and it's considered legacy.
set(USE_MANAGED_ARRAYS_DESC
	"Use managed CUDA arrays (instead of manual CUDA host and device memory) to store data"
	"Switching it on makes the project operable on a Linux platform (excluding WSL) only."
)
option(USE_MANAGED_ARRAYS "${USE_MANAGED_ARRAYS_DESC}" OFF)

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

# Global fix for CUDA/HIP language bug
if(USE_HIP)
	# Add HIP include directories for all targets (both CXX and HIP)
	include_directories(${HIP_INCLUDE_DIR})
else()
	include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()


#------------------------------------------------------------
# Install destinations for data and demo programs
#------------------------------------------------------------

set(DEME_INSTALL_DEMO "bin")

# ---------------------------------------------------------------------------- #
# Source-level configuration
# ---------------------------------------------------------------------------- #

add_subdirectory(src/core)
add_subdirectory(src/DEM)
add_subdirectory(src/algorithms)

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

# If use ChPF, inform the source
if(USE_CHPF)
    target_compile_definitions(simulator_multi_gpu PUBLIC DEME_USE_CHPF)
	set(USE_CHPF_STR "ON")

	if(USE_HIP)
		find_library(HIPRTC_LIB hiprtc HINTS ${ROCM_PATH}/lib /opt/rocm/lib REQUIRED)
		target_link_libraries(simulator_multi_gpu
			PUBLIC hip::host
			PUBLIC ${HIPRTC_LIB}
			PUBLIC ${ChPF_IMPORTED_NAME}
			PUBLIC DEMERuntimeDataHelper
		)
	else()
		target_link_libraries(simulator_multi_gpu
			PUBLIC CUDA::cudart
			PUBLIC CUDA::nvrtc
			PUBLIC CUDA::cuda_driver
			PUBLIC ${ChPF_IMPORTED_NAME}
			PUBLIC DEMERuntimeDataHelper
		)
	endif()
else()
	set(USE_CHPF_STR "OFF")

	if(USE_HIP)
		find_library(HIPRTC_LIB hiprtc HINTS ${ROCM_PATH}/lib /opt/rocm/lib REQUIRED)
		target_link_libraries(simulator_multi_gpu
			PUBLIC hip::host
			PUBLIC ${HIPRTC_LIB}
			PUBLIC DEMERuntimeDataHelper
		)
	else()
		target_link_libraries(simulator_multi_gpu
			PUBLIC CUDA::cudart
			PUBLIC CUDA::nvrtc
			PUBLIC CUDA::cuda_driver
			PUBLIC DEMERuntimeDataHelper
		)
	endif()
endif()

# If use managed arrays, define a macro
if(USE_MANAGED_ARRAYS)
	target_compile_definitions(simulator_multi_gpu PUBLIC DEME_USE_MANAGED_ARRAYS)
endif()

# Specific to Windows...
if(WIN32)
	target_link_libraries(simulator_multi_gpu 
		PUBLIC dbghelp)
endif()

# Attach include directories to the top-level library target
if(USE_HIP)
	set_target_properties(simulator_multi_gpu
		PROPERTIES
		LINKER_LANGUAGE HIP
	)
else()
	set_target_properties(simulator_multi_gpu
		PROPERTIES
		LINKER_LANGUAGE CUDA
	)
endif()

if(DEME_BUILD_PYTHON)
	pybind11_add_module(_deme
		src/DEM/PyDEME.cpp
		$<TARGET_OBJECTS:core>
		$<TARGET_OBJECTS:DEM>
		$<TARGET_OBJECTS:algorithms>
	)
	target_include_directories(_deme PRIVATE ${ProjectIncludeSource} ${ProjectIncludeGenerated})
	target_link_libraries(_deme PRIVATE
		CUDA::cudart
		CUDA::nvrtc
		CUDA::cuda_driver
		DEMERuntimeDataHelper_python
		pybind11::headers
	)
	if(USE_CHPF)
		target_link_libraries(_deme PRIVATE ${ChPF_IMPORTED_NAME})
	endif()
	set_target_properties(_deme PROPERTIES CXX_STANDARD ${CXXSTD_SUPPORTED} POSITION_INDEPENDENT_CODE TRUE)
	install(TARGETS _deme LIBRARY DESTINATION deme COMPONENT python)
	install(FILES src/DEM/python/deme/__init__.py DESTINATION deme COMPONENT python)
	install(FILES src/DEM/python/DEME.py DESTINATION . COMPONENT python)
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
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/data/ DESTINATION ${CMAKE_INSTALL_DATADIR}/DEME/data/)

# ---------------------------------------------------------------------------- #
# Copy kernels over
# ---------------------------------------------------------------------------- #
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/kernel/ DESTINATION ${CMAKE_BINARY_DIR}/kernel/)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/DEM/Defines.h DESTINATION ${CMAKE_BINARY_DIR}/DEM)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/DEM/VariableTypes.h DESTINATION ${CMAKE_BINARY_DIR}/DEM)
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
