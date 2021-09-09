# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

# ---------------------------------------------------------------------------- #
# CMake Project Settings
# ---------------------------------------------------------------------------- #

cmake_minimum_required(VERSION 3.18)

# Version information
set(VERSION_MAJOR   0)
set(VERSION_MINOR   0)
set(VERSION_PATCH   0)

project(
	SBEL-GPU-Physics 
	VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}
	LANGUAGES CXX CUDA
)

# ---------------------------------------------------------------------------- #
# Additional Packages
# ---------------------------------------------------------------------------- #
include(cmake/FixNinjaColors.cmake)
fix_ninja_colors()

find_package(CUDAToolkit REQUIRED)

# Find CUB library (this might need to be done in source-level config)
find_package(
	CUB REQUIRED
	PATHS ${CMAKE_CURRENT_LIST_DIR}/thirdparty/cub/cub
)

find_path(
	NVIDIAJitifyPath
	NAMES jitify.hpp
	PATHS "${CMAKE_CURRENT_LIST_DIR}/thirdparty/jitify"
)

set(NVIDIAMathDir ${CMAKE_CURRENT_LIST_DIR}/thirdparty/nvidia_helper_math)

# ---------------------------------------------------------------------------- #
# Global Configuration
# ---------------------------------------------------------------------------- #

# The compiler will build against this C++ standard, if available.
set(TargetCXXStandard "STD_AUTODETECT" CACHE STRING "The C++ standard used by the compiler")
set_property(
	CACHE TargetCXXStandard
	PROPERTY
	STRINGS STD_AUTODETECT STD_CXX11 STD_CXX14 STD_CXX17 STD_CXX20
)

# Convert the standard into something CMake will understand
if(TargetCXXStandard STREQUAL STD_CXX11)
	set(GLOBAL_cxxver 11 CACHE INTERNAL "")
elseif(TargetCXXStandard STREQUAL STD_CXX14)
	set(GLOBAL_cxxver 14 CACHE INTERNAL "")
elseif(TargetCXXStandard STREQUAL STD_CXX17)
	set(GLOBAL_cxxver 17 CACHE INTERNAL "")
elseif(TargetCXXStandard STREQUAL STD_CXX20)
	set(GLOBAL_cxxver 20 CACHE INTERNAL "")
else()
	message(SEND_ERROR "STD_AUTODETECT is not yet supported! Set TargetCXXStandard to an appropriate value to continue.")
	# TODO: Perform C++ version autodetection based on the compiler ID
endif()


# Allow the use of #include <> for project headers (allowing cleaner better relative pathing)
set(ProjectIncludeRoot "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/src")

# Global fix for CUDA language bug
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# ---------------------------------------------------------------------------- #
# Source-level configuration
# ---------------------------------------------------------------------------- #

add_subdirectory(src/core)
add_subdirectory(src/granular)
add_subdirectory(src/sph)

# ---------------------------------------------------------------------------- #
# Final Library Generation
# ---------------------------------------------------------------------------- #

add_library(
	sbel_gpu_physics 
	SHARED
	$<TARGET_OBJECTS:core>
	$<TARGET_OBJECTS:granular>
	$<TARGET_OBJECTS:sph>
	)

# Extract include directories from the object bundles
get_target_property(CORE_INTERFACE core INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(GRANULAR_INTERFACE granular INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(SPH_INTERFACE sph INTERFACE_INCLUDE_DIRECTORIES)


# Attach include directories to the top-level library target
set_target_properties(sbel_gpu_physics
	PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES "${CORE_INTERFACE} ${GRANULAR_INTERFACE} ${SPH_INTERFACE}"
	INTERFACE_LINK_LIBRARIES "CUDA::cudart;CUDA::nvrtc"
)
	
add_subdirectory(src/demo)