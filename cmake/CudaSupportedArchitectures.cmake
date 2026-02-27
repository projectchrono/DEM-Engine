# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# 
#	SPDX-License-Identifier: BSD-3-Clause


# The original source code of this file was authored by Colin Vanden Heuvel,
# who also worked on this project, and the upstream version is also available
# under a permissive (Zlib) license.
#
# https://gitlab.com/thepianoboy/cmake-nuggets


# Determine the GPU architectures supported by the active version of NVCC

# Effects:
#
# Populates the cache variable CUDASUP_ARCHITECTURES with a list of numeric
# values representing the compute capabilities supported by the current 
# version of the CUDA Toolkit
#
# Minimum CUDA version: 7.0
# Note: newer CUDA versions are supported as long as NVCC supports
# `--list-gpu-arch` (preferred) or the fallback version map below includes them.

function(cuda_supported_architectures)

	# Prefer querying NVCC directly so we don't have to guess which SMs each
	# CUDA toolkit version supports (this avoids regressions when a machine has
	# a CUDA 12.x toolkit that does *not* yet support the newest architectures).
	if (DEFINED CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_CUDA_COMPILER}")
		execute_process(
			COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-arch
			RESULT_VARIABLE _cudasup_nvcc_res
			OUTPUT_VARIABLE _cudasup_nvcc_out
			ERROR_VARIABLE _cudasup_nvcc_err
			OUTPUT_STRIP_TRAILING_WHITESPACE
		)

		if (_cudasup_nvcc_res EQUAL 0)
			# NVCC prints one `compute_XX` per line; extract the numeric suffixes.
			string(REGEX MATCHALL "compute_([0-9]+)" _cudasup_matches "${_cudasup_nvcc_out}")
			set(_cudasup_archs "")
			foreach(_m ${_cudasup_matches})
				string(REPLACE "compute_" "" _a "${_m}")
				list(APPEND _cudasup_archs "${_a}")
			endforeach()

			list(REMOVE_DUPLICATES _cudasup_archs)
			list(SORT _cudasup_archs COMPARE NATURAL)

			if (_cudasup_archs)
				set(CUDASUP_ARCHITECTURES ${_cudasup_archs} CACHE INTERNAL "")
				return()
			endif()
		endif()
	endif()

	# Supported CUDA compute capabilities by toolkit version
	set(cu7 20 30 35 50 52)
	set(cu8 20 30 35 50 52 60 61)
	set(cu9 30 35 50 52 60 61 70 72)
	set(cu10 30 35 50 52 60 61 70 72 75)
	set(cu11 35 50 52 60 61 70 72 75 80)
	set(cu11_x 35 50 52 60 61 70 72 75 80 86)
	set(cu12_x 50 52 60 61 70 72 75 80 86 89 90)
	set(cu12_8_plus 50 52 60 61 70 72 75 80 86 89 90 100 101 120)
	set(cu13_x 75 80 86 89 90 100 101 120 121)

	if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 7)
		set(CUDASUP_ARCHITECTURES ${cu7} CACHE INTERNAL "")
	endif()

	if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 8)
		set(CUDASUP_ARCHITECTURES ${cu8} CACHE INTERNAL "")
	endif()

	if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
		set(CUDASUP_ARCHITECTURES ${cu9} CACHE INTERNAL "")
	endif()

	if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
		set(CUDASUP_ARCHITECTURES ${cu10} CACHE INTERNAL "")
	endif()

	if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
		set(CUDASUP_ARCHITECTURES ${cu11} CACHE INTERNAL "")
	endif()

	if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
		set(CUDASUP_ARCHITECTURES ${cu11_x} CACHE INTERNAL "")
	endif()

	if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
		set(CUDASUP_ARCHITECTURES ${cu12_x} CACHE INTERNAL "")
	endif()

	if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
		set(CUDASUP_ARCHITECTURES ${cu12_8_plus} CACHE INTERNAL "")
	endif()
	
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
        set(CUDASUP_ARCHITECTURES ${cu13_x} CACHE INTERNAL "")
    endif()

	if (NOT DEFINED CUDASUP_ARCHITECTURES)
		message(SEND_ERROR "[CUDASUP] Could not determine device architectures supported by the CUDA toolkit!")
	endif()

endfunction()	
