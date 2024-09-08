# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# 
#	SPDX-License-Identifier: BSD-3-Clause


# The original source code of this file was authored by Colin Vanden Heuvel,
# who also worked on this project, and the upstream version is also available
# under a permissive (Zlib) license.
#
# https://gitlab.com/thepianoboy/cmake-nuggets


# Fix Ninja Colors

# Effects:
#
# Sets the appropriate compiler flag to produce color-highlighted diagnostic
# messages when using the Ninja Generator. Results may vary if the output
# medium is not a terminal.
#
# Supported compilers:
# - GNU C++ Compiler
# - LLVM Clang

function(fix_ninja_colors)

	# Check if override is set
	if(${FORCE_PLAIN_OUTPUT})
		return()
	endif()

	# Check for Ninja
	if(${CMAKE_GENERATOR} STREQUAL "Ninja")

		# Fix GCC colors
		if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
			#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fdiagnostics-color=always" PARENT_SCOPE)
			list(APPEND CMAKE_CXX_FLAGS "-fdiagnostics-color=always")	

		# Fix Clang colors
		elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
			#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fcolor-diagnostics" PARENT_SCOPE)
			list(APPEND CMAKE_CXX_FLAGS "-fcolor-diagnostics")

		endif()
		
		list(REMOVE_DUPLICATES CMAKE_CXX_FLAGS)
		set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} PARENT_SCOPE)
	endif()

endfunction()
