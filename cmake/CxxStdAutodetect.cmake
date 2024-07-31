# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# 
#	SPDX-License-Identifier: BSD-3-Clause


# The original source code of this file was authored by Colin Vanden Heuvel,
# who also worked on this project, and the upstream version is also available
# under a permissive (Zlib) license.
#
# https://gitlab.com/thepianoboy/cmake-nuggets

# Determine a suitable C++ standard version for the active compiler 

# Effects: 
#
# Sets the cache variable CXXSTD_SUPPORTED with the numeric portion of the most
# recent ISO C++ standard revision supported by the compiler. Throws an error
# if a suitable version is not detected.
# 
# If the variable CXXSTD_MAX is defined, the script will only select versions
# AT OR BELOW the given maximum.
# 
#    NOTE: The value of CXXSTD_MAX should be no higher than 97 (the default)
#
# If the variable CXXSTD_NO_ERROR is defined, the script will not throw an
# error and will instead generate a warning message.

function(cxx_std_autodetect)

	set(mprefix "[C++STD Autodetect]")	
	message("${mprefix} Detecting level of C++ support...")

	if (NOT DEFINED CXXSTD_MAX)
		set (CXXSTD_MAX 97)
	endif()
		
	foreach (local_cxxver RANGE ${CXXSTD_MAX})
		if ("cxx_std_${local_cxxver}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
			message("${mprefix} Compiler supports C++${local_cxxver}...")
			set(CXXSTD_SUPPORTED ${local_cxxver} CACHE INTERNAL "")
		endif()
	endforeach()

	if (NOT DEFINED CXXSTD_SUPPORTED)
		if ("cxx_std_98" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
			message("${mprefix} Compiler supports C++98...")
			set (CXXSTD_SUPPORTED 98 CACHE INTERNAL "")
		else()
			#if (DEFINED CXXSTD_NO_ERROR)
			#	message(WARNING "${mprefix} Could not determine a suitable C++ standard!")
			#else()
			#	message(SEND_ERROR "${mprefix} Could not determine a suitable C++ standard!")
			#endif()
			message(WARNING "${mprefix} Could not determine a suitable C++ standard! We will try C++17...")
			set (CXXSTD_SUPPORTED 17 CACHE INTERNAL "")
		endif()
	endif()

endfunction()

