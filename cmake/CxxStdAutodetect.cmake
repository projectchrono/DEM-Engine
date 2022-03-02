# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# All rights reserved.

# Determine a suitable C++ standard version for the active compiler 

function(CxxStdAutodetect)

	set(mprefix "[C++ STD_AUTODETECT]")	
	message("${mprefix} Detecting level of C++ support...")

	foreach(local_cxxver RANGE 97)
		if("cxx_std_${local_cxxver}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
			message("${mprefix} Compiler supports C++${local_cxxver}...")
			set(GLOBAL_cxxver ${local_cxxver} CACHE INTERNAL "")
		endif()
	endforeach()

	if (local_cxxver EQUAL 97)
		message(SEND_ERROR "${mprefix} Could not detect a suitable C++ standard")
	endif()

endfunction()
