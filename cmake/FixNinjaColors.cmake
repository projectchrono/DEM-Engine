# Fix Ninja Colors

function(fix_ninja_colors)

	# Check if override is set
	if(${FORCE_PLAIN_OUTPUT})
		return()
	endif()

	# Check for Ninja
	if(${CMAKE_GENERATOR} STREQUAL "Ninja")

		# Fix GCC colors
		if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
			set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fdiagnostics-color=always" PARENT_SCOPE)

		# Fix Clang colors
		elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
			set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fcolor-diagnostics" PARENT_SCOPE)

		endif()

	endif()

endfunction()
