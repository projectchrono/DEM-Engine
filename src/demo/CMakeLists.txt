# ------------------------------------------------------------------------------
# Additional include paths and libraries
# ------------------------------------------------------------------------------

# INCLUDE_DIRECTORIES(${ProjectIncludeRoot})

SET(LIBRARIES
	sbel_gpu_physics
)

# ------------------------------------------------------------------------------
# List of all executables
# ------------------------------------------------------------------------------

SET(DEMOS
        demo1
		sphdemo
)

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

message(STATUS "Demo programs for Gpu module...")

FOREACH(PROGRAM ${DEMOS})
		
		message(STATUS "...add ${PROGRAM}")

		add_executable(${PROGRAM}  "${PROGRAM}.cpp")
		
		source_group("" FILES "${PROGRAM}.cpp")
		
		target_link_libraries(${PROGRAM} PUBLIC ${LIBRARIES})

		add_dependencies(${PROGRAM} ${LIBRARIES})

ENDFOREACH(PROGRAM)

