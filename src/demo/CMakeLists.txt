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
)

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

MESSAGE(STATUS "Demo programs for Gpu module...")

FOREACH(PROGRAM ${DEMOS})
		MESSAGE(STATUS "...add ${PROGRAM}")

		ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
		
		target_include_directories(${PROGRAM} PUBLIC ${ProjectIncludeRoot})

		SOURCE_GROUP(""  FILES "${PROGRAM}.cpp")

		TARGET_LINK_LIBRARIES(${PROGRAM} ${LIBRARIES})
		ADD_DEPENDENCIES(${PROGRAM} ${LIBRARIES})
ENDFOREACH(PROGRAM)

