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
		# DEMdemo_Pile
		# DEMdemo_Sieve
		# DEMdemo_SingleSphereCollide
		# DEMdemo_TestPack
		# DEMdemo_RotatingDrum
		# DEMdemo_Centrifuge
		# DEMdemo_GameOfLife
		DEMdemo_BallDrop
		# DEMdemo_PolydisperseGeneration
		# DEMdemo_RoverWheel
		# DEMdemo_GRCPrep_Part1
		# DEMdemo_GRCPrep_Part2
		# DEMdemo_GRCPrep_Part3
		sphdemo
		hello_world
)

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

message(STATUS "Demo programs for Gpu module...")

FOREACH(PROGRAM ${DEMOS})
		
		message(STATUS "...add ${PROGRAM}")

		add_executable(${PROGRAM}  "${PROGRAM}.cpp")
		
		source_group("" FILES "${PROGRAM}.cpp")
		
		target_link_libraries(${PROGRAM} 
			PUBLIC ${LIBRARIES}
			PUBLIC ${EXTERNAL_LIBRARIES}
		)
		
		add_dependencies(${PROGRAM} ${LIBRARIES})

		set_target_properties(
			${PROGRAM} PROPERTIES
			CXX_STANDARD ${CXXSTD_SUPPORTED}
		)

ENDFOREACH(PROGRAM)

