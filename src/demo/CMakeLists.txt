# ------------------------------------------------------------------------------
# Additional include paths and libraries
# ------------------------------------------------------------------------------

# INCLUDE_DIRECTORIES(${ProjectIncludeRoot})

SET(LIBRARIES
	simulator_multi_gpu
)

# ------------------------------------------------------------------------------
# List of all executables
# ------------------------------------------------------------------------------

SET(DEMOS
		DEMdemo_Pile
		DEMdemo_Sieve
		DEMdemo_SingleSphereCollide
		DEMdemo_TestPack
		DEMdemo_TestRestart
		DEMdemo_RotatingDrum
		DEMdemo_Centrifuge
		DEMdemo_GameOfLife
		DEMdemo_BallDrop
		DEMdemo_Mixer
		DEMdemo_PolydisperseGeneration
		DEMdemo_RoverWheel
		DEMdemo_GRCPrep_Part1
		DEMdemo_GRCPrep_Part2
		DEMdemo_GRCPrep_Part3
		DEMdemo_ConePenetration
)

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

message(STATUS "Demo programs for DEM solver...")

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

