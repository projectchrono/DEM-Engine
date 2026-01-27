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
		DEMdemo_Repose
		DEMdemo_Repose2D
		DEMdemo_ContactChain
		DEMdemo_Plow
		DEMdemo_Sieve
		DEMdemo_SingleSphereCollide
		DEMdemo_MeshCollide
		DEMdemo_MeshFalling
		DEMdemo_TestPack
		DEMdemo_TestRestart
		DEMdemo_RotatingDrum
		DEMdemo_DrumCubes
		DEMdemo_Centrifuge
		DEMdemo_DrumCubes
		DEMdemo_ResponseAngleMesh
		DEMdemo_GameOfLife
		DEMdemo_BallDrop
		DEMdemo_BallDrop2D
		DEMdemo_Mixer
		DEMdemo_PolydisperseGeneration
		DEMdemo_RoverWheel
		DEMdemo_GRCPrep_Part1
		DEMdemo_GRCPrep_Part2
		DEMdemo_GRCPrep_Part3
		DEMdemo_ConePenetration
		DEMdemo_Shake
		DEMdemo_WheelDP
		DEMdemo_WheelDPSimplified
		DEMdemo_WheelSlopeSlip
		DEMdemo_Indentation
		DEMdemo_SolarSystem
		DEMdemo_Electrostatic
		DEMdemo_FlexibleMesh
		DEMdemo_Hopper_Sphere_Cylinder
		DEMdemo_Fracture_Box
)

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

message(STATUS "Demo programs for DEM solver...")

FOREACH(PROGRAM ${DEMOS})
		
		message(STATUS "...add ${PROGRAM}")

		add_executable(${PROGRAM}  "${PROGRAM}.cpp")

		set_target_properties(
			${PROGRAM} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${DEME_INSTALL_DEMO}"
		)

		if(WIN32)
			# set_property(TARGET ${PROGRAM} 
			# 	PROPERTY VS_DEBUGGER_ENVIRONMENT "PATH=$<TARGET_FILE_DIR:DEMERuntimeDataHelper_install>;%PATH%"
			# )
			# set_property(TARGET ${PROGRAM} 
			# 	PROPERTY VS_DEBUGGER_ENVIRONMENT "PATH=$<TARGET_FILE_DIR:DEMERuntimeDataHelper>;%PATH%"
			# )
			add_custom_command(TARGET ${PROGRAM} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "$<TARGET_FILE_DIR:DEMERuntimeDataHelper_install>/DEMERuntimeDataHelper_install.dll"
					   "$<TARGET_FILE_DIR:DEMERuntimeDataHelper_install>/DEMERuntimeDataHelper.dll"
                       "$<TARGET_FILE_DIR:${PROGRAM}>")
		endif()
		
		source_group("" FILES "${PROGRAM}.cpp")
		
		target_link_libraries(${PROGRAM} 
			PUBLIC ${LIBRARIES}
			PUBLIC ${EXTERNAL_LIBRARIES}
		)
		
		add_dependencies(${PROGRAM} ${LIBRARIES})

		if (CXXSTD_SUPPORTED)
			set_target_properties(
				${PROGRAM} PROPERTIES
				CXX_STANDARD ${CXXSTD_SUPPORTED}
			)
		endif()

		# install(TARGETS ${PROGRAM} DESTINATION ${DEME_INSTALL_DEMO})

ENDFOREACH(PROGRAM)

# ------------------------------------------------------------------------------
# Add modular tests subdirectory
# ------------------------------------------------------------------------------

add_subdirectory(ModularTests)
