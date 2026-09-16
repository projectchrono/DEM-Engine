# ------------------------------------------------------------------------------
# Additional include paths and libraries
# ------------------------------------------------------------------------------

SET(LIBRARIES
		simulator_multi_gpu
)

# ------------------------------------------------------------------------------
# List of all test executables
# ------------------------------------------------------------------------------

SET(MODULAR_TESTS
		DEMTest_StateFinite
		DEMTest_AdaptiveTimestep
		DEMTest_AnalyticalCone
		DEMTest_AnalyticalConePour
		DEMTest_CombinedOwners
		DEMTest_ContactIslandFlooding
		DEMTest_ConeGrazingPlane
		DEMTest_ConeGrazingSphere
		DEMTest_CylinderCrossRotation
		DEMTest_CudaHeaderDiscovery
		DEMTest_DeviceDataRetrieval
		DEMTest_DeviceSelection
		DEMTest_MeshBox1Particle
		DEMTest_MeshSlidingOnSlope
		DEMTest_MeshTemplate
		DEMTest_PatchLocations
		DEMTest_RuntimeSimParams
		DEMTest_MeshPatch
		DEMTest_SimpleCollisions
		DEMTest_SpherePlanePeneArea
		DEMTest_SphereSphereArea
		DEMTest_TrianglePVTracking
		DEMTest_VisualizationSnapshot
		DEMTest_VisualizationScene
)

if(DEME_BUILD_VISUALIZER)
	list(APPEND MODULAR_TESTS DEMTest_Visualizer)
endif()

# ------------------------------------------------------------------------------
# Add all test executables
# ------------------------------------------------------------------------------

message(STATUS "Modular test programs for DEM solver...")

FOREACH(PROGRAM ${MODULAR_TESTS})
		
		message(STATUS "...add ${PROGRAM}")

		add_executable(${PROGRAM}  "${PROGRAM}.cpp")

		set_target_properties(
			${PROGRAM} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${DEME_INSTALL_DEMO}"
		)

		if(WIN32)
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
		if(PROGRAM STREQUAL "DEMTest_Visualizer")
			target_link_libraries(${PROGRAM} PRIVATE DEMEVisualizer)
		endif()
		
		add_dependencies(${PROGRAM} ${LIBRARIES})

		if (CXXSTD_SUPPORTED)
			set_target_properties(
				${PROGRAM} PROPERTIES
				CXX_STANDARD ${CXXSTD_SUPPORTED}
			)
		endif()

ENDFOREACH(PROGRAM)
