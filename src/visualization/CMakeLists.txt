# The visualizer stays in a separate object target so the C++ library and Python extension can both include it without
# forcing raylib into DEME's solver internals.
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/imgui)
install(FILES ${IMGUI_DIR}/LICENSE.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/DEME/licenses RENAME imgui-LICENSE.txt)
add_library(visualization_bridge OBJECT VisualizerBridge.cpp)
target_include_directories(visualization_bridge PRIVATE ${ProjectIncludeSource} ${ProjectIncludeGenerated})
set_target_properties(visualization_bridge PROPERTIES POSITION_INDEPENDENT_CODE TRUE CXX_STANDARD ${CXXSTD_SUPPORTED})
add_library(visualization OBJECT DEMVisualizer.cpp NativeRenderer.cpp
	${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp
	${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp
	${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)

target_include_directories(visualization
	PUBLIC ${ProjectIncludeSource}
	PUBLIC ${ProjectIncludeGenerated}
	PRIVATE ${IMGUI_DIR} ${IMGUI_DIR}/backends
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/raylib/src/external
)

target_link_libraries(visualization PUBLIC raylib)

set_target_properties(visualization PROPERTIES
	POSITION_INDEPENDENT_CODE TRUE
	CXX_STANDARD ${CXXSTD_SUPPORTED}
)
