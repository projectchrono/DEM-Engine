# Copy the selected Jitify header without changing the third-party checkout. On MSVC, the optional upstream
# fallback needs the same DbgHelp serialization as DEME's vendored variant. Reject unfamiliar fallback layouts
# rather than silently producing an unprotected Windows build.
function(deme_prepare_jitify input_header output_header msvc_build)
	set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${input_header}")
	file(READ "${input_header}" jitify_source)
	if(msvc_build AND NOT jitify_source MATCHES "DEME serializes DbgHelp")
		set(demangle_signature "inline std::string demangle_native_type(const std::type_info& typeinfo) {")
		string(FIND "${jitify_source}" "${demangle_signature}" signature_offset)
		if(signature_offset EQUAL -1)
			message(FATAL_ERROR "Cannot add MSVC DbgHelp protection to ${input_header}; use DEME's vendored Jitify header.")
		endif()
		# This signature also occurs in the non-MSVC implementation. Guard the inserted code explicitly so the
		# generated header remains usable on other platforms with no locking or extra includes there.
		set(demangle_guard "\n#ifdef _MSC_VER
    // DEME serializes DbgHelp across all reflected types and worker threads.
    static std::mutex dbghelp_mutex;
    std::lock_guard<std::mutex> lock(dbghelp_mutex);
#endif
")
		string(REPLACE "${demangle_signature}" "${demangle_signature}${demangle_guard}" jitify_source "${jitify_source}")
		set(jitify_source "#ifdef _MSC_VER\n#include <mutex>\n#endif\n${jitify_source}")
		# COPYONLY keeps the final header's timestamp stable when configure is rerun without content changes.
		set(patched_header "${output_header}.msvc-input")
		file(WRITE "${patched_header}" "${jitify_source}")
		configure_file("${patched_header}" "${output_header}" COPYONLY)
	else()
		configure_file("${input_header}" "${output_header}" COPYONLY)
	endif()
endfunction()
