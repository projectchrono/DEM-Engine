//	Copyright (c) 2021, SBEL GPU Development Team
//	Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_JIT_HELPER_H
#define DEME_JIT_HELPER_H

#include <filesystem>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <utility>
#include <cctype>

#include <jitify/jitify.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>

#if defined(_WIN32) || defined(_WIN64)
    #undef max
    #undef min
    #undef strtok_r
#endif

class JitHelper {
  public:
    class CachedProgram;

    class Header {
      public:
        Header(const std::filesystem::path& sourcefile);
        const std::string& getSource();
        void substitute(const std::string& symbol, const std::string& value);

      private:
        std::string _source;
    };

    static CachedProgram buildProgram(
        const std::string& name,
        const std::filesystem::path& source,
        std::unordered_map<std::string, std::string> substitutions = std::unordered_map<std::string, std::string>(),
        std::vector<std::string> flags = std::vector<std::string>());

    //// I'm pretty sure C++17 auto-converts this
    // static CachedProgram buildProgram(
    // 	const std::string& name, const std::string& code,
    // 	std::vector<Header> headers = 0,
    // 	std::vector<std::string> flags = 0
    // );

    static const std::filesystem::path KERNEL_DIR;
    static const std::filesystem::path KERNEL_INCLUDE_DIR;
    static const std::filesystem::path CACHE_DIR;

  private:
    static std::string hashString(const std::string& in);
    static std::string toHex(uint64_t value);
    static std::filesystem::path resolveCacheDir();

    inline static std::string loadSourceFile(const std::filesystem::path& sourcefile) {
        std::string code;
        // If the file exists, read in the entire thing.
        if (std::filesystem::exists(sourcefile)) {
            std::ifstream input(sourcefile);
            std::getline(input, code, std::string::traits_type::to_char_type(std::string::traits_type::eof()));
        }
        return code;
    };
};

class JitHelper::CachedProgram {
  public:
    class Kernel;

    Kernel kernel(const std::string& name, std::vector<std::string> options = {}) const;

    const std::string& key() const { return m_storage->programHash; }

  private:
    struct ProgramStorage {
        ProgramStorage(std::string code_in, std::vector<std::string> flags_in);
        std::unique_ptr<jitify::experimental::Program> program;
        std::string code;
        std::vector<std::string> flags;
        std::string programHash;
        std::filesystem::path cacheDir;
        int device = 0;
        std::string archTag;
        std::unordered_map<std::string, std::shared_ptr<jitify::experimental::KernelInstantiation>> kernelCache;
        std::mutex mutex;
    };

    std::shared_ptr<ProgramStorage> m_storage;
    explicit CachedProgram(std::shared_ptr<ProgramStorage> storage);

    friend class JitHelper;
};

class JitHelper::CachedProgram::Kernel {
  public:
    class KernelInstantiation;

    KernelInstantiation instantiate(std::vector<std::string> template_args = {}) const;

    template <typename... TemplateArgs>
    KernelInstantiation instantiate(TemplateArgs... targs) const;

  private:
    std::shared_ptr<ProgramStorage> m_storage;
    std::string m_name;
    std::vector<std::string> m_options;

    Kernel(std::shared_ptr<ProgramStorage> storage, std::string name, std::vector<std::string> options);

    std::shared_ptr<jitify::experimental::KernelInstantiation> getKernelInstantiation(
        const std::vector<std::string>& template_args) const;

    friend class CachedProgram;
};

class JitHelper::CachedProgram::Kernel::KernelInstantiation {
  public:
    class KernelLauncher;

    KernelLauncher configure(dim3 grid, dim3 block, unsigned int smem = 0, cudaStream_t stream = 0) const;
    KernelLauncher configure_1d_max_occupancy(int max_block_size = 0,
                                              unsigned int smem = 0,
                                              CUoccupancyB2DSize smem_callback = 0,
                                              cudaStream_t stream = 0,
                                              unsigned int flags = 0) const;

    CUdeviceptr get_global_ptr(const char* name, size_t* size = nullptr) const {
        return m_impl->get_global_ptr(name, size);
    }

    template <typename T>
    CUresult get_global_array(const char* name, T* data, size_t count, CUstream stream = 0) const {
        return m_impl->get_global_array(name, data, count, stream);
    }

    template <typename T>
    CUresult get_global_value(const char* name, T* value, CUstream stream = 0) const {
        return get_global_array(name, value, 1, stream);
    }

    template <typename T>
    CUresult set_global_array(const char* name, const T* data, size_t count, CUstream stream = 0) const {
        return m_impl->set_global_array(name, data, count, stream);
    }

    template <typename T>
    CUresult set_global_value(const char* name, const T& value, CUstream stream = 0) const {
        return set_global_array(name, &value, 1, stream);
    }

    const std::string& mangled_name() const { return m_impl->mangled_name(); }
    const std::string& ptx() const { return m_impl->ptx(); }

    const std::vector<std::string>& link_files() const { return m_impl->link_files(); }
    const std::vector<std::string>& link_paths() const { return m_impl->link_paths(); }

  private:
    std::shared_ptr<jitify::experimental::KernelInstantiation> m_impl;

    KernelInstantiation(std::shared_ptr<jitify::experimental::KernelInstantiation> impl);

    friend class Kernel;
};

class JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher {
  public:
    KernelLauncher(std::shared_ptr<jitify::experimental::KernelInstantiation> inst,
                   jitify::experimental::KernelLauncher launcher)
        : m_inst(std::move(inst)), m_launcher(std::move(launcher)) {}

    CUresult launch(std::vector<void*> arg_ptrs = {}, std::vector<std::string> arg_types = {}) const {
        return m_launcher.launch(std::move(arg_ptrs), std::move(arg_types));
    }

    template <typename... ArgTypes>
    CUresult launch(const ArgTypes&... args) const {
        return launch(std::vector<void*>({(void*)&args...}),
                      {jitify::reflection::reflect<ArgTypes>()...});
    }

    void safe_launch(std::vector<void*> arg_ptrs = {}, std::vector<std::string> arg_types = {}) const {
        m_launcher.safe_launch(std::move(arg_ptrs), std::move(arg_types));
    }

    template <typename... ArgTypes>
    void safe_launch(const ArgTypes&... args) const {
        safe_launch(std::vector<void*>({(void*)&args...}),
                    {jitify::reflection::reflect<ArgTypes>()...});
    }

  private:
    std::shared_ptr<jitify::experimental::KernelInstantiation> m_inst;
    jitify::experimental::KernelLauncher m_launcher;
};

template <typename... TemplateArgs>
inline JitHelper::CachedProgram::Kernel::KernelInstantiation
JitHelper::CachedProgram::Kernel::instantiate(TemplateArgs... targs) const {
    return instantiate(std::vector<std::string>({jitify::reflection::reflect(targs)...}));
}

inline JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher
JitHelper::CachedProgram::Kernel::KernelInstantiation::configure(dim3 grid,
                                                                 dim3 block,
                                                                 unsigned int smem,
                                                                 cudaStream_t stream) const {
    return KernelLauncher(m_impl, m_impl->configure(grid, block, smem, stream));
}

inline JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher
JitHelper::CachedProgram::Kernel::KernelInstantiation::configure_1d_max_occupancy(int max_block_size,
                                                                                  unsigned int smem,
                                                                                  CUoccupancyB2DSize smem_callback,
                                                                                  cudaStream_t stream,
                                                                                  unsigned int flags) const {
    return KernelLauncher(m_impl, m_impl->configure_1d_max_occupancy(max_block_size, smem, smem_callback, stream, flags));
}

#endif
