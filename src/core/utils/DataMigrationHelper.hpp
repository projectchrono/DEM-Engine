//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_DATA_MIGRATION_HPP
#define DEME_DATA_MIGRATION_HPP

#include <cassert>
#include <optional>
#include <unordered_map>

#include "GpuError.h"
#include "../../DEM/VariableTypes.h"

namespace deme {

// A to-device memcpy wrapper
template <typename T>
void CudaCopyToDevice(T* pD, T* pH) {
    DEME_GPU_CALL(cudaMemcpy(pD, pH, sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T>
void CudaCopyToDevice(T* pD, T* pH, size_t n) {
    DEME_GPU_CALL(cudaMemcpy(pD, pH, n * sizeof(T), cudaMemcpyHostToDevice));
}

// A to-host memcpy wrapper
template <typename T>
void CudaCopyToHost(T* pH, T* pD) {
    DEME_GPU_CALL(cudaMemcpy(pH, pD, sizeof(T), cudaMemcpyDeviceToHost));
}
template <typename T>
void CudaCopyToHost(T* pH, T* pD, size_t n) {
    DEME_GPU_CALL(cudaMemcpy(pH, pD, n * sizeof(T), cudaMemcpyDeviceToHost));
}

// ptr being a reference to a pointer is crucial
template <typename T>
inline void DevicePtrDealloc(T*& ptr) {
    if (!ptr)
        return;
    cudaPointerAttributes attrib;
    DEME_GPU_CALL(cudaPointerGetAttributes(&attrib, ptr));

    if (attrib.type != cudaMemoryType::cudaMemoryTypeUnregistered)
        DEME_GPU_CALL(cudaFree(ptr));
}

// You have to deal with it yourself if ptr is an already-used device pointer
template <typename T>
inline void DevicePtrAlloc(T*& ptr, size_t size) {
    DEME_GPU_CALL(cudaMalloc((void**)&ptr, size * sizeof(T)));
}

template <typename T>
inline void HostPtrDealloc(T*& ptr) {
    if (!ptr)
        return;
    cudaPointerAttributes attrib;
    DEME_GPU_CALL(cudaPointerGetAttributes(&attrib, ptr));

    if (attrib.type != cudaMemoryType::cudaMemoryTypeUnregistered)
        DEME_GPU_CALL(cudaFreeHost(ptr));
}
template <typename T>
inline void HostPtrAlloc(T*& ptr, size_t size) {
    DEME_GPU_CALL(cudaMallocHost((void**)&ptr, size * sizeof(T)));
}

// Managed advise doesn't seem to do anything...
#define DEME_ADVISE_DEVICE(vec, device) \
    { advise(vec, ManagedAdvice::PREFERRED_LOC, device); }
#define DEME_MIGRATE_TO_DEVICE(vec, device, stream) \
    { migrate(vec, device, stream); }

// DEME_DUAL_ARRAY_RESIZE is a reminder for developers that a work array is resized, and this may automatically change
// the external device pointer this array's bound to. Therefore, after this call, syncing the data pointer bundle
// (granData) to device may be needed, and you remember to cudaSetDevice beforehand so it allocates to correct places.
#define DEME_DUAL_ARRAY_RESIZE(vec, newsize, val) \
    { vec.resize(newsize, val); }
#define DEME_DUAL_ARRAY_RESIZE_NOVAL(vec, newsize) \
    { vec.resize(newsize); }

// Simply a reminder that this is a device array resize, to distinguish from some general .resize calls
#define DEME_DEVICE_ARRAY_RESIZE(vec, newsize) \
    { vec.resize(newsize); }

// Use (void) to silence unused warnings.
// #define assertm(exp, msg) assert(((void)msg, exp))

// We protect GPU-related data types with NonCopyable, because the device pointers inside these data types are too
// fragile for copying. If say a shallow copy is enforced to our array types in a vector-of-arrays resizing, then if you
// check the copied array's pointer from host, CUDA might not recognize it properly. The best practice is just using
// unique_ptr to manage these array classes if you expect to put them in places where some under-the-hood copying could
// happen.
class NonCopyable {
  protected:
    NonCopyable() = default;
    ~NonCopyable() = default;

    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
};

// Used for wrapping data structures so they become usable on GPU
template <typename T>
class DualStruct : private NonCopyable {
  private:
    T* host_data;           // Pointer to host memory (pinned)
    T* device_data;         // Pointer to device memory
    bool modified_on_host;  // Flag to track if host data has been modified
  public:
    // Constructor: Initialize and allocate memory for both host and device
    DualStruct() : modified_on_host(false) {
        DEME_GPU_CALL(cudaMallocHost((void**)&host_data, sizeof(T)));
        DEME_GPU_CALL(cudaMalloc((void**)&device_data, sizeof(T)));
    }

    // Constructor: Initialize and allocate memory for both host and device with init values
    DualStruct(T init_val) : modified_on_host(false) {
        DEME_GPU_CALL(cudaMallocHost((void**)&host_data, sizeof(T)));
        DEME_GPU_CALL(cudaMalloc((void**)&device_data, sizeof(T)));

        *host_data = init_val;

        toDevice();
    }

    // Destructor: Free memory
    ~DualStruct() { free(); }

    void free() {
        HostPtrDealloc(host_data);      // Free pinned memory
        DevicePtrDealloc(device_data);  // Free device memory
    }

    // Synchronize changes from host to device
    void toDevice() {
        DEME_GPU_CALL(cudaMemcpy(device_data, host_data, sizeof(T), cudaMemcpyHostToDevice));
        modified_on_host = false;
    }

    // Synchronize changes from device to host
    void toHost() { DEME_GPU_CALL(cudaMemcpy(host_data, device_data, sizeof(T), cudaMemcpyDeviceToHost)); }

    // // Synchronize change of one field of the struct to device
    // template <typename MemberType>
    // void syncMemberToDevice(ptrdiff_t offset) {
    //     DEME_GPU_CALL(cudaMemcpy(reinterpret_cast<char*>(device_data) + offset,
    //                              reinterpret_cast<char*>(host_data) + offset, sizeof(MemberType),
    //                              cudaMemcpyHostToDevice));
    // }

    // // Synchronize change of one field of the struct to host
    // template <typename MemberType>
    // void syncMemberToHost(ptrdiff_t offset) {
    //     DEME_GPU_CALL(cudaMemcpy(reinterpret_cast<char*>(host_data) + offset,
    //                              reinterpret_cast<char*>(device_data) + offset, sizeof(MemberType),
    //                              cudaMemcpyDeviceToHost));
    // }

    // Check if host data has been modified and not synced
    bool checkNoPendingModification() { return !modified_on_host; }

    void markModified() { modified_on_host = true; }

    void unmarkModified() { modified_on_host = false; }

    // Accessor for host data (using the arrow operator)
    T* operator->() { return host_data; }

    // Accessor for host data (using the arrow operator)
    T* operator->() const { return host_data; }

    // Dereference operator for simple types (like float) to access the value directly
    T& operator*() {
        return *host_data;  // Return a reference to the value
    }

    // Dereference operator for simple types (like float) to access the value directly
    T& operator*() const {
        return *host_data;  // Return a reference to the value
    }

    // Overloaded operator& for device pointer access
    T* operator&() const {
        return device_data;  // Return device pointer when using &
    }

    // Getter for the device pointer
    T* getDevicePointer() { return device_data; }

    // Getter for the host pointer
    T* getHostPointer() { return host_data; }

    // Get host or device size in bytes
    size_t getNumBytes() const { return sizeof(T); }
};

#ifndef DEME_USE_MANAGED_ARRAYS
// CPU--GPU unified array, leveraging pinned memory
template <typename T>
class DualArray : private NonCopyable {
  public:
    using PinnedVector = std::vector<T, PinnedAllocator<T>>;

    explicit DualArray(size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        ensureHostVector();
    }

    DualArray(size_t n, size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        resize(n);
    }

    DualArray(size_t n, T val, size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        resize(n, val);
    }

    DualArray(const std::vector<T>& vec,
              size_t* host_external_counter = nullptr,
              size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        attachHostVector(&vec, /*deep_copy=*/true);
    }

    ~DualArray() { free(); }

    void resize(size_t n) {
        assert(m_host_vec_ptr == m_pinned_vec.get() && "resize() requires internal host ownership");
        resizeHost(n);
        resizeDevice(n);
    }

    // This resize flavor fills host values only!
    void resize(size_t n, const T& val) {
        assert(m_host_vec_ptr == m_pinned_vec.get() && "resize() requires internal host ownership");
        resizeHost(n, val);
        resizeDevice(n);
    }

    void resizeHost(size_t n) {
        ensureHostVector();  // allocates pinned vec if null
        size_t old_bytes = m_host_vec_ptr->size() * sizeof(T);
        m_host_vec_ptr->resize(n);
        size_t new_bytes = m_host_vec_ptr->size() * sizeof(T);
        updateHostMemCounter(static_cast<ssize_t>(new_bytes) - static_cast<ssize_t>(old_bytes));
    }

    void resizeHost(size_t n, const T& val) {
        ensureHostVector();  // allocates pinned vec if null
        size_t old_bytes = m_host_vec_ptr->size() * sizeof(T);
        m_host_vec_ptr->resize(n, val);
        size_t new_bytes = m_host_vec_ptr->size() * sizeof(T);
        updateHostMemCounter(static_cast<ssize_t>(new_bytes) - static_cast<ssize_t>(old_bytes));
    }

    // m_device_capacity is allocated memory, not array usable data range.
    // Also, this method preserves already-existing device data.
    void resizeDevice(size_t n, bool allow_shrink = false) {
        if (!allow_shrink && m_device_capacity >= n)
            return;

        T* new_device_ptr = nullptr;
        DevicePtrAlloc(new_device_ptr, n);

        // If previous data exists, copy the minimum amount
        if (m_device_ptr && m_device_capacity > 0) {
            size_t copy_count = std::min(n, m_device_capacity);
            DEME_GPU_CALL(cudaMemcpy(new_device_ptr, m_device_ptr, copy_count * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        // Free old memory and update bookkeeping
        updateDeviceMemCounter(-(ssize_t)(m_device_capacity * sizeof(T)));
        DevicePtrDealloc(m_device_ptr);

        m_device_ptr = new_device_ptr;
        updateBoundDevicePointer();

        updateDeviceMemCounter(static_cast<ssize_t>(n * sizeof(T)));
        m_device_capacity = n;
    }

    void freeHost() {
        if (m_host_vec_ptr) {
            updateHostMemCounter(-(ssize_t)(m_host_vec_ptr->size() * sizeof(T)));
        }
        m_pinned_vec.reset();
        m_host_vec_ptr = nullptr;
        m_host_dirty = false;
    }

    void freeDevice() {
        DevicePtrDealloc(m_device_ptr);
        updateDeviceMemCounter(-(ssize_t)(m_device_capacity * sizeof(T)));
        m_device_ptr = nullptr;
        m_device_capacity = 0;
        updateBoundDevicePointer();
    }

    void free() {
        freeDevice();
        freeHost();
    }

    void toDevice() {
        assert(m_host_vec_ptr);
        size_t count = size();
        if (count > m_device_capacity)
            resizeDevice(count);
        DEME_GPU_CALL(cudaMemcpy(m_device_ptr, m_host_vec_ptr->data(), count * sizeof(T), cudaMemcpyHostToDevice));
        m_host_dirty = false;
    }

    void toDevice(size_t start, size_t n) {
        assert(m_host_vec_ptr && m_device_ptr);
        // Partial flavor aims for speed, no size check
        DEME_GPU_CALL(
            cudaMemcpy(m_device_ptr + start, m_host_vec_ptr->data() + start, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    void toDeviceAsync(cudaStream_t& stream) {
        assert(m_host_vec_ptr);
        size_t count = size();
        if (count > m_device_capacity)
            resizeDevice(count);
        DEME_GPU_CALL(
            cudaMemcpyAsync(m_device_ptr, m_host_vec_ptr->data(), count * sizeof(T), cudaMemcpyHostToDevice, stream));
        m_host_dirty = false;
    }

    // And partial update methods...
    // Normally this is preferred when they are used in tracker implementation
    void toDeviceAsync(cudaStream_t& stream, size_t start, size_t n) {
        assert(m_host_vec_ptr && m_device_ptr);
        // Partial flavor aims for speed, no size check
        DEME_GPU_CALL(cudaMemcpyAsync(m_device_ptr + start, m_host_vec_ptr->data() + start, n * sizeof(T),
                                      cudaMemcpyHostToDevice, stream));
    }

    void toHost() {
        assert(m_device_ptr && m_host_vec_ptr);
        DEME_GPU_CALL(cudaMemcpy(m_host_vec_ptr->data(), m_device_ptr, size() * sizeof(T), cudaMemcpyDeviceToHost));
        m_host_dirty = false;
    }

    void toHost(size_t start, size_t n) {
        assert(m_device_ptr && m_host_vec_ptr);
        DEME_GPU_CALL(
            cudaMemcpy(m_host_vec_ptr->data() + start, m_device_ptr + start, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void toHostAsync(cudaStream_t& stream) {
        assert(m_host_vec_ptr && m_device_ptr);
        DEME_GPU_CALL(
            cudaMemcpyAsync(m_host_vec_ptr->data(), m_device_ptr, size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        m_host_dirty = false;
    }

    void toHostAsync(cudaStream_t& stream, size_t start, size_t n) {
        assert(m_host_vec_ptr && m_device_ptr);
        // Async partial flavor aims for speed, no size check
        DEME_GPU_CALL(cudaMemcpyAsync(m_host_vec_ptr->data() + start, m_device_ptr + start, n * sizeof(T),
                                      cudaMemcpyDeviceToHost, stream));
    }

    T getVal(size_t start) {
        toHost(start, 1);  // sync from device to host
        return (*m_host_vec_ptr)[start];
    }

    std::vector<T> getVal(size_t start, size_t n) {
        toHost(start, n);  // sync from device to host
        return std::vector<T>(m_host_vec_ptr->begin() + start, m_host_vec_ptr->begin() + start + n);
    }

    void setVal(const T& data, size_t start) {
        (*m_host_vec_ptr)[start] = data;
        toDevice(start, 1);
    }

    void setVal(const std::vector<T>& data, size_t start, size_t n = 0) {
        size_t count = (n > 0) ? n : data.size();
        // Copy to host vector
        std::copy(data.begin(), data.begin() + count, m_host_vec_ptr->begin() + start);
        toDevice(start, count);
    }

    void setVal(cudaStream_t& stream, const T& data, size_t start) {
        (*m_host_vec_ptr)[start] = data;
        toDeviceAsync(stream, start, 1);
    }

    void setVal(cudaStream_t& stream, const std::vector<T>& data, size_t start, size_t n = 0) {
        size_t count = (n > 0) ? n : data.size();
        // Copy to host vector
        std::copy(data.begin(), data.begin() + count, m_host_vec_ptr->begin() + start);
        toDeviceAsync(stream, start, count);
    }

    void markHostModified() { m_host_dirty = true; }
    void unmarkHostModified() { m_host_dirty = false; }

    // Array's in-use data range is always stored on host by size()
    size_t size() const { return m_host_vec_ptr ? m_host_vec_ptr->size() : 0; }

    // Get host or device size in bytes
    size_t getNumBytes() const { return m_host_vec_ptr ? m_host_vec_ptr->size() * sizeof(T) : 0; }

    T* host() { return m_host_vec_ptr ? m_host_vec_ptr->data() : nullptr; }

    T* device() { return m_device_ptr; }

    // Overloaded operator& for device pointer access
    T* operator&() const { return m_device_ptr; }

    // data() returns device data for the ease of packing pointers
    T* data() { return device(); }

    PinnedVector& getHostVector() { return *m_host_vec_ptr; }

    void bindDevicePointer(T** external_ptr_to_ptr) {
        m_bound_device_ptr = external_ptr_to_ptr;
        updateBoundDevicePointer();
    }

    void unbindDevicePointer() { m_bound_device_ptr = nullptr; }

    void setHostMemoryCounter(size_t* counter) { m_host_mem_counter = counter; }
    void setDeviceMemoryCounter(size_t* counter) { m_device_mem_counter = counter; }
    // You can use nullptr to unbind

    void attachHostVector(const std::vector<T>* external_vec, bool deep_copy = true) {
        freeHost();  // discard internal memory and update memory tracker
        if (deep_copy) {
            m_pinned_vec = std::make_unique<PinnedVector>(external_vec->begin(), external_vec->end());
            m_host_vec_ptr = m_pinned_vec.get();
            updateHostMemCounter(static_cast<ssize_t>(m_host_vec_ptr->size() * sizeof(T)));
        } else {
            m_host_vec_ptr = const_cast<PinnedVector*>(reinterpret_cast<const PinnedVector*>(external_vec));
        }
        m_host_dirty = true;
    }

    T& operator[](size_t i) { return (*m_host_vec_ptr)[i]; }
    const T& operator[](size_t i) const { return (*m_host_vec_ptr)[i]; }
    T operator()(size_t i) { return getVal(i); }

  private:
    std::unique_ptr<PinnedVector> m_pinned_vec = nullptr;
    PinnedVector* m_host_vec_ptr = nullptr;

    size_t* m_host_mem_counter = nullptr;
    size_t* m_device_mem_counter = nullptr;

    T* m_device_ptr = nullptr;
    size_t m_device_capacity = 0;

    T** m_bound_device_ptr = nullptr;

    bool m_host_dirty = false;

    void ensureHostVector(size_t n = 0) {
        if (!m_host_vec_ptr) {
            m_pinned_vec = std::make_unique<PinnedVector>(n);
            m_host_vec_ptr = m_pinned_vec.get();
        }
    }

    void updateBoundDevicePointer() {
        if (m_bound_device_ptr)
            *m_bound_device_ptr = m_device_ptr;
    }

    void updateHostMemCounter(ssize_t delta) {
        if (m_host_mem_counter)
            *m_host_mem_counter += delta;
    }

    void updateDeviceMemCounter(ssize_t delta) {
        if (m_device_mem_counter)
            *m_device_mem_counter += delta;
    }
};
#else
// CPU--GPU unified array, leveraging managed memory
template <typename T>
class DualArray : private NonCopyable {
  public:
    using ManagedVector = std::vector<T, ManagedAllocator<T>>;

    explicit DualArray(size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        ensureHostVector();
    }

    DualArray(size_t n, size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        resize(n);
    }

    DualArray(size_t n, T val, size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        resize(n, val);
    }

    ~DualArray() { free(); }

    void resize(size_t n) {
        assert(m_host_vec_ptr == m_pinned_vec.get() && "resize() requires internal host ownership");
        resizeHost(n);
        resizeDevice(n);
    }

    // This resize flavor fills host values only!
    void resize(size_t n, const T& val) {
        assert(m_host_vec_ptr == m_pinned_vec.get() && "resize() requires internal host ownership");
        resizeHost(n, val);
        resizeDevice(n);
    }

    void resizeHost(size_t n) {
        ensureHostVector();  // allocates pinned vec if null
        size_t old_bytes = m_host_vec_ptr->size() * sizeof(T);
        m_host_vec_ptr->resize(n);
        size_t new_bytes = m_host_vec_ptr->size() * sizeof(T);
        updateMemCounter(static_cast<ssize_t>(new_bytes) - static_cast<ssize_t>(old_bytes));
        updateBoundDevicePointer();
    }

    void resizeHost(size_t n, const T& val) {
        ensureHostVector();  // allocates pinned vec if null
        size_t old_bytes = m_host_vec_ptr->size() * sizeof(T);
        m_host_vec_ptr->resize(n, val);
        size_t new_bytes = m_host_vec_ptr->size() * sizeof(T);
        updateMemCounter(static_cast<ssize_t>(new_bytes) - static_cast<ssize_t>(old_bytes));
        updateBoundDevicePointer();
    }

    // m_device_capacity is allocated memory, not array usable data range
    void resizeDevice(size_t n, bool allow_shrink = false) {}

    void freeHost() {
        if (m_host_vec_ptr) {
            updateMemCounter(-(ssize_t)(m_host_vec_ptr->size() * sizeof(T)));
        }
        m_pinned_vec.reset();
        m_host_vec_ptr = nullptr;
        updateBoundDevicePointer();
    }

    void freeDevice() {}

    void free() {
        freeDevice();
        freeHost();
    }

    void toDevice() {}

    void toDevice(size_t start, size_t n) {}

    void toDeviceAsync(cudaStream_t& stream) {}

    void toDeviceAsync(cudaStream_t& stream, size_t start, size_t n) {}

    void toHost() {}

    void toHost(size_t start, size_t n) {}

    void toHostAsync(cudaStream_t& stream) {}

    void toHostAsync(cudaStream_t& stream, size_t start, size_t n) {}

    T getVal(size_t start) { return (*m_host_vec_ptr)[start]; }

    std::vector<T> getVal(size_t start, size_t n) {
        return std::vector<T>(m_host_vec_ptr->begin() + start, m_host_vec_ptr->begin() + start + n);
    }

    void setVal(const T& data, size_t start) { (*m_host_vec_ptr)[start] = data; }

    void setVal(const std::vector<T>& data, size_t start, size_t n = 0) {
        size_t count = (n > 0) ? n : data.size();
        std::copy(data.begin(), data.begin() + count, m_host_vec_ptr->begin() + start);
    }

    void setVal(cudaStream_t& stream, const T& data, size_t start) { (*m_host_vec_ptr)[start] = data; }

    void setVal(cudaStream_t& stream, const std::vector<T>& data, size_t start, size_t n = 0) {
        size_t count = (n > 0) ? n : data.size();
        std::copy(data.begin(), data.begin() + count, m_host_vec_ptr->begin() + start);
    }

    void markHostModified() { m_host_dirty = true; }
    void unmarkHostModified() { m_host_dirty = false; }

    // Array's in-use data range is always stored on host by size()
    size_t size() const { return m_host_vec_ptr ? m_host_vec_ptr->size() : 0; }

    // Get host or device size in bytes
    size_t getNumBytes() const { return m_host_vec_ptr ? m_host_vec_ptr->size() * sizeof(T) : 0; }

    T* host() { return m_host_vec_ptr ? m_host_vec_ptr->data() : nullptr; }

    T* device() { return host(); }

    // Overloaded operator& for device pointer access
    T* operator&() const { return host(); }

    // data() returns device data for the ease of packing pointers
    T* data() { return host(); }

    ManagedVector& getHostVector() { return *m_host_vec_ptr; }

    void bindDevicePointer(T** external_ptr_to_ptr) {
        m_bound_device_ptr = external_ptr_to_ptr;
        updateBoundDevicePointer();
    }

    void unbindDevicePointer() { m_bound_device_ptr = nullptr; }

    void setHostMemoryCounter(size_t* counter) { m_host_mem_counter = counter; }
    void setDeviceMemoryCounter(size_t* counter) { m_device_mem_counter = counter; }
    // You can use nullptr to unbind

    T& operator[](size_t i) { return (*m_host_vec_ptr)[i]; }
    const T& operator[](size_t i) const { return (*m_host_vec_ptr)[i]; }
    T operator()(size_t i) { return getVal(i); }

  private:
    std::unique_ptr<ManagedVector> m_pinned_vec = nullptr;
    ManagedVector* m_host_vec_ptr = nullptr;

    size_t* m_host_mem_counter = nullptr;
    size_t* m_device_mem_counter = nullptr;

    T** m_bound_device_ptr = nullptr;

    bool m_host_dirty = false;

    void ensureHostVector(size_t n = 0) {
        if (!m_host_vec_ptr) {
            m_pinned_vec = std::make_unique<ManagedVector>(n);
            m_host_vec_ptr = m_pinned_vec.get();
        }
    }

    void updateBoundDevicePointer() {
        if (m_bound_device_ptr)
            *m_bound_device_ptr = host();
    }

    void updateMemCounter(ssize_t delta) {
        if (m_host_mem_counter)
            *m_host_mem_counter += delta;
        if (m_device_mem_counter)
            *m_device_mem_counter += delta;
    }
};
#endif

// Pure device data type, usually used for scratching space
template <typename T>
class DeviceArray : private NonCopyable {
  public:
    DeviceArray(size_t* external_counter = nullptr) : m_mem_counter(external_counter) {}

    explicit DeviceArray(size_t n, size_t* external_counter = nullptr) : m_mem_counter(external_counter) { resize(n); }

    ~DeviceArray() { free(); }

    // In practice, we use device array as temp arrays so we never really resize, let alone preserving existing data
    void resize(size_t n, bool allow_shrink = false) {
        if (!allow_shrink && m_capacity >= n)
            return;
        T* new_device_ptr = nullptr;
        DevicePtrAlloc(new_device_ptr, n);

        // If previous data exists, copy the minimum amount
        if (m_data && m_capacity > 0) {
            size_t copy_count = std::min(n, m_capacity);
            DEME_GPU_CALL(cudaMemcpy(new_device_ptr, m_data, copy_count * sizeof(T), cudaMemcpyDeviceToDevice));
        }
        // Free old memory and update bookkeeping
        updateMemCounter(-(ssize_t)(m_capacity * sizeof(T)));
        DevicePtrDealloc(m_data);

        m_data = new_device_ptr;

        updateMemCounter(static_cast<ssize_t>(n * sizeof(T)));
        m_capacity = n;
    }

    void free() {
        DevicePtrDealloc(m_data);
        updateMemCounter(-(ssize_t)(m_capacity * sizeof(T)));
        m_data = nullptr;
        m_capacity = 0;
    }

    size_t size() const { return m_capacity; }

    // Get host or device size in bytes
    size_t getNumBytes() const { return m_capacity * sizeof(T); }

    T* data() { return m_data; }

    const T* data() const { return m_data; }

    void setMemoryCounter(size_t* counter) { m_mem_counter = counter; }

  private:
    T* m_data = nullptr;
    size_t m_capacity = 0;
    size_t* m_mem_counter = nullptr;

    void updateMemCounter(ssize_t delta) {
        if (m_mem_counter)
            *m_mem_counter += delta;
    }
};

/// @brief General abstraction of vector pool
/// @tparam T Array data type
/// @tparam DerivedEnclosedData The data type of the enclosed arrays in the derived pool class
template <typename T, typename DerivedEnclosedData>
class ResourcePool : private NonCopyable {
  protected:
    std::vector<std::unique_ptr<DerivedEnclosedData>> vectors;
    std::vector<std::optional<std::string>> in_use;
    std::unordered_map<std::string, size_t> name_to_index;

  public:
    virtual ~ResourcePool() { releaseAll(); }

    void resize(const std::string& name, size_t new_size) {
        auto it = name_to_index.find(name);
        if (it == name_to_index.end()) {
            throw std::runtime_error("Cannot resize: name not found");
        }
        vectors[it->second]->resize(new_size);
    }

    void unclaim(const std::string& name) {
        auto it = name_to_index.find(name);
        if (it == name_to_index.end())
            return;
        in_use[it->second] = std::nullopt;
        name_to_index.erase(it);
    }

    void release(const std::string& name) {
        auto it = name_to_index.find(name);
        if (it == name_to_index.end()) {
            throw std::runtime_error("Cannot release: name not found");
        }
        vectors[it->second]->free();
        in_use[it->second] = std::nullopt;
        name_to_index.erase(it);
    }

    void releaseAll() {
        for (size_t i = 0; i < vectors.size(); ++i) {
            vectors[i]->free();
        }
        in_use.clear();
        name_to_index.clear();
    }

    void printStatus() const {
        for (size_t i = 0; i < in_use.size(); ++i) {
            if (in_use[i]) {
                const std::string& name = *in_use[i];
                const size_t index = name_to_index.at(name);
                const auto& vec = vectors[index];
                std::cout << "Storage vector[" << i << "] in use as \"" << name << "\", using " << vec->getNumBytes()
                          << " bytes.\n";
            } else {
                std::cout << "Storage vector[" << i << "] is free\n";
            }
        }
    }
};

/// @brief A pool that dispatches device arrays
/// @tparam T Array data type
template <typename T>
class DeviceVectorPool : public ResourcePool<T, DeviceArray<T>> {
  public:
    explicit DeviceVectorPool(size_t* external_counter = nullptr) : m_mem_counter(external_counter) {}

    T* claim(const std::string& name, size_t size, bool allow_duplicate = false) {
        // Referring to the base class that correspond to me
        using Base = ResourcePool<T, DeviceArray<T>>;

        if (Base::name_to_index.count(name)) {
            if (allow_duplicate) {
                size_t index = Base::name_to_index[name];
                Base::vectors[index]->resize(size);
                return Base::vectors[index]->data();
            } else {
                throw std::runtime_error("Name already claimed: " + name);
            }
        }

        for (size_t i = 0; i < Base::in_use.size(); ++i) {
            if (!Base::in_use[i]) {
                Base::vectors[i]->resize(size);
                Base::in_use[i] = name;
                Base::name_to_index[name] = i;
                return Base::vectors[i]->data();
            }
        }

        Base::vectors.emplace_back(std::make_unique<DeviceArray<T>>(size, m_mem_counter));
        Base::in_use.emplace_back(name);
        size_t new_index = Base::vectors.size() - 1;
        Base::name_to_index[name] = new_index;
        return Base::vectors[new_index]->data();
    }

    T* get(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            throw std::runtime_error("Name not found: " + name);
        return this->vectors[it->second]->data();
    }

    void setMemoryCounter(size_t* counter) {
        m_mem_counter = counter;
        for (auto& vec : this->vectors) {
            vec->setMemoryCounter(counter);
        }
    }

  private:
    size_t* m_mem_counter = nullptr;
};

/// @brief A pool that dispatches dual arrays
/// @tparam T Array data type
template <typename T>
class DualArrayPool : public ResourcePool<T, DualArray<T>> {
  public:
    explicit DualArrayPool(size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {}

    DualArray<T>* claim(const std::string& name, size_t size, bool allow_duplicate = false) {
        // Referring to the base class that correspond to me
        using Base = ResourcePool<T, DualArray<T>>;

        if (Base::name_to_index.count(name)) {
            if (allow_duplicate) {
                size_t index = Base::name_to_index[name];
                Base::vectors[index]->resize(size);
                return Base::vectors[index].get();
            } else {
                throw std::runtime_error("Name already claimed: " + name);
            }
        }

        for (size_t i = 0; i < Base::in_use.size(); ++i) {
            if (!Base::in_use[i]) {
                Base::vectors[i]->resize(size);
                Base::in_use[i] = name;
                Base::name_to_index[name] = i;
                return Base::vectors[i].get();
            }
        }

        Base::vectors.emplace_back(std::make_unique<DualArray<T>>(size, m_host_mem_counter, m_device_mem_counter));
        Base::in_use.emplace_back(name);
        size_t new_index = Base::vectors.size() - 1;
        Base::name_to_index[name] = new_index;
        return Base::vectors[new_index].get();
    }

    DualArray<T>* get(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            throw std::runtime_error("Name not found: " + name);
        return this->vectors[it->second].get();
    }

    T* getHost(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            throw std::runtime_error("Name not found: " + name);
        return this->vectors[it->second]->host();
    }

    T* getDevice(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            throw std::runtime_error("Name not found: " + name);
        return this->vectors[it->second]->device();
    }

    void setMemoryCounter(size_t* host_external_counter, size_t* device_external_counter) {
        m_host_mem_counter = host_external_counter;
        m_device_mem_counter = device_external_counter;
        for (auto& vec : this->vectors) {
            vec->setHostMemoryCounter(m_host_mem_counter);
            vec->setDeviceMemoryCounter(m_device_mem_counter);
        }
    }

  private:
    size_t* m_host_mem_counter = nullptr;
    size_t* m_device_mem_counter = nullptr;
};

/// @brief A pool that dispatches dual structs
/// @tparam T Struct data type
template <typename T>
class DualStructPool : public ResourcePool<T, DualStruct<T>> {
  public:
    explicit DualStructPool() {}

    DualStruct<T>* claim(const std::string& name, bool allow_duplicate = false) {
        // Referring to the base class that correspond to me
        using Base = ResourcePool<T, DualStruct<T>>;

        if (Base::name_to_index.count(name)) {
            if (allow_duplicate) {
                size_t index = Base::name_to_index[name];
                return Base::vectors[index].get();
            } else {
                throw std::runtime_error("Name already claimed: " + name);
            }
        }

        for (size_t i = 0; i < Base::in_use.size(); ++i) {
            if (!Base::in_use[i]) {
                Base::in_use[i] = name;
                Base::name_to_index[name] = i;
                return Base::vectors[i].get();
            }
        }

        Base::vectors.emplace_back(std::make_unique<DualStruct<T>>());
        Base::in_use.emplace_back(name);
        size_t new_index = Base::vectors.size() - 1;
        Base::name_to_index[name] = new_index;
        return Base::vectors[new_index].get();
    }

    DualStruct<T>* get(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            throw std::runtime_error("Name not found: " + name);
        return this->vectors[it->second].get();
    }

    T* getHost(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            throw std::runtime_error("Name not found: " + name);
        return this->vectors[it->second]->getHostPointer();
    }

    T* getDevice(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            throw std::runtime_error("Name not found: " + name);
        return this->vectors[it->second]->getDevicePointer();
    }
};

// A conceptual note: In DEME, DualStruct-managed simParams, granData etc. are in general considered "host-major",
// meaning that we generally believe the data on the host is more fresh. So, we change host data freely, copy host data
// to device more freely, but generally do not copy data from device to host. In contrast, DualArray-managed simulation
// status or work arrays are "device-major", meaning the device copy is believed to be more fresh. So, we copy from
// device to host more freely and concern-free, but when copying from host to device, that's either in a centrialized
// initialization stage, or we do piecemeal and fine-grain copying which reflects the user's forced system updates only.

}  // namespace deme

#endif
