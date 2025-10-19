//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_CUDALLOC_HPP
#define DEME_CUDALLOC_HPP

#include <core/ApiVersion.h>

#include <cuda_runtime_api.h>
#include <climits>
#include <iostream>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace deme {

// Unified memory allocator
template <class T>
struct ManagedAllocator {
  public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

#if CXX_OLDER(STD_CXX20)
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template <class U>
    struct rebind {
        typedef typename deme::ManagedAllocator<U> other;
    };
#endif

#if CXX_EQ_NEWER(STD_CXX14)
    using propagate_on_container_move_assignment = std::false_type;

    // These aren't required by the standard, but some implementations
    // care about them...
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;
#endif

#if CXX_EQ_NEWER(STD_CXX17)
    using is_always_equal = std::true_type;
#endif

#if CXX_EQ_OLDER(STD_CXX17)  // C++17 or older
    ManagedAllocator() noexcept {}
    ManagedAllocator(const ManagedAllocator& other) noexcept {}

    template <class U>
    ManagedAllocator(const ManagedAllocator<U>& other) noexcept {}
#else  // CXX_EQ_NEWER(STD_CXX20) C++20 or newer
    constexpr ManagedAllocator() noexcept {};
    constexpr ManagedAllocator(const ManagedAllocator& other) noexcept {}

    template <class U>
    constexpr ManagedAllocator(const ManagedAllocator<U>& other) noexcept {}
#endif

#if CXX_OLDER(STD_CXX20)  // before C++20
    value_type* address(value_type& x) const noexcept {
        return &x;
    }

    size_type max_size() const noexcept {
        return ULLONG_MAX / sizeof(value_type);
    }

    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        ::new ((void*)p) U(std::forward<Args>(args)...);
    }

    template <class U>
    void destroy(U* p) {
        p->~U();
    }
#endif

// #if CXX_OLDER(STD_CXX17)
//     pointer allocate(size_type n, std::allocator<void>::const_pointer hint = 0) { return this->__alloc_impl(n); }
// #elif CXX_EQUAL(STD_CXX17)
#if CXX_OLDER(STD_CXX20)
    T* allocate(std::size_t n) {
        return this->__alloc_impl(n);
    }
    T* allocate(std::size_t n, const void* hint) {
        return this->__alloc_impl(n);
    }
#else
    [[nodiscard]] constexpr T* allocate(std::size_t n) {
        return this->__alloc_impl(n);
    }
#endif
    // #elif CXX_EQ_NEWER(STD_CXX20)
    //     [[nodiscard]] constexpr T* allocate(std::size_t n) { return this->__alloc_impl(n); }
    // #endif

#if CXX_OLDER(STD_CXX20)
    void deallocate(T* p, std::size_t n) {
        cudaFree(p);
    }
#else  // CXX_EQ_NEWER(STD_CXX20)
    constexpr void deallocate(T* p, std::size_t n) {
        cudaFree(p);
    }
#endif

#if CXX_OLDER(STD_CXX20)
    template <class T2>
    bool operator==(const ManagedAllocator<T2>& other) noexcept {
        return true;
    }

    template <class T2>
    bool operator!=(const ManagedAllocator<T2>& other) noexcept {
        return false;
    }

#else  // CXX_EQ_NEWER(STD_CXX20)
    template <class T2>
    constexpr bool operator==(const ManagedAllocator<T2>& other) noexcept {
        return true;
    }
#endif

  private:
    constexpr T* __alloc_impl(std::size_t n) {
        void* vptr = nullptr;
        cudaError_t err = cudaMallocManaged(&vptr, n * sizeof(T), cudaMemAttachGlobal);
        if (err == cudaErrorMemoryAllocation || err == cudaErrorNotSupported) {
            throw std::bad_alloc();
        }
        return (T*)vptr;
    }
};

// Host pinned memory allocator
template <class T>
struct PinnedAllocator {
  public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

#if CXX_OLDER(STD_CXX20)
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template <class U>
    struct rebind {
        typedef typename deme::PinnedAllocator<U> other;
    };
#endif

#if CXX_EQ_NEWER(STD_CXX14)
    using propagate_on_container_move_assignment = std::false_type;

    // These aren't required by the standard, but some implementations
    // care about them...
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;
#endif

#if CXX_EQ_NEWER(STD_CXX17)
    using is_always_equal = std::true_type;
#endif

#if CXX_EQ_OLDER(STD_CXX17)  // C++17 or older
    PinnedAllocator() noexcept {}
    PinnedAllocator(const PinnedAllocator& other) noexcept {}

    template <class U>
    PinnedAllocator(const PinnedAllocator<U>& other) noexcept {}
#else  // CXX_EQ_NEWER(STD_CXX20) C++20 or newer
    constexpr PinnedAllocator() noexcept {};
    constexpr PinnedAllocator(const PinnedAllocator& other) noexcept {}

    template <class U>
    constexpr PinnedAllocator(const PinnedAllocator<U>& other) noexcept {}
#endif

#if CXX_OLDER(STD_CXX20)  // before C++20
    value_type* address(value_type& x) const noexcept {
        return &x;
    }

    size_type max_size() const noexcept {
        return ULLONG_MAX / sizeof(value_type);
    }

    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        ::new ((void*)p) U(std::forward<Args>(args)...);
    }

    template <class U>
    void destroy(U* p) {
        p->~U();
    }
#endif

// #if CXX_OLDER(STD_CXX17)
//     pointer allocate(size_type n, std::allocator<void>::const_pointer hint = 0) { return this->__alloc_impl(n); }
// #elif CXX_EQUAL(STD_CXX17)
#if CXX_OLDER(STD_CXX20)
    T* allocate(std::size_t n) {
        return this->__alloc_impl(n);
    }
    T* allocate(std::size_t n, const void* hint) {
        return this->__alloc_impl(n);
    }
#else
    [[nodiscard]] constexpr T* allocate(std::size_t n) {
        return this->__alloc_impl(n);
    }
#endif
    // #elif CXX_EQ_NEWER(STD_CXX20)
    //     [[nodiscard]] constexpr T* allocate(std::size_t n) { return this->__alloc_impl(n); }
    // #endif

#if CXX_OLDER(STD_CXX20)
    void deallocate(T* p, std::size_t n) {
        cudaFreeHost(p);
    }
#else  // CXX_EQ_NEWER(STD_CXX20)
    constexpr void deallocate(T* p, std::size_t n) {
        cudaFreeHost(p);
    }
#endif

#if CXX_OLDER(STD_CXX20)
    template <class T2>
    bool operator==(const PinnedAllocator<T2>& other) noexcept {
        return true;
    }

    template <class T2>
    bool operator!=(const PinnedAllocator<T2>& other) noexcept {
        return false;
    }

#else  // CXX_EQ_NEWER(STD_CXX20)
    template <class T2>
    constexpr bool operator==(const PinnedAllocator<T2>& other) noexcept {
        return true;
    }
#endif

  private:
    constexpr T* __alloc_impl(std::size_t n) {
        T* ptr = nullptr;
        cudaError_t err = cudaHostAlloc((void**)&ptr, n * sizeof(T), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            throw std::bad_alloc();
        }
        return ptr;
    }
};

}  // END namespace deme

#endif
