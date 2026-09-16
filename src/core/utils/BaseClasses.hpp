//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_BASE_CLASSES_HPP
#define DEME_BASE_CLASSES_HPP

namespace deme {

// -----------------------------
// Utility: Non-copyable class
// -----------------------------
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

// -----------------------------
// Utility: CRTP Singleton
// -----------------------------
template <typename T>
class Singleton {
  public:
    static T& GetInstance() {
        static T instance;  // Guaranteed to be lazy and thread-safe in C++11+
        return instance;
    }

  protected:
    Singleton() = default;
    ~Singleton() = default;
};

}  // namespace deme

#endif
