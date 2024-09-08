/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

////////////////////////////////////////////////////////////////////////////////
// Weirdly, a few float3 and double3 operators are not in the cuda toolkit and
// I have difficulty including thridparty cuda helper math here. Even if I can,
// I suspect namespace problems. So they will for now just be defined here manually.
////////////////////////////////////////////////////////////////////////////////

#ifndef DEME_CUDA_HELPER_MATH_CU
#define DEME_CUDA_HELPER_MATH_CU

#include <DEM/Defines.h>

inline __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline __device__ double3 cross(double3 a, double3 b) {
    return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __device__ double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __device__ float dot(double3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline __device__ double dot(double4 a, double4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline __device__ float dot(double4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __device__ float length(float3 v) {
    return sqrt(dot(v, v));
}
inline __device__ double length(double3 v) {
    return sqrt(dot(v, v));
}
inline __device__ float length(float4 v) {
    return sqrt(dot(v, v));
}
inline __device__ double length(double4 v) {
    return sqrt(dot(v, v));
}

// Addition and subtraction

inline __device__ float3 operator-(float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}
inline __device__ int3 operator-(int3& a) {
    return make_int3(-a.x, -a.y, -a.z);
}

inline __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ double3 operator+(double3 a, double3 b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __device__ double3 operator-(double3 a, double3 b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator+(double3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __device__ float3 operator-(double3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ void operator+=(float3& a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __device__ void operator-=(float3& a, float3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __device__ void operator+=(float3& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
}
inline __device__ void operator-=(float3& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __device__ void operator+=(double3& a, double3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __device__ void operator-=(double3& a, double3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __device__ void operator+=(double3& a, double b) {
    a.x += b;
    a.y += b;
    a.z += b;
}
inline __device__ void operator-=(double3& a, double b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __device__ void operator+=(float3& a, double3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __device__ void operator-=(float3& a, double3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __device__ void operator+=(float3& a, double b) {
    a.x += b;
    a.y += b;
    a.z += b;
}
inline __device__ void operator-=(float3& a, double b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __device__ void operator+=(double3& a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __device__ void operator-=(double3& a, float3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __device__ void operator+=(double3& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
}
inline __device__ void operator-=(double3& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

// Multiplication

inline __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __device__ void operator*=(float3& a, float3 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __device__ float3 operator*(float b, float3 a) {
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __device__ void operator*=(float3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __device__ double3 operator*(double3 a, double3 b) {
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __device__ void operator*=(double3& a, double3 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __device__ double3 operator*(double3 a, double b) {
    return make_double3(a.x * b, a.y * b, a.z * b);
}
inline __device__ double3 operator*(double b, double3 a) {
    return make_double3(b * a.x, b * a.y, b * a.z);
}
inline __device__ void operator*=(double3& a, double b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

// Division

inline __device__ float3 operator/(float3 a, float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __device__ void operator/=(float3& a, float3 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __device__ float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __device__ float3 operator/(float3 a, double b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __device__ void operator/=(float3& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

inline __device__ double3 operator/(double3 a, double3 b) {
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __device__ void operator/=(double3& a, double3 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __device__ double3 operator/(double3 a, double b) {
    return make_double3(a.x / b, a.y / b, a.z / b);
}
inline __device__ double3 operator/(double3 a, float b) {
    return make_double3(a.x / b, a.y / b, a.z / b);
}
inline __device__ void operator/=(double3& a, double b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

inline __device__ float4 operator/(float4 a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __device__ float4 operator/(float4 a, double b) {
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __device__ void operator/=(float4& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

inline __device__ double4 operator/(double4 a, float b) {
    return make_double4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __device__ double4 operator/(double4 a, double b) {
    return make_double4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __device__ void operator/=(double4& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

inline __device__ float3 normalize(float3 v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __device__ double3 normalize(double3 v) {
    double invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// Assignment

template <typename T1>
inline __device__ float3 to_float3(const T1& a) {
    float3 b;
    b.x = a.x;
    b.y = a.y;
    b.z = a.z;
    return b;
}

template <typename T1>
inline __device__ double3 to_double3(const T1& a) {
    double3 b;
    b.x = a.x;
    b.y = a.y;
    b.z = a.z;
    return b;
}

template <typename T1, typename T2>
inline __device__ T2 to_real3(const T1& a) {
    T2 b;
    b.x = a.x;
    b.y = a.y;
    b.z = a.z;
    return b;
}

// Cause an error inside a kernel
#define DEME_ABORT_KERNEL(...) \
    {                          \
        printf(__VA_ARGS__);   \
        __threadfence();       \
        asm volatile("trap;"); \
    }

#endif
