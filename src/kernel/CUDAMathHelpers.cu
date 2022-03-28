#include <granular/DataStructs.h>
#include <granular/GranularDefines.h>

////////////////////////////////////////////////////////////////////////////////
// Weirdly, a few float3 and double3 operators are not in the cuda toolkit and I have difficulty including thridparty
// cuda helper math here. Even if I can I suspect namespace problems. So they will for now just be defined here
// manually.
////////////////////////////////////////////////////////////////////////////////

inline __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Addition and subtraction

inline __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator+(double3 a, double3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(double3 a, double3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
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

inline __host__ __device__ float3 operator/(float3 a, float3 b) {
    return ::make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3& a, float3 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b) {
    return ::make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
