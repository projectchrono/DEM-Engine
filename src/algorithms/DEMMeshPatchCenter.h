// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DEME_MESH_PATCH_CENTER_H
#define DEME_MESH_PATCH_CENTER_H
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace deme {
// Shared host/device policy. These settings belong to the solver, not mass properties or the contact kernel.
struct MeshPatchCenterOptions {
    float alpha = 1.f;
    float beta = 0.01f;
    float gamma = 0.8f;
};

// Accumulate centroid coordinates, oriented twice-area vector, and total twice-area in double precision.
// Cross products are evaluated after promotion so small triangles and large coordinates do not overflow float
// intermediates. Keeping the scalar area separately lets us detect cancellation independently of mesh scale.
__host__ __device__ inline void AccumulatePatchTriangle(double* sum, float3 a, float3 b, float3 c) {
    const double ux = double(b.x) - a.x, uy = double(b.y) - a.y, uz = double(b.z) - a.z;
    const double vx = double(c.x) - a.x, vy = double(c.y) - a.y, vz = double(c.z) - a.z;
    const double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    sum[0] += (double(a.x) + b.x + c.x) / 3.;
    sum[1] += (double(a.y) + b.y + c.y) / 3.;
    sum[2] += (double(a.z) + b.z + c.z) / 3.;
    sum[3] += nx;
    sum[4] += ny;
    sum[5] += nz;
    sum[6] += sqrt(nx * nx + ny * ny + nz * nz);
}

// A surface centroid is a poor reference for the depth/center-distance rejection guard: legitimate contacts
// close to that centroid have almost zero permitted penetration. Move automatic multi-patch centers behind
// their oriented surface instead. Both an exterior face and an inward-facing container wall move along -n;
// testing which side contains the CoM is unnecessary. Local coordinates have their implicit CoM at the origin.
// h = |(CoM-c).n| uses only normal distance, avoiding offsets caused by tangential displacement from the CoM.
// d = alpha * max(gamma*h, beta*L), where L is the CURRENT whole-mesh local bounding-box diagonal (including
// unused vertices). The fallback handles planar meshes whose CoM lies in the plane, and does not shrink with
// triangle subdivision. This is a heuristic, not a solid-interior or visibility test: inconsistent winding,
// folded/concave patches and thin walls can still make a center inappropriate. Larger offsets also admit more
// deep contacts. Explicit centers bypass this policy, and alpha=0 restores the historical averaged centers.
// A resultant normal below 1e-6 of the sum of triangle areas is unreliable; leave such a patch unshifted.
// Single-patch meshes are handled by callers and retain their local-origin center. No mass properties change.
__host__ __device__ inline float3 FinishPatchCenter(const double* sum,
                                                    size_t count,
                                                    double diagonal,
                                                    MeshPatchCenterOptions options) {
    if (!count)
        return make_float3(0, 0, 0);
    double x = sum[0] / count, y = sum[1] / count, z = sum[2] / count;
    const double norm = sqrt(sum[3] * sum[3] + sum[4] * sum[4] + sum[5] * sum[5]);
    if (options.alpha > 0 && norm > 1.e-6 * sum[6]) {
        const double nx = sum[3] / norm, ny = sum[4] / norm, nz = sum[5] / norm;
        const double h = fabs(x * nx + y * ny + z * nz);
        const double d = double(options.alpha) * fmax(double(options.gamma) * h, double(options.beta) * diagonal);
        x -= d * nx;
        y -= d * ny;
        z -= d * nz;
    }
    return make_float3(float(x), float(y), float(z));
}
}  // namespace deme
#endif
