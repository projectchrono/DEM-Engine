//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <DEM/utils/MeshFrameUtils.hpp>

#include <DEM/BdrsAndObjs.h>
#include <DEM/utils/HostSideHelpers.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace deme {

namespace {

inline float4 normalizeQuatSafe(const float4& q) {
    const double n2 = static_cast<double>(q.x) * q.x + static_cast<double>(q.y) * q.y + static_cast<double>(q.z) * q.z +
                      static_cast<double>(q.w) * q.w;
    if (!(n2 > 0.0) || !std::isfinite(n2)) {
        return make_float4(0, 0, 0, 1);
    }
    const float inv_n = static_cast<float>(1.0 / std::sqrt(n2));
    return make_float4(q.x * inv_n, q.y * inv_n, q.z * inv_n, q.w * inv_n);
}

inline bool inferUniformScale(const float3& target, const float3& source, double& out_scale) {
    constexpr double tiny = 1e-20;
    const double src[3] = {source.x, source.y, source.z};
    const double dst[3] = {target.x, target.y, target.z};
    double ratios[3] = {0.0, 0.0, 0.0};
    int n = 0;
    for (int i = 0; i < 3; i++) {
        if (!std::isfinite(src[i]) || !std::isfinite(dst[i])) {
            return false;
        }
        if (std::abs(src[i]) <= tiny) {
            return false;
        }
        const double r = dst[i] / src[i];
        if (!(r > tiny) || !std::isfinite(r)) {
            return false;
        }
        ratios[n++] = r;
    }
    if (n != 3) {
        return false;
    }
    const double mean = (ratios[0] + ratios[1] + ratios[2]) / 3.0;
    const double denom = std::max(std::abs(mean), tiny);
    double max_rel_dev = 0.0;
    for (int i = 0; i < 3; i++) {
        max_rel_dev = std::max(max_rel_dev, std::abs(ratios[i] - mean) / denom);
    }
    if (max_rel_dev > 2e-2) {
        return false;
    }
    out_scale = mean;
    return true;
}

inline bool jacobiEigenSymmetric3(const double in_A[3][3], double eigvals[3], double eigvecs[3][3]) {
    double A[3][3];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            A[r][c] = in_A[r][c];
            eigvecs[r][c] = (r == c) ? 1.0 : 0.0;
        }
    }

    constexpr int max_iters = 32;
    constexpr double eps = 1e-18;
    for (int it = 0; it < max_iters; it++) {
        int p = 0;
        int q = 1;
        double max_off = std::abs(A[0][1]);
        const double off_02 = std::abs(A[0][2]);
        const double off_12 = std::abs(A[1][2]);
        if (off_02 > max_off) {
            p = 0;
            q = 2;
            max_off = off_02;
        }
        if (off_12 > max_off) {
            p = 1;
            q = 2;
            max_off = off_12;
        }

        const double diag_scale = std::abs(A[0][0]) + std::abs(A[1][1]) + std::abs(A[2][2]) + eps;
        if (max_off <= diag_scale * 1e-14) {
            break;
        }

        const double app = A[p][p];
        const double aqq = A[q][q];
        const double apq = A[p][q];
        if (std::abs(apq) <= eps) {
            continue;
        }

        const double tau = (aqq - app) / (2.0 * apq);
        const double t =
            (tau >= 0.0) ? (1.0 / (tau + std::sqrt(1.0 + tau * tau))) : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        for (int k = 0; k < 3; k++) {
            if (k == p || k == q) {
                continue;
            }
            const double aik = A[k][p];
            const double akq = A[k][q];
            A[k][p] = c * aik - s * akq;
            A[p][k] = A[k][p];
            A[k][q] = c * akq + s * aik;
            A[q][k] = A[k][q];
        }

        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p][q] = 0.0;
        A[q][p] = 0.0;

        for (int k = 0; k < 3; k++) {
            const double vkp = eigvecs[k][p];
            const double vkq = eigvecs[k][q];
            eigvecs[k][p] = c * vkp - s * vkq;
            eigvecs[k][q] = s * vkp + c * vkq;
        }
    }

    eigvals[0] = A[0][0];
    eigvals[1] = A[1][1];
    eigvals[2] = A[2][2];
    return std::isfinite(eigvals[0]) && std::isfinite(eigvals[1]) && std::isfinite(eigvals[2]);
}

inline float4 quatFromRotationMatrix(const double R[3][3]) {
    float4 q = make_float4(0, 0, 0, 1);
    const double trace = R[0][0] + R[1][1] + R[2][2];
    if (trace > 0.0) {
        const double s = std::sqrt(trace + 1.0) * 2.0;
        q.w = static_cast<float>(0.25 * s);
        q.x = static_cast<float>((R[2][1] - R[1][2]) / s);
        q.y = static_cast<float>((R[0][2] - R[2][0]) / s);
        q.z = static_cast<float>((R[1][0] - R[0][1]) / s);
    } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
        const double s = std::sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0;
        q.w = static_cast<float>((R[2][1] - R[1][2]) / s);
        q.x = static_cast<float>(0.25 * s);
        q.y = static_cast<float>((R[0][1] + R[1][0]) / s);
        q.z = static_cast<float>((R[0][2] + R[2][0]) / s);
    } else if (R[1][1] > R[2][2]) {
        const double s = std::sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0;
        q.w = static_cast<float>((R[0][2] - R[2][0]) / s);
        q.x = static_cast<float>((R[0][1] + R[1][0]) / s);
        q.y = static_cast<float>(0.25 * s);
        q.z = static_cast<float>((R[1][2] + R[2][1]) / s);
    } else {
        const double s = std::sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0;
        q.w = static_cast<float>((R[1][0] - R[0][1]) / s);
        q.x = static_cast<float>((R[0][2] + R[2][0]) / s);
        q.y = static_cast<float>((R[1][2] + R[2][1]) / s);
        q.z = static_cast<float>(0.25 * s);
    }
    return normalizeQuatSafe(q);
}

}  // namespace

MeshCanonicalizationResult canonicalizeMeshOwnerFrame(DEMMesh& mesh) {
    MeshCanonicalizationResult result;

    double volume = 0.0;
    float3 center = make_float3(0, 0, 0);
    float3 inertia = make_float3(0, 0, 0);
    float3 inertia_products = make_float3(0, 0, 0);
    mesh.ComputeMassProperties(volume, center, inertia, inertia_products);
    if (!(volume > 0.0) || !std::isfinite(volume)) {
        return result;
    }

    result.center_norm = std::sqrt(center.x * center.x + center.y * center.y + center.z * center.z);
    const double diag_norm = std::abs(inertia.x) + std::abs(inertia.y) + std::abs(inertia.z);
    const double offdiag_norm = std::sqrt(static_cast<double>(inertia_products.x) * inertia_products.x +
                                          static_cast<double>(inertia_products.y) * inertia_products.y +
                                          static_cast<double>(inertia_products.z) * inertia_products.z);
    result.offdiag_rel = offdiag_norm / std::max(diag_norm, 1e-20);

    float bound_r = 0.f;
    for (const auto& v : mesh.m_vertices) {
        bound_r = std::max(bound_r, length(v));
    }
    const double center_tol = std::max(1e-7, 1e-5 * std::max(1e-3, static_cast<double>(bound_r)));
    constexpr double offdiag_tol = 1e-6;
    if (result.center_norm <= center_tol && result.offdiag_rel <= offdiag_tol) {
        return result;
    }

    double A[3][3] = {
        {inertia.x, inertia_products.x, inertia_products.z},
        {inertia_products.x, inertia.y, inertia_products.y},
        {inertia_products.z, inertia_products.y, inertia.z},
    };
    double eigvals[3] = {0.0, 0.0, 0.0};
    double eigvecs[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    if (!jacobiEigenSymmetric3(A, eigvals, eigvecs)) {
        return result;
    }

    std::array<int, 3> order = {0, 1, 2};
    std::sort(order.begin(), order.end(), [&](int l, int r) { return eigvals[l] < eigvals[r]; });

    double R[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    for (int col = 0; col < 3; col++) {
        const int src_col = order[col];
        double nrm = 0.0;
        for (int row = 0; row < 3; row++) {
            R[row][col] = eigvecs[row][src_col];
            nrm += R[row][col] * R[row][col];
        }
        nrm = std::sqrt(std::max(nrm, 1e-30));
        for (int row = 0; row < 3; row++) {
            R[row][col] /= nrm;
        }
    }

    const double det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
                       R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
                       R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
    if (det < 0.0) {
        for (int row = 0; row < 3; row++) {
            R[row][2] = -R[row][2];
        }
    }

    const float4 principal_q = quatFromRotationMatrix(R);
    float3 principal_inertia = make_float3(static_cast<float>(std::max(0.0, eigvals[order[0]])),
                                           static_cast<float>(std::max(0.0, eigvals[order[1]])),
                                           static_cast<float>(std::max(0.0, eigvals[order[2]])));

    const float3 old_init_pos = mesh.init_pos;
    const float4 old_init_ori = mesh.init_oriQ;

    mesh.InformCentroidPrincipal(center, principal_q);
    if (mesh.patch_locations_explicitly_set) {
        for (auto& p_loc : mesh.m_patch_locations) {
            applyFrameTransformGlobalToLocal(p_loc, center, principal_q);
        }
    }

    float3 new_init_pos = center;
    applyFrameTransformLocalToGlobal(new_init_pos, old_init_pos, old_init_ori);
    mesh.init_pos = new_init_pos;
    mesh.init_oriQ = normalizeQuatSafe(hostHamiltonProduct(old_init_ori, principal_q));

    double scale_from_old = 0.0;
    double scale_from_principal = 0.0;
    const bool moi_from_old = inferUniformScale(mesh.MOI, inertia, scale_from_old);
    const bool moi_from_principal = inferUniformScale(mesh.MOI, principal_inertia, scale_from_principal);
    if (moi_from_old && !moi_from_principal) {
        mesh.MOI = principal_inertia * static_cast<float>(scale_from_old);
        result.moi_rescaled = true;
    } else if (!moi_from_principal && result.offdiag_rel > 1e-4) {
        result.moi_inconsistent = true;
    }

    result.transformed = true;
    return result;
}

}  // namespace deme
