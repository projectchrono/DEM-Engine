//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// This file contains modifications of the code by Radu Serban as a part
// of Project Chrono
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt. A copy of the license is below.

// Copyright (c) 2014 projectchrono.org
// All Rights Reserved.

// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
// following conditions are met:

//  - Redistributions of source code must retain the above copyright notice, this list of conditions and the following
//  disclaimer.
//  - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
//  following disclaimer in the documentation and/or other materials provided with the distribution.
//  - Neither the name of the nor the names of its contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef DEME_SAMPLERS_HPP
#define DEME_SAMPLERS_HPP

#include <cmath>
#include <list>
#include <random>
#include <utility>
#include <vector>
#include <DEM/HostSideHelpers.hpp>

namespace deme {

/// Volumetric sampling method.
enum class SamplingType {
    REGULAR_GRID,  ///< Regular (equidistant) grid
    POISSON_DISK,  ///< Poisson Disk sampling
    HCP_PACK       ///< Hexagonally Close Packing
};

/// Base class for different types of point samplers.
class Sampler {
  public:
    Sampler(float separation) : m_separation(separation) {}
    virtual ~Sampler() {}

    /// Return points sampled from the specified box volume.
    std::vector<float3> SampleBox(const float3& center, const float3& halfDim) {
        m_center = center;
        m_size = halfDim;
        return Sample(BOX);
    }

    /// Return points sampled from the specified spherical volume.
    std::vector<float3> SampleSphere(const float3& center, float radius) {
        m_center = center;
        m_size = host_make_float3(radius, radius, radius);
        return Sample(SPHERE);
    }

    /// Return points sampled from the specified X-aligned cylindrical volume.
    std::vector<float3> SampleCylinderX(const float3& center, float radius, float halfHeight) {
        m_center = center;
        m_size = host_make_float3(halfHeight, radius, radius);
        return Sample(CYLINDER_X);
    }

    /// Return points sampled from the specified Y-aligned cylindrical volume.
    std::vector<float3> SampleCylinderY(const float3& center, float radius, float halfHeight) {
        m_center = center;
        m_size = host_make_float3(radius, halfHeight, radius);
        return Sample(CYLINDER_Y);
    }

    /// Return points sampled from the specified Z-aligned cylindrical volume.
    std::vector<float3> SampleCylinderZ(const float3& center, float radius, float halfHeight) {
        m_center = center;
        m_size = host_make_float3(radius, radius, halfHeight);
        return Sample(CYLINDER_Z);
    }

    /// Get the current value of the minimum separation.
    virtual float GetSeparation() const { return m_separation; }

    /// Change the minimum separation for subsequent calls to Sample.
    virtual void SetSeparation(float separation) { m_separation = separation; }

  protected:
    enum VolumeType { BOX, SPHERE, CYLINDER_X, CYLINDER_Y, CYLINDER_Z };

    /// Worker function for sampling the given domain.
    /// Implemented by concrete samplers.
    virtual std::vector<float3> Sample(VolumeType t) = 0;

    /// Utility function to check if a point is inside the sampling volume.
    bool accept(VolumeType t, const float3& p) const {
        float3 vec = p - m_center;
        float fuzz = (m_size.x < 1) ? (float)1e-6 * m_size.x : (float)1e-6;

        switch (t) {
            case BOX:
                return (std::abs(vec.x) <= m_size.x + fuzz) && (std::abs(vec.y) <= m_size.y + fuzz) &&
                       (std::abs(vec.z) <= m_size.z + fuzz);
            case SPHERE:
                return (length(vec) * length(vec) <= m_size.x * m_size.x);
            case CYLINDER_X:
                return (vec.y * vec.y + vec.z * vec.z <= m_size.y * m_size.y) && (std::abs(vec.x) <= m_size.x + fuzz);
            case CYLINDER_Y:
                return (vec.z * vec.z + vec.x * vec.x <= m_size.z * m_size.z) && (std::abs(vec.y) <= m_size.y + fuzz);
            case CYLINDER_Z:
                return (vec.x * vec.x + vec.y * vec.y <= m_size.x * m_size.x) && (std::abs(vec.z) <= m_size.z + fuzz);
            default:
                return false;
        }
    }

    float m_separation;  ///< inter-particle separation
    float3 m_center;     ///< center of the sampling volume
    float3 m_size;       ///< half dimensions of the bounding box of the sampling volume
};

class HCPSampler : public Sampler {
  public:
    HCPSampler(float separation) : Sampler(separation) {}

  private:
    /// Worker function for sampling the given domain.
    virtual std::vector<float3> Sample(VolumeType t) override {
        std::vector<float3> out_points;

        float3 bl = this->m_center - this->m_size;  // start corner of sampling domain

        float dx = this->m_separation;                                  // distance between two points in X direction
        float dy = this->m_separation * (float)(std::sqrt(3.0) / 2);    // distance between two rows in Y direction
        float dz = this->m_separation * (float)(std::sqrt(2.0 / 3.0));  // distance between two layers in Z direction

        int nx = (int)(2 * this->m_size.x / dx) + 1;
        int ny = (int)(2 * this->m_size.y / dy) + 1;
        int nz = (int)(2 * this->m_size.z / dz) + 1;

        for (int k = 0; k < nz; k++) {
            // Y offsets for alternate layers
            float offset_y = (k % 2 == 0) ? 0 : dy / 3;
            for (int j = 0; j < ny; j++) {
                // X offset for current row and layer
                float offset_x = ((j + k) % 2 == 0) ? 0 : dx / 2;
                for (int i = 0; i < nx; i++) {
                    float3 p = bl + host_make_float3(offset_x + i * dx, offset_y + j * dy, k * dz);
                    if (this->accept(t, p))
                        out_points.push_back(p);
                }
            }
        }

        return out_points;
    }
};

class GridSampler : public Sampler {
  public:
    GridSampler(float separation) : Sampler(separation) {
        m_sep3D.x = separation;
        m_sep3D.y = separation;
        m_sep3D.z = separation;
    }
    GridSampler(const float3& separation) : Sampler(separation.x) { m_sep3D = separation; }

    /// Change the minimum separation for subsequent calls to Sample.
    virtual void SetSeparation(float separation) override {
        m_sep3D = host_make_float3(separation, separation, separation);
    }

  private:
    /// Worker function for sampling the given domain.
    virtual std::vector<float3> Sample(VolumeType t) override {
        std::vector<float3> out_points;

        float3 bl = this->m_center - this->m_size;

        int nx = (int)(2 * this->m_size.x / m_sep3D.x) + 1;
        int ny = (int)(2 * this->m_size.y / m_sep3D.y) + 1;
        int nz = (int)(2 * this->m_size.z / m_sep3D.z) + 1;

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    float3 p = bl + host_make_float3(i * m_sep3D.x, j * m_sep3D.y, k * m_sep3D.z);
                    if (this->accept(t, p))
                        out_points.push_back(p);
                }
            }
        }

        return out_points;
    }

    float3 m_sep3D;
};

/// A wrapper for a grid sampler of a box domain
inline std::vector<float3> DEMBoxGridSampler(float3 BoxCenter,
                                             float3 HalfDims,
                                             float GridSizeX,
                                             float GridSizeY = -1.0,
                                             float GridSizeZ = -1.0) {
    if (GridSizeY < 0)
        GridSizeY = GridSizeX;
    if (GridSizeZ < 0)
        GridSizeZ = GridSizeX;
    GridSampler sampler(host_make_float3(GridSizeX, GridSizeY, GridSizeZ));
    return sampler.SampleBox(BoxCenter, HalfDims);
}

/// A wrapper for a HCP sampler of a box domain
inline std::vector<float3> DEMBoxHCPSampler(float3 BoxCenter, float3 HalfDims, float GridSize) {
    HCPSampler sampler(GridSize);
    return sampler.SampleBox(BoxCenter, HalfDims);
}

/// A light-weight sampler that generates a shell made of particles that resembles a cylindrical surface
inline std::vector<float3> DEMCylSurfSampler(float3 CylCenter,
                                             float3 CylAxis,
                                             float CylRad,
                                             float CylHeight,
                                             float ParticleRad,
                                             float spacing = 1.2f) {
    std::vector<float3> points;
    float perimeter = 2.0 * PI * CylRad;
    unsigned int NumRows = perimeter / (spacing * ParticleRad);
    float RadIncr = 2.0 * PI / (float)(NumRows);
    float SideIncr = spacing * ParticleRad;
    float3 UnitCylAxis = normalize(CylAxis);
    float3 RadDir = findPerpendicular<float3>(UnitCylAxis);
    for (unsigned int i = 0; i < NumRows; i++) {
        std::vector<float3> thisRow;
        float3 thisRowSt = CylCenter + UnitCylAxis * (CylHeight / 2.) + RadDir * CylRad;
        for (float d = 0.; d <= CylHeight; d += SideIncr) {
            float3 point;
            point = thisRowSt + UnitCylAxis * (-d);
            thisRow.push_back(point);
        }
        points.insert(points.end(), thisRow.begin(), thisRow.end());
        RadDir = Rodrigues(RadDir, UnitCylAxis, RadIncr);
    }
    return points;
}

}  // namespace deme

#endif
