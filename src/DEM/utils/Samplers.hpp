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

#include "../HostSideHelpers.hpp"

namespace deme {

// -----------------------------------------------------------------------------
// Construct a single random engine (on first use)
//
// Note that this object is never destructed (but this is OK)
// -----------------------------------------------------------------------------
inline std::default_random_engine& rengine() {
    static std::default_random_engine* re = new std::default_random_engine;
    return *re;
}

// -----------------------------------------------------------------------------
// sampleTruncatedDist
//
// Utility function for generating samples from a truncated normal distribution.
// -----------------------------------------------------------------------------
template <typename T>
inline T sampleTruncatedDist(std::normal_distribution<T>& distribution, T minVal, T maxVal) {
    T val;

    do {
        val = distribution(rengine());
    } while (val < minVal || val > maxVal);

    return val;
}

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
    std::vector<std::vector<float>> SampleBox(const std::vector<float>& center, const std::vector<float>& halfDim) {
        assertThreeElements(center, "SampleBox", "center");
        assertThreeElements(halfDim, "SampleBox", "halfDim");
        m_center = make_float3(center[0], center[1], center[2]);
        m_size = make_float3(halfDim[0], halfDim[1], halfDim[2]);
        auto res = Sample(BOX);
        std::vector<std::vector<float>> xyz(res.size(), std::vector<float>(3, 0.));
        for (size_t i = 0; i < res.size(); i++) {
            xyz[i][0] = res[i].x;
            xyz[i][1] = res[i].y;
            xyz[i][2] = res[i].z;
        }
        return xyz;
    }

    /// Return points sampled from the specified spherical volume.
    std::vector<float3> SampleSphere(const float3& center, float radius) {
        m_center = center;
        m_size = make_float3(radius, radius, radius);
        return Sample(SPHERE);
    }
    std::vector<std::vector<float>> SampleSphere(const std::vector<float>& center, float radius) {
        assertThreeElements(center, "SampleSphere", "center");
        m_center = make_float3(center[0], center[1], center[2]);
        m_size = make_float3(radius, radius, radius);
        auto res = Sample(SPHERE);
        std::vector<std::vector<float>> xyz(res.size(), std::vector<float>(3, 0.));
        for (size_t i = 0; i < res.size(); i++) {
            xyz[i][0] = res[i].x;
            xyz[i][1] = res[i].y;
            xyz[i][2] = res[i].z;
        }
        return xyz;
    }

    /// Return points sampled from the specified X-aligned cylindrical volume.
    std::vector<float3> SampleCylinderX(const float3& center, float radius, float halfHeight) {
        m_center = center;
        m_size = make_float3(halfHeight, radius, radius);
        return Sample(CYLINDER_X);
    }

    std::vector<std::vector<float>> SampleCylinderX(const std::vector<float>& center, float radius, float halfHeight) {
        assertThreeElements(center, "SampleCylinderX", "center");
        m_center = make_float3(center[0], center[1], center[2]);
        m_size = make_float3(halfHeight, radius, radius);
        auto res = Sample(CYLINDER_X);
        std::vector<std::vector<float>> xyz(res.size(), std::vector<float>(3, 0.));
        for (size_t i = 0; i < res.size(); i++) {
            xyz[i][0] = res[i].x;
            xyz[i][1] = res[i].y;
            xyz[i][2] = res[i].z;
        }
        return xyz;
    }

    /// Return points sampled from the specified Y-aligned cylindrical volume.
    std::vector<float3> SampleCylinderY(const float3& center, float radius, float halfHeight) {
        m_center = center;
        m_size = make_float3(radius, halfHeight, radius);
        return Sample(CYLINDER_Y);
    }

    std::vector<std::vector<float>> SampleCylinderY(const std::vector<float>& center, float radius, float halfHeight) {
        assertThreeElements(center, "SampleCylinderY", "center");
        m_center = make_float3(center[0], center[1], center[2]);
        m_size = make_float3(radius, halfHeight, radius);
        auto res = Sample(CYLINDER_Y);
        std::vector<std::vector<float>> xyz(res.size(), std::vector<float>(3, 0.));
        for (size_t i = 0; i < res.size(); i++) {
            xyz[i][0] = res[i].x;
            xyz[i][1] = res[i].y;
            xyz[i][2] = res[i].z;
        }
        return xyz;
    }

    /// Return points sampled from the specified Z-aligned cylindrical volume.
    std::vector<float3> SampleCylinderZ(const float3& center, float radius, float halfHeight) {
        m_center = center;
        m_size = make_float3(radius, radius, halfHeight);
        return Sample(CYLINDER_Z);
    }
    std::vector<std::vector<float>> SampleCylinderZ(const std::vector<float>& center, float radius, float halfHeight) {
        assertThreeElements(center, "SampleCylinderZ", "center");
        m_center = make_float3(center[0], center[1], center[2]);
        m_size = make_float3(radius, radius, halfHeight);
        auto res = Sample(CYLINDER_Z);
        std::vector<std::vector<float>> xyz(res.size(), std::vector<float>(3, 0.));
        for (size_t i = 0; i < res.size(); i++) {
            xyz[i][0] = res[i].x;
            xyz[i][1] = res[i].y;
            xyz[i][2] = res[i].z;
        }
        return xyz;
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

class PDGrid {
  public:
    typedef std::pair<float3, bool> Content;

    PDGrid() {}

    int GetDimX() const { return m_dimX; }
    int GetDimY() const { return m_dimY; }
    int GetDimZ() const { return m_dimZ; }

    void Resize(int dimX, int dimY, int dimZ) {
        m_dimX = dimX;
        m_dimY = dimY;
        m_dimZ = dimZ;
        m_data.resize(dimX * dimY * dimZ, Content(make_float3(0, 0, 0), true));
    }

    void SetCellPoint(int i, int j, int k, const float3& p) {
        int ii = index(i, j, k);
        m_data[ii].first = p;
        m_data[ii].second = false;
    }

    const float3& GetCellPoint(int i, int j, int k) const { return m_data[index(i, j, k)].first; }

    bool IsCellEmpty(int i, int j, int k) const {
        if (i < 0 || i >= m_dimX || j < 0 || j >= m_dimY || k < 0 || k >= m_dimZ)
            return true;

        return m_data[index(i, j, k)].second;
    }

    Content& operator()(int i, int j, int k) { return m_data[index(i, j, k)]; }
    const Content& operator()(int i, int j, int k) const { return m_data[index(i, j, k)]; }

  private:
    int index(int i, int j, int k) const { return i * m_dimY * m_dimZ + j * m_dimZ + k; }

    int m_dimX;
    int m_dimY;
    int m_dimZ;
    std::vector<Content> m_data;
};

// PD
class PDSampler : public Sampler {
  public:
    typedef std::vector<float3> PointVector;
    typedef std::list<float3> PointList;

    /// Construct a Poisson Disk sampler with specified minimum distance.
    PDSampler(float separation, int pointsPerIteration = m_ppi_default)
        : Sampler(separation), m_ppi(pointsPerIteration), m_realDist(0.0, 1.0) {
        m_gridLoc.resize(3);
        m_realDist.reset();
        rengine().seed(0);
    }

    /// Set the current state of the random-number engine (default: 0).
    void SetRandomEngineSeed(unsigned int seed) { rengine().seed(seed); }

  private:
    enum Direction2D { NONE, X_DIR, Y_DIR, Z_DIR };

    /// Worker function for sampling the given domain.
    virtual PointVector Sample(VolumeType t) override {
        PointVector out_points;

        // Check 2D/3D. If the size in one direction (e.g. z) is less than the
        // minimum distance, we switch to a 2D sampling. All sample points will
        // have p.z = m_center.z
        if (this->m_size.z < this->m_separation) {
            m_2D = Z_DIR;
            m_cellSize = this->m_separation / std::sqrt(2.0);
            this->m_size.z = 0;
        } else if (this->m_size.y < this->m_separation) {
            m_2D = Y_DIR;
            m_cellSize = this->m_separation / std::sqrt(2.0);
            this->m_size.y = 0;
        } else if (this->m_size.x < this->m_separation) {
            m_2D = X_DIR;
            m_cellSize = this->m_separation / std::sqrt(2.0);
            this->m_size.x = 0;
        } else {
            m_2D = NONE;
            m_cellSize = this->m_separation / std::sqrt(3.0);
        }

        m_bl = this->m_center - this->m_size;
        m_tr = this->m_center + this->m_size;

        m_grid.Resize((int)(2 * this->m_size.x / m_cellSize) + 1, (int)(2 * this->m_size.y / m_cellSize) + 1,
                      (int)(2 * this->m_size.z / m_cellSize) + 1);

        // Add the first output point (and initialize active list)
        AddFirstPoint(t, out_points);

        // As long as there are active points...
        while (m_active.size() != 0) {
            // ... select one of them at random
            std::uniform_int_distribution<int> intDist(0, (int)m_active.size() - 1);

            typename PointList::iterator point = m_active.begin();
            std::advance(point, intDist(rengine()));

            // ... attempt to add points near the active one
            bool found = false;

            for (int k = 0; k < m_ppi; k++)
                found |= AddNextPoint(t, *point, out_points);

            // ... if not possible, remove the current active point
            if (!found)
                m_active.erase(point);
        }

        return out_points;
    }

    /// Add the first point in the volume (selected randomly).
    void AddFirstPoint(VolumeType t, PointVector& out_points) {
        float3 p;

        // Generate a random point in the domain
        do {
            p.x = m_bl.x + m_realDist(rengine()) * 2 * this->m_size.x;
            p.y = m_bl.y + m_realDist(rengine()) * 2 * this->m_size.y;
            p.z = m_bl.z + m_realDist(rengine()) * 2 * this->m_size.z;
        } while (!this->accept(t, p));

        // Place the point in the grid, add it to the active list, and add it
        // to output.
        MapToGrid(p);

        m_grid.SetCellPoint(m_gridLoc[0], m_gridLoc[1], m_gridLoc[2], p);
        m_active.push_back(p);
        out_points.push_back(p);
    }

    /// Attempt to add a new point, close to the specified one.
    bool AddNextPoint(VolumeType t, const float3& point, PointVector& out_points) {
        // Generate a random candidate point in the neighborhood of the
        // specified point.
        float3 q = GenerateRandomNeighbor(point);

        // Check if point is in the domain.
        if (!this->accept(t, q))
            return false;

        // Check distance from candidate point to any existing point in the grid
        // (note that we only need to check 5x5x5 surrounding grid cells).
        MapToGrid(q);

        for (int i = m_gridLoc[0] - 2; i < m_gridLoc[0] + 3; i++) {
            for (int j = m_gridLoc[1] - 2; j < m_gridLoc[1] + 3; j++) {
                for (int k = m_gridLoc[2] - 2; k < m_gridLoc[2] + 3; k++) {
                    if (m_grid.IsCellEmpty(i, j, k))
                        continue;
                    float3 dist = q - m_grid.GetCellPoint(i, j, k);
                    if (dot(dist, dist) < this->m_separation * this->m_separation)
                        return false;
                }
            }
        }

        // The candidate point is acceptable.
        // Place it in the grid, add it to the active list, and add it to the
        // output.
        m_grid.SetCellPoint(m_gridLoc[0], m_gridLoc[1], m_gridLoc[2], q);
        m_active.push_back(q);
        out_points.push_back(q);

        return true;
    }

    /// Return a random point in spherical anulus between sep and 2*sep centered at given point.
    float3 GenerateRandomNeighbor(const float3& point) {
        float x, y, z;

        switch (m_2D) {
            case Z_DIR: {
                float radius = this->m_separation * (1 + m_realDist(rengine()));
                float angle = 2 * PI * m_realDist(rengine());
                x = point.x + radius * std::cos(angle);
                y = point.y + radius * std::sin(angle);
                z = this->m_center.z;
            } break;
            case Y_DIR: {
                float radius = this->m_separation * (1 + m_realDist(rengine()));
                float angle = 2 * PI * m_realDist(rengine());
                x = point.x + radius * std::cos(angle);
                y = this->m_center.y;
                z = point.z + radius * std::sin(angle);
            } break;
            case X_DIR: {
                float radius = this->m_separation * (1 + m_realDist(rengine()));
                float angle = 2 * PI * m_realDist(rengine());
                x = this->m_center.x;
                y = point.y + radius * std::cos(angle);
                z = point.z + radius * std::sin(angle);
            } break;
            default:
            case NONE: {
                float radius = this->m_separation * (1 + m_realDist(rengine()));
                float angle1 = 2 * PI * m_realDist(rengine());
                float angle2 = 2 * PI * m_realDist(rengine());
                x = point.x + radius * std::cos(angle1) * std::sin(angle2);
                y = point.y + radius * std::sin(angle1) * std::sin(angle2);
                z = point.z + radius * std::cos(angle2);
            } break;
        }

        return make_float3(x, y, z);
    }

    /// Map point location to a 3D grid location.
    void MapToGrid(float3 point) {
        m_gridLoc[0] = (int)((point.x - m_bl.x) / m_cellSize);
        m_gridLoc[1] = (int)((point.y - m_bl.y) / m_cellSize);
        m_gridLoc[2] = (int)((point.z - m_bl.z) / m_cellSize);
    }

    PDGrid m_grid;
    PointList m_active;

    Direction2D m_2D;  ///< 2D or 3D sampling
    float3 m_bl;       ///< bottom-left corner of sampling domain
    float3 m_tr;       ///< top-right corner of sampling domain      REMOVE?
    float m_cellSize;  ///< grid cell size

    std::vector<int> m_gridLoc;
    int m_ppi;  ///< maximum points per iteration

    /// Generate real numbers uniformly distributed in (0,1)
    std::uniform_real_distribution<float> m_realDist;

    static const int m_ppi_default = 30;
};

/// Poisson Disk sampler for sampling a 3D box in layers.
/// The computational efficiency of PD sampling degrades as points are added, especially for large volumes.
/// This class provides an alternative sampling method where PD sampling is done in 2D layers, separated by a specified
/// distance (padding_factor * diam). This significantly improves computational efficiency of the sampling but at the
/// cost of discarding the PD uniform distribution properties in the direction orthogonal to the layers.
std::vector<float3> PDLayerSampler_BOX(float3 center,                ///< Center of axis-aligned box to fill
                                       float3 hdims,                 ///< Half-dimensions along the x, y, and z axes
                                       float diam,                   ///< Particle diameter
                                       float padding_factor = 1.02,  ///< Multiplier on particle diameter for spacing
                                       bool verbose = false          ///< Output progress during generation
) {
    float fill_bottom = center.z - hdims.z;
    float fill_top = center.z + hdims.z;

    // set center to bottom
    center.z = fill_bottom;
    // 2D layer
    hdims.z = 0;

    PDSampler sampler(diam * padding_factor);
    std::vector<float3> points_full;
    while (center.z < fill_top) {
        if (verbose) {
            std::cout << "Create layer at " << center.z << std::endl;
        }
        auto points = sampler.SampleBox(center, hdims);
        points_full.insert(points_full.end(), points.begin(), points.end());
        center.z += diam * padding_factor;
    }
    return points_full;
}

// HCP
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
                    float3 p = bl + make_float3(offset_x + i * dx, offset_y + j * dy, k * dz);
                    if (this->accept(t, p))
                        out_points.push_back(p);
                }
            }
        }

        return out_points;
    }
};

// Grid
class GridSampler : public Sampler {
  public:
    GridSampler(float separation) : Sampler(separation) {
        m_sep3D.x = separation;
        m_sep3D.y = separation;
        m_sep3D.z = separation;
    }
    GridSampler(const float3& separation) : Sampler(separation.x) { m_sep3D = separation; }

    /// Change the minimum separation for subsequent calls to Sample.
    virtual void SetSeparation(float separation) override { m_sep3D = make_float3(separation, separation, separation); }

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
                    float3 p = bl + make_float3(i * m_sep3D.x, j * m_sep3D.y, k * m_sep3D.z);
                    if (this->accept(t, p))
                        out_points.push_back(p);
                }
            }
        }

        return out_points;
    }

    float3 m_sep3D;
};

/// A wrapper for a grid sampler of a box domain.
inline std::vector<float3> DEMBoxGridSampler(float3 BoxCenter,
                                             float3 HalfDims,
                                             float GridSizeX,
                                             float GridSizeY = -1.0,
                                             float GridSizeZ = -1.0) {
    if (GridSizeY < 0)
        GridSizeY = GridSizeX;
    if (GridSizeZ < 0)
        GridSizeZ = GridSizeX;
    GridSampler sampler(make_float3(GridSizeX, GridSizeY, GridSizeZ));
    return sampler.SampleBox(BoxCenter, HalfDims);
}
/// A wrapper for a grid sampler of a box domain.
inline std::vector<std::vector<float>> DEMBoxGridSampler(const std::vector<float>& BoxCenter,
                                                         const std::vector<float>& HalfDims,
                                                         float GridSizeX,
                                                         float GridSizeY = -1.0,
                                                         float GridSizeZ = -1.0) {
    assertThreeElements(BoxCenter, "DEMBoxGridSampler", "BoxCenter");
    assertThreeElements(HalfDims, "DEMBoxGridSampler", "HalfDims");

    std::vector<std::vector<float>> return_vec;

    std::vector<float3> return_val =
        DEMBoxGridSampler(make_float3(BoxCenter[0], BoxCenter[1], BoxCenter[2]),
                          make_float3(HalfDims[0], HalfDims[1], HalfDims[2]), GridSizeX, GridSizeY, GridSizeZ);

    for (int i = 0; i < return_val.size(); i++) {
        std::vector<float> tmp;

        tmp.push_back(return_val[i].x);
        tmp.push_back(return_val[i].y);
        tmp.push_back(return_val[i].z);

        return_vec.push_back(tmp);
    }

    return return_vec;
}

/// A wrapper for a HCP sampler of a box domain.
inline std::vector<float3> DEMBoxHCPSampler(float3 BoxCenter, float3 HalfDims, float GridSize) {
    HCPSampler sampler(GridSize);
    return sampler.SampleBox(BoxCenter, HalfDims);
}

/// A wrapper for a HCP sampler of a box domain.
inline std::vector<std::vector<float>> DEMBoxHCPSampler(const std::vector<float>& BoxCenter,
                                                        const std::vector<float>& HalfDims,
                                                        float GridSize) {
    assertThreeElements(BoxCenter, "DEMBoxHCPSampler", "BoxCenter");
    assertThreeElements(HalfDims, "DEMBoxHCPSampler", "HalfDims");

    std::vector<std::vector<float>> return_vec;

    std::vector<float3> return_val = DEMBoxHCPSampler(make_float3(BoxCenter[0], BoxCenter[1], BoxCenter[2]),
                                                      make_float3(HalfDims[0], HalfDims[1], HalfDims[2]), GridSize);

    for (int i = 0; i < return_val.size(); i++) {
        std::vector<float> tmp;

        tmp.push_back(return_val[i].x);
        tmp.push_back(return_val[i].y);
        tmp.push_back(return_val[i].z);

        return_vec.push_back(tmp);
    }

    return return_vec;
}

/// A light-weight sampler that generates a shell made of particles that resembles a cylindrical surface.
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
/// A light-weight sampler that generates a shell made of particles that resembles a cylindrical surface.
inline std::vector<std::vector<float>> DEMCylSurfSampler(const std::vector<float>& CylCenter,
                                                         const std::vector<float>& CylAxis,
                                                         float CylRad,
                                                         float CylHeight,
                                                         float ParticleRad,
                                                         float spacing = 1.2f) {
    assertThreeElements(CylCenter, "DEMCylSurfSampler", "CylCenter");
    assertThreeElements(CylAxis, "DEMCylSurfSampler", "CylAxis");
    std::vector<float3> res_float3 =
        DEMCylSurfSampler(make_float3(CylCenter[0], CylCenter[1], CylCenter[2]),
                          make_float3(CylAxis[0], CylAxis[1], CylAxis[2]), CylRad, CylHeight, ParticleRad, spacing);
    std::vector<std::vector<float>> res(res_float3.size());
    for (size_t i = 0; i < res_float3.size(); i++) {
        res[i] = {res_float3[i].x, res_float3[i].y, res_float3[i].z};
    }
    return res;
}

}  // namespace deme

#endif
