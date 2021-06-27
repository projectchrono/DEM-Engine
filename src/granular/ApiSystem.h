//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <core/ApiVersion.h>
#include <granular/PhysicsSystem.h>
namespace sgps {

class SGPS_impl;

class SGPS_api {
public:
  SGPS_api(float rad);
  virtual ~SGPS_api();

protected:
  SGPS_api() : m_sys(nullptr) {}
  SGPS_impl *m_sys;
};

} // namespace sgps
