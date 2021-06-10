//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#include <core/ApiVersion.h>
#include "ApiSystem.h"

namespace sgps {

SGPS_api::SGPS_api(float rad) {
	m_sys = new SGPS_impl(rad);
}

SGPS_api::~SGPS_api() {
	delete m_sys;
}

}

