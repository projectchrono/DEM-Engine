//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once 

#include <core/ApiVersion.h>
namespace sgps {

class SGPS_impl {
	public:
		virtual ~SGPS_impl();
		friend class SGPS_api;
	protected:
		SGPS_impl() = delete;
		SGPS_impl(float sphere_rad);
		float sphereUU;
};

}

