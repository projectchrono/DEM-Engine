//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//  All rights reserved.

#pragma once

#include <stdio.h>
#include <stdlib.h>

#define SGPS_ERROR(...)                  \
    {                                    \
        printf("ERROR!");                \
        printf("\n%s", __VA_ARGS__);     \
        printf("\n%s", __func__);        \
        printf(": EXITING SGPS SIM.\n"); \
        exit(1);                         \
    }
