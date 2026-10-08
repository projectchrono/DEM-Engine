// Copyright (c) 2026, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ostream>
#include <set>
#include <sstream>
#include <string>

#include "../Structs.h"

namespace deme {

// Serialize an owned contact snapshot using the matching column configuration. This helper deliberately has no
// solver access, so asynchronous formatting and disk I/O cannot observe a later frame or resized host arrays.
inline void writeContactSnapshotAsCsv(std::ostream& ptFile,
                                      const ContactInfoContainer& contact_info,
                                      unsigned int output_flags,
                                      const std::set<std::string>& wildcard_names) {
    std::ostringstream outstrstream;

    outstrstream << OUTPUT_FILE_CNT_TYPE_NAME;
    if (output_flags & CNT_OUTPUT_CONTENT::OWNER) {
        outstrstream << "," + OUTPUT_FILE_OWNER_1_NAME + "," + OUTPUT_FILE_OWNER_2_NAME;
    }
    if (output_flags & CNT_OUTPUT_CONTENT::GEO_ID) {
        outstrstream << "," + OUTPUT_FILE_GEO_ID_1_NAME + "," + OUTPUT_FILE_GEO_ID_2_NAME;
    }
    if (output_flags & CNT_OUTPUT_CONTENT::FORCE) {
        outstrstream << "," + OUTPUT_FILE_FORCE_X_NAME + "," + OUTPUT_FILE_FORCE_Y_NAME + "," +
                            OUTPUT_FILE_FORCE_Z_NAME;
    }
    if (output_flags & CNT_OUTPUT_CONTENT::CNT_POINT) {
        outstrstream << "," + OUTPUT_FILE_X_COL_NAME + "," + OUTPUT_FILE_Y_COL_NAME + "," + OUTPUT_FILE_Z_COL_NAME;
    }
    // if (output_flags & CNT_OUTPUT_CONTENT::COMPONENT) {
    //     outstrstream << ","+OUTPUT_FILE_COMP_1_NAME+","+OUTPUT_FILE_COMP_2_NAME;
    // }
    // if (output_flags & CNT_OUTPUT_CONTENT::NICKNAME) {
    //     outstrstream << ","+OUTPUT_FILE_OWNER_NICKNAME_1_NAME+","+OUTPUT_FILE_OWNER_NICKNAME_2_NAME;
    // }
    if (output_flags & CNT_OUTPUT_CONTENT::NORMAL) {
        outstrstream << "," + OUTPUT_FILE_NORMAL_X_NAME + "," + OUTPUT_FILE_NORMAL_Y_NAME + "," +
                            OUTPUT_FILE_NORMAL_Z_NAME;
    }
    if (output_flags & CNT_OUTPUT_CONTENT::TORQUE) {
        outstrstream << "," + OUTPUT_FILE_TORQUE_X_NAME + "," + OUTPUT_FILE_TORQUE_Y_NAME + "," +
                            OUTPUT_FILE_TORQUE_Z_NAME;
    }
    if (output_flags & CNT_OUTPUT_CONTENT::CNT_WILDCARD) {
        // Write all wildcard names as header
        for (const auto& w_name : wildcard_names) {
            outstrstream << "," + w_name;
        }
    }
    outstrstream << "\n";

    for (size_t i = 0; i < contact_info.Size(); i++) {
        outstrstream << contact_info.Get<std::string>("ContactType")[i];

        // (Internal) ownerID and/or geometry ID
        if (output_flags & CNT_OUTPUT_CONTENT::OWNER) {
            outstrstream << "," << contact_info.Get<bodyID_t>("AOwner")[i] << ","
                         << contact_info.Get<bodyID_t>("BOwner")[i];
        }
        if (output_flags & CNT_OUTPUT_CONTENT::GEO_ID) {
            outstrstream << "," << contact_info.Get<bodyID_t>("AGeo")[i] << ","
                         << contact_info.Get<bodyID_t>("BGeo")[i];
        }

        // Force is already in global...
        if (output_flags & CNT_OUTPUT_CONTENT::FORCE) {
            outstrstream << "," << contact_info.Get<float3>("Force")[i].x << ","
                         << contact_info.Get<float3>("Force")[i].y << "," << contact_info.Get<float3>("Force")[i].z;
        }

        if (output_flags & CNT_OUTPUT_CONTENT::CNT_POINT) {
            // oriQ is updated already... whereas the contact point is effectively last step's... That's unfortunate.
            // Should we do somthing ahout it?
            outstrstream << "," << contact_info.Get<float3>("Point")[i].x << ","
                         << contact_info.Get<float3>("Point")[i].y << "," << contact_info.Get<float3>("Point")[i].z;
        }

        if (output_flags & CNT_OUTPUT_CONTENT::NORMAL) {
            outstrstream << "," << contact_info.Get<float3>("Normal")[i].x << ","
                         << contact_info.Get<float3>("Normal")[i].y << "," << contact_info.Get<float3>("Normal")[i].z;
        }

        // Torque is in global already...
        if (output_flags & CNT_OUTPUT_CONTENT::TORQUE) {
            outstrstream << "," << contact_info.Get<float3>("Torque")[i].x << ","
                         << contact_info.Get<float3>("Torque")[i].y << "," << contact_info.Get<float3>("Torque")[i].z;
        }

        // Contact wildcards
        if (output_flags & CNT_OUTPUT_CONTENT::CNT_WILDCARD) {
            // The order shouldn't be an issue... the same set is being processed here and in equip_contact_wildcards,
            // see Model.h
            for (const auto& name : wildcard_names) {
                outstrstream << "," << contact_info.Get<float>(name)[i];
            }
        }

        outstrstream << "\n";
    }

    ptFile << outstrstream.str();
}

}  // namespace deme
