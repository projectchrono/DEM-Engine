// Creative Commons Legal Code
//
// CC0 1.0 Universal
//
//     CREATIVE COMMONS CORPORATION IS NOT A LAW FIRM AND DOES NOT PROVIDE
//     LEGAL SERVICES. DISTRIBUTION OF THIS DOCUMENT DOES NOT CREATE AN
//     ATTORNEY-CLIENT RELATIONSHIP. CREATIVE COMMONS PROVIDES THIS
//     INFORMATION ON AN "AS-IS" BASIS. CREATIVE COMMONS MAKES NO WARRANTIES
//     REGARDING THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS
//     PROVIDED HEREUNDER, AND DISCLAIMS LIABILITY FOR DAMAGES RESULTING FROM
//     THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS PROVIDED
//     HEREUNDER.
//
// Statement of Purpose
//
// The laws of most jurisdictions throughout the world automatically confer
// exclusive Copyright and Related Rights (defined below) upon the creator
// and subsequent owner(s) (each and all, an "owner") of an original work of
// authorship and/or a database (each, a "Work").
//
// Certain owners wish to permanently relinquish those rights to a Work for
// the purpose of contributing to a commons of creative, cultural and
// scientific works ("Commons") that the public can reliably and without fear
// of later claims of infringement build upon, modify, incorporate in other
// works, reuse and redistribute as freely as possible in any form whatsoever
// and for any purposes, including without limitation commercial purposes.
// These owners may contribute to the Commons to promote the ideal of a free
// culture and the further production of creative, cultural and scientific
// works, or to gain reputation or greater distribution for their Work in
// part through the use and efforts of others.
//
// For these and/or other purposes and motivations, and without any
// expectation of additional consideration or compensation, the person
// associating CC0 with a Work (the "Affirmer"), to the extent that he or she
// is an owner of Copyright and Related Rights in the Work, voluntarily
// elects to apply CC0 to the Work and publicly distribute the Work under its
// terms, with knowledge of his or her Copyright and Related Rights in the
// Work and the meaning and intended legal effect of CC0 on those rights.
//
// 1. Copyright and Related Rights. A Work made available under CC0 may be
// protected by copyright and related or neighboring rights ("Copyright and
// Related Rights"). Copyright and Related Rights include, but are not
// limited to, the following:
//
//   i. the right to reproduce, adapt, distribute, perform, display,
//      communicate, and translate a Work;
//  ii. moral rights retained by the original author(s) and/or performer(s);
// iii. publicity and privacy rights pertaining to a person's image or
//      likeness depicted in a Work;
//  iv. rights protecting against unfair competition in regards to a Work,
//      subject to the limitations in paragraph 4(a), below;
//   v. rights protecting the extraction, dissemination, use and reuse of data
//      in a Work;
//  vi. database rights (such as those arising under Directive 96/9/EC of the
//      European Parliament and of the Council of 11 March 1996 on the legal
//      protection of databases, and under any national implementation
//      thereof, including any amended or successor version of such
//      directive); and
// vii. other similar, equivalent or corresponding rights throughout the
//      world based on applicable law or treaty, and any national
//      implementations thereof.
//
// 2. Waiver. To the greatest extent permitted by, but not in contravention
// of, applicable law, Affirmer hereby overtly, fully, permanently,
// irrevocably and unconditionally waives, abandons, and surrenders all of
// Affirmer's Copyright and Related Rights and associated claims and causes
// of action, whether now known or unknown (including existing as well as
// future claims and causes of action), in the Work (i) in all territories
// worldwide, (ii) for the maximum duration provided by applicable law or
// treaty (including future time extensions), (iii) in any current or future
// medium and for any number of copies, and (iv) for any purpose whatsoever,
// including without limitation commercial, advertising or promotional
// purposes (the "Waiver"). Affirmer makes the Waiver for the benefit of each
// member of the public at large and to the detriment of Affirmer's heirs and
// successors, fully intending that such Waiver shall not be subject to
// revocation, rescission, cancellation, termination, or any other legal or
// equitable action to disrupt the quiet enjoyment of the Work by the public
// as contemplated by Affirmer's express Statement of Purpose.
//
// 3. Public License Fallback. Should any part of the Waiver for any reason
// be judged legally invalid or ineffective under applicable law, then the
// Waiver shall be preserved to the maximum extent permitted taking into
// account Affirmer's express Statement of Purpose. In addition, to the
// extent the Waiver is so judged Affirmer hereby grants to each affected
// person a royalty-free, non transferable, non sublicensable, non exclusive,
// irrevocable and unconditional license to exercise Affirmer's Copyright and
// Related Rights in the Work (i) in all territories worldwide, (ii) for the
// maximum duration provided by applicable law or treaty (including future
// time extensions), (iii) in any current or future medium and for any number
// of copies, and (iv) for any purpose whatsoever, including without
// limitation commercial, advertising or promotional purposes (the
// "License"). The License shall be deemed effective as of the date CC0 was
// applied by Affirmer to the Work. Should any part of the License for any
// reason be judged legally invalid or ineffective under applicable law, such
// partial invalidity or ineffectiveness shall not invalidate the remainder
// of the License, and in such case Affirmer hereby affirms that he or she
// will not (i) exercise any of his or her remaining Copyright and Related
// Rights in the Work or (ii) assert any associated claims and causes of
// action with respect to the Work, in either case contrary to Affirmer's
// express Statement of Purpose.
//
// 4. Limitations and Disclaimers.
//
//  a. No trademark or patent rights held by Affirmer are waived, abandoned,
//     surrendered, licensed or otherwise affected by this document.
//  b. Affirmer offers the Work as-is and makes no representations or
//     warranties of any kind concerning the Work, express, implied,
//     statutory or otherwise, including without limitation warranties of
//     title, merchantability, fitness for a particular purpose, non
//     infringement, or the absence of latent or other defects, accuracy, or
//     the present or absence of errors, whether or not discoverable, all to
//     the greatest extent permissible under applicable law.
//  c. Affirmer disclaims responsibility for clearing rights of other persons
//     that may apply to the Work or any use thereof, including without
//     limitation any person's Copyright and Related Rights in the Work.
//     Further, Affirmer disclaims responsibility for obtaining any necessary
//     consents, permissions or other rights required for any use of the
//     Work.
//  d. Affirmer understands and acknowledges that Creative Commons is not a
//     party to this document and has no duty or obligation with respect to
//     this CC0 or use of the Work.

/********************************************************/
/* AABB-triangle overlap test code                      */
/* originally by Tomas Akenine-Möller                   */
/* modified by Ruochun Zhang for DEM-Engine             */
/* Function: int triBoxOverlap(float boxcenter[3],      */
/*          float boxhalfsize[3],float triverts[3][3]); */
/* History:                                             */
/*   2001-03-05: released the code in its first version */
/*   2001-06-18: changed the order of the tests, faster */
/*                                                      */
/* Acknowledgement: Many thanks to Pierre Terdiman for  */
/* suggestions and discussions on how to optimize code. */
/* Thanks to David Hunt for finding a ">="-bug!         */
/********************************************************/

#pragma once
// Fast triangle-box (cube) overlap predicate for the *union* of two triangles:
//  - Triangle A: (v0, v1, v2) in box-centered coordinates
//  - Triangle B: (v0+t, v1+t, v2+t) where t is a constant translation in the same coordinate system
//
// Key idea (Option C):
//  - Compute edges/abs/normal once (from A in box frame).
//  - For the 2nd triangle, reuse SAT projections by shifting min/max with the translation-induced delta.
//  - This avoids running a full second SAT for the sandwich triangle.
//
// The function supports bounds gating via testA/testB: if one is false it is skipped.

#include <cuda_runtime.h>

__device__ __forceinline__ bool _deme_sep_axis_fp32(float mn, float mx, float rad, float eps) {
    // Separating axis exists if interval [mn,mx] is entirely outside [-rad,rad]
    return (mn > rad + eps) || (mx < -rad - eps);
}

__device__ __forceinline__ bool triBoxOverlapBinLocalEdgesUnionShiftFP32(const float3& v0,
                                                                         const float3& v1,
                                                                         const float3& v2,
                                                                         const float3& t,
                                                                         float h,
                                                                         bool testA,
                                                                         bool testB,
                                                                         float eps = 0.0f) {
    bool okA = testA;
    bool okB = testB;

    if (!okA && !okB)
        return false;

    // Edges in box-centered frame (numerically aligned with reference)
    const float3 e0 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
    const float3 e1 = {v2.x - v1.x, v2.y - v1.y, v2.z - v1.z};
    const float3 e2 = {v0.x - v2.x, v0.y - v2.y, v0.z - v2.z};

    const float3 ae0 = {fabsf(e0.x), fabsf(e0.y), fabsf(e0.z)};
    const float3 ae1 = {fabsf(e1.x), fabsf(e1.y), fabsf(e1.z)};
    const float3 ae2 = {fabsf(e2.x), fabsf(e2.y), fabsf(e2.z)};

    float p0, p1, p2, mn, mx, rad;

    // ---- 9 cross-axis tests ----
    // Edge e0: X01
    p0 = e0.z * v0.y - e0.y * v0.z;
    p2 = e0.z * v2.y - e0.y * v2.z;
    mn = fminf(p0, p2);
    mx = fmaxf(p0, p2);
    rad = h * (ae0.z + ae0.y);
    if (okA && _deme_sep_axis_fp32(mn, mx, rad, eps))
        okA = false;
    if (okB) {
        const float d = e0.z * t.y - e0.y * t.z;
        if (_deme_sep_axis_fp32(mn + d, mx + d, rad, eps))
            okB = false;
    }
    if (!okA && !okB)
        return false;

    // Edge e0: Y02
    p0 = -e0.z * v0.x + e0.x * v0.z;
    p2 = -e0.z * v2.x + e0.x * v2.z;
    mn = fminf(p0, p2);
    mx = fmaxf(p0, p2);
    rad = h * (ae0.z + ae0.x);
    if (okA && _deme_sep_axis_fp32(mn, mx, rad, eps))
        okA = false;
    if (okB) {
        const float d = -e0.z * t.x + e0.x * t.z;
        if (_deme_sep_axis_fp32(mn + d, mx + d, rad, eps))
            okB = false;
    }
    if (!okA && !okB)
        return false;

    // Edge e0: Z12
    p1 = e0.y * v1.x - e0.x * v1.y;
    p2 = e0.y * v2.x - e0.x * v2.y;
    mn = fminf(p1, p2);
    mx = fmaxf(p1, p2);
    rad = h * (ae0.y + ae0.x);
    if (okA && _deme_sep_axis_fp32(mn, mx, rad, eps))
        okA = false;
    if (okB) {
        const float d = e0.y * t.x - e0.x * t.y;
        if (_deme_sep_axis_fp32(mn + d, mx + d, rad, eps))
            okB = false;
    }
    if (!okA && !okB)
        return false;

    // Edge e1: X01
    p0 = e1.z * v0.y - e1.y * v0.z;
    p2 = e1.z * v2.y - e1.y * v2.z;
    mn = fminf(p0, p2);
    mx = fmaxf(p0, p2);
    rad = h * (ae1.z + ae1.y);
    if (okA && _deme_sep_axis_fp32(mn, mx, rad, eps))
        okA = false;
    if (okB) {
        const float d = e1.z * t.y - e1.y * t.z;
        if (_deme_sep_axis_fp32(mn + d, mx + d, rad, eps))
            okB = false;
    }
    if (!okA && !okB)
        return false;

    // Edge e1: Y02
    p0 = -e1.z * v0.x + e1.x * v0.z;
    p2 = -e1.z * v2.x + e1.x * v2.z;
    mn = fminf(p0, p2);
    mx = fmaxf(p0, p2);
    rad = h * (ae1.z + ae1.x);
    if (okA && _deme_sep_axis_fp32(mn, mx, rad, eps))
        okA = false;
    if (okB) {
        const float d = -e1.z * t.x + e1.x * t.z;
        if (_deme_sep_axis_fp32(mn + d, mx + d, rad, eps))
            okB = false;
    }
    if (!okA && !okB)
        return false;

    // Edge e1: Z0 (original uses v0 and v1)
    p0 = e1.y * v0.x - e1.x * v0.y;
    p1 = e1.y * v1.x - e1.x * v1.y;
    mn = fminf(p0, p1);
    mx = fmaxf(p0, p1);
    rad = h * (ae1.y + ae1.x);
    if (okA && _deme_sep_axis_fp32(mn, mx, rad, eps))
        okA = false;
    if (okB) {
        const float d = e1.y * t.x - e1.x * t.y;
        if (_deme_sep_axis_fp32(mn + d, mx + d, rad, eps))
            okB = false;
    }
    if (!okA && !okB)
        return false;

    // Edge e2: X2 (original uses v0 and v1)
    p0 = e2.z * v0.y - e2.y * v0.z;
    p1 = e2.z * v1.y - e2.y * v1.z;
    mn = fminf(p0, p1);
    mx = fmaxf(p0, p1);
    rad = h * (ae2.z + ae2.y);
    if (okA && _deme_sep_axis_fp32(mn, mx, rad, eps))
        okA = false;
    if (okB) {
        const float d = e2.z * t.y - e2.y * t.z;
        if (_deme_sep_axis_fp32(mn + d, mx + d, rad, eps))
            okB = false;
    }
    if (!okA && !okB)
        return false;

    // Edge e2: Y1
    p0 = -e2.z * v0.x + e2.x * v0.z;
    p1 = -e2.z * v1.x + e2.x * v1.z;
    mn = fminf(p0, p1);
    mx = fmaxf(p0, p1);
    rad = h * (ae2.z + ae2.x);
    if (okA && _deme_sep_axis_fp32(mn, mx, rad, eps))
        okA = false;
    if (okB) {
        const float d = -e2.z * t.x + e2.x * t.z;
        if (_deme_sep_axis_fp32(mn + d, mx + d, rad, eps))
            okB = false;
    }
    if (!okA && !okB)
        return false;

    // Edge e2: Z12 (original uses v1 and v2)
    p1 = e2.y * v1.x - e2.x * v1.y;
    p2 = e2.y * v2.x - e2.x * v2.y;
    mn = fminf(p1, p2);
    mx = fmaxf(p1, p2);
    rad = h * (ae2.y + ae2.x);
    if (okA && _deme_sep_axis_fp32(mn, mx, rad, eps))
        okA = false;
    if (okB) {
        const float d = e2.y * t.x - e2.x * t.y;
        if (_deme_sep_axis_fp32(mn + d, mx + d, rad, eps))
            okB = false;
    }
    if (!okA && !okB)
        return false;

    // ---- Case 1) overlap in X/Y/Z ----
    float minx = fminf(v0.x, fminf(v1.x, v2.x));
    float maxx = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    if (okA && (minx > h + eps || maxx < -h - eps))
        okA = false;
    if (okB) {
        const float minxB = minx + t.x;
        const float maxxB = maxx + t.x;
        if (minxB > h + eps || maxxB < -h - eps)
            okB = false;
    }
    if (!okA && !okB)
        return false;

    float miny = fminf(v0.y, fminf(v1.y, v2.y));
    float maxy = fmaxf(v0.y, fmaxf(v1.y, v2.y));
    if (okA && (miny > h + eps || maxy < -h - eps))
        okA = false;
    if (okB) {
        const float minyB = miny + t.y;
        const float maxyB = maxy + t.y;
        if (minyB > h + eps || maxyB < -h - eps)
            okB = false;
    }
    if (!okA && !okB)
        return false;

    float minz = fminf(v0.z, fminf(v1.z, v2.z));
    float maxz = fmaxf(v0.z, fmaxf(v1.z, v2.z));
    if (okA && (minz > h + eps || maxz < -h - eps))
        okA = false;
    if (okB) {
        const float minzB = minz + t.z;
        const float maxzB = maxz + t.z;
        if (minzB > h + eps || maxzB < -h - eps)
            okB = false;
    }
    if (!okA && !okB)
        return false;

    // ---- Case 2) plane test ----
    // normal = cross(e0, v2-v0)
    const float3 v20 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
    const float3 n = {e0.y * v20.z - e0.z * v20.y, e0.z * v20.x - e0.x * v20.z, e0.x * v20.y - e0.y * v20.x};

    const float distA = n.x * v0.x + n.y * v0.y + n.z * v0.z;
    const float rPlane = h * (fabsf(n.x) + fabsf(n.y) + fabsf(n.z)) + eps;

    if (okA && fabsf(distA) > rPlane)
        okA = false;
    if (okB) {
        const float distB = distA + (n.x * t.x + n.y * t.y + n.z * t.z);
        if (fabsf(distB) > rPlane)
            okB = false;
    }

    return okA || okB;
}
