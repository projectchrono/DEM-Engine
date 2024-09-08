//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// This file contains modifications of the code by Alessandro Tasora as a part
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

#ifndef DEME_TIMER_HPP
#define DEME_TIMER_HPP

#include <chrono>

namespace deme {

template <class seconds_type = double>
class Timer {
  private:
    std::chrono::high_resolution_clock::time_point m_start;
    std::chrono::high_resolution_clock::time_point m_end;
    std::chrono::duration<seconds_type> m_total;

  public:
    Timer() { m_total = std::chrono::duration<seconds_type>(0); }

    /// Start the timer
    void start() { m_start = std::chrono::high_resolution_clock::now(); }

    /// Stops the timer
    void stop() {
        m_end = std::chrono::high_resolution_clock::now();
        m_total += m_end - m_start;
    }

    /// Reset the total accumulated time (when repeating multiple start() stop() start() stop() )
    void reset() { m_total = std::chrono::duration<seconds_type>(0); }

    /// Returns the time in [ms].
    /// Use start()..stop() before calling this.
    unsigned long long GetTimeMilliseconds() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(m_total).count();
    }

    /// Returns the time in [ms] since start(). It does not require stop().
    unsigned long long GetTimeMillisecondsIntermediate() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                     m_start)
            .count();
    }

    /// Returns the time in [us].
    /// Use start()..stop() before calling this.
    unsigned long long GetTimeMicroseconds() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(m_total).count();
    }

    /// Returns the time in [us] since start(). It does not require stop().
    unsigned long long GetTimeMicrosecondsIntermediate() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() -
                                                                     m_start)
            .count();
    }

    /// Returns the time in [s], with real_type precision
    /// Use start()..stop() before calling this.
    seconds_type GetTimeSeconds() const { return m_total.count(); }

    /// Returns the time in [s] since start(). It does not require stop().
    seconds_type GetTimeSecondsIntermediate() const {
        std::chrono::duration<seconds_type> int_time = std::chrono::high_resolution_clock::now() - m_start;
        return int_time.count();
    }

    /// Get the last timer value, in seconds, with the () operator.
    seconds_type operator()() const { return GetTimeSeconds(); }
};

}  // namespace deme

#endif
