#cython: language_level=3
"""
Original code from which this Cython code is derived:
Copyright (c) 2011-2020, Stan Developers and their Assignees.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
"""
from libc.math cimport pi, ceil, exp, floor, log, sin, sqrt, fmax
cimport cython
import numpy as np


cdef double WIENER_ERR = 0.000001
cdef double SQRT_PI = sqrt(pi)
cdef double PI_TIMES_WIENER_ERR = pi * WIENER_ERR
cdef double LOG_PI_LOG_WIENER_ERR = log(pi) + log(WIENER_ERR)
cdef double TWO_TIMES_SQRT_TWO_PI_TIMES_WIENER_ERR = 2.0 * sqrt(2 * pi) * WIENER_ERR
cdef double LOG_TWO_OVER_TWO_PLUS_LOG_SQRT_PI = log(2) / 2 + log(sqrt(pi))
cdef double SQUARE_PI_OVER_TWO = (pi ** 2) * 0.5
cdef double TWO_TIMES_LOG_SQRT_PI = 2.0 * log(sqrt(pi))


cpdef double wiener_lpdf(double y, double alpha, double tau,
        double beta, double delta):
    if y <= tau:
        return -log(1e-7)

    cdef double one_minus_beta = 1.0 - beta
    cdef double alpha2 = alpha ** 2
    cdef double x = (y - tau) / alpha2

    cdef double kl, ks, tmp = 0
    cdef double k, K
    cdef double sqrt_x = sqrt(x)
    cdef double log_x = log(x)
    cdef double one_over_pi_times_sqrt_x = 1.0 / pi * sqrt_x

    cdef double tmp_expr0, tmp_expr1, tmp_expr2
    cdef double lp

    # calculate number of terms needed for large t:
    # if error threshold is set low enough
    if PI_TIMES_WIENER_ERR * x < 1:
        # compute bound
        kl = sqrt(-2.0 * SQRT_PI * (LOG_PI_LOG_WIENER_ERR + log_x)) / sqrt_x
        # ensure boundary conditions met
        kl = fmax(kl, one_over_pi_times_sqrt_x)
    else:
        # set to boundary condition
        kl = one_over_pi_times_sqrt_x

    # calculate number of terms needed for small t:
    # if error threshold is set low enough
    tmp_expr0 = TWO_TIMES_SQRT_TWO_PI_TIMES_WIENER_ERR * sqrt_x
    if tmp_expr0 < 1:
        # compute bound
        ks = 2.0 + sqrt_x * sqrt(-2.0 * log(tmp_expr0))
        # ensure boundary conditions met
        ks = fmax(ks, sqrt_x + 1.0)
    else:  # if error threshold was set too high
        # minimal kappa for that case
        ks = 2.0

    if ks < kl:  # small t
        K = ceil(ks)  # round to smallest integer meeting error
        tmp_expr1 = (K - 1.0) / 2.0
        tmp_expr2 = ceil(tmp_expr1)
        k = -floor(tmp_expr1)
        while k <= tmp_expr2:
            tmp += (one_minus_beta + 2.0 * k) * \
                    exp(-((one_minus_beta + 2.0 * k) ** 2.0) * 0.5 / x)
            k += 1
        tmp = log(tmp) - LOG_TWO_OVER_TWO_PLUS_LOG_SQRT_PI - 1.5 * log_x
    else:  # if larger t is better...
        K = ceil(kl)  # round to smallest integer meeting error
        k = 1
        while k <= K:
            tmp += k * exp(-(k ** 2) * (SQUARE_PI_OVER_TWO * x)) * \
                    sin(k * pi * one_minus_beta)
            k += 1
        tmp = log(tmp) + TWO_TIMES_LOG_SQRT_PI

    # convert to f(t|v,a,w) and return result
    lp = delta * alpha * one_minus_beta - \
            (delta ** 2) * x * alpha2 / 2.0 - log(alpha2) + tmp

    return lp
