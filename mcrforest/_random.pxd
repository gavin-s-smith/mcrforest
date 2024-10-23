# Authors: Arnaud Joly
#
# License: BSD 3 clause


import numpy as np
cimport numpy as np
ctypedef np.npy_uint32 UINT32_t

cdef inline UINT32_t DEFAULT_SEED = 1

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

# intp_t/np.intp should be used to index arrays in a platform dependent way.
# Storing arrays with platform dependent dtypes as attribute on picklable
# objects is not recommended as it requires special care when loading and
# using such datastructures on a host with different bitness. Instead one
# should rather use fixed width integer types such as int32 or uint32 when we know
# that the number of elements to index is not larger to 2 or 4 billions.
ctypedef Py_ssize_t intp_t
    
# Compatibility type to always accept the default int type used by NumPy, both
# before and after NumPy 2. On Windows, `long` does not always match `inp_t`.
# See the comments in the `sample_without_replacement` Python function for more
# details.
ctypedef fused default_int:
    intp_t
    long
    
cpdef sample_without_replacement(default_int n_population,
                                 default_int n_samples,
                                 method=*,
                                 random_state=*)

# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    """Generate a pseudo-random np.uint32 from a np.uint32 seed"""
    # seed shouldn't ever be 0.
    if (seed[0] == 0): seed[0] = DEFAULT_SEED

    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    # Note: we must be careful with the final line cast to np.uint32 so that
    # the function behaves consistently across platforms.
    #
    # The following cast might yield different results on different platforms:
    # wrong_cast = <UINT32_t> RAND_R_MAX + 1
    #
    # We can use:
    # good_cast = <UINT32_t>(RAND_R_MAX + 1)
    # or:
    # cdef np.uint32_t another_good_cast = <UINT32_t>RAND_R_MAX + 1
    return seed[0] % <UINT32_t>(RAND_R_MAX + 1)
