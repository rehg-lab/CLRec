# distutils: language = c++

import cython
import numpy as np
cimport numpy as np


# declare the interface to the C code
cdef extern from "cjitter.h":
	void cjitter (float* img, int height, int width, float h_change, float s_change, float l_change)

@cython.boundscheck(False)
@cython.wraparound(False)
def jitter(np.ndarray[float, ndim=3, mode="c"] img not None, float h_change, float s_change, float l_change):
	cdef int height, width
	height, width = img.shape[1], img.shape[2]
	cjitter(&img[0, 0, 0], height, width, h_change, s_change, l_change)

	return None