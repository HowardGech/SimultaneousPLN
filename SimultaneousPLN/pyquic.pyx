# The original algorithm was coded by Manjari Narayan and Jason Laska.
# The implementation was modified by Changhao Ge.
# This code is under the MIT license.

import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, int32_t
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libcpp.memory cimport unique_ptr, allocator

cdef extern from "QUIC.h":
    void QUIC(char mode, uint32_t& p, double* S, double* Lambda0,
	  uint32_t& pathLen, double* path, double& tol,
	  int32_t& msg, uint32_t& maxIter,
	  double* X, double* W, double* opt, double* cputime,
	  uint32_t* iter, double* dGap) nogil


def quic(
        list S_l, 
        list L_l,
        list X_l, 
        list W_l,
        double tol, int max_iter, int msg = 0
        ):
    """
    """
    cdef char* mode = 'default'
    cdef uint32_t _p = S_l[0].shape[0]
    cdef int n = len(S_l)
    cdef uint32_t pathLen = 1
    cdef int32_t _msg = msg
    cdef uint32_t _max_iter = max_iter
    cdef double[:,:] temp
    cdef allocator[double*] alloc
    cdef allocator[uint32_t] alloc_uint
    cdef unique_ptr[double*] S_ptr, L_ptr, X_ptr, W_ptr
    cdef double* path = NULL
    cdef double* opt = NULL
    cdef double* cputime = NULL
    cdef double* dGap = NULL
    cdef unique_ptr[uint32_t] iter
    iter.reset(alloc_uint.allocate(n * sizeof(uint32_t)))
    S_ptr.reset(alloc.allocate(n * sizeof(double*)))
    L_ptr.reset(alloc.allocate(n * sizeof(double*)))
    X_ptr.reset(alloc.allocate(n * sizeof(double*)))
    W_ptr.reset(alloc.allocate(n * sizeof(double*)))
    cdef int i
    for i in range(len(S_l)):
        temp = S_l[i]
        S_ptr.get()[i] = &temp[0,0]
        temp = L_l[i]
        L_ptr.get()[i] = &temp[0,0]
        temp = X_l[i]
        X_ptr.get()[i] = &temp[0,0]
        temp = W_l[i]
        W_ptr.get()[i] = &temp[0,0]
        iter.get()[i] = 0

    for i in range(n):
        QUIC(mode[0], _p, S_ptr.get()[i], L_ptr.get()[i], pathLen, path, tol, _msg, _max_iter, 
                X_ptr.get()[i], W_ptr.get()[i], opt, cputime, &iter.get()[i], dGap)

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def quic_par(
        list S_l, 
        list L_l,
        list X_l, 
        list W_l,
        double tol, int max_iter, int threads, int msg = 0
        ):
    """
    """
    cdef char* mode = 'default'
    cdef uint32_t _p = S_l[0].shape[0]
    cdef int n = len(S_l)
    cdef uint32_t pathLen = 1
    cdef int32_t _msg = msg
    cdef uint32_t _max_iter = max_iter
    cdef double[:,:] temp
    cdef allocator[double*] alloc
    cdef allocator[uint32_t] alloc_uint
    cdef unique_ptr[double*] S_ptr, L_ptr, X_ptr, W_ptr
    cdef double* path = NULL
    cdef double* opt = NULL
    cdef double* cputime = NULL
    cdef double* dGap = NULL
    cdef unique_ptr[uint32_t] iter
    iter.reset(alloc_uint.allocate(n * sizeof(uint32_t)))
    S_ptr.reset(alloc.allocate(n * sizeof(double*)))
    L_ptr.reset(alloc.allocate(n * sizeof(double*)))
    X_ptr.reset(alloc.allocate(n * sizeof(double*)))
    W_ptr.reset(alloc.allocate(n * sizeof(double*)))
    cdef int i
    for i in range(len(S_l)):
        temp = S_l[i]
        S_ptr.get()[i] = &temp[0,0]
        temp = L_l[i]
        L_ptr.get()[i] = &temp[0,0]
        temp = X_l[i]
        X_ptr.get()[i] = &temp[0,0]
        temp = W_l[i]
        W_ptr.get()[i] = &temp[0,0]
        iter.get()[i] = 0
    if threads == 0:
        for i in prange(n, nogil=True):
            QUIC(mode[0], _p, S_ptr.get()[i], L_ptr.get()[i], pathLen, path, tol, _msg, _max_iter, 
                    X_ptr.get()[i], W_ptr.get()[i], opt, cputime, &iter.get()[i], dGap)
    else:
        for i in prange(n, nogil=True, num_threads=threads):
            QUIC(mode[0], _p, S_ptr.get()[i], L_ptr.get()[i], pathLen, path, tol, _msg, _max_iter, 
                    X_ptr.get()[i], W_ptr.get()[i], opt, cputime, &iter.get()[i], dGap)

    return