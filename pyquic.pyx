# The original algorithm was coded by Manjari Narayan and Jason Laska.
# The implementation was modified by Changhao Ge.
# This code is under the MIT license.
# import numpy as np
# cimport numpy as np
# from libc.stdint cimport uint32_t, int32_t
# cimport cython
# from cython.parallel cimport prange
# from libc.stdlib cimport malloc, free

# cdef extern from "QUIC.h":
#     void QUIC(char mode, uint32_t& p, double* S, double* Lambda0,
# 	  uint32_t& pathLen, double* path, double& tol,
# 	  int32_t& msg, uint32_t& maxIter,
# 	  double* X, double* W, double* opt, double* cputime,
# 	  uint32_t* iter, double* dGap) nogil

# # def quic(char* mode, int p, 
# #         double* S, 
# #         double* L,
# #         int pathLen,
# #         double* path,
# #         double tol, int msg, int max_iter, 
# #         double* X, 
# #         double* W,
# #         double* opt,
# #         double* cputime,
# #         uint32_t* _iter,
# #         double* dGap
# #         ):
# #     """
# #     """
# #     cdef uint32_t _p = p
# #     cdef uint32_t _pathLen = pathLen
# #     cdef int32_t _msg = msg
# #     cdef uint32_t _max_iter = max_iter
# #     QUIC(mode[0], _p, S, L, _pathLen, path, tol, _msg, _max_iter, 
# #             X, W, opt, cputime, _iter, dGap)

# #     return


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def quic_py_par(
#         list S_l, 
#         list L_l,
#         list X_l, 
#         list W_l,
#         double tol, int max_iter, int threads, int msg = 0
#         ):
#     """
#     """
#     cdef uint32_t p = S_l[0].shape[0]
#     cdef char* mode = 'default'
#     cdef uint32_t pathLen = <uint32_t>1
#     cdef double* path = NULL
#     cdef double* opt = NULL
#     cdef double* cputime = NULL
#     cdef int I = len(S_l)
#     cdef Py_ssize_t i
#     cdef double* dGap = NULL
#     cdef uint32_t* iter = <uint32_t*>malloc(I * sizeof(uint32_t))
#     cdef double** S_ptr = <double**>malloc(I * sizeof(double*))
#     cdef double** X_ptr = <double**>malloc(I * sizeof(double*))
#     cdef double** W_ptr = <double**>malloc(I * sizeof(double*))
#     cdef double** L_ptr = <double**>malloc(I * sizeof(double*))
#     cdef double[:,:] arr_temp
#     cdef int32_t _msg = msg
#     cdef uint32_t _max_iter = max_iter
#     for i in range(I):
#         arr_temp = S_l[i]
#         S_ptr[i] = &arr_temp[0,0]
#         arr_temp = X_l[i]
#         X_ptr[i] = &arr_temp[0,0]
#         arr_temp = W_l[i]
#         W_ptr[i] = &arr_temp[0,0]
#         arr_temp = L_l[i]
#         L_ptr[i] = &arr_temp[0,0]
#         iter[i] = 0
    
#     if threads == 0:
#         for i in prange(I, nogil=True):
#             QUIC(mode[0], p, S_ptr[i], 
#                 L_ptr[i],
#                 pathLen, 
#                 path, 
#                 tol, _msg, _max_iter, 
#                 X_ptr[i], 
#                 W_ptr[i], 
#                 opt, 
#                 cputime, 
#                 &iter[i], 
#                 dGap)
#     else:
#         for i in prange(I, nogil=True, num_threads=threads):
#             QUIC(mode[0], p, S_ptr[i], 
#                 L_ptr[i],
#                 pathLen, 
#                 path, 
#                 tol, _msg, _max_iter, 
#                 X_ptr[i], 
#                 W_ptr[i], 
#                 opt, 
#                 cputime, 
#                 &iter[i], 
#                 dGap)

#     free(S_ptr)
#     free(X_ptr)
#     free(W_ptr)
#     free(L_ptr)
#     free(iter)

#     return


# def quic_py(
#         list S_l, 
#         list L_l,
#         list X_l, 
#         list W_l,
#         double tol, int max_iter, int msg = 0
#         ):
#     """
#     """
#     cdef uint32_t p = S_l[0].shape[0]
#     cdef char* mode = 'default'
#     cdef int pathLen = 1
#     cdef double* path = NULL
#     cdef double* opt = NULL
#     cdef double* cputime = NULL
#     cdef int I = len(S_l)
#     cdef Py_ssize_t i
#     cdef double* dGap = NULL
#     cdef uint32_t* iter = <uint32_t*>malloc(I * sizeof(uint32_t))
#     cdef double** S_ptr = <double**>malloc(I * sizeof(double*))
#     cdef double** X_ptr = <double**>malloc(I * sizeof(double*))
#     cdef double** W_ptr = <double**>malloc(I * sizeof(double*))
#     cdef double** L_ptr = <double**>malloc(I * sizeof(double*))
#     cdef double[:,:] arr_temp
#     cdef int32_t _msg = msg
#     cdef uint32_t _max_iter = max_iter
#     for i in range(I):
#         arr_temp = S_l[i]
#         S_ptr[i] = &arr_temp[0,0]
#         arr_temp = X_l[i]
#         X_ptr[i] = &arr_temp[0,0]
#         arr_temp = W_l[i]
#         W_ptr[i] = &arr_temp[0,0]
#         arr_temp = L_l[i]
#         L_ptr[i] = &arr_temp[0,0]
#         iter[i] = 0
#     for i in range(I):
#         QUIC(mode[0], p, S_ptr[i], 
#             L_ptr[i],
#             pathLen, 
#             path, 
#             tol, _msg, _max_iter, 
#             X_ptr[i], 
#             W_ptr[i], 
#             opt, 
#             cputime, 
#             &iter[i], 
#             dGap)
#     free(S_ptr)
#     free(X_ptr)
#     free(W_ptr)
#     free(L_ptr)
#     free(iter)
#     return
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