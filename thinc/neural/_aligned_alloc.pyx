# Copyright 2014 Knowledge Economy Developments Ltd
# Copyright 2014 David Wells
#
# Henry Gomersall
# heng@kedevelopments.co.uk
# David Wells
# drwells <at> vt.edu
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

from bisect import bisect_left
cimport numpy as np
import numpy
from libc.stdint cimport intptr_t
import warnings

#cimport cpu
cdef int _simd_alignment = 32 #cpu.simd_alignment()

#: A tuple of simd alignments that make sense for this cpu
if _simd_alignment == 16:
    _valid_simd_alignments = (16,)

elif _simd_alignment == 32:
    _valid_simd_alignments = (16, 32)

else:
    _valid_simd_alignments = ()


cpdef byte_align(array, n=None, dtype=None):
    '''byte_align(array, n=None, dtype=None)

    Function that takes a numpy array and checks it is aligned on an n-byte
    boundary, where ``n`` is an optional parameter. If ``n`` is not provided
    then this function will inspect the CPU to determine alignment. If the
    array is aligned then it is returned without further ado.  If it is not
    aligned then a new array is created and the data copied in, but aligned
    on the n-byte boundary.

    ``dtype`` is an optional argument that forces the resultant array to be
    of that dtype.
    '''

    if not isinstance(array, np.ndarray):
        raise TypeError('Invalid array: byte_align requires a subclass '
                'of ndarray')

    if n is None:
        n = _simd_alignment

    if dtype is not None:
        if not array.dtype == dtype:
            update_dtype = True

    else:
        dtype = array.dtype
        update_dtype = False

    # See if we're already n byte aligned. If so, do nothing.
    offset = <intptr_t>np.PyArray_DATA(array) %n

    if offset is not 0 or update_dtype:

        _array_aligned = empty_aligned(array.shape, dtype, n=n)

        _array_aligned[:] = array

        array = _array_aligned.view(type=array.__class__)

    return array


cpdef is_byte_aligned(array, n=None):
    ''' is_n_byte_aligned(array, n=None)

    Function that takes a numpy array and checks it is aligned on an n-byte
    boundary, where ``n`` is an optional parameter, returning ``True`` if it is,
    and ``False`` if it is not. If ``n`` is not provided then this function will
    inspect the CPU to determine alignment.
    '''
    if not isinstance(array, np.ndarray):
        raise TypeError('Invalid array: is_n_byte_aligned requires a subclass '
                'of ndarray')

    if n is None:
        n = _simd_alignment

    # See if we're n byte aligned.
    offset = <intptr_t>np.PyArray_DATA(array) %n

    return not bool(offset)


cpdef is_n_byte_aligned(array, n):
    ''' is_n_byte_aligned(array, n)
    **This function is deprecated:** ``is_byte_aligned`` **should be used
    instead.**

    Function that takes a numpy array and checks it is aligned on an n-byte
    boundary, where ``n`` is a passed parameter, returning ``True`` if it is,
    and ``False`` if it is not.
    '''
    return is_byte_aligned(array, n=n)


cpdef empty_aligned(shape, dtype='float32', order='C', n=None):
    '''empty_aligned(shape, dtype='float32', order='C', n=None)

    Function that returns an empty numpy array that is n-byte aligned,
    where ``n`` is determined by inspecting the CPU if it is not
    provided.

    The alignment is given by the final optional argument, ``n``. If
    ``n`` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.empty`.
    '''
    if n is None:
        n = _simd_alignment

    itemsize = np.dtype(dtype).itemsize

    # Apparently there is an issue with numpy.prod wrapping around on 32-bits
    # on Windows 64-bit. This shouldn't happen, but the following code
    # alleviates the problem.
    if not isinstance(shape, (int, numpy.integer)):
        array_length = 1
        for each_dimension in shape:
            array_length *= each_dimension

    else:
        array_length = shape

    # Allocate a new array that will contain the aligned data
    _array_aligned = numpy.empty(array_length*itemsize+n, dtype='int8')

    # We now need to know how to offset _array_aligned
    # so it is correctly aligned
    _array_aligned_offset = (n-<intptr_t>np.PyArray_DATA(_array_aligned))%n

    array = numpy.frombuffer(
            _array_aligned[_array_aligned_offset:_array_aligned_offset-n].data,
            dtype=dtype).reshape(shape, order=order)

    return array


cpdef zeros_aligned(shape, dtype='float32', order='C', n=None):
    '''zeros_aligned(shape, dtype='float64', order='C', n=None)

    Function that returns a numpy array of zeros that is n-byte aligned,
    where ``n`` is determined by inspecting the CPU if it is not
    provided.

    The alignment is given by the final optional argument, ``n``. If
    ``n`` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.zeros`.
    '''
    array = empty_aligned(shape, dtype=dtype, order=order, n=n)
    array.fill(0)
    return array


cpdef ones_aligned(shape, dtype='float32', order='C', n=None):
    '''ones_aligned(shape, dtype='float32', order='C', n=None)

    Function that returns a numpy array of ones that is n-byte aligned,
    where ``n`` is determined by inspecting the CPU if it is not
    provided.

    The alignment is given by the final optional argument, ``n``. If
    ``n`` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.ones`.
    '''
    array = empty_aligned(shape, dtype=dtype, order=order, n=n)
    array.fill(1)
    return array
