# ------------------------------------------------------------------------------
#  File: ndarray.py
#  Author: Jan Kukacka
#  Date: 6/2018
# ------------------------------------------------------------------------------
#  Functions to help handling ndarrays
# ------------------------------------------------------------------------------

import numpy as np


def diag(array):
    '''
    Convert last axis of an array to a diagonal matrix, leave preceding
    dimensions unchanged.

    # Arguments:
        - array: numpy array of shape [..., N]

    # Returns:
        - array: numpy array of shape [..., N, N] with elements of the input
            array on the diagonal of the last 2 dimensions

    # Example:
    >>  diag(np.array([[1,2,3],
                       [4,5,6]]))
        [[[1, 0, 0],
          [0, 2, 0],
          [0, 0, 3]],
         [[4, 0, 0],
          [0, 5, 0],
          [0, 0, 6]]]
    '''
    array_flat = np.reshape(array, (-1, array.shape[-1]))
    result = np.empty(array_flat.shape + (array.shape[-1],))
    for i in range(array_flat.shape[0]):
        result[i] = np.diag(array_flat[i])
    return np.reshape(result, array.shape + (array.shape[-1],))


def collapse_diag(array):
    '''
    Convert matrix represented by the last 2 axes of an array to 1D array
    containing the diagonal of the matrix, leave preceding dimensions unchanged.

    # Arguments:
        - array: numpy array of shape [..., N, N]

    # Returns:
        - array: numpy array of shape [..., N] with elements of the diagonal of
            the last 2 dimensions in the last dimension

    # Example:
    >>  collapse_diag(np.array([[[1, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 3]],
                                [[4, 0, 0],
                                 [0, 5, 0],
                                 [0, 0, 6]]]))
         [[1,2,3],
          [4,5,6]]
    '''
    array_flat = np.reshape(array, (-1, array.shape[-1], array.shape[-1]))
    result = np.empty(array_flat.shape[:-1])
    for i in range(array_flat.shape[0]):
        result[i] = np.diag(array_flat[i])
    return np.reshape(result, array.shape[:-1])


def collapse_axes(a, preserve=None):
    '''
    Collapses (flattens) all the array except for axes specified in the parameter
    `preserve`, which will be kept. Dimensions between preserved axes will be
    collapsed.

    # Arguments:
        - a: input numpy array of shape [n1, n2, n3, ..., nk]
        - preserve: int or iterable of axes that should be preserved. Negative
            numbers may specify axes indexed from the end

    # Returns:
        - reshaped array with collapsed axes not specified in the preserve param

    # Example:
    >>  a = np.zeros((5,3,8,6,6,10))
    >>  collapse_axes(a, preserve=(1,-1)).shape
        (5,3,288,10)
    >>  collapse_axes(a, preserve=(3)).shape
        (320,6,60)
    '''
    if preserve is None:
        return a.ravel()

    from .misc import ensure_list
    preserve = ensure_list(preserve)

    n_dim = a.ndim
    shape = a.shape

    ## Convert preserved axes to positive integers
    preserve = [i if i >= 0 else n_dim+i for i in preserve]
    ## Sort dimensions
    preserve = sorted(list(set(preserve)))

    ## Compute new array dimensions
    new_shape = []
    accumulator = 1
    collapse = False
    for axis, size in enumerate(shape):
        ## Check if current axis should be preserved
        ## (it would be at the start of the current preserved list)
        if len(preserve) == 0 or axis < preserve[0]:
            ## When collapsing an axis, increase accumulator size and raise a flag
            accumulator = accumulator * size
            collapse = True
        else:
            ## This axis should be preserved
            if collapse:
                ## If there were some collapsed axes before this axis, add their
                ## accumulated shape to the new shape and reset flags
                new_shape.append(accumulator)
                accumulator = 1
                collapse = False
            ## Add preserved axes shape and remove it from the preserved list
            new_shape.append(size)
            del preserve[0]
    if collapse:
        ## Do not forget to append remaining axes at the end
        new_shape.append(accumulator)

    return a.reshape(new_shape)
