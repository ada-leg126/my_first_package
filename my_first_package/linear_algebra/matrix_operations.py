"""Matrix operations for my_first_package."""

import numpy as np


def matrix_addition(matrix_a, matrix_b):
    """Add two matrices element-wise.

    Parameters
    ----------
    matrix_a : numpy.ndarray
        The first input matrix.
    matrix_b : numpy.ndarray
        The second input matrix. It must have a compatible shape with
        ``matrix_a``.

    Returns
    -------
    numpy.ndarray
        A matrix containing the element-wise sum of ``matrix_a`` and
        ``matrix_b``.

    Raises
    ------
    ValueError
        If the input matrices have incompatible shapes.

    Examples
    --------
    >>> import numpy as np
    >>> matrix_addition(np.array([[1, 2]]), np.array([[3, 4]]))
    array([[4, 6]])
    """
    return np.add(matrix_a, matrix_b)


def matrix_subtraction(matrix_a, matrix_b):
    """Subtract the second matrix from the first element-wise.

    Parameters
    ----------
    matrix_a : numpy.ndarray
        The matrix from which to subtract.
    matrix_b : numpy.ndarray
        The matrix to subtract. It must have a compatible shape with
        ``matrix_a``.

    Returns
    -------
    numpy.ndarray
        A matrix containing the element-wise difference between
        ``matrix_a`` and ``matrix_b``.

    Raises
    ------
    ValueError
        If the input matrices have incompatible shapes.

    Examples
    --------
    >>> import numpy as np
    >>> matrix_subtraction(np.array([[5, 6]]), np.array([[2, 4]]))
    array([[3, 2]])
    """
    return np.subtract(matrix_a, matrix_b)
