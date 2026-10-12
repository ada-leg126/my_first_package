"""Element-wise array operations for my_first_package."""

import numpy as np


def elementwise_multiplication(array_a, array_b):
    """Multiply two arrays element-wise.

    Parameters
    ----------
    array_a : numpy.ndarray
        The first input array.
    array_b : numpy.ndarray
        The second input array. It must have a compatible shape with
        ``array_a`` according to NumPy broadcasting rules.

    Returns
    -------
    numpy.ndarray
        An array containing the element-wise product of ``array_a`` and
        ``array_b``.

    Examples
    --------
    >>> import numpy as np
    >>> elementwise_multiplication(np.array([1, 2, 3]), np.array([4, 5, 6]))
    array([ 4, 10, 18])
    """
    return np.multiply(array_a, array_b)


def elementwise_division(array_a, array_b):
    """Divide one array by another element-wise.

    Parameters
    ----------
    array_a : numpy.ndarray
        The numerator array.
    array_b : numpy.ndarray
        The denominator array. It must have a compatible shape with
        ``array_a`` according to NumPy broadcasting rules.

    Returns
    -------
    numpy.ndarray
        An array containing the element-wise quotient of ``array_a`` and
        ``array_b``.

    Raises
    ------
    ZeroDivisionError
        Not raised for NumPy arrays; division by zero typically produces
        ``inf`` or ``nan`` and may emit a NumPy runtime warning.

    Examples
    --------
    >>> import numpy as np
    >>> elementwise_division(np.array([8, 10, 12]), np.array([2, 5, 3]))
    array([4., 2., 4.])
    """
    return np.divide(array_a, array_b)
