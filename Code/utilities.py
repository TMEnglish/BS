import math
from scipy.integrate import solve_ivp
from numba import njit
from os import mkdir
# ASSUME: import numpy as np
# ASSUME: from mpmath import mp
    

# Make some mpmath functions into quasi-ufuncs taking either scalar
# or array arguments.

mp_float = np.frompyfunc(mp.mpf, 1, 1)
mp_exp = np.frompyfunc(mp.exp, 1, 1)
fabs = np.frompyfunc(mp.fabs, 1, 1)
erf = np.frompyfunc(mp.erf, 1, 1)
erfc = np.frompyfunc(mp.erfc, 1, 1)
gamma = np.frompyfunc(mp.gamma, 1, 1)
rgamma = np.frompyfunc(mp.rgamma, 1, 1) # reciprocal gamma
hyp1f1 = np.frompyfunc(mp.hyp1f1, 3, 1) # confluent hypergeometric 1_F_1
hyperu = np.frompyfunc(mp.hyperu, 3, 1) # confluent hypergeometric 2_F_2


# Provide alternatives for some NumPy functions that do not handle
# mpmath's multiprecision floating-point type.

def linspace(a, b, nsteps):
    """
    Tries to return result of NumPy linspace, falls back on mpmath version.
    """
    try:
        return np.linspace(a, b, nsteps)
    except:
        return np.array(mp.linspace(a, b, nsteps))

def exp(x):
    """
    Tries to return result of NumPy exp, falls back on mpmath version.
    """
    try:
        return np.exp(x)
    except:
        return mp_exp(x)

    
# Generalized fsum and frexp work with float and multiprecision float.

def fsum(a):
    """
    Return either mp.fsum(a) or math.fsum(a), depending on type of `a`.
    """
    if type(a[0]) is mp.mpf:
        return mp.fsum(a)
    else:
        return math.fsum(a)

def frexp(x):
    """
    Return either mp.frexp(x) or math.frexp(x), depending on type of `x`.
    """
    if type(x) is mp.mpf:
        return mp.frexp(x)
    else:
        return math.frexp(x)


def convert(x, basetype):
    """
    Returns `x` converted to array with elements of type `basetype`.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    shape = x.shape
    return np.array([basetype(e) for e in np.ravel(x)]).reshape(shape)


def ivp_solution(derivative, initial_frequencies, times, max_step=1/128):
    """
    Use the Runge-Kutta 4(5) method to solve for frequencies at specified
    `times`, given the frequencies at the first of those times.
    """
    interval = (times[0], times[-1] + max_step)
    result = solve_ivp(derivative, interval, initial_frequencies,
                       method='RK45', t_eval=times, max_step=max_step,
                       rtol=1e-13, atol=1e-11)
    return result.y.T, result


def bias_exponents(array, max_exponent, current_max=None):
    """
    Scales elements in `array` by an integer power of 2.

    On return, the internal representation of the maximum element of
    `array` has `max_exponent` as its exponent. There is no change in
    the mantissas of the elements. The given `array` is returned.
    """
    if current_max is None:
        _, current_max = frexp(array.max())
    array *= 2.0 ** (max_exponent - current_max)
    return array


def accurate_sum(a):
    """
    Returns sum of elements in array `a` as multiprecision float.
    """
    return mp.fsum(a)


def moment(frequency, x, n):
    """
    Returns n-th raw moment as multiprecision float.
    """
    x = mp_float(x)
    x **= n
    return mp.fsum(np.multiply(frequency, x)) / mp.fsum(frequency)


def mean(frequency, x):
    """
    Returns mean for `frequency` distribution over `x`.
    
    Sums are calculated accurately using `fsum`.
    """
    return fsum(frequency * x) / fsum(frequency)


def mean_var(frequency, x):
    """
    Returns mean and variance for `frequency` distribution over `x`.
    
    Sums are calculated accurately using `fsum`.
    """
    if type(frequency[0]) is mp.mpf:
        if not type(x[0]) is mp.mpf:
            x = mp_float(x)
    elif type(x[0]) is mp.mpf:
        frequency = mp_float(frequency)
    norm = fsum(frequency)
    mom1 = fsum(frequency * x) 
    mom2 = fsum(frequency * x**2)
    var = (mom2 - mom1**2 / norm) / norm
    mean = mom1 / norm
    return mean, var

   
def mean_and_variance(x, p):
    """
    Return (mean, var) for probability distribution(s) `p` over `x`.
    
    Also works if x has the same shape as p.
    """
    axis = max(x.ndim, p.ndim) - 1
    mean = np.sum(np.multiply(p, x), axis=axis)
    variance = np.sum(np.multiply(p, np.square(x)), axis=axis)
    variance -= np.square(mean)
    return mean, variance


def regress_through_origin(x, y):
    """
    Return slope obtained by regression through the origin.
    """
    sum_x = fsum(x)
    sum_y = fsum(y)
    n = len(x)
    cov = fsum(x * y) - sum_x * sum_y / n
    var = fsum(x * x) - sum_x * sum_x / n
    return cov / var 
    

def relative_error(actual, desired, absolute=False):
    """
    Returns (actual - desired) / desired, with 0/0 defined equal to 0.
    """
    if np.shape(actual) != np.shape(desired):
        raise ValueError('Arguments are not identical in shape')
    result = np.zeros_like(desired)
    key = actual != desired
    if np.any(desired[key] == 0):
        return np.inf
    result[key] = (actual[key] - desired[key]) / desired[key]
    if absolute:
        np.abs(result, out=result)
    return result


def maximum_absolute_relative_error(actual, desired):
    return np.max(relative_error(actual, desired, absolute=True))


def min_and_max(a):
    """
    Returns the pair (min(a), max(a)).
    """
    return np.min(a), np.max(a)


@njit
def trim(array, threshold):
    for left, value in enumerate(array):
        if value >= threshold:
            break
    for right, value in enumerate(array[::-1]):
        if value >= threshold:
            break
    n = len(array) - right
    array[:left] = 0
    array[n:] = 0
    return left, right

@njit
def support_range(p, start, stop):
    for left, value in enumerate(p):
        if value > 0:
            break
    for right, value in enumerate(p[::-1]):
        if value > 0:
            break
    return start+left, stop-right

def slice_to_support(p, sliced=None):
    """
    Returns slice excluding zeros in the tails of distribution p.
    """
    if sliced is None:
        indices = support_range(p, 0, len(p))
    else:
        start, stop = sliced.start, sliced.stop
        indices = support_range(p, start, stop)
    return slice(*indices)


def ensure_directory_exists(path):
    """
    Create directory with given `path` if it does not exist already.
    """
    try:
        mkdir(path)
    except FileExistsError:
        pass