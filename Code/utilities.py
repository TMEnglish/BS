import pickle

DATA_DIR = './Data/'

# Make some multiprecision scalar functions into NumPy ufuncs.
# The ufuncs take either scalar or array arguments where the
# original functions take only scalars.
mp_exp = np.frompyfunc(mp.exp, 1, 1)
mp_erfc = np.frompyfunc(mp.erfc, 1, 1)
mp_sqrt = np.frompyfunc(mp.sqrt, 1, 1)
mp_frexp = np.frompyfunc(mp.frexp, 1, 2)
mp_ldexp = np.frompyfunc(mp.ldexp, 2, 1)


def raveled(a):
    # Raveling `a` allows it to be processed as a flat array, even
    # if it is given as a scalar. If `a` is not given as an array,
    # then we convert it to an array of base type `object`, thus
    # preventing conversion of integers to NumPy 64-bit integers.
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=object)
    return a.ravel(), a.shape

def shaped(a, shape):
    # Returns `a` as an array with the given `shape`.
    if shape == ():
        return a[0]
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    return a.reshape(shape)


def to_mpf(a):
    """
    Converts (array) `a` to (array of) `mpmath.mpf` object(s).
    """
    a, shape = raveled(a)
    try:
        a = [mp.mpf(x) for x in a]
    except TypeError:
        a = [mp.mpf(x) for x in a.astype(str)]
    return shaped(a, shape)


def to_fraction(a):
    """
    Converts (array) `a` to (array of) `Fraction` object(s).
    """
    # If the Fraction constructor is supplied with an integer, it
    # uses the given integer as the numerator. To avoid overflow
    # in operations on Fractions, we convert NumPy 64-bit integers
    # to Python `int` objects (which are unlimited in size).
    if isinstance(a, np.ndarray):
        if a.dtype is np.dtype('int64'):
            a = a.astype(object)
    elif type(a) is np.int64:
        a = int(a)
    a, shape = raveled(a)
    #
    # First try passing objects in `a` to the `Fraction` constructor.
    # This will fail if `a` contains a multiprecision float.
    try:
        a = [Fraction(x) for x in a]
        return shaped(a, shape)
    except TypeError:
        pass
    #
    # Fall back to treating elements of `a` as multiprecision floats.
    # Represent `a` as `c * 2**power`, where all elements of `power` are 
    # integers and all elements of `c` are multiprecision floats in the
    # interval [0.5, 1).
    c, power = mp_frexp(a)
    numerator_power = max(mp.prec, np.max(power))
    denominator_power = numerator_power - power
    #
    #       c * 2**power 
    #    == c / 2**-power
    #    == c * 2**numerator_power / 2**(numerator_power-power)
    #    == c * 2**numerator_power / 2**denominator_power
    #
    # Elements of `c` are calculated with the current working precision,
    # `mp.prec`, so scaling `c` by 
    #
    #     2**numerator_power >= 2**mp.prec
    #
    # yields an array of integers.
    numerators = [int(p) for p in mp_ldexp(c, numerator_power)]
    #
    # Array `2**denom_power` contains only integers because `denom_power`
    # contains only non-negative integers.
    denominators = 2**denominator_power
    #
    # Construct Fractions from corresponding numerators and denominators.
    a = [Fraction(p, q) for p, q in zip(numerators, denominators)]
    return shaped(a, shape)


#def is_rational(x):
#    """
#    Returns result of check that `x` is a rational number.
#    """
#    # Exception wrongly occurs when x is a string.
#    try:
#        result = isinstance(x, numbers.Rational)
#    except:
#        result = False
#    return result


def exactly(*specs):
    """
    Return a `Fraction` for each specification in `specs`.
    
    An exception is raised if a specification is neither a string nor
    a rational number.
    """
    exact = [isinstance(x, str) or isinstance(x, numbers.Rational)
                for x in specs]
    if not all(exact):
        raise TypeError('exactly: spec neither rational nor string')
    if len(specs) == 1:
        specs = specs[0]
    return to_fraction(specs)


def equispaced(n, spacing='1', start='0', warn=False):
    """
    Return an array of `n` exactly equispaced numbers of type `Fraction`.
    
    * `n`      : number of elements in the array
    * `spacing`: difference an element and its predecessor (`str`)
    * `start`  : value of the first element of the array (`str`)
    """
    spacing, start = exactly(spacing, start)
    return spacing * to_fraction(range(n)) + start


def fsum(a):
    """
    Returns an accurate sum of elements of `a`.
    
    - `sum` is applied if the first element is rational
    - `mpmath.fsum` is applied if the first element is multiprecision
    - `math.fsum` is applied otherwise
    """
    a, _ = raveled(a)
    if isinstance(a[0], numbers.Rational):
        return sum(a)
    if isinstance(a[0], mp.mpf):
        return mp.fsum(a)
    return math.fsum(a)


# PROBLEM: uses `mp`
def bias_exponents(a, max_exponent):
    """
    Scales array `a` of floating-point numbers by an integer power of 2.
    
    Returns the integer power `n` of the scalar `2**n`.

    On return, the maximum element of `a` has `max_exponent` as its
    exponent. If, before and after the scaling operation, no element
    of `a` is subnormal (with leading zeros in its mantissa), then the
    mantissas of all elements are unchanged, and the operation is
    precisely invertible. The elements of `a` may be multiprecision
    floats.
    """
    a_max = np.max(a)
    basetype = type(a_max)
    unused_mantissa, current_max_exponent = mp.frexp(a_max) # WRONG
    power = max_exponent - current_max_exponent
    if power != 0.0:
        a *= basetype(2.0)**power
    return power


def mean_var(frequency, x):
    """
    Returns mean and variance for `frequency` distribution over `x`.
    
    The elements of array `frequency` are not necessarily integers. All
    calculations are done with numbers of type `Fraction`.
    """
    frequency = to_fraction(frequency)
    x = to_fraction(x)
    norm = sum(frequency)
    mean = sum(frequency * x) / norm
    var = sum(frequency * x**2) / norm - mean**2
    return mean, var

   
def mean_and_variance(x, p):
    """
    Return (mean, variance) for distribution(s) p of probability over x.
    
    Also works if x has the same shape as p.
    """
    axis = max(x.ndim, p.ndim) - 1
    mean = np.sum(np.multiply(p, x), axis=axis)
    variance = np.sum(np.multiply(p, np.square(x)), axis=axis)
    variance -= np.square(mean)
    return mean, variance


def print3D(a, field_format=' {:4.2e}', col_space=' '*2):
    """
    Print 3D array in 2D.
    """
    a = np.array(a)
    form = col_space + field_format * a.shape[2]
    form = form * a.shape[1]
    for row in a:
        print(form.format(*row.flatten()))
        

########################################################################
def dump_load_path(name):
    return DATA_DIR + name + '.p'

def dump_load_report(action, name, obj):
    done = obj.stoptime
    wait = done - obj.starttime
    print("{} '{}' done {} wait {}".format(action, name, done, wait))
    
def dump(obj, name):
    with open(dump_load_path(name), "wb") as file:
        pickle.dump(obj, file)
    dump_load_report('Dump', name, obj)
    
def load(name):
    with open(dump_load_path(name), "rb") as file:
        obj = pickle.load(file)
    dump_load_report('Load', name, obj)
    return obj
