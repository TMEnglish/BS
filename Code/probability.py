def binned_normal(bin_walls, mean='0.044', std='0.005', normed=True):
    """
    Returns a binned normal distribution.

    The elements of the array `bin_walls` must be strictly ascending.
    The i-th element of the returned array is the probability mass
    for the interval `(bin_walls[i], bin_walls[i+1])`. Calculations
    are done with multiprecision floats.  If `normed` is true, the
    returned probability masses are normalized.
    """
    # Evaluate the CDF and the complementary CDF at the bin walls.
    cdf = normal_cdf(bin_walls, mean, std)
    ccdf = normal_ccdf(bin_walls, mean, std)
    #
    # Calculate bin masses by differencing the CDF values, and also
    # by differencing the complementary CDF values.
    per_cdf = cdf[1:] - cdf[:-1]
    per_ccdf = ccdf[:-1] - ccdf[1:]
    #
    # In the lower (upper) tail, use differences of the CDF (CCDF).
    p = np.where(bin_walls[1:] < mp_float(mean), per_cdf, per_ccdf)
    if normed:
        p /= mp.fsum(p)
    return p


def binned_mixture(comp_cdf, bin_walls, weight, normed=True):
    """
    Returns a mixture of a binned distribution and its reflection.
    
    The complementary cumulative distribution function `comp_cdf` is
    for a continuous probability distribution with the positive real
    numbers in its support. Elements of the array `bin_walls` must be
    positive and strictly ascending. Function `comp_cdf` must accept
    arrays of points as arguments.
    
    The axis of reflection is x = 0, and the center bin of the mixture
    has endpoints `(-bin_walls[0], bin_walls[0])`. Probability masses
    for positive and negative bins in the mixture are scaled,
    respectively, by `weight` and `1 - weight`. An array of masses is
    returned. If normed is true, the masses are normalized.
    """
    # Introduce a bin wall at zero.
    assert bin_walls[0] > 0
    walls = np.concatenate(([0], bin_walls))
    #
    # Difference the values of the complementary CDF at the bin walls
    # to obtain the probability masses of bins.
    complementary_cdf = comp_cdf(walls)
    p = complementary_cdf[:-1] - complementary_cdf[1:]
    tail = p[1:]
    #
    # Scale the tail of `p` by `weight` and `1-weight`, respectively,
    # to obtain the upper and (reversed) lower tails of the mixture
    # `q`. The mass for the middle bin reduces to the difference of
    # the complementary CDF at 0 and `bin_walls[0]`. We have already
    # set `p[0]` to that value.
    lower_tail = (1 - weight) * tail[::-1]
    upper_tail = weight * tail
    q = np.concatenate((lower_tail, [p[0]], upper_tail))
    if normed:
        q /= fsum(q)
    return q


def normal_cdf(x, mean='0.044', std='0.005'):
    """
    Gaussian cumulative distribution function.
    
    The argument `x` must be either a scalar or a NumPy array. The
    result of calculations with multiprecision floats is returned.
    """
    arg = (x - mp_float(mean)) / (mp_float(std) * mp_float(2.0)**0.5)
    return 0.5 * (1 + erf(arg))


def normal_ccdf(x, mean='0.044', std='0.005'):
    """
    Gaussian complementary cumulative distribution function.
    
    The argument `x` must be either a scalar or a NumPy array. The
    result of calculations with multiprecision floats is returned.
    """
    arg = (x - mp_float(mean)) / (mp_float(std) * mp.mpf(2.0)**0.5)
    return 0.5 * erfc(arg)


def bs_gamma_ccdf(x):
    """
    Gamma complementary cumulative distribution function.
    
    As for Basener and Sanford, shape alpha=0.5 and rate beta=500. The
    argument `x` must be either a scalar or a NumPy array. The result
    of calculations with multiprecision floats is returned.
    """
    # Use of erfc when alpha is 0.5 improves speed and accuracy.
    beta = mp.mpf(500)
    result = np.where(x > 0, erfc((beta * x) ** 0.5), mp.mpf(1))
    #
    # If the result is a 0-d array, return a plain scalar instead.
    if result.ndim == 0:
        result = result.item()
    return result