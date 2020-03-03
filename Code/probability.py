def normal_pdf(x, mean='0', std='1'):
    """
    Returns result of multiprecision calculation of Gaussian density.
    """
    x = mp_float(x)
    mean = mp_float(mean)
    std = mp_float(std)
    z = (x - mean) / std
    result = exp(-0.5 * z**2) 
    result /= std * (2 * mp.pi)**0.5
    return result


def normal_cdf(x, mean='0', std='1'):
    """
    Returns result of multiprecision calculation of Gaussian CDF.
    
    The argument `x` may be an array.
    """
    x = mp_float(x)
    arg = (x - mp_float(mean)) / (mp_float(std) * mp_float(2.0)**0.5)
    return 0.5 * (1 + erf(arg))


def normal_ccdf(x, mean='0', std='1'):
    """
    Returns Gaussian complementary CDF value as multiprecision float.
    
    The argument `x` may be an array.
    """
    x = mp_float(x)
    arg = (x - mp_float(mean)) / (mp_float(std) * mp_float(2.0)**0.5)
    return 0.5 * erfc(arg)


def gamma_pdf(x, alpha='0.5', beta='500'):
    """
    Returns Gamma probability density as multiprecision float.
    
    The intended use is in numerical integration, to check other
    calculations. Integrate with an expression like
    
        `mp.quad(gamma_pdf, [a, b])`,
        
    where `a` and `b` are the limits of the integral.
    """
    x = mp_float(x)
    alpha = mp_float(alpha)
    beta = mp_float(beta)
    result = beta ** alpha / gamma(alpha) * x ** (alpha - 1)
    result *= mp_exp(-beta * mp_float(x))
    return result

    
def gamma_cdf(x, alpha='0.5', beta='500'):
    """
    Returns multiprecision value of the Gamma CDF.
    """
    return regularized_lower_incomplete_gamma(x, alpha, beta)

    
def gamma_ccdf(x, alpha='0.5', beta='500'):
    """
    Returns multiprecision value of the Gamma complementary CDF.
    """
    return regularized_upper_incomplete_gamma(x, alpha, beta)


def regularized_upper_incomplete_gamma(x, alpha=0.5, beta=500, 
                                       allow_special=True):
    """
    Returns multiprecision value of the Gamma complementary CDF.

    The result is the value of the regularized (upper) incomplete gamma
    function with arguments alpha and z = beta * x. The Boolean value
    of `allow_special` determines whether special cases get special
    handling.
    """
    a = mp_float(alpha)
    z = mp_float(beta) * mp_float(x)
    #
    # Use of erfc when alpha is 0.5 improves speed and accuracy.
    if a == 0.5 and allow_special:
        return erfc(z ** 0.5)
    #
    # http://functions.wolfram.com/GammaBetaErf/Gamma2/26/01/03/0001/
    # gives the formula for the incomplete gamma function (with U
    # denoting the the Tricomi confluent hypergeometric function).
    return rgamma(a) * exp(-z) * hyperu(1 - a, 1 - a, z)

        
def regularized_lower_incomplete_gamma(x, alpha=0.5, beta=500,
                                       allow_special=True):
    """
    Returns multiprecision value of the Gamma CDF.
    
    We don't expect to experiment with values of `alpha` other than
    0.5, which is a very nice case: the standard `erf()` is applied to
    the square root of `beta * x`. In other cases, the calculations
    take some seconds to complete.
    """
    x = mp_float(x)
    alpha = mp_float(alpha)
    beta = mp_float(beta)
    z = np.multiply(x, beta)
    #
    # Use of erfc when alpha is 0.5 improves speed and accuracy.
    if alpha == 0.5 and allow_special:
        return erf(z ** 0.5)
    unregularized = hypergeometric_lower_incomplete_gamma(alpha, z)
    return mp.rgamma(alpha) * unregularized

    
def hypergeometric_lower_incomplete_gamma(s, z):
    """
    Returns value of the lower incomplete gamma function.
    
    The result is not regularized. Kummer's confluent hypergeometric
    function is used in the calculation. (See "Incomplete gamma
    function" in Wikipedia for details.)

    To calculate the cumulative distribution function of the Gamma
    distribution, set `s` equal to alpha, and `z` equal to the product
    of beta and the array of x values for which results are desired.
    Multiply the result by `mp.rgamma(alpha)`, the reciprocal Gamma
    function, to regularize.
    """
    s = mp_float(s)
    z = mp_float(z)
    return z ** s / s * hyp1f1(s, s + 1, -z)


def binned_normal(bin_walls, mean='0', std='1', normed=True):
    """
    Returns binned normal distribution for given `bin_walls`.
    
    The endpoints of n-th bin are `bin_walls[n:n+1]`. The masses of
    the bins are multiprecision floating point numbers.
    """
    # Work with at least 300 digits of precision.        
    with mp.workdps(max(mp.dps, 300)):
        # Calculate bin masses by differencing the CDF.
        cdf = normal_cdf(bin_walls, mean, std)
        per_cdf = cdf[1:] - cdf[:-1]
        #
        # Calculate bin masses by differencing the complementary CDF.
        ccdf = normal_ccdf(bin_walls, mean, std)
        per_ccdf = ccdf[:-1] - ccdf[1:]
        #
        # In the lower tail, the result of differencing the CDF is more
        # accurate than the result of differencing the complementary
        # CDF. The opposite holds in the upper tail.
        p = np.where(bin_walls[1:] < mp_float(mean), per_cdf, per_ccdf)
        #
        # Issue a warning if any of the bin masses is zero.
        if np.any(p == 0):
            message = 'bin mass of 0 due to insufficient precision'
            warnings.warn(message)
        #
        # Conditionally scale the masses so that they sum to 1.
        if normed:
            p /= mp.fsum(p)
        return p

    
def binned_mixture(comp_cdf, n_bins, positive_weight, bin_width='5e-4',
                   normed=True):
    """
    Returns a binned mixture of a distribution and its reflection.
    
    The distribution is defined over (0, infinity). Its complementary 
    cumulative distribution function is `comp_cdf`.
    
    The middle of the `n_bins` bins is centered on zero. In the mixture
    distribution, the weight of the distribution over positive numbers
    is `weight`, and the weight of the reflection of that distribution
    is `1 - weight`. If `normed` is true, then the bin probabilities
    are scaled so that they sum to 1.
    """
    # Work with at least 300 digits of precision.        
    with mp.workdps(max(mp.dps, 300)):
        # Calculate the endpoints of width-w bins centered on points w,
        # 2w, ..., nw, where n is the number of bins in the upper tail.
        n = n_bins // 2
        w = mp_float(bin_width)
        bin_endpoints = w * np.linspace(1/2, n + 1/2, n + 1)
        #
        # Difference the complementary CDF at the endpoints of the bins
        # to obtain the probability masses of the bins.
        complementary_cdf = comp_cdf(bin_endpoints)
        tail_masses = complementary_cdf[:-1] - complementary_cdf[1:]
        #
        # Weight the masses (reversed for the lower tail) to obtain
        # masses for the bins in the tails of the mixture distribution.
        weight = mp_float(positive_weight)
        upper_tail = weight * tail_masses
        lower_tail = (1 - weight) * tail_masses[::-1]
        #
        # The bin centered on zero is [-w/2, w/2] has mass CDF(w/2).
        mass_at_zero = 1 - complementary_cdf[0]
        #
        # Assemble the bin masses into a single array, and issue a
        # warning if any of them is equal to zero.
        p = np.concatenate((lower_tail, [mass_at_zero], upper_tail))
        if np.any(p == 0):
            warnings.warn('bin mass of 0 due to insufficient precision')
        #
        # Conditionally scale the masses so that they sum to 1.
        if normed:
            p /= mp.fsum(p)
        return p


def bs_initial_frequencies(rates):
    """
    Returns binned normal distribution with mean=0.044 and std=0.005.
    
    Probability is distributed over fitness. The parameters come from
    Section 5 of Basener and Sanford.
    """
    fit = rates.fitness
    half_w = rates.bin_width / 2
    bin_walls = np.concatenate(([fit[0] - half_w], fit + half_w))
    return binned_normal(bin_walls, mean='0.044', std='0.005')


def bs_mutation_probabilities(rates, beneficial_weight='1e-3'):
    """
    Returns binned mixture of a Gamma distribution and its reflection.
    
    Probability is distributed over mutational effects. The shape and
    rate parameters of the Gamma distribution are alpha=0.5 and beta=500,
    respectively. The parameter settings come from Section 5 of Basener
    and Sanford. The weight of the Gamma distribution in the mixture is
    `beneficial_weight`, which Basener and Sanford set to 1e-3.
    """
    nbins = len(rates.effects)
    bin_width = rates.bin_width
    #
    # The bin masses are obtained by differencing the complementary CDF
    # of the Gamma distribution.
    comp_cdf = lambda x: gamma_ccdf(x, alpha=0.5, beta=500)
    return binned_mixture(comp_cdf, nbins, beneficial_weight, bin_width)