def binned_mixture(comp_cdf, n_bins, weight, bin_width='5e-4', normed=True):
    """
    Returns a binned mixture of a distribution and its reflection.
    
    The distribution is defined over (0, infinity). Its complementary 
    cumulative distribution function is `comp_cdf`.
    
    The middle of the `n_bins` bins is centered on zero. The weight of
    the positive distribution in the mixture is `weight`, and the weight
    of the reflection of the distribution is `1 - weight`. If `normed` is
    true, then the bin probabilities are scaled so that they sum to 1.
    """
    # Calculate endpoints of width-w bins centered on points w, 2w, ...,
    # nw, where n is the number of bins in the upper tail.
    n = n_bins // 2
    w = mp_float(bin_width)
    bin_endpoints = w * np.linspace(1/2, n + 1/2, n + 1)
    #
    # Difference the complementary CDF at the endpoints of the bins to
    # obtain the probability masses of the bins.
    complementary_cdf = comp_cdf(bin_endpoints)
    tail_masses = complementary_cdf[:-1] - complementary_cdf[1:]
    #
    # Weight the masses (reversed for the lower tail) to obtain masses
    # for the bins in the tails of the mixture distribution.
    weight = mp_float(weight)
    upper_tail = weight * tail_masses
    lower_tail = (1 - weight) * tail_masses[::-1]
    #
    # The mass of the bin centered on zero, [-w/2, w/2], is CDF(w/2).
    mass_at_zero = 1 - complementary_cdf[0]
    #
    # Assemble the bin masses into a single array.
    p = np.concatenate((lower_tail, [mass_at_zero], upper_tail))
    #
    if normed:
        # Scale the masses so that they sum to 1.
        p /= mp.fsum(p)
    return p