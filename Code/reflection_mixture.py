def reflection_mixture(ccdf, weight, n, delta, normed=False):
    """
    Return discretized mixture of a distribution and its reflection.
    
    * `ccdf`  : complementary CDF specifying a continuous probability
                distribution over the positive (or non-negative) reals
    * `weight`: weighting (exact) of the distribution in a mixture with
                its reflection over negative reals (0 <= `weight` <= 1)
    * `n`     : probability masses are calculated at 2n + 1 evenly
                spaced points, the (n+1)-st of which is zero
    * `delta` : spacing of points (exact)
    * `normed`: determines whether the masses are normalized
    
    "Exact" parameter values are of type `str`, `int`, or `Fraction`.
    The values returned by function `ccdf` should be of type `Fraction`.
    """
    # For array `a`, `a[1:]` excludes the first element, `a[:-1]` 
    # excludes the last element, and `a[::-1]` reverses the elements.
    delta, weight = exactly(delta, weight)
    bins = equispaced(n + 1, spacing=delta, start=delta/2)
    ccdf_values = ccdf(bins)
    mass_at_zero = 1 - ccdf_values[0]
    unweighted_tail_masses = ccdf_values[:-1] - ccdf_values[1:]
    upper_tail = weight * unweighted_tail_masses
    lower_tail = (1 - weight) * unweighted_tail_masses[::-1]
    pmf = np.concatenate((lower_tail, [mass_at_zero], upper_tail))
    if normed:
        pmf /= sum(pmf)
    return pmf