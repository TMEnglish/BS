def binned(pdf, centers, normed=True):
    """
    Returns the result of binning probability density function `pdf`.
    
    The bin width is derived from the given bin `centers`, which are
    assumed to be equispaced and strictly ascending. The returned
    probability masses are multiprecision floats. If `normed` is
    true, the masses are scaled so that they sum to one.
    """
    n = len(centers)
    p = np.empty(n, dtype=object)
    #
    # Integrating the probability density function with twice the
    # current precision `mp.dps` usually yields masses with `mp.dps`
    # accurate digits.
    with mp.workdps(2 * mp.dps):
        # Define `n + 1` endpoints of `n` bins with given `centers`.
        width = (mp_float(centers[-1]) - centers[0]) / (n - 1)
        least_endpoint = centers[0] - width/2
        other_endpoints = centers + width/2
        endpoints = np.concatenate(([least_endpoint], other_endpoints))
        #
        # If a bin does not contain the point zero, then integrate the
        # density from its lower endpoint to its upper endpoint.
        # Otherwise, split the bin at zero, integrate the density over
        # the two subintervals, and sum the results.
        for i in range(n):
            a, b = endpoints[i:i+2]
            if a * b > 0:
                p[i] = mp.quad(pdf, [a, b])
            else:
                p[i] = mp.quad(pdf, [a, 0]) + mp.quad(pdf, [0, b])
        if normed:
            p /= mp.fsum(p)
    return p


class Normal(object):
    """
    Callable object calculating Gaussian probability density function.
    
    The parameters are set when the class is instantiated. All
    calculations are done with multiprecision floatint-point numbers.
    """
    def __init__(self, mean='0.044', std='0.005'):
        """
        Sets the parameters of the distribution.
        """
        self.mean = mp_float(mean)
        self.std = mp_float(std)

    def __call__(self, x):
        """
        Returns the probability density at `x` as multiprecision float.
        """
        z = (x - self.mean) / self.std
        result = exp(-0.5 * z**2) 
        result /= self.std * (2 * mp.pi)**0.5
        return result


class GammaMixture(object):
    """
    Callable object calculating probability density for Gamma mixture.
    
    The distribution is a mixture of a Gamma distribution and its
    reflection. Parameters are set when the class is instantiated. All
    calculations are done with multiprecision floatint-point numbers.
    """
    def __init__(self, alpha='0.5', beta='500', weight='1e-3'):
        """
        Sets the parameters of the distribution.
        
        Here `alpha` and `beta` are, respectively, the shape and rate
        parameters of the Gamma distribution. The given `weight` is
        for the distribution over positive numbers.
        """
        self.a = mp_float(alpha)
        self.b = mp_float(beta)
        self.weight = mp_float(weight)
        # Compute the normalizing constant only once.
        self.norm = self.b**self.a * mp.rgamma(self.a)
        
    def __call__(self, x):
        """
        Returns the probability density at `x` as a multiprecision float.

        For positive `x`, the probability density is `weight` times
        the Gamma density at `x`. For negative `x`, the density is 
        `1 - weight` times the Gamma density at `-x`. If `x` is zero,
        then infinity is returned.
        """
        if x < 0:
            w = 1 - self.weight
            x = -x
        else:
            w = self.weight
        return w * self.norm * x**(self.a - 1) * mp.exp(-self.b * x)