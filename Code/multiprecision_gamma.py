class MultiprecisionGamma(object):
    """
    A limited implementation of the Gamma distribution in multiprecision math.
    
    All numerical operations are performed at arbitrary precision, in the global
    context `mp`.
    """
    def __init__(self, alpha=0.5, beta=500):
        """
        Sets the Gamma rate and shape to `alpha` and `beta`, respectively.
        """
        self.alpha = mp_float(alpha)
        self.beta = mp_float(beta)
    
    def pdf(self, x):
        """
        Returns the value of the Gamma density function at positive `x`.
        """
        x = mp_float(x)
        a = self.alpha
        b = self.beta
        return b ** a * rgamma(a) * exp(-b * x) * x ** (a - 1)
    
    def sf(self, x, allow_special=True):
        """
        Returns the value of the Gamma complementary CDF at non-negative `x`.
        
        The result is the value of the regularized (upper) incomplete gamma
        function with arguments alpha and z = beta * x. The Boolean value of
        `allow_special` determines whether special cases get special handling.
        """
        a = mp_float(self.alpha)
        z = self.beta * mp_float(x)
        
        # Use of erfc in this special case improves speed and accuracy.
        if a == 0.5 and allow_special:
            return erfc(z ** 0.5)
        
        # See http://functions.wolfram.com/GammaBetaErf/Gamma2/26/01/03/0001/
        # for the formula used to calculate the incomplete gamma function, and
        # note that U is the Tricomi confluent hypergeometric function.
        return rgamma(a) * exp(-z) * hyperu(1 - a, 1 - a, z)
    
    def mass(self, endpoints):
        """
        Returns probability masses of intervals with given `endpoints`.
        
        The masses are obtained by differencing the complementary CDF at the
        points in the iterable `endpoints`.
        """
        return np.abs(np.diff(self.sf(endpoints)))

    def density_integral(self, a, b):
        """
        Numerically integrates the density over `(a, b)`, and returns result.
        """
        # Double working precision to obtain `mp.dps` accurate digits in result.
        with mp.workdps(2 * mp.dps):
            return mp.quad(self.pdf, [a, b])
    
    def raw_moment(self, power):
        """
        Integrates x ** power * pdf(x) over (0, inf), and returns result.
        """
        # Double working precision to obtain `mp.dps` accurate digits in result.
        with mp.workdps(2 * mp.dps):
            return mp.quad(lambda x: x ** power * self.pdf(x), [0, mp.inf])