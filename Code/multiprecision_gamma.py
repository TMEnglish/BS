class MultiprecisionGamma(object):
    """
    A limited implementation of the Gamma distribution.
    
    All floating-point operations are performed at arbitrary precision
    specified in the global context `mp`. 
    """
    def __init__(self, alpha='0.5', beta='500'):
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
        
        # Use of erfc when alpha is 0.5 improves speed and accuracy.
        if a == 0.5 and allow_special:
            return erfc(z ** 0.5)
        
        # See http://functions.wolfram.com/GammaBetaErf/Gamma2/26/01/03/0001/
        # for the formula used to calculate the incomplete gamma function, and
        # note that U is the Tricomi confluent hypergeometric function.
        return rgamma(a) * exp(-z) * hyperu(1 - a, 1 - a, z)
    
    def cdf(self, x):
        """
        Returns the value of the Gamma CDF at non-negative `x`.
        """
        return 1 - self.sf(mp_float(x))
        
    def mean(self):
        """
        Returns mean of the distribution.
        """
        return self.alpha / self.beta
    
    def variance(self):
        """
        Returns variance of the distribution.
        """
        return self.alpha / self.beta ** 2
    
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
        # Double working precision to get `mp.dps` accurate digits in result.
        with mp.workdps(2 * mp.dps):
            return mp.quad(self.pdf, [a, b])
    
    def raw_moment(self, power):
        """
        Integrates x ** power * pdf(x) over (0, inf), and returns result.
        """
        # Double working precision to get `mp.dps` accurate digits in result.
        with mp.workdps(2 * mp.dps):
            return mp.quad(lambda x: x ** power * self.pdf(x), [0, mp.inf])
    
    def discretized_mixture(self, n, delta='5e-4', weight='1e-3', normed=True):
        """
        Returns discretized mixture of the distribution and its reflection.

        That is, an array of 2 * `n` + 1 probability masses is returned. The 
        masses correspond to equispaced points ranging from `-n * delta` to
        `n * delta`. The unnormalized mass of each nonzero point x is the
        integral of the Gamma density over the width-`delta` interval 
        centered on |x|, multiplied by `weight` if x > 0, and by 1 - `weight`
        if x < 0. The mass of point 0 is `cdf(delta / 2)`. The `weight` must
        be in the closed interval [0, 1].

        The value of `normed` determines whether masses are normalized.
        """
        d = mp.mpf(delta)
        w = mp.mpf(weight)
        if w < 0 or w > 1:
            raise ValueError('Weight {} is out of range'.format(float(w)))
        #
        # Calculate the n + 1 endpoints of the width-d intervals centered on
        # points d, 2 * d, ..., n * d. Obtain the masses distributed over the
        # intervals by differencing the values of the survival function sf() 
        # at the endpoints. Construct the upper and lower tails of the mixture
        # distribution by weighting the masses (reversed for the lower tail).
        # The mass at point 0 is (1-w) cdf(d/2) + w cdf(d/2) = cdf(d/2).
        #
        endpoints = d * np.arange(1, n + 2) - d / 2
        sf = self.sf(endpoints)
        gamma_mass = sf[:-1] - sf[1:]
        upper_tail = w * gamma_mass
        lower_tail = (1 - w) * gamma_mass[::-1]
        mass_at_zero = self.cdf(d / 2)
        mixture = np.concatenate((lower_tail, [mass_at_zero], upper_tail))
        #
        if normed:
            return mixture / mp.fsum(mixture)
        return mixture 
    
    # Internal testing of class methods
    def _internal_tests(self, tolerance = mp.mpf('1e-20'), verbose=False):
        self._overall_result = True
        #
        def test(name, success):
            if success:
                if verbose:
                    print(name+':', 'pass')
            else:
                print(name+':', 'FAIL')
                self._overall_result = False
        #        
        def re(a, b):
            return abs((a - b) / b)
        #
        def comp(a, a_name, b, b_name):
            name = ' '.join(['Compare', a_name, 'to', b_name])
            test(name, re(a, b) < tolerance)
        #
        # General tests
        mean = self.mean()
        var = self.variance()
        m1 = self.raw_moment(1)
        m2 = self.raw_moment(2) - m1 ** 2
        comp(m1, 'calculated 1st moment', mean, 'analytic mean')
        comp(m2, 'calculated 2nd moment', var, 'analytic variance')
        special = self.sf(mean)
        general = self.sf(mean, allow_special=False)
        comp(special, 'special sf() for alpha=0.5', general, 'general sf()')
        #
        # Tests involving normalized mixture
        w = mp.mpf('1e-12')
        n = 500
        delta = mp.mpf('0.0005')
        p = self.discretized_mixture(n, delta, w, normed=True)
        comp(mp.fsum(p), 'sum of normed mixture probs', 1, '1')
        desired_ratio = (1 - w) / w
        ratios = p[:n] / p[:-(n+1):-1]
        test('Check ratios of all probs in tails of normed mixture',
             (re(ratios, desired_ratio) < tolerance).all())
        #
        # Tests involving unnormalized mixture
        p = self.discretized_mixture(n, delta, w, normed=False)
        p_0 = self.density_integral(0, delta / 2)
        comp(p[n], 'unnormed mixture prob of 0', p_0, 'integrated density')
        for i in range(1, n+1):
            interval = delta * np.array([i, i+1]) - delta / 2 
            integral = w * self.density_integral(*interval)
            if re(p[n+i], integral) >= tolerance:
                name = 'Compare unnormed_mixture[{}] to integrated density'
                test(name.format(n + i), False)
                abs_rel_error = float(re(p[n+i], integral))
                print('*** Absolute relative error:', abs_rel_error)
                break
        #
        if self._overall_result:
            print('Passed all tests.')
        else:
            print('FAILED one or more tests.')
