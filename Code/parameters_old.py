class Parameters(object):
    """
    Wraps all parameter settings of the infinite-population model.
    
    There are correspondingly named instance methods for q, F, M, and W.
    The distribution of fitness effects is calculated by the instance
    method `dfe`.
    
    There are correspondingly named members for scalar parameters n,
    d, w, gamma, beta, and mu; and for vectorial parameters b and m.
    The base-2 logarithm of the number L of loci, an integer power of 2,
    is assigned to member `log_L`. All numbers other than integers `n`
    and `log_L` are represented exactly as objects in class `Fraction`. 
    """
    # Numerical accuracy is of great concern. The general strategy is to
    # do exact calculations with rational numbers (Python type Fraction)
    # whenever practical, and to work with multiprecision floating-point
    # numbers otherwise. However, the L-th order convolution of an array
    # of numbers with itself is practical only with 64-bit floating-point
    # numbers.
    #
    def __init__(self, b_max='0.25', d='0.1', w='5e-4', gamma='1e-3',
                       beta='500.0', mu='1.0', log_L=0):
        """
        Sets elementary parameters of the infinite-population model.
                
        Parameters (all but the last are given as strings)
        `b_max`: maximum birth parameter, evenly divisible by `w`
        `d`    : death parameter
        `w`    : bin width
        `gamma`: weighting of advantageous mutational effects in
                 Sanford's distribution of fitness effects (DFE)
        `beta` : rate parameter of the Gamma distribution (with shape
                 parameter alpha=0.5) in Sanford's DFE
        `mu`   : probability that mutation occurs in an offspring
        `log_L`: base-2 logarithm of the number of loci in genotypes
        
        Default settings come from Section 5 of Basener and Sanford.
        """
        # Verify that all non-integer arguments are given as strings.
        args = [b_max, d, w, gamma, beta, mu]
        assert all(isinstance(x, str) for x in args)
        self.log_L = log_L
        #
        # Convert parameters to Fraction objects.
        self.d = Fraction(d)
        self.w = Fraction(w)
        self.beta = Fraction(beta)
        self.gamma = Fraction(gamma)
        self.mu = Fraction(mu)
        #
        # Derive the number n of classes from the given maximum of the
        # birth rate parameters and the bin width w.
        n = Fraction(b_max) / self.w + 1
        assert n.denominator == 1, 'b_max is not evenly divisible by w'
        self.n = n.numerator
        #
        # Create an array of n+1 evenly spaced Fractions ranging from 0
        # to n*w. First create an array [0, 1, ..., n] of integers, and
        # then scale each element of the array by Fraction w, producing
        # a new array of Fractions.
        base = self.w * np.arange(self.n+1)
        #
        # The first n elements of array `base` are the exact birth rate
        # parameters.
        self.b = base[:self.n]
        #
        # Calculate exactly the n+1 walls of the width-w bins centered
        # on the birth rate parameters: subtract Fraction w/2 from each
        # element of `base`, producing a new array of Fraction objects.
        self.b_walls = base - self.w / 2
        #
        # Calculate exactly the fitnesses and the walls of bins centered
        # on the fitnesses by subtracting Fraction d from, respectively,
        # the birth rate parameters and the walls of bins centered on the
        # birth rate parameters.
        self.m = self.b - self.d
        self.m_walls = self.b_walls - self.d
        
    def W(self, as_float=True):
        """
        Returns the derivative operator W = M - d I as a square array.
        
        Array elements are initially numbers of type `Fraction`, and are
        ultimately converted to 64-bit floats if `as_float` is true. Some
        of the calculations are done in floating-point arithmetic, so the
        results are not exact in any case.
        """
        # Subtract the death rate parameter from all elements of the main
        # diagonal of matrix `M()`. The numbers are all of type Fraction.
        w = self.M(as_float=False)
        w[np.diag_indices(self.n)] -= self.d
        if as_float:
            w = w.astype(float)
        return w

    def M(self, as_float=True):
        """
        Returns mutant-birth rate parameters M = F B as a square array.
        
        For column array P specifying the numbers of organisms in classes,
        the array product M P gives the rates of birth of organisms into
        those classes, taking mutation into account.
        
        Array elements are initially numbers of type `Fraction`, and are
        ultimately converted to 64-bit floats if `as_float` is true. Some
        of the calculations are done in floating-point arithmetic, so the
        results are not exact in any case.
        """
        # Scale the columns of matrix `F()` by the corresponding birth
        # rate parameters. This operation is equivalent to multiplication
        # of `F()` by the square matrix `diag(self.b)`. The numbers are
        # all of type Fraction.
        m = self.F() * self.b
        if as_float:
            m = m.astype(float)
        return m
        
    def F(self):
        """
        Returns the array F of distributions of offspring over classes.
        
        The array is n-by-n, where n is the number of classes, `self.n`.
        Each of the columns represents a probability distribution. The
        i,j-the element is the probability that a birth to a parent in
        class j belongs to class i. 
        
        Each off-diagonal element (i != j) is set to the probability that
        mutation at one or more loci changes fitness by (i - j)w units,
        where w is the bin width, `self.w`. This probability does not
        depend on the class of the parent. Elements of the main diagonal
        (i == j) are set to make all column sums equal to 1. Thus the
        probability that a birth is identical in class to its parent
        generally does depend on the class of the parent.
        
        The probabilities are numbers of type `Fraction`. However, some
        of the calculations are done in floating-point arithmetic. Thus
        the probabilities are not exact.
        """
        # Set the i,j-element to the probability of fitness difference
        # m[i] - m[j], which is offset by n - 1 elements in array q[].
        q = self.q()
        F = np.array([[q[(i - j) + (self.n - 1)]
                        for j in range(self.n)]
                            for i in range(self.n)])
        #
        # Reset the elements of the main diagonal of f to make the sum
        # of elements in each column equal to 1. That is, subtract the 
        # column sums from 1, and add the results to the corresponding
        # elements of the main diagonal of f.
        F[np.diag_indices(self.n)] += 1 - F.sum(axis=0)
        return F

    def q(self):
        """
        Returns a probability distribution over fitness differences.
        
        The distribution is derived from the L-fold convolution of the
        distribution over possible effects of mutation on fitness at each
        of L loci, where L is 2**`self.log_L`. The result depends on the
        per-locus mutation rate, `self.mu`/L, and the distribution of
        fitness effects for a single mutation, calculated by instance
        function `dfe()`.
        
        The probability distribution is represented as an array of 2n - 1
        positive numbers of type Fraction, where n is the number of
        classes, `self.n`. The numbers sum exactly to 1. The probability
        of fitness difference m[i] - m[j] is indexed (i - j) + (n - 1).
        """
        # Convert mutation rate for the entire genome to mutation rate at
        # each of the L loci.
        mu = self.mu / 2**self.log_L
        #
        # The distribution of probability over fitness (non)changes due
        # to (non)mutation at a single locus is an array of 4n - 1
        # Fractions, with the middle element giving the probability of 
        # no change.
        q = mu * self.dfe(2*self.n)
        q[2*self.n - 1] += 1 - mu
        #
        # If there is more than one locus, i.e., if L > 1, take the L-
        # fold convolution of the distribution with itself to obtain the
        # distribution of the sum of L i.i.d. contributions to change in
        # fitness. In each convolution, the result is restricted to
        # 4n - 1 elements, the sum of which is stored as an element of
        # array `self._convolution_mass` to facilitate validation of the
        # code. Work with 64-bit floats, and convert the numbers in the
        # final distribution to type Fraction.
        self._convolution_mass = np.empty(self.log_L+1)
        if self.log_L > 0:
            q = q.astype(float)
            for i in range(self.log_L):
                q = np.convolve(q, q, 'same')
                self._convolution_mass[i] = fsum(q)
                q /= self._convolution_mass[i]
            q = to_fraction(q)
        # Trim excess elements from the tails of the distribution, 
        # reducing the number of elements to 2n - 1, and then normalize.
        # The mass of the unnormalized distribution (with tails trimmed)
        # is stored as the last element of `self._convolution_mass`. 
        excess = (len(q) - (2 * self.n - 1)) // 2
        q = q[excess:-excess]
        assert len(q) == 2 * self.n - 1, 'q: Bad length of result'
        q_norm = sum(q)
        self._convolution_mass[-1] = q_norm
        q /= q_norm
        return q
        
    def dfe(self, n, normed=True):
        """
        Returns Sanford's distribution of fitness effects, discretized.
        
        Writing w for `self.w`, the bin width, the possible effects of
        a single mutation on fitness are
        
            (1 - n)w, ..., -w, 0, w, ..., (n - 1)w.
        
        The unnormalized probability of mutational effect x is 
        
            G(x + w/2) - G(x - w/2),
        
        where G is the cumulative distribution function of the weighted
        mixture of a Gamma distribution (shape parameter alpha=0.5, rate
        parameter `self.beta`) and its reflection. The weighting of the
        Gamma distribution in the mixture is `self.gamma`, a number
        in the closed interval [0, 1].
        
        The returned array, containing 2n - 1 numbers of type `Fraction`,
        is normalized if `normed` is true (in which case the numbers sum
        exactly to unity). Some calculations are done in floating-point
        arithmetic with precision determined by the `mpmath` environment
        from which this method is called. Thus the probabilities are not
        exact.
        """
        # Approach:
        # 1. Discretize the Gamma distribution over the positive
        #    mutational effects, i.e., produce an array of masses for
        #    the width-w bins centered on the positive effects.
        # 2. Reverse the array to get the unweighted masses for the
        #    negative mutational effects.
        # 3. Weight the masses for positive and negative effects by
        #    gamma and 1 - gamma, respectively.
        # 4. Calculate the mass of the width-2 bin centered on zero
        #    effect as the value of the Gamma cumulative distribution
        #    function at w/2.
        # 5. Assemble the weighted masses into a single array, and
        #    normalize conditionally.
        #
        # Notes:
        # 1. The positive mutational effects are identical to the
        #    positive birth parameters.
        # 2. Slices [1:] and [:-1] of an array reference, respectively,
        #    all but the first and all but the last of the elements.
        # 3. Slice [::-1] of an array references all of the elements in
        #    reverse order.
        # 4. With shape parameter alpha=0.5, the complementary CDF of
        #    the Gamma distribution may be calculated in terms of the
        #    erfc function.
        #
        # Evaluate the complementary CDF of the Gamma distribution at
        # the walls of bins centered on the positive effects. The
        # calculation of the square root and the erfc functions is done
        # in floating-point arithmetic, with precision determined by the
        # `mpmath` context in which this method is called. The results
        # are converted exactly to Fractions.
        walls = self.w * np.arange(n) + self.w / 2
        z = to_mpf(self.beta*walls)
        comp_cdf = to_fraction(mp_erfc(mp_sqrt(z)))
        #
        # Calculate unweighted bin masses by differencing the comple-
        # mentary CDF at the bin walls. If any of the masses is zero, then
        # there was insufficient precision in the calculation of the
        # complementary CDF.
        masses = comp_cdf[:-1] - comp_cdf[1:]
        assert all(masses > 0), 'Insufficient precision in dfe()'
        #
        # Weight the (reversed) masses to obtain masses for the upper
        # (lower) tail of the mixture distribution. 
        upper_tail = self.gamma * masses
        lower_tail = (1 - self.gamma) * masses[::-1]
        #
        # Assemble the masses into a single array, with the mass for
        # zero mutational effect in the middle. Conditionally normalize
        # the resulting distribution.
        dfe = np.concatenate((lower_tail, [1-comp_cdf[0]], upper_tail))
        if normed:
            dfe /= sum(dfe)
        return dfe

    def normal_freqs(self, mean='0.044', std='0.005', as_float=True):
        """
        Returns a binned Gaussian distribution over fitness.
        
        The unregularized probability of fitness `m[i]` is
        
            `F(m[i] + self.w/2) - F(m[i] - self.w/2)`,
            
        where `F` is the cumulative distribution function of the
        normal distribution with the given `mean` and standard
        deviation `std`. The default settings of the parameters come
        from Section 5 of Basener and Sanford.
        
        The probability distribution is calculated as an array of
        `self.n` numbers of type Fraction. Probabilities are then
        converted to 64-bit floats if `as_float` is true. The
        distribution is regularized.
        """
        # The mean and standard deviation must be given as strings.
        assert isinstance(mean, str) and isinstance(std, str)
        mean = Fraction(mean)
        std = Fraction(std)
        #
        # Evaluate the CDF and the complementary CDF at the walls of
        # width-w bins centered on the fitnesses. Note that accuracy is
        # much greater using erfc() than using erf(). The calculation of
        # z is exact until the division by the square root of 2.
        z = to_mpf((self.m_walls - mean) / std) / mp.sqrt(2)
        cdf = Fraction('1/2') * to_fraction(mp_erfc(-z))
        ccdf = Fraction('1/2') * to_fraction(mp_erfc(z))
        #
        # Calculate bin masses by differencing CDF values at the bin
        # walls, and also by differencing complementary CDF values at
        # the bin walls. The slice [1:] includes all array elements
        # except the first, and the slice [:-1] includes all array
        # elements except the last.
        per_cdf = cdf[1:] - cdf[:-1]
        per_ccdf = ccdf[:-1] - ccdf[1:]
        #
        # For accuracy, use the CDF differences for bins with upper
        # walls no greater than the mean, and the complementary CDF
        # differences for other bins.
        freqs = np.where(self.m_walls[1:] <= mean, per_cdf, per_ccdf)
        assert all(freqs > 0), 'Insufficient precision in normal_freqs()'
        if as_float:
            freqs = freqs.astype(float)
            freqs /= math.fsum(freqs)
        else:
            freqs /= sum(freqs)
        return freqs