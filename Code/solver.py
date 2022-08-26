class Solver(object):
    """
    A solver for relative frequencies in the modified model.
    
    A modification of the forward Euler method of numerical integration
    is applied. Solutions for relative frequencies are set to zero when
    they fall below a given threshold, and subsequently are held at zero.
    Only end-of-year solutions are stored, beginning with the solution
    for year 0 (derived from the given initial frequencies). Note that 
    solutions for the infinite-population model can be obtained by
    setting the threshold to zero.
    
    The solver is run by calling this object. Each call extends the 
    solutions by a given number of years. The end-of-year solutions are
    retrieved by indexing this object.
    """ 
    def __init__(self, W, initial_freqs, log_steps_per_year=10,
                       threshold=1e-9):
        """
        Initialize the solver.
        
        Parameter `W` is a derivative matrix operator. The value of
        `log_steps_per_year` must be a non-negative integer. When the
        solver is run, there are `2 ** log_steps_per_year` integration
        steps  per year. The step size is the reciprocal of the number
        of steps per year.
        
        The solution for year 0 is `initial_freqs` with frequencies
        less than `threshold` times the sum of the initial frequencies
        set to zero.
        """
        self.W = np.array(W)
        assert type(log_steps_per_year) is int
        assert log_steps_per_year >= 0
        self.steps_per_year = 2 ** log_steps_per_year
        self.step_size = 1 / self.steps_per_year
        self.threshold = threshold
        #
        # Array `s` always contains the latest solution for frequencies
        # of classes, scaled by `2**s_bias` to avoid overflow and
        # underflow in calculations. The setting of `max_exponent`,
        # which specifies the exponent of the maximum element of `s`, 
        # is small enough to ensure that the sum of squared elements of
        # `s` can be calculated with 64-bit floats. The base type of
        # `s` is that of `initial_freqs`.
        self.s = np.array(initial_freqs)
        self.max_exponent = 510 - math.ceil(math.log2(len(self.s)))
        self.s_bias = bias_exponents(self.s, self.max_exponent)
        #
        # Zero initial frequencies that are below threshold.
        self._zero()
        #
        # Array `solutions` contains the solutions for end-of-year
        # relative frequencies of classes. The base type is float. There
        # is one row for each year.
        self.n_solutions = 1
        self.solutions = np.empty((self.n_solutions, len(self.s)))
        self.solutions[0] = self.s / fsum(self.s)

    def _zero_subthreshold_frequencies(self):
        """
        Zeroes calculated frequencies that are below threshold.
        
        Returns a Boolean array indicating which frequencies are zero.
        """
        # Determine which calculated frequencies are below threshold.
        # The array `subthreshold` contains Boolean values.
        subthreshold = self.s < self.threshold * self.s.sum()
        #
        # Using Boolean indexing, zero the subthreshold elements of `s`.
        self.s[subthreshold] = 0.0
        return subthreshold
        
    def _zero(self):
        """
        Zeroes subthreshold calculated frequencies and their derivatives.
        """
        if self.threshold > 0.0:
            # Array `zeroed` indicates which calculated frequencies are
            # zero. Rows of the derivative operator `W` corresponding to
            # zeroed frequencies are zeroed.
            zeroed = self._zero_subthreshold_frequencies()
            self.W[zeroed,:] = 0.0
        
    def __call__(self, n_years=1000):
        """
        Solve for `n_years` end-of-year relative frequencies.
        """
        # Extend the `solutions` array to hold an additional `n_years`
        # solutions for end-of-year relative frequencies.
        self._extend_storage(n_years)
        #
        for _ in range(n_years):
            # Perform `steps_per_year` numerical integration steps.
            for _ in range(self.steps_per_year):
                # Multiply derivative operator `W` by the calculated
                # frequencies `s` to obtain derivatives of frequencies.
                # Scale the derivatives by the step size, and add the
                # result to `s`.
                self.s += self.step_size * (self.W @ self.s)
                #
                # Zero subthreshold elements of `s`. Derivatives of
                # zeroed frequencies are set to zero unless the `_zero`
                # method is overridden.
                self._zero()
            # Bias exponents of the current solution to avoid overflow
            # and underflow. Keep track of the cumulative bias.
            self.s_bias += bias_exponents(self.s, self.max_exponent)
            #
            # Store the solution for end-of-year relative frequencies.
            self.solutions[self.n_solutions] = self.s / fsum(self.s)
            self.n_solutions += 1

    def get_last_solution(self):
        """
        Returns unnormalized solution for frequencies in the last year.
        
        The elements of the returned array are multiprecision floats.
        """
        return mp.mpf(2.0)**-self.s_bias * self.s

    def __getitem__(self, key):
        # Returns the result of indexing end-of-year solutions by `key`.
        return self.solutions[key]

    def __len__(self):
        # Returns the number of stored solutions (one per year).
        return len(self.solutions)

    def _extend_storage(self, n):
        # Allocate storage for solutions for an additional `n` years.
        rows, cols = self.solutions.shape
        new = np.zeros((rows+n, cols), dtype=float)
        new[:rows] = self.solutions
        self.solutions = new

        
class PoorSolver(Solver):
    """
    A poor solver for frequencies in the modified model.
    """
    def _zero(self):
        """
        Zeroes calculated frequencies that are below threshold.
        
        Derivatives of zeroed frequencies are NOT set to zero.
        """
        if self.threshold > 0.0:
            self._zero_subthreshold_frequencies()