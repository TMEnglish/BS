class ModifiedEuler(object):
    """
    Modification of Euler forward method for numerical integration.
    
    EXPAND:
    1. Euler forward method is applied when `threshold` is zero.
    2. Callable and indexable.
    3. Customized to application. It is assumed that subthreshold
    frequencies are at the tails of the frequency distribution.
    """
    def __init__(self, derivative, initial_freqs, threshold=0):
        """
        NEED
        """
        self.derivative = derivative
        self.threshold = threshold
        #
        # Array `s` always contains the current solution. The base type
        # of `s` is the base type of `initial_value`. Subthreshold
        # frequencies are zeroed.
        self.s = initial_freqs + 0
        self.s[self.s < threshold * np.sum(self.s)] = 0
        #
        # Slice `support` indicates which classes have positive
        # frequencies, i.e., which classes to include in the calculation
        # of derivatives.
        self.support = slice_to_support(self.s)
        #
        # Array `solutions` contains the end-of-year solutions. The base
        # type is float.
        self.solutions = np.empty((1, len(self.s)), dtype=float)
        self.solutions[0] = self.s
        self.n_solutions = 1
        
    def __call__(self, n_years=1000, steps_per_year=2**7):
        """
        NEED
        
        There are `steps_per_year` numerical integration steps per year.
        At the end of each year, the current solution is stored.
        """
        self._extend_storage(n_years)
        step_size = 1 / steps_per_year
        for _ in range(n_years):
            for _ in range(steps_per_year):
                self.integration_step(step_size)
            self.scale_current_solution()
            self.solutions[self.n_solutions] = self.s
            self.n_solutions += 1

    def integration_step(self, step_size):
        """
        Performs an integration step; zeroes subthreshold frequencies.
        """
        # The derivative calculation is restricted to classes in the
        # support of the frequency distribution. 
        self.s += step_size * self.derivative(0, self.s, self.support)
        #
        # Zero frequencies that have fallen below threshold.
        included = self.s[self.support]
        included[included < self.threshold * fsum(included)] = 0
        #
        # Update the support if any frequencies of classes in the
        # support have been zeroed.
        if included[-1] == 0 or included[0] == 0:
            self.support = slice_to_support(self.s)
    
    def scale_current_solution(self):
        """
        Scales the current solution to avoid overflow and underflow.
        """
        # Make the maximum frequency of the solution have an exponent
        # of 768. Scaling a floating point number by an integer power
        # of 2 does not change its mantissa. Thus there is no loss in
        # precision.
        _, max_exponent = frexp(self.s.max())
        self.s *= 2.0 ** (768 - max_exponent)
        
    def __getitem__(self, which):
        """
        Returns the result of indexing solutions by `which`.
        
        The frequency vector for each year is normalized.
        """
        s = self.solutions[which]
        if s.ndim == 1:
            return s / fsum(s)
        return s / s.sum(axis=1)[:,None]

    def __len__(self):
        """
        Returns the number of stored solutions (one per year).
        """
        return len(self.solutions)

    def _extend_storage(self, n):
        """
        Allocate storage for solutions for an additional `n` years.
        """
        rows, cols = self.solutions.shape
        new = np.zeros((rows+n, cols), dtype=float)
        new[:rows] = self.solutions
        self.solutions = new


class PoorlyModifiedEuler(ModifiedEuler):
    def integration_step(self, step_size):
        """
        Incorrectly performs an integration step.
        
        Derivatives should not be calculated for frequencies that are
        below threshold, but they are. All frequencies are updated, and
        only those that are below threshold after the update are zeroed.
        That is, previously zeroed frequencies are not held at zero.
        """
        self.s += step_size * self.derivative(0, self.s)
        self.s[self.s < self.threshold * fsum(self.s)] = 0