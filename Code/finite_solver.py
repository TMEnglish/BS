class FiniteSolver(object):
    """
    A solver for relative frequencies in the finite-population model.
    
    A modification of the Euler forward method of numerical integration
    is applied. Solutions for relative frequencies are set to zero when
    they fall below a given threshold, and subsequently are held at zero.
    Only end-of-year solutions are stored, beginning with the solution
    for year 0 (derived from the given initial frequencies). Note that 
    solutions for the infinite-population model can be obtained by setting
    the threshold to zero.
    
    Assumption: Frequencies fall below threshold only at the tail ends
    of the frequency distribution.
    
    The solver is run by calling this object. Each call extends the 
    solutions by a given number of years. The end-of-year solutions are
    retrieved by indexing this object.
    """ 
    def __init__(self, derivative, log_steps_per_year, initial_freqs,
                       threshold=1e-9):
        """
        Initialize the solver.
        
        The `derivative` operator is an instance of class Derivative. The
        parameter `log_steps_per_year` must be an integer. When the solver
        is run, there are `2 ** log_steps_per_year` integration steps per
        year. The step size is the reciprocal of the number of steps per
        year.
        
        The solution for year 0 is `initial_freqs` with frequencies less
        than `threshold` times the sum of `initial_freqs` set to zero. The
        given `threshold` is used similarly in each integration step.
        """
        self.derivative = derivative
        assert type(log_steps_per_year) is int
        self.steps_per_year = 2 ** log_steps_per_year
        self.step_size = 1 / self.steps_per_year
        self.threshold = threshold
        #
        # Array `s` always contains the current solution for frequencies
        # of classes. The base type of `s` is that of `initial_value`.
        self.s = np.array(initial_freqs)
        #
        # Zero subthreshold elements at the left and right ends of the
        # solution array.
        left, right = trim(self.s, self.threshold * self.s.sum())
        #
        # Slice `included` indicates which elements of the solution to
        # include in integration steps. Here we include only nonzero
        # elements of the solution. A subclass initializer may change
        # the setting of `included`.
        self.included = slice(left, len(self.s) - right)
        #
        # Array `solutions` contains the solutions for end-of-year
        # relative frequencies of classes. The base type is float. There
        # is one row for each year.
        self.n_solutions = 1
        self.solutions = np.empty((self.n_solutions, len(self.s)))
        self.solutions[0] = self.s / fsum(self.s)
        
    def __call__(self, n_years=1000):
        """
        Solve for `n_years` end-of-year relative frequencies.
        """
        # Extend the `solutions` array to hold an additional `n_years`
        # solutions for end-of-year relative frequencies.
        self._extend_storage(n_years)
        #
        for _ in range(n_years):
            # Scale the current solution in order to avoid overflow and
            # underflow in calculations.
            bias_exponents(self.s, max_exponent=512)
            #
            # Perform `steps_per_year` numerical integration steps.
            for _ in range(self.steps_per_year):
                self.included = self._step(self.included)
            #
            # Store the solution for end-of-year relative frequencies.
            self.solutions[self.n_solutions] = self.s / fsum(self.s)
            self.n_solutions += 1

    def _step(self, included):
        """
        Performs an integration step, returns an update of `included`.
        
        Slice `included` indicates which elements of the solution to
        update in the integration step. The `included` elements that fall
        below threshold after the integration step are set to zero. The
        returned slice is a narrowing of `included` to exclude all newly
        zeroed elements of the solution.
        """
        # Restrict the integration step to `included` solution elements.
        s = self.s[included]
        s += self.step_size * self.derivative(None, s, included)
        #
        # Zero solution elements that fall below threshold.
        l, r = trim(s, self.threshold * s.sum())
        #
        # Narrow the `included` slice to exclude newly zeroed elements.
        return slice(included.start + l, included.stop - r)

    def __getitem__(self, which):
        # Returns the result of indexing solutions by `which`.
        return self.solutions[which]

    def __len__(self):
        # Returns the number of stored solutions (one per year).
        return len(self.solutions)

    def _extend_storage(self, n):
        # Allocate storage for solutions for an additional `n` years.
        rows, cols = self.solutions.shape
        new = np.zeros((rows+n, cols), dtype=float)
        new[:rows] = self.solutions
        self.solutions = new

        
class BotchedSolver(FiniteSolver):
    """
    An erroneous solver for frequencies in the finite-population model.
    
    Zeroed elements of the solution are updated in integration steps,
    rather than held at zero.
    """
    def __init__(self, derivative, log_steps_per_year, initial_freqs,
                       threshold=1e-9, margin=2):
        """
        Initialize the solver, mostly as is done for the parent class.
        
        The additional `margin` parameter determines how many zeroed
        elements of the solution are included in the integration step.
        In general, `2 * margin` zeroed elements are included, half of
        them to the immediate left, and half of them to the immediate
        right, of the nonzero elements. However, all elements of the
        solution are included in the first integration step.
        """
        super().__init__(derivative, log_steps_per_year, initial_freqs,
                         threshold)
        #
        # All elements are included in the first integration step.
        self.included = slice(0, len(self.s))
        self.margin = margin

    def _step(self, included):
        """
        Performs an integration step, returns an updated of `included`.
        
        Slice `included` indicates which elements of the solution to
        update in the integration step. The `included` elements that fall
        below threshold after the integration step are set to zero. The
        returned slice includes all elements that remain positive, and
        also includes zeroed elements that stand a chance of rising to
        threshold in the next integration step. 
        
        An exception is raised if the `margin` parameter of the solver,
        which indicates how many zeroes to the left and right of the
        nonzero solution elements to include, is set too small.
        """
        s = self.s[included]
        #
        # Use the `_step` method of the parent class to perform an
        # integration step and zero subthreshold solution elements.
        new_included = super()._step(included)
        #
        # The leftmost and rightmost of the `included` elements of the 
        # solution should remain zero after the integration step, except
        # in special cases.
        lower_error = s[0] != 0 and included.start != 0
        upper_error = s[-1] != 0 and included.stop != len(self.s)
        if lower_error or upper_error:
            raise Exception('`margin` is set too small')
        
        # The next integration step includes all elements of the solution
        # that stand a chance of rising to threshold.
        start = max(0, new_included.start - self.margin)
        stop = min(len(self.s), new_included.stop + self.margin)
        return slice(start, stop)