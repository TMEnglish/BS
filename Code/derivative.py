class Derivative(object):
    """
    Wraps the matrix operator `W` of Basener and Sanford (Section 4).
    
    An object `d` in this class is called `d(t, s)`, where `s` is the
    solution for frequencies at time `t`, to calculate the derivative
    of the frequencies.  The derivative is the product of the n-by-n
    matrix `W` and the n-by-1 vector `s`. The time `t` is ignored, 
    but the signature `d(t, s)` is required by standard ODE solvers.
    
    Alternatively, a call make take the form `d(t, s, include)`. Then
    the result is the product of submatrix `W[include, include]` and
    vector `s`.
    """
    def __init__(self, params, basetype=float):
        """
        Prepare the matrix operator `W`.
        
        The `params` argument is an instance of class `Parameters`. It
        provides the birth rates, death rate, and mutation matrix
        required to calculate `W`. When the calculation of `W` is
        complete, its elements are converted to the given `basetype`.
        """
        # Scale each column of the mutation matrix by the corresponding
        # birth rate. Subtract the death rate from elements on the main
        # diagonal of the result.
        self.W = params.f * params.b
        self.W[np.diag_indices(params.N)] -= params.d
        #
        # Ensure that the elements of `W` are of the given `basetype`.
        if not type(self.W[0,0]) is basetype:
            self.W = convert(self.W, basetype)
        #
        # Internally, the square submatrix matrix `Wp`, which may
        # exclude some rows and columns of `W`, is used in calculating
        # derivatives. Initially, all rows and columns are included. 
        self.Wp = self.W
        self.included = slice(0, n)
    
    def __call__(self, ignored_time, s, include=None):
        """
        Returns the vector of derivatives at `s`.
        
        If `include` is `None`, then the result is the product of the
        wrapped matrix `W` and given vector `s`. Otherwise, the result
        is the product of `W[include, include]` and `s`.
        """
        if include is None:
            self.Wp = W
        elif include != self.included:
            self.Wp = np.array(self.W[include, include])
            self.included = include
        return np.matmul(self.Wp, s)
    
    def __getitem__(self, which):
        """
        Returns `W[which]`, where `W` is the wrapped matrix.
        """
        return self.W[which]