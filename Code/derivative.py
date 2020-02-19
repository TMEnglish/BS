class Derivative(object):
    """
    Wrap the W matrix of Basener and Sanford (Section 4).
    
    An object d in this class is called d(t, s), where s is the state
    of the system at time t, to obtain the derivative at time t.
    However, the time t is ignored. The derivative is the product of
    the n-by-n matrix W and the n-by-1 vector s.
    """
    @staticmethod
    def mutation_matrix(p):
        """
        Returns 2-D array of mutation rates, denoted f by B&S.

        The 1-D array `p` is of length 2n + 1. The probability is
        `p[i-j+n]` that the effect of mutation on fitness is i - j 
        units.

        The returned matrix f is n-by-n. For off-diagonal elements,
        f[i, j] is equal to `p[i-j+n]`. The value of f[j, j] is set to
        make the sum of elements of the j-th column equal to 1.
        """
        n = len(p) // 2 + 1
        f = np.empty((n, n), dtype=type(p[0]))
        for j in range(n):
            col = f[:,n-j-1]
            col[:] = p[j:n+j] 
            col[n-j-1] += 1 - fsum(col)
        return f

    def __init__(self, mutation_probs, factors, basetype=float):
        """
        Prepare the W matrix of Basener and Sanford Section 4.
        
        The initial type of the elements of W is determined by the
        types of the arguments `mutation_probs` and `factors`. When
        the calculation of W is complete, the elements are converted
        to the given `basetype`.
        """
        n = len(factors.birth)
        self.factors = factors
        self.basetype = basetype
        #
        # Columns in the mutation matrix are scaled by corresponding
        # birth factors. Then the death factor is subtracted from
        # elements on the main diagonal.
        self.W = Derivative.mutation_matrix(mutation_probs)
        self.W *= factors.birth
        self.W[np.diag_indices(n)] -= factors.death
        if not type(self.W[0,0]) is basetype:
            self.W = self.W.astype(basetype)
    
    def __call__(self, ignored_time, state, support=None):
        """
        Returns vector of derivatives at `state`.
        
        If `support` is provided, then the calculation of derivatives
        is restricted to `state[support]`. For excluded components, the
        derivatives are set to zero.
        
        The elements of `state` are converted to match the base type of
        this operator.
        """
        if not type(state) is self.basetype:
            state = state.astype(self.basetype)
        if support is None:
            # Multiply n-by-n W by n-by-1 state.
            return np.dot(self.W, state)
        #
        # Multiply restricted W matrix by restricted state vector.
        d = np.zeros_like(state)
        np.dot(self.W[support, support], state[support], out=d[support])
        return d
        
    def equilibrium(self, v0=None, maxiter=None, npower=1000):
        """
        Returns real part of the dominant eigenvector of W, normalized.
                
        The initial vector `v0` and and the maximum number of iterations 
        `maxiter` are passed under the same names to the `eigs` function
        of cipy.sparse.linalg. This calculation is done in type float.
        The solution returned by `eigs` is improved by subsequent
        application of the power method for `npower` iterations. The
        calculation is done in the basetype of this object.
        """
        # The `eigs` function does not work when supplied multiprecision
        # floats. We get a quick solution with ordinary floating point.
        if v0 is None:
            walls = self.factors.fitness_walls
            mean = fsum(self.factors.fitness) / len(self.factors.fitness)
            var = 2 * self.factors.bin_width
            v0 = binned_normal(walls, mean, var).astype(float)
        else:
            v0 = v0.astype(float)
        if self.basetype is mp.mpf:
            W = self.W.astype(float)
        else:
            W = self.W
        try:
            _, e_vectors = eigs(W, 1, which='LR', v0=v0, maxiter=maxiter)
            v = e_vectors[:, 0].real
            v0 = convert(np.abs(v), self.basetype)
        except:
            v0 = v0.astype(self.basetype)
        #
        # Now use the power method to improve the solution. Note that
        # self.W may contain multiprecision floats.
        v = np.empty_like(v0)
        w = np.empty_like(v0)
        v[:] = v0
        v0 /= fsum(v0)
        for _ in range(100):
            for _ in range(2000):
                _, max_exponent = frexp(v.max())
                v *= 2.0 ** (512 - max_exponent)
                v += np.dot(self.W, v)
            w = v / fsum(v)
            error = maximum_absolute_relative_error(w, v0)
            if error < 1e-16:
                return w
            v0[:] = w
        warnings.warn('equilibrium relative error {}'.format(error))
        return w