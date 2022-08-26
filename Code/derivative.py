class Derivative(object):
    """
    Supplies the matrix operator W of the mutation-selection model.
    
    When instance `W` is initialized, the death-rate parameter d is
    set to zero. The matrix (2-D array of floating-point numbers) is
    obtained by calling the instance, e.g., `W()` or `W(d=0.1)`. The
    latter example shows how to change the value of d (not recommended).
    """
    def __init__(self, q, n):
        """
        Calculates derivative operator W, assuming death rates of zero.
                
        Parameters
        `q` : distribution of probability over mutational effects
        `n` : number of types (less than `q.K`)
        """
        # Truncate q to the 2n - 1 elements centered on q[q.K]. Then the
        # columns of F are, from left to right, the length-n spans of q,
        # from right to left. The j-th column of M is initially set to
        # the j-th column of F, scaled by b[j]. Then M[j,j] is adjusted 
        # to make the column sum equal to b[j].
        b = equispaced(n, spacing=q.w, start='0').astype(float)
        self.q = q
        q = q[q.K-(n-1):q.K+n]
        q = (q / sum(q)).astype(float)
        columns = [b * q[n-1-j:2*n-1-j] for j, b in enumerate(b)]
        colsums = [math.fsum(col) for col in columns]
        self.M = np.transpose(columns)
        self.M[np.diag_indices(n)] += b - colsums
        self.n = n
        self.b = b
        
    def __call__(self, d=0.0):
        """
        Returns the derivative operator as a square array of floats.
        """
        # Subtract death-rate parameter(s) from the main diagonal.
        W = self.M.astype(float)
        W[np.diag_indices(self.n)] -= d
        return W
    
    def __getattr__(self, name):
        # Some parameters are attributes of distribution `self.q`. Create
        # the illusion that they are attributes of `self`.
        return getattr(self.q, name)