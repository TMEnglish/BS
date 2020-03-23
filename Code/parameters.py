##### THIS IS OUT OF DATE!!!!!
class Parameters(object):
    """
    Stores all parameter settings of the infinite-population model.    
    
    Members `b`, `d`, `m`, `w`, `gamma`, `N`, `p`, and `f` are named
    as in our article. In addition, `initial_freqs` is the vector of
    initial frequencies P(0).
    
    All numbers other than `N` are multiprecision floats, with
    precision determined by the `mpmath` environment in which the
    class is instantiated.
    """
    def __init__(self, b_max='0.1', d='0.1', w='5e-4', gamma='1e-3'):
        """
        Sets all parameters of the infinite-population model.
        
        Parameters are derived from the maximum birth rate `b_max`,
        the death rate `d`, the bin width `w`, and the weighting
        `gamma` of beneficial mutational effects. The maximum birth
        rate must be an integer multiple of the bin width.
        """
        # Use `mp_float()` for explicit conversion to multiprecision
        # float. In mixed-mode operations, (arrays of) integers and
        # floats are promoted to (arrays of) multiprecision floats.
        b_max = mp_float(b_max)
        self.d = mp_float(d)
        self.w = mp_float(w)
        self.gamma = mp_float(gamma)
        #
        # Scale the array [0, 1, ..., N-1] by the bin width to get the
        # array of birth rates. Subtract the death rate from the array
        # of birth rates to get the array of fitnesses.
        self.N = int(mp.nint(b_max / self.w)) + 1
        self.b = self.w * np.arange(self.N)
        self.d = mp_float(d)
        self.m = self.b - self.d
        #
        # Scale the array [-1/2, 1/2, ..., N-1/2] by the bin width to
        # get the N+1 walls of bins centered on the N birth rates.
        # Subtract the death rate from the bin walls for the birth
        # rates to get the bin walls for the fitnesses.
        b_walls = self.w * np.arange(-1/2, self.N, 1)
        m_walls = b_walls - self.d
        assert len(b_walls) == self.N + 1
        #
        # Set the initial frequencies as in Section 5 of B&S, binning
        # a normal distribution of probability over fitness.
        self.initial_freqs = binned_normal(m_walls, mean='0.044', 
                                                    std='0.005')
        assert fsum(self.initial_freqs) == 1
        #
        # The probability distribution `p` over mutational effects is
        # a mixture of a binned Gamma distribution over beneficial
        # effects and its reflection, with weighting `gamma` of the
        # distribution over beneficial effects. The beneficial effects
        # are identical to the positive birth rates.
        self.p = binned_mixture(bs_gamma_ccdf, b_walls[1:], self.gamma)
        assert fsum(self.p) == 1
        #
        # Set each off-diagonal element `f[i,j]` of matrix `f` to
        # `p[i-j+n-1]`. Set each element `f[j,j]` of the main diagonal
        # to make the sum of elements in column `j` equal to 1.
        self.f = np.empty((self.N, self.N), dtype=type(p[0]))
        for i in range(self.N):
            # Traverse columns of `f` from left to right.
            j = (self.N - 1) - i
            # Set column `j` of `f` to `p[i], p[i+1], ..., p[i+N-1]`.
            self.f[:,j] = self.p[i:i+self.N]
            self.f[j,j] += 1 - fsum(self.f[:,j])        