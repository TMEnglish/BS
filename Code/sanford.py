class Sanford(object):
    """
    Array representation of the discrete form of Sanford's DFE.
    
    Probability masses are associated with points in
    
        D_K = {(1 - K)w, ..., -w, 0, w, ..., (K - 1)w}.
    
    For an instance `d` of this class, the value of `d[k + d.K - 1]` is
    the probability of effect k w in D_K. Probabilities are of type
    `Fraction`, though they are calculated inexactly.
    """
    def __init__(self, K, w='5e-4', gamma='1e-3', beta='500', normed=True):
        """
        Calculates probabilities for the discrete form of Sanford's DFE.

        * `K`     : probability masses are calculated at 2K - 1 evenly
                    spaced points, the K-th of which is zero
        * `w`     : spacing of points (exact)
        * `gamma` : weighting of positive effects (exact)
        * `beta`  : rate parameter (exact)
        * `normed`: determines whether the masses are normalized
        
        "Exact" supplied values are of type `str`, `int`, or `Fraction`.
        """
        # Validate "exact" parameters and convert them to `Fraction`.
        self.gamma, self.w, self.beta = all_exact([gamma, w, beta])
        self.K = K
        self.dfe = reflection_mixture(self.gamma_ccdf, gamma, K, w)
        if normed:
            self.dfe /= sum(self.dfe)
            
    def gamma_ccdf(self, x):
        """
        Return values of the Gamma complementary CDF at points in `x`.
        
        Values are calculated as multiprecision floats, and returned as
        `Fraction` objects.
        """
        # Shape alpha = 0.5, and rate beta is set on initialization.
        z = to_mpf(self.beta * x)
        return to_fraction(mp_erfc(mp_sqrt(z)))
            
    def __getitem__(self, key):
        # Index the array representing the DFE.
        return self.dfe[key]
    
    def __len__(self):
        # Return the length of the array representing the DFE.
        return len(self.dfe)