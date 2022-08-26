class Sanford(object):
    """
    The discrete form of Sanford's DFE, represented as an array.
    
    The domain of the distribution is restricted to the set
    
        D_K = {(1 - K)w, ..., -w, 0, w, ..., (K - 1)w}.
    
    For instance `d`, the probability of mutational effect k w in D_K is
    `d[k + d.K - 1]`. Probabilities are of type Fraction.
    """
    def __init__(self, K, w='5e-4', gamma='1e-3', beta='500',
                 normed=True):
        """
        Calculates probabilities for the discrete form of Sanford's DFE.

        Parameters
        * `K`     : probabilities are calculated at 2K - 1 equispaced
                    points, the K-th of which is zero
        * `w`     : exact spacing of points (bin width)
        * `gamma` : exact mixture weight in Sanford's DFE
        * `beta`  : exact rate parameter in Sanford's DFE
        * `normed`: determines whether the probabilities are normalized

        Parameters designated exact must be specified by strings or by
        numbers of type Fraction.

        Probabilities are assigned to the 2K - 1 mutational effects 

            (1 - K)w, ..., -w, 0, w, ..., (K - 1)w.

        The unnormalized probability of mutational effect x is

            G_gamma(x + w/2) - G_gamma(x - w/2),

        where G_gamma is the cumulative distribution function of
        Sanford's DFE (shape alpha=1/2, rate `beta`, mixture weight
        `gamma`).

        The probabilities are of type `Fraction`, but are not exact: some
        intermediate calculations are done in multiprecision floating-
        point arithmetic, with the number of digits of working precision
        specified by `mp.mps`.
        """
        # Operate on numbers of type Fraction, producing exact results,
        # whenever practical. Otherwise, operate on multiprecision floats,
        # and convert the results to type Fraction as soon as possible.
        #
        # Verify that "exact" arguments are specified correctly, and then
        # convert them exactly to type Fraction.
        exact_args = [gamma, w, beta]
        assert all(type(x) in [str, Fraction] for x in exact_args), \
               'Inexact argument supplied to Sanford'
        self.gamma, self.w, self.beta = to_fraction(exact_args)
        self.K = K
        #
        # Calculate exactly the K endpoints of the length-w intervals
        # centered on the positive effects 1w, 2w, ..., (K - 1)w: create
        # an array of Fractions [0/1, 1/1, (K-1)/1], scale all elements
        # of the array by Fraction w, and finally add the Fraction w/2
        # to all elements. All of the operations are exact.
        endpoints = self.w * to_fraction(range(self.K)) + self.w / 2
        #
        # Evaluate the Gamma complementary CDF at the endpoints. Square
        # root and erfc calculations are done in multiprecision floating-
        # point arithmetic with `mp.dps` digits of working precision.
        # Results are converted immediately to type Fraction, preventing
        # accumulation of further imprecision in subsequent operations.
        z = self.beta * endpoints
        gamma_ccdf = to_fraction(mp_erfc(mp_sqrt(to_mpf(z))))
        self.dps = mp.dps
        #
        # Differencing values of the complementary CDF (type Fraction) at
        # the endpoints of intervals gives the masses distributed over
        # the intervals. The differences (type Fraction) are exact,
        # though the differenced values are inexact. The slices [:-1] and
        # [1:] of array `gamma_ccdf` refer, respectively, to all elements
        # but the last and all elements but the first.
        unweighted_tail = gamma_ccdf[:-1] - gamma_ccdf[1:]
        zero_mass = 1 - gamma_ccdf[0]
        #
        # Weight the (reversed) tail masses to obtain masses for the
        # upper (lower) tail of the distribution over mutational effects.
        # All operands are Fractions, and thus the operations are exact.
        upper_tail = self.gamma * unweighted_tail
        lower_tail = (1 - self.gamma) * unweighted_tail[::-1]
        #
        # Assemble the probability masses into a single array.
        self.dfe = np.concatenate((lower_tail, [zero_mass], upper_tail))
        #
        # Issue a warning if the sum of unnormalized probabilities is not
        # exactly equal to the calculated value of G(Kw - w/2).
        self.norm = sum(self.dfe)
        if self.norm != 1 - gamma_ccdf[-1]:
            warnings.warn('Sanford: some loss of significance')
        if normed:
            self.dfe /= self.norm
            
    def __getitem__(self, key):
        # Makes the instance indexable.
        return self.dfe[key]
    
    def __len__(self):
        # The length of the instance is the size of the domain.
        return len(self.dfe)