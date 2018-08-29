class WeightedDoubleGamma(EffectsDistribution, MultiprecisionGamma):
    """
    Double gamma distribution over mutation effects, weighted and discretized.
    
    The distribution is essentially the combination of a Gamma distribution 
    (over positive effects), weighted by p, with its reflection (over negative
    effects), weighted by 1 - p, with 0 <= p <= 1.
    
    The discrete mutation effects, including zero, are equispaced. For spacing
    delta, the probability of effect x is
    
               |      p  * [F(x + delta / 2)  - F(x - delta / 2)],  x > 0
               |
        P(x) = | (1 - p) * [F(-x + delta / 2) - F(-x - delta / 2)], x < 0
               |
               |            F(delta / 2),                           x = 0
        
    where F is the cumulative distribution function of the Gamma distribution.
    """
    def __init__(self, factors, alpha=0.5, beta=500, weight=mp_float('1e-3'),
                       density=False):
        """
        Discretizes a weighted double-gamma distribution over mutation effects.
        
        The sequence of effects, `factors.effects`, is assumed to be symmetric,
        with zero at the center, and with points spaced by `factors.delta`. The
        parameter `weight` gives the weighting of the Gamma(`alpha`, `beta`) 
        distribution over positive effects. The weighting of the reflection of
        the Gamma distribution (over negative effects) is 1 - `weight`.
        
        The probability of an effect is the integral of the weighted double-
        gamma density function over the length-`factors.delta` subinterval
        centered on it.  However, if `density` is true, then the probability of
        each nonzero effect is approximated as as the product of `factors.delta`
        and the weighted double-gamma density at the effect. The probability of
        zero effect is what ever probability mass is not allocated to nonzero
        effects, i.e, unity minus the sum of probabilities of nonzero effects.
        
        Calculations are performed with `mp.dps` digits of precision, where `mp`
        is the global context for multiprecision math operations. The resulting
        probabilities are converted to the base type of `factors.effects`. 
        """
        # Initialize default distribution over discrete effects.
        EffectsDistribution.__init__(self, factors)
        # Instantiate continuous distribution with `pdf` and `mass` methods.
        MultiprecisionGamma.__init__(self, alpha, beta)
        
        self.weight = weight
        effects = factors.effects  # abbreviate

        # Calculate probability masses for positive and zero effects.
        # The subinterval centered on 0 has average weight of 1/2. Thus the
        # mass of the Gamma distribution over the upper half is the mass of
        # the weighted double-gamma distribution over the entire subinterval.
        if density:
            # Calculate delta-density products for positive effects.
            masses = self.delta * self.pdf(effects[effects > 0])
            # Assign all unallocated probability mass to zero effect.
            self.p[effects == 0] = 1 - accurate_sum(masses)
        else:
            # Calculate endpoints of subintervals centered on positive effects.
            endpoints = effects[effects >= 0] + mp_float(self.delta) / 2
            # Obtain masses of the Gamma distribution over positive subintervals.
            masses = self.mass(endpoints)
            # Assign mass of Gamma distribution over (0, delta/2] to zero effect.
            self.p[effects == 0] = self.mass([0, endpoints[0]])

        # Reflect the probability masses. Weight the masses of nonzero effects.
        self.p[effects > 0] = weight * masses
        self.p[effects < 0] = (1 - weight) * masses[::-1]

    def gimmick(self):
        """
        Sets probability of effect zero to the density-delta product at -delta.
        
        Here `delta` refers to the spacing of discrete mutation effects. The
        density of the weighted double-gamma distribution at `-delta` is
        multiplied by `delta`, and is assigned to effect zero.
        """
        self.p[self.effect == 0] = (1 - self.weight) * self.pdf(delta)
        self.p[self.effect == 0] *= self.delta