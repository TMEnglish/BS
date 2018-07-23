class Gamma(EffectsDistribution):
    """
    A discretized Gamma distribution and its reflection in the zero-effect axis.
    """
    def __init__(self, factors, alpha=0.5, beta=0.5/0.001, density=True,
                       normed=False):
        """
        Defines a symmetric Gamma distribution with the given parameters.
        
        The `factors` object is an instance of class `Factors`. The Boolean
        parameters `density` and `normed` determine, respectively, whether the
        density function or the cumulative distribution function is used in
        assignment of probability masses to effects, and whether the resulting
        distribution is normalized.
        """
        rv = stats.gamma(alpha, scale=1/beta)
        super().__init__(factors, rv, density=density, normed=normed)
    
    def zero_neutral(self):
        """
        Sets to zero the probability that mutation has no effect on fitness.
        """
        self.p[self.effect == 0] = 0
        
    def gimmick(self):
        """
        Assigns the probability of minimally deleterious effect to zero effect.
        """
        self.p[self.effect == 0] = self.p[self.effect < 0][-1]

    def reweight(self, beneficial=0.001, botched=False):
        """
        Sets the probability of positive effect to `beneficial` unless botched.
        
        If `botched` is false, then the probabilities of positive mutation
        effects are scaled so that they sum to `beneficial`, and the
        probabilities of non-positive effects are scaled so that they sum to
        1 - `beneficial`.
        
        If `botched` is true, then the probabilities of positive and non-
        positive effects are simply multiplied by `beneficial` and 1 -
        `beneficial`, respectively.
        """
        effect = self.effect
        if not botched:
            self.p[effect > 0] /= float(mp.fsum(self.p[effect > 0]))
            self.p[effect <= 0] /= float(mp.fsum(self.p[effect <= 0]))
        self.p[effect > 0] *= beneficial
        self.p[effect <= 0] *= 1 - beneficial