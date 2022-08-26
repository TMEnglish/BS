def zero_loci_prob(q):
    L = q.L
    U = float(q.U)
    mu = U / L
    p = (1 - mu)**L
    p += L * mu * (1 - mu)**(L-1) * q.dfe[q.k]
    p += L*(L-1)/2 * mu**2 * (1-mu)**(L-2) *q.dfe[q.k]**2
    return p, float(q[q.k])


class AltSanford(Sanford):
    def gamma_ccdf(self, x):
        """
        Return values of the Gamma complementary CDF at points in `x`.
        
        Values are calculated as multiprecision floats, and returned as
        `Fraction` objects.
        """
        # Shape alpha = 0.5, and rate beta is set on initialization.
        z = to_mpf(self.beta * x)
        return to_fraction(mp_erfc(mp_sqrt(z)))


def check_dfe(dfeclass, K, gamma='1e-3', w='4e-5'):
    #
    dfe = dfeclass(K, gamma=gamma, w=w, normed=False)
    max_effect = (dfe.K - 1) * dfe.w
    weight_ratio = (1 - dfe.gamma) / dfe.gamma
    ratio_holds = all(dfe[:K-1][::-1] / dfe[-(K-1):] == weight_ratio)
    tail = np.array([max_effect + dfe.w/Fraction(2)])
    excluded_mass = dfe.gamma_ccdf(tail)[0]
    #
    print('(K, w, gamma):', (len(dfe)//2 + 1, str(dfe.gamma), str(dfe.w)))
    print('weight ratio :', weight_ratio)
    print('ratio holds  :', ratio_holds)
    print('max effect   :', max_effect)
    print('mass at max  :', float(dfe[K-1+K-1]))
    print('mass at zero :', float(dfe[K-1]))
    print('mass at -w   :', float(dfe[K-2]))
    print('excluded mass:', float(excluded_mass))
    print('== 1-sum(dfe):', excluded_mass == (1 - sum(dfe[:])))

class AltSolver(Solver):
    def get_last_solution(self):
        """
        Return the most recent solution, without normalization.
        
        The returned array is obtained by converting the internally
        stored state of the population, i.e, the number of individuals
        of each type (not the relative frequencies of the types) at the
        end of the most recent run, to an array of multiprecision
        floating-point numbers.
        """
        return mp.mpf(2.0)**-self.s_bias * self.s

