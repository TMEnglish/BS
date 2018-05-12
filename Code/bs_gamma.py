def bs_gamma(factors, alpha=0.5, beta=0.5/0.001, percent_beneficial=0.001, gimmick=True):
    effects = factors.effects
    p = stats.gamma(alpha, scale=1/beta).pdf(np.abs(effects))
    if gimmick:
        p[effects == 0] = p[effects < 0][-1]
        p[effects > 0] *= percent_beneficial
        p[effects <= 0] *= 1 - percent_beneficial
        p *= factors.delta
    else:
        p[effects != 0] *= factors.delta
        p[effects == 0] = 2 - fsum(p[effects != 0])
        p[effects > 0] *= percent_beneficial / fsum(p[effects > 0])
        p[effects <= 0] *= (1 - percent_beneficial) / fsum(p[effects <= 0])
    return p