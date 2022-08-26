def basener_init(n, m_lim=(-0.1, 0.15), mean=0.044, std=0.005):
    """
    Return initial frequencies and fitnesses calculated as by Basener.
    
    Parameters
    `n`    : number of types, set 1 greater than by Basener
    `m_lim`: (lower, upper) limits on fitness. Basener excludes the
             upper limit, but it is included here for convenience
    `mean` : mean of the normal distribution discretized to obtain
             initial frequencies of types
    `std`  : standard deviation of the normal distribution
    """
    w = (m_lim[1] - m_lim[0]) / (n - 1)
    m = w * np.arange(float(n)) + m_lim[0]
    p = np.exp(-0.5 * ((m - mean) / std)**2)
    p[abs(m - mean) > 11.2 * std] = 0
    p /= math.fsum(p)
    return p, m