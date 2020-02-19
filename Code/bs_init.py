def bs_init(fitnesses, mean=0.044, std=0.005, z_limit=11.2, norm=1):
    """
    Returns a discretized Gaussian distribution over growth factors.

    Growth factors differing from the given `mean` by more than `z_limit`
    standard deviations are assigned probability zero. Each of the other
    growth factors m has probability proportional to f(m), where f is the
    probability density function of the distribution N(mean, std). The
    resulting distribution is scale to have the given `norm`.
    """
    z = (fitnesses - mean) / std
    frequencies = np.where(np.abs(z) > z_limit, 0, np.exp(-0.5 * z**2))
    return norm * frequencies / fsum(frequencies)