def bs_annual_update(factors, frequencies, mutations,
                     threshold_norm=None, updates_per_year=1):
    birth_factors = factors.birth / updates_per_year
    death_factor = factors.death / updates_per_year
    f = np.array(frequencies)
    births = np.empty_like(f)
    for _ in range(updates_per_year):
        births[:] = np.convolve(birth_factors * f, mutations, mode='valid')
        f *= (1 - death_factor)
        f += births
    if not threshold_norm is None:
        f[f <= 1e-9 * threshold_norm(f)] = 0
    return f