#from scipy.fftpack import rfft, irfft
from scipy.signal import fftconvolve

def fast_annual_update(growth_factors, frequencies, mutations,
                       updates_per_year=1):
    birth_factors = (growth_factors + DEATH_FACTOR) / updates_per_year
    death_factor = DEATH_FACTOR / updates_per_year
    n = len(frequencies)
    #fft_mutations = np.zeros(3 * n - 2)
    #fft_mutations[n/2 : n/2 + 2*n - 1] = mutations
    #fft_mutations = rfft(fft_mutations)
    f = np.array(frequencies)
    births = np.empty_like(f)
    for _ in range(updates_per_year):
        births[:] = fftconvolve(birth_factors * f, mutations, mode='valid')
        f *= (1 - death_factor)
        f += births
    return f