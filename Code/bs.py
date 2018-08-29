import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as la
import math
import json
import gzip
import pickle
import warnings
from matplotlib import animation, rc
from mpmath import mp


# Set the default number of digits of precision in mpmath multiprecision
# operations.
mp.dps = 50

# Make some mpmath functions into quasi-ufuncs taking either scalar or
# array arguments.

mp_float = np.frompyfunc(mp.mpf, 1, 1)
erf = np.frompyfunc(mp.erf, 1, 1)
erfc = np.frompyfunc(mp.erfc, 1, 1)
exp = np.frompyfunc(mp.exp, 1, 1)
gamma = np.frompyfunc(mp.gamma, 1, 1)
rgamma = np.frompyfunc(mp.rgamma, 1, 1) # reciprocal gamma
hyp1f1 = np.frompyfunc(mp.hyp1f1, 3, 1) # confluent hypergeometric 1_F_1
hyperu = np.frompyfunc(mp.hyperu, 3, 1) # confluent hypergeometric 2_F_2


# Generate JS/HTML animations.
# HTML5 animations require (sometimes tricky) FFmpeg installation on the host.
plt.rcParams['animation.html'] = 'jshtml'


# Use the Seaborn package to generate plots.
sns.set() 
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
sns.set_style("darkgrid", {"axes.facecolor": ".92"})
sns.set_palette(sns.color_palette("Set2", 4))



class Factors(object):
    """
    Stores Malthusian parameters exactly equal to those in Basener's code.
    """
    def __init__(self, n_types, death=0.1, max_growth=0.15, exclude_max=False):
        self.n_types = n_types
        self.death = death
        self.max_growth = max_growth
        self.exclude_max = exclude_max
        if exclude_max:
            self.delta = (max_growth + death) / n_types
            self.growth = np.linspace(-death, max_growth, n_types+1)[:-1]
        else:
            self.delta = (max_growth + death) / (n_types - 1)
            self.growth = np.linspace(-death, max_growth, n_types)
        self.birth = self.growth + death
        assert self.birth[0] == 0
        self.effects = np.concatenate((-self.birth[::-1], self.birth[1:]))


################################################################################
#             Parameters of the Basener-Sanford experiments
################################################################################

# Ideal spacing of points in the intervals of birth factors and growth factors
# Would be actual spacing if the values had exact binary representations
N_TYPES = {
            'NoneExact' : 626,   # Sects 5.1, 5.2
            'Gaussian'  : 251,   # Sect 5.3
            'Gamma'     : 501    # Sect 5.4
        }
N_TYPES['None'] = N_TYPES['NoneExact']


# I don't include year 0 in the count of years as Basener does.
N_YEARS = {
            'NoneExact' : 3500,
            'Gaussian'  : 300,
            'Gamma'     : 2500
          }
N_YEARS['None'] = N_YEARS['NoneExact']



################################################################################
#          Classes for populations and evolutionary trajectories
################################################################################


class Evolution(object):
    """
    Record the evolution of a Population instance.
    
    The n-th element of the evolutionary trajectory gives the frequencies of
    fitnesses (alternatively, growth factors) in the population after n epochs.
    """
    def __init__(self, population, n_epochs=0, years_per_epoch=1, x_stride=1):
        """
        Records the trajectory of `population` over `n_epochs` of evolution.
        """
        self.p = population
        self.trajectory = np.array([self.p[::x_stride]])
        self.sums = np.array([np.sum(self.p[:])])
        self.years_per_epoch = years_per_epoch
        self.xstride = x_stride
        self(n_epochs)
   
    def __call__(self, n_epochs=1):
        """
        Extends the evolutionary trajectory by the given number of epochs.
        """
        if n_epochs < 1:
            return
        n = len(self.trajectory)
        new_trajectory = np.empty((n_epochs + n, len(self.p[::self.xstride])))
        new_trajectory[:n] = self.trajectory
        self.trajectory = new_trajectory
        new_sums = np.empty(n_epochs + n)
        new_sums[:n] = self.sums
        self.sums = new_sums
        for i in range(n, n + n_epochs):
            for _ in range(self.years_per_epoch):
                self.p.annual_update()
            self.trajectory[i] = self.p[::self.xstride]
            self.sums[i] = np.sum(self.p[:])

    def __getitem__(self, index_or_slice):
        """
        Index or slice the evolutionary trajectory.
        """
        return self.trajectory[index_or_slice]
    
    def __len__(self):
        """
        Returns the length of the evolutionary trajectory.
        """
        return len(self.trajectory)
    
    def __str__(self):
        """
        Returns the string that labels the population/process.
        """
        return self.p.label
    
    def n_years(self):
        """
        Returns the number of years in the evolutionary process.
        """
        return self.per_epoch * (len(self.trajectory) - 1)
    
    def last_valid_epoch(self):
        """
        Returns the last epoch in which population size is a valid number.
        """
        sums = self.sums
        i = np.argmax(np.isnan(sums))
        j = np.argmax(np.isinf(sums))
        k = np.argmax(sums == 0)
        where = np.array([i, j, k])
        if np.max(where) > 0:
            where = where[np.nonzero(where)]
            first_invalid_epoch = np.min(where) - 1
            return first_invalid_epoch - 1
        return len(self) - 1
    
    def normalized(self, begin=None, end=None, stride=None):
        """
        Returns the trajectory with each point normalized.
        """
        # t = self.trajectory[begin:end:stride]
        # return (t.T / np.sum(t, axis=1).T).T
        return (self.trajectory[begin:end:stride].T / self.sums[begin:end:stride]).T
    
    def growth_factors(self, effective=False):
        return self.p.growth_factors(effective)[::self.xstride]
    
    def birth_factors(self, effective=False):
        return self.p.birth_factors(effective)[::self.xstride]
    
    def mean_and_variance(self, effective=False):
        """
        Returns mean and variance of fitnesses at each point in the trajectory.
        """
        return mean_and_variance(self.growth_factors(effective),
                                 self.normalized())


class BS_Evolution(Evolution):
    """
    Output of Basener's code, wrapped in an `Evolution` instance.
    """
    def __init__(self, bs, label=''):
        """
        Converts `bs` output of Basener's code to `Evolution` instance.
        """
        self.p = BS_Population(bs, label)
        self.trajectory = bs['Psolution']        

        
class Population(object):
    def __init__(self, initial_freqs, mutations, updates_per_year=1,
                       norm=None, threshold=1e-9, lossy=False, label='',
                       matrix=False):
        self.initial_freqs = initial_freqs
        self.freqs = np.array(initial_freqs)
        self.births = np.empty_like(self.freqs)
        self.annual_factors = initial_freqs.factors
        self.death_factor = self.annual_factors.death / updates_per_year
        self.updates_per_year = updates_per_year
        n = len(initial_freqs)
        self.birth_factors = self.annual_factors.birth / updates_per_year
        self.mutations = mutations
        if matrix:
            self.birthing = mutations.matrix(lossy) * self.birth_factors
            m = [mp.log(mp.fsum(col) + 1 - self.death_factor)
                     for col in self.birthing.T]
            self.effective_growth = np.multiply(m, self.updates_per_year)
            self.effective_birth = self.effective_growth - self.effective_growth[0]
        else:
            """HACK HACK HACK"""
            self.effective_growth = self.annual_factors.growth
            self.effective_birth = self.annual_factors.birth
            """HACK HACK HACK"""
        self.norm = norm
        self.threshold = threshold
        self.lossy = lossy
        self.label = label
        self.zero = self.freqs[0] * 0
        
    def solve(self):
        W = self.birthing - self.death_factor * np.eye(len(self))
        self.values, self.vectors = np.eig(W)
        self.c = la.solve(self.e_vectors, self.initial_freqs)
        
    def solution(self, t, normed=False):
        P_t = np.dot(self.e_vectors, self.c * np.exp(t * self.e_values)).real
        if normed:
            P_t /= float(mp.fsum(P_t))
        return P_t
    
    def update(self):
        try:
            np.dot(self.birthing, self.freqs, out=self.births)
        except:
            np.multiply(self.freqs, self.birth_factors, out=self.births)
            self.births = self.mutations(self.births, self.lossy)
        self.freqs *= 1 - self.death_factor
        self.freqs += self.births
        
    def annual_update(self):
        for _ in range(self.updates_per_year):
            self.update()
        if not self.norm is None:
            relatively_small = self.freqs <= self.threshold * self.norm(self.freqs)
            self.freqs[relatively_small] = self.zero  
        return self.freqs
    
    def birth_factors(self, effective=False):
        """        
        Returns growth factors minus the smallest of the growth factors.
        
        This assumes that the effective death factor does not depend on the
        type of organism. The fit to Equation 3.4 of the article is good, but
        conceivably would be improved by calculation of different death factors
        for the the different types.
        """
        if effective:
            return self.effective_birth
        return self.annual_factors.birth

    def growth_factors(self, effective=False):
        if effective:
            return self.effective_growth
        return self.annual_factors.growth

    def get_frequencies(self, normed=False):
        """
        Returns the frequencies of fitnesses in the current population.
        
        If `normalize` is true, proportions are returned instead of frequencies.
        """
        if normed:
            return self.freqs / self.size()
        return self.freqs
    
    def size(self):
        """
        Returns the size of the population, relative to its size in year 0.
        
        The size of the population is the sum of the frequencies of the discrete
        fitnesses. The size of the initial population is equated with 1. Thus if
        the current frequences sum to F, then the size of the population has
        changed by a factor of F since year 0.
        """
        return accurate_sum(self.freqs)
    
    def mean(self, effective=False):
        """
        TO DO: TEST
        """
        return moment(self.freqs, self.growth_factors(effective), 1)
        # return mp.dot(self.freqs, self.growth_factors(effective)) / self.size()
    
    def mean_and_variance(self, effective=False):
        factors = self.growth_factors(effective)
        mean = moment(self.freqs, factors, 1)
        var = moment(self.freqs, factors, 2) - mean ** 2
        return mean, var
    
    def set_label(self, label):
        """
        Sets the label of the population used in plotting.
        """
        self.label = label
    
    def __getitem__(self, index_or_slice):
        """
        Index or slice the  frequencies of the fitnesses.
        """
        return self.freqs[index_or_slice]
        
    def __len__(self):
        """
        Returns the number of discrete growth factors (and frequencies).
        """
        return len(self.freqs)
    
    def __str__(self):
        """
        Returns the label of the population.
        """
        return self.label

    
class BS_Population(Population):
    def __init__(self, bs, label=''):
        self.freqs = np.array(bs['Psolution'][-1])
        self.births = np.empty_like(self.freqs)
        self.annual_factors = Factors(bs['numIncrements'])
        self.death_factor = self.annual_factors.death
        self.updates_per_year = 1
        self.birthing = bs['MP']
        self.norm = np.max
        self.lossy = True
        self.label = label
        self.zero = 0


class XBS_Population(Population):
    """
    Override the `annual_update` method of the superclass.
    """
    BS_THRESHOLD = 1e-9
    
    def __init__(self, initial_distribution, births_redistribution, label='',
                 n_updates_per_year=1, threshold_norm=None, endpoint=True):
        """
        Register subclass-specific params; invoke the superclass initializer.
        
        The `threshold_norm` ... `endpoint` ...
        """
        self.threshold_norm = threshold_norm
        self.endpoint = endpoint
        super().__init__(initial_distribution, births_redistribution, 
                         label=label, n_updates_per_year=n_updates_per_year)
        if not self.endpoint:
            self.freqs[-1] = 0

    def annual_update(self):
        """
        Treat factors as linear factors, set subthreshold frequencies to zero.
        
        As in Basener's script, births that are distributed outside the range
        of growth factors are discarded. Also, the death rate and birth factors are
        treated as linear rather than logarithmic. The error is small when the
        factors are close to zero. A question here is whether increasing the
        number of updates per year, and scaling the factors inversely, increases
        the accuracy (when there is no thresholding).
        
        If the instance was created with `endpoint=False`, then the uppermost
        growth factor is set to zero at the end of each iteration.
        
        If the instance was created with `threshold_norm` ...
        """
        for _ in range(self.n_updates_per_year):
            births = self.freqs * self.birth_factors
            births = np.convolve(births, self.redistribution[:], mode='valid')
            self.freqs *= 1.0 - self.death_factor
            self.freqs += births
            if not self.threshold_norm is None:
                norm = self.threshold_norm(self.freqs)
                above_threshold = self.freqs >= self.BS_THRESHOLD * norm
                self.freqs *= above_threshold
            if not self.endpoint:
                self.freqs[-1] = 0
        self.year += 1



################################################################################
#                  Base class for discrete distributions
################################################################################


class Distribution(object):
    """
    A distribution of probability mass over a partition of an interval.
    
    Basener approximates the probability mass for each subinterval in the
    partition, multiplying the probability density at the center by the length
    of the subinterval. The only case in which he normalizes a distribution is
    in intialization of the population.
    """
    def __init__(self, domain, delta, label=None):
        """
        Initialize with all probability mass on the point closest to zero.
        
        The type of probabilities is the type of `domain` elements.
        """
        self.domain = domain
        self.delta = delta
        self.label = label
        n_points = len(domain)
        self.zero_centered = domain[0] == -domain[-1] and n_points % 2 == 1
        if self.zero_centered:
            self.zero_index = n_points // 2
            assert self.domain[self.zero_index] == 0
        else:
            self.zero_index = np.argmin(np.abs(self.domain))
            if self.domain[self.zero_index] != 0:
                warnings.warn('Zero is not in the domain')
        self.p = np.zeros_like(self.domain)  # p has base type of domain
        self.p[self.zero_index] = 1
        
    def convert(self, new_basetype):
        """
        Convert type of internally stored probabilities.
        """
        self.p = convert(self.p, new_basetype)
        
    def masses(self, distribution, domain, approximate=False):
        """
        Returns probabilites distributed over intervals with equispaced centers.
        
        The `distribution` is a scipy.stats frozen rv, e.g., stats.norm(0, 1).
        """
        if approximate:
            return distribution.pdf(domain) * self.delta
        lower = domain - self.delta / 2
        upper = domain + self.delta / 2
        upper[:-1] = lower[1:]
        return np.where(lower > distribution.median(),
                        distribution.sf(lower) - distribution.sf(upper),
                        distribution.cdf(upper) - distribution.cdf(lower))
        
    def gaussian(self, mean, std, approximate=False):
        self.rv = stats.norm(mean, std)
        self.p[:] = self.masses(self.rv, self.domain, approximate)
        if self.zero_centered and mean == 0:
            self.p[:] = (self.p + self.p[::-1]) / 2
    
    def symmetrized_gamma(self, alpha, beta, approximate=False):
        """
        In all cases, the Gamma CDF is used to set the probability of 0.
        """
        if not self.zero_centered:
            raise Exception('Zero is not at the center of the domain')
        self.rv = stats.gamma(alpha, scale=1/beta)
        x = self.domain
        self.p[x > 0] = self.masses(self.rv, x[x > 0], approximate) / 2
        self.p[x < 0] = self.p[x > 0][::-1]
        self.p[x == 0] = self.rv.cdf(self.delta / 2)
    
    def threshold(self, theta, normed=True):
        """
        Sets all probabilities less than `theta` to 0.
        
        The resulting distribution is normalized by default.
        """
        self.p[self.p < theta] = 0
        if normed:
            self.normalize()
    
    def support(self):
        """
        Returns array of domain elements with probability greater than zero.
        """
        return self.domain[self.p > 0]
    
    def norm(self):
        """
        Returns the sum of the probabilities as a multiprecision float.
        """
        return mp.fsum(np.sort(self.p))
    
    def normalize(self):
        """
        Divides all probabilities by their (accurate) sum.
        """
        self.p[:] = self.p / self.norm()
    
    def set_label(self, label):
        """
        Sets the identifier used in legends of plots of the distribution.
        """
        self.label = label

    def get_label(self):
        """
        Returns a string describing the distribution.
        """
        if self.label is None:
            return type(self).__name__
        return str(self.label)
    
    def moment(self, n):
        """
        Returns the n-th raw moment of the distribution as multiprecision float.
        """
        return moment(self.p, self.domain, n)
    
    def mean(self):
        """
        Returns the mean value of the distribution as multiprecision float.
        """
        return self.moment(1)
    
    def variance(self):
        """
        Returns the variance of the distribution as multiprecision float.
        """
        return self.moment(2) - self.moment(1) ** 2
    
    def mean_and_variance(self):
        """
        Returns (mean, variance) of the distribution as multiprecision floats.
        """
        mean = self.moment(1)
        variance = self.moment(2) - mean ** 2
        return mean, variance
            
    def vlines(self, axes, x_offset=0, label=None, color='k', lw=3):
        """
        Returns vlines object plotted on the given `axes`.
        
        When plotting vertical lines for two distributions on the same axes,
        shift one plot slightly to the left with a negative `x_offset`, and the
        other slightly to the right with a positive `x_offset`.
        """
        if label is None:
            label = self.get_label()
        axes.set_xlabel('Change in Nominal Fitness')
        axes.set_ylabel('Probability')
        v = axes.vlines(self.domain + x_offset, 0, self.p, label=label)
        v.set(color=color, linewidth=lw, alpha=1)
        return v
    
    def line(self, axes, label=None):
        """
        Plot the distribution on the axes as a line, return the line object.
        """
        if label is None:
            label = self.get_label()
        line, = axes.plot(self.domain, self.p, label=label)
        return line
    
    def __len__(self):
        """
        Returns the number of elements in the domain of the distribution.
        """
        return len(self.domain)

    def __getitem__(self, index_or_slice):
        """
        Index or slice the distribution.
        """
        return self.p[index_or_slice]



################################################################################
#          Initial distributions of the population over growth factors
################################################################################


class Frequencies(Distribution):
    """
    Base class for probability distributions over discrete growth factors.
    
    Although the primary use of this class is as a base, it can be instantiated
    directly, in which case all of the probability mass is associated with the
    growth factor closest to 0.
    """
    def __init__(self, factors):
        """
        Initializes with all probability mass on growth factor closest to zero.
        """
        self.factors = factors
        super().__init__(factors.growth, factors.delta)

        

class GaussianFrequencies(Frequencies):
    """
    A discretized Gaussian distribution of probability over growth factors.
    """
    def __init__(self, factors, mean=0.044, std=0.005, crop=11.2, density=True):
        """
        Sets distribution to discretized Normal(mean, std) over growth factors.
        
        Given in `factors.growth` are equispaced centers of subintervals of
        length `factors.delta`. Growth factors differing from the given `mean`
        by more than `crop` standard deviations are assigned zero mass. The
        Boolean parameter `density` determines whether we follow Basener's
        script in setting the probability masses of subintervals proportional
        to the probability density at their centers.
        
        The distribution of probability mass is normalized.
        """
        super().__init__(factors)
        self.given_mean = mean
        self.given_std = std
        z = (self.domain - mean) / std
        if density:
            self.p[:] = np.exp(-0.5 * z ** 2)
        else:
            # Difference the cumulative distribution function at the endpoints
            # of subintervals centered on equispaced growth factors, using
            # multiprecision floating point numbers in the calculations. This
            # is equivalent to integrating the density over each subinterval.
            delta_z = mp.mpf(factors.delta) / std
            ends = np.concatenate(([z[0] - delta_z / 2], z + delta_z / 2))
            cdf = 0.5 * (1 + erf(ends / mp.sqrt(2)))
            self.p[:] = cdf[1:] - cdf[:-1]
        self.p[np.abs(z) > crop] = 0
        self.normalize()


################################################################################
#          Distributions of births over mutation effects on growth rate
################################################################################


class EffectsDistribution(Distribution):
    """
    Distribution of probability over mutation effects on fitness.
    
    The initial distribution is symmetric by construction, with probabilities
    of positive effects reflected in the zero-effect axis. There are various
    methods for altering the distribution.
    """
    def __init__(self, factors, rv=None, density=False, normed=True):
        """
        Sets a symmetric distribution of probability over mutation effects.
        
        Parameter `factors` is an instance of class `Factors`. Parameter `rv`
        is either `None`, in which case probability mass of 1 is assigned to
        effect 0, or a scipy.stats "frozen rv," e.g., stats.norm(0, 0.002) for
        the Gaussian case of the article. The Boolean `density` determines
        whether probability densities are used instead of probability masses
        in construction of the distribution. The Boolean `normed` determines
        whether or not the constructed distribution is normalized. 
        
        In any case, a symmetric distribution is constructed by reflecting
        the probabilities of positive effects in the zero-effect axis. 
        
        If `density` is true, then the probability of a non-negative effect is
        the probability density at the effect, multiplied by the length of the
        subinterval centered on the effect. However, it may be that the
        density is undefined at zero, in which case the probability of zero
        effect is set as when `density` is false.
        
        If `density` is false, then the probability of a positive effect is
        the probability mass of the subinterval centered on the effect. The
        probability of zero effect is twice the mass of (0, delta/2], where
        delta is the length of the subintervals.
        """
        self.factors = factors
        self.rv = rv
        super().__init__(factors.effects, factors.delta)
        self.effect = self.domain
        if rv is None:
            return
        x = self.effect
        if density:
            self.p[x > 0] = rv.pdf(x[x > 0]) * self.delta
        else:
            self.p[x > 0] = self.masses(rv, x[x > 0])
        if density and not np.isinf(rv.pdf(0)):
            self.p[x == 0] = rv.pdf(0) * self.delta
        else:
            self.p[x == 0] = 2 * (rv.cdf(self.delta / 2) - rv.cdf(0))
        self.p[x < 0] = self.p[x > 0][::-1]
        if normed:
            self.normalize()
    
    def growth_factors(self):
        return self.factors.growth
    
    def birth_factors(self):
        return self.factors.birth
        
    def matrix(self, lossy=False):
        n = (len(self.p) + 1) // 2
        c = np.empty((n, n), dtype=type(self.p[0]))
        for i in range(n):
            c[i] = self.p[i:i+n][::-1]
        if not lossy:
            for i in range(1, n):
                c[0, :i] += self.p[:i][::-1]
            for i in range(n-1, 0, -1):
                c[-1, -i:] += self.p[-i:][::-1]
        return c
    
    def gimmick(self):
        """
        Assigns the probability of minimally delerious effect to zero effect.
        """
        zero = len(self) // 2
        self.p[zero] = self.p[zero - 1]
    
    def reweight(self, percent_beneficial):
        """
        Sets the probability of positive effect to `percent_beneficial`.
        
        That is, the probabilities of positive effects are scaled so that they
        sum to `percent_beneficial`, and the probabilities of negative effects
        are scaled so that they sum to 1 - `percent_beneficial`.
        """
        raise Exception('this is wrong')
        x = self.domain
        self.p[x > 0] *= percent_beneficial / float(mp.fsum(self.p[x > 0][::-1]))
        self.p[x <= 0] *= (1 - percent_beneficial) / float(mp.fsum(self.p[x <= 0]))
        self.normalize()
    
    def iid_effects(self, number_of_mutations=1, log_number_of_loci=0,
                          truncate_self_convolution=False):
        """
        With the default settings, the distribution is unchanged. If the number
        of mutations is zero, then the probability of zero effect is 1.
        """
        self.log_number_of_loci = log_number_of_loci
        self.number_of_mutations = number_of_mutations
        self.mu = number_of_mutations / 2 ** log_number_of_loci
        self.p *= self.mu
        self.p[self.zero_index] += 1 - self.mu
        self.self_convolve(log_number_of_loci, truncate_self_convolution)
    
    def convolve(self, x, discard_excess=False):
        """
        Returns specially handled convolution of x and self.
        
        The length of the returned array is equal to the length of x. If
        `discard_excess` is false, then the out-of-range components are lumped
        with the endpoints of the result. Otherwise, the out-of-range components
        are simply excluded from the result, as in Basener's script.
        """
        n = len(self.p) // 2
        result = np.convolve(x, self.p)
        if not discard_excess:
            result[n] += np.sum(result[:n])
            result[-n-1] += np.sum(result[-n:])
        return result[n:-n]
    
    def self_convolve(self, n_times=1, discard_excess=False):
        """
        Set distribution to L-fold convolution of itself, where L=2^n_times.
        
        If discard_excess is true, then the distribution is renormalized in each
        iteration.
        """
        for _ in range(n_times):
            self.p[:] = self.convolve(self.p, discard_excess)
            if discard_excess:
                self.p /= np.sum(self.p)
                
    def plot(self, title, x_offset=0, label=None):
        """
        Return a figure containing a vlines plot of the distribution.
        """
        fig, ax = plt.subplots()
        vlines = self.vlines(ax, x_offset=x_offset, label=label)
        mean, variance = self.mean_and_variance()
        std = np.sqrt(variance)
        subtitle = '\nMean {0}, Standard Deviation {1}'.format(mean, std)
        ax.set_title(title + subtitle)
        ax.set_xlabel('Change in Nominal Growth Factor')
        ax.set_ylabel('Probability [CORRECT LABEL?]')
        return fig 
    
    def __call__(self, other, discard_excess=False):
        return self.convolve(other, discard_excess)



################################################################################
#                            Plots and animations
################################################################################


class Comparison(object):
    """
    Container of multiple evolutionary processes under comparison.
    """
    def __init__(self, processes, subtitle='\n[set_subtitle]',
                       colors=None, linestyles=None):
        """
        Stores the evolutionary `processes` along with plot settings.
        
        Shorter processes are run, when necessary, to make them equal in
        length to the longest of the processes. The subtitle and graphics
        settings are used in plots and animations.
        """
        self.processes = list(processes)
        self.set_subtitle(subtitle)
        lengths = [len(p) for p in processes]
        self.length = np.min(lengths)
        self.run_until_epoch(np.max(lengths) - 1)
        if colors is None:
            colors = sns.color_palette()
        self.set_colors(colors)
        if linestyles is None:
            linestyles = ['-']
        self.set_linestyles(linestyles)

    def run_until_epoch(self, epoch):
        """
        Makes the processes extend logically to the given `epoch`.
        
        Processes are run as long as necessary (perhaps not at all) to make
        them extend to the given epoch. Then the length of this object is set
        unconditionally to `epoch` + 1.
        """
        new_length = epoch + 1
        if new_length > self.length:
            for ev in self.processes:
                ev(new_length - len(ev))
        self.length = new_length
    
    def __len__(self):
        """
        Return the length of the evolutionary processes (including year 0).
        """
        return self.length

    def set_subtitle(self, subtitle):
        assert isinstance(subtitle, str)
        self.subtitle = subtitle
            
    def set_linestyles(self, linestyles):
        n = len(self.processes)
        ls = [linestyles[i % len(linestyles)] for i in range(n)]
        self.linestyles = np.array(ls)
        
    def set_colors(self, colors):
        n = len(self.processes)
        self.colors = np.array([colors[i % len(colors)] for i in range(n)])
        
    def select(self, index_list):
        """
        Returns new `Comparison` of processes indicated by `index_list`.
        """
        i = index_list
        processes = [self.processes[j] for j in i]
        return Comparison(processes, self.subtitle, colors=self.colors[i],
                          linestyles=self.linestyles[i])
    
    def mean_variance_plots(self, line_styles=None):
        return mean_variance_plots(self.processes, line_styles=line_styles,
                                   subtitle=self.subtitle)
    
    def animate(self, nframes=100, duration=10000, effective=True):
        """
        Returns animation of evolutionary processes.
        
        If the number of frames, `nframes`, is 0, then a static figure is
        returned instead of an animation. The `duration` of the animation
        is given in milliseconds. The Boolean `effective` determines whether
        effective growth rates (fitnesses) are displayed instead of nominal
        fitnesses.
        """
        length = len(self)
        n_years = length - 1
        if nframes < 1:
            stride = None
        else:
            nframes = min(nframes, length)
            duration = max(duration, nframes)
            stride = length // nframes
            interval = round(duration / nframes)
        labels = [str(p.p) for p in self.processes]
        
        # Each process has its own growth factors
        g = [p.growth_factors(effective) for p in self.processes] 
        procs = [p[:length:stride] for p in self.processes]          # Introduce stride in views of unnormalized
        procs_last = [p[length-1:length] for p in self.processes]          # Views of last unnormalized frames
        normed = [p.normalized(end=length, stride=stride) for p in self.processes]
        normed_last = [p / np.sum(p) for p in procs_last]
        normed = [np.concatenate((a, b)) for a, b in zip(normed, normed_last)]
        procs = [np.concatenate((a, b)) for a, b in zip(procs, procs_last)]
        p = [normed[:], procs[:]]
        n_frames = len(p[0][0])
        lines = np.empty((2, len(p[0])), dtype=object)
        is_interactive = plt.isinteractive()
        plt.interactive(False)
        fig, ax = plt.subplots(2, sharex=True)
        
        # Construct lines by plotting them for the final year
        for n in range(2):
            for i in range(len(g)):
                w = p[n][i][-1] > 0  # Boolean indices of support
                lines[n][i], = ax[n].plot(g[i][w], p[n][i][-1][w], 
                                          label=labels[i], lw=4, zorder=10,
                                          ls=self.linestyles[i],
                                          c=self.colors[i],)
        for n in range(2):
            for i in range(len(g)):
                w = p[n][i][0] > 0
                ax[n].plot(g[i][w], p[n][i][0][w], c='black', lw=1, alpha=0.5,
                           zorder=11)
                w = p[n][i][-1] > 0
                ax[n].plot(g[i][w], p[n][i][-1][w], c=lines[n][i].get_c(),
                           lw=1, alpha=1)
        
        title = 'Evolution for {0} Years'.format(n_years)
        fig.suptitle(title + self.subtitle)
        if effective:
            ax[1].set_xlabel('Effective Fitness')
        else:
            ax[1].set_xlabel('Nominal Fitness')
        ax[1].set_yscale('log')
        ax[1].set_ylabel('Frequency')
        ax[0].set_ylabel('Proportion')
        ax[0].legend(loc='upper left')
        plt.interactive(is_interactive)
        
        def adjust_ylim(ax, data):
            raveled = np.ravel(data)
            y_min, y_max = min_and_max(raveled[np.nonzero(raveled)])
            y_lim = ax.get_ylim()
            if not np.isinf(y_max) and not np.isnan(y_max):
                y_max = max(y_max, y_lim[1])
            if not np.isinf(y_min) and not np.isnan(y_min):
                y_min = min(y_min, y_lim[0])
            try:
                ax.set_ylim(y_min, y_max)
            except:
                pass
            
        def initializer():
            for n in range(2):
                for line, x, y in zip(lines[n], g, p[n]):
                    line.set_xdata(x[y[0] > 0])
                    line.set_ydata(y[0][y[0] > 0])
                    line.set_lw(1)
            return lines.flatten()

        def animator(i):
            for n in range(2):
                for line, x, y in zip(lines[n], g, p[n]):
                    line.set_xdata(x[y[i] > 0])
                    line.set_ydata(y[i][y[i] > 0])
                    if i == 1:
                        line.set_lw(4)
            return lines.flatten()
        
        if stride is None:
            out = fig
        else:
            adjust_ylim(ax[1], procs)
            adjust_ylim(ax[0], normed) 
            out = animation.FuncAnimation(fig, animator, init_func=initializer,
                                          frames=n_frames, interval=interval,
                                          blit=True, repeat_delay=2000)
            plt.close()
        return out
    
    def Xanimate(self, nframes=100, duration=10000, effective=True):
        """
        Returns animation of evolutionary processes.
        
        If the number of frames, `nframes`, is 0, then a static figure is
        returned instead of an animation. The `duration` of the animation
        is given in milliseconds. The Boolean `effective` determines whether
        effective growth rates (fitnesses) are displayed instead of nominal
        fitnesses.
        """
        length = len(self)
        n_years = length - 1
        if nframes < 1:
            stride = None
        else:
            nframes = min(nframes, length)
            duration = max(duration, nframes)
            stride = length // nframes
            interval = round(duration / nframes)
        labels = [str(p.p) for p in self.processes]
        g = [p.p.growth_factors(effective) for p in self.processes]
        procs = [p[:length] for p in self.processes]                  # Introduce stride here?
        normed = [p.normalized(end=length) for p in self.processes]   # Need a normalization generator?
        p = [normed[:], procs[:]]
        n = length
        if not stride is None:
            for i in range(2):
                p[i] = [np.concatenate((y[:n:stride], [y[n-1]])) for y in p[i]]  # Is contenation necessary?
        #
        # MOVE NORMALIZATION HERE
        #
        n_frames = len(p[0][0])
        lines = np.empty((2, len(p[0])), dtype=object)
        is_interactive = plt.isinteractive()
        plt.interactive(False)
        fig, ax = plt.subplots(2, sharex=True)
        
        # Construct lines by plotting them for the final year
        for n in range(2):
            for i in range(len(g)):
                w = p[n][i][-1] > 0  # Boolean indices of support
                lines[n][i], = ax[n].plot(g[i][w], p[n][i][-1][w], 
                                          label=labels[i], lw=4, zorder=10,
                                          ls=self.linestyles[i],
                                          c=self.colors[i],)
        for n in range(2):
            for i in range(len(g)):
                w = p[n][i][0] > 0
                ax[n].plot(g[i][w], p[n][i][0][w], c='black', lw=1, alpha=0.5,
                           zorder=11)
                w = p[n][i][-1] > 0
                ax[n].plot(g[i][w], p[n][i][-1][w], c=lines[n][i].get_c(),
                           lw=1, alpha=1)
        
        title = 'Evolution for {0} Years'.format(n_years)
        fig.suptitle(title + self.subtitle)
        if effective:
            ax[1].set_xlabel('Effective Fitness')
        else:
            ax[1].set_xlabel('Nominal Fitness')
        ax[1].set_yscale('log')
        ax[1].set_ylabel('Frequency')
        ax[0].set_ylabel('Proportion')
        ax[0].legend(loc='upper left')
        plt.interactive(is_interactive)
        
        def adjust_ylim(ax, data):
            raveled = np.ravel(data)
            y_min, y_max = min_and_max(raveled[np.nonzero(raveled)])
            y_lim = ax.get_ylim()
            if not np.isinf(y_max) and not np.isnan(y_max):
                y_max = max(y_max, y_lim[1])
            if not np.isinf(y_min) and not np.isnan(y_min):
                y_min = min(y_min, y_lim[0])
            try:
                ax.set_ylim(y_min, y_max)
            except:
                pass
            
        def initializer():
            for n in range(2):
                for line, x, y in zip(lines[n], g, p[n]):
                    line.set_xdata(x[y[0] > 0])
                    line.set_ydata(y[0][y[0] > 0])
                    line.set_lw(1)
            return lines.flatten()

        def animator(i):
            for n in range(2):
                for line, x, y in zip(lines[n], g, p[n]):
                    line.set_xdata(x[y[i] > 0])
                    line.set_ydata(y[i][y[i] > 0])
                    if i == 1:
                        line.set_lw(4)
            return lines.flatten()
        
        if stride is None:
            out = fig
        else:
            adjust_ylim(ax[1], procs)
            adjust_ylim(ax[0], normed)        
            out = animation.FuncAnimation(fig, animator, init_func=initializer,
                                          frames=n_frames, interval=interval,
                                          blit=True, repeat_delay=2000)
            plt.close()
        return out




def mean_variance_plots(evs, labels=None, line_styles=None, subtitle=''):
    """
    Generate some BS-like plots for Evolution instances `evs`.
    
    The iterable `evs` must be of the same length as the corresponding `labels`
    used in the legends of plots.
    
    TO DO: Add number of years to "Mean and Variance" title
    TO DO: The "delta" plots of BS
    """
    if labels is None:
        labels = np.array([str(ev.p) for ev in evs])
    if line_styles is None:
        line_styles = np.array(['-' for ev in evs])
    assert len(evs) == len(labels)
    #
    def plot(title, xlabel, ylabel, xs=None, ys=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if not xs is None and not ys is None:
            for x, y, label, ls in zip(xs, ys, labels, line_styles):
                ax.plot(x, y, label=label, ls=ls)
                fig.canvas.draw()
        ax.set_title('{0}{1}'.format(title, subtitle))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        return fig, ax
    #
    means_variances = [ev.mean_and_variance() for ev in evs]
    means = [mv[0] for mv in means_variances]
    variances = [mv[1] for mv in means_variances]
    years = [np.arange(len(evs[0]))] * len(evs)
    plot('Mean and Variance', 'Variance in Fitness', 'Mean Fitness',
            variances, means)
    plot('Mean Fitness', 'Year', 'Mean Fitness', years, means)
    plot('Upward Fitness Pressure: Variance in Fitness',
            'Year', 'Variance in Fitness', years, variances)



################################################################################
#                              Utility functions
################################################################################


def convert(iterable, new_basetype):
    """
    Returns Numpy array with objects in iterable converted to new type.
    """
    return np.array([new_basetype(x) for x in iterable])


def accurate_sum(a, dps=mp.dps):
    """
    Returns sum of elements in array `a` as multiprecision float.
    """
    with mp.workdps(dps):
        i = np.argsort(np.abs(a))
        return mp.fsum(a[i])


def moment(frequency, x, n, dps=mp.dps):
    """
    Returns n-th raw moment as multiprecision float.
    """
    with mp.workdps(dps):
        x = convert(x, mp.mpf) ** n
        f = convert(frequency, mp.mpf)
        return accurate_sum(np.multiply(f, x), dps) / accurate_sum(f, dps)


def accurate_mean(frequency, x, dps=mp.dps):
    """
    Returns mean value of x as multiprecision float.
    """
    return moment(frequency, x, 1, dps)


def accurate_variance(frequency, x, dps=mp.dps):
    """
    Returns variance of x as multiprecision float.
    """
    return moment(frequency, x, 2, dps) - moment(frequency, x, 1, dps) ** 2

   
def mean_and_variance(x, p):
    """
    Return (mean, variance) for distribution(s) p of probability over x.
    
    Also works if x has the same shape as p.
    """
    axis = max(x.ndim, p.ndim) - 1
    mean = np.sum(np.multiply(p, x), axis=axis)
    variance = np.sum(np.multiply(p, np.square(x)), axis=axis)
    variance -= np.square(mean)
    return mean, variance


def relative_error(actual, desired, absolute=False):
    """
    Returns (actual - desired) / desired, with 0/0 defined equal to 0.
    """
    if np.shape(actual) != np.shape(desired):
        raise ValueError('Arguments are not identical in shape')
    result = np.where(actual == desired, 0, (actual - desired) / desired)
    if absolute:
        np.abs(result, out=result)
    return result


def maximum_absolute_relative_error(actual, desired):
    return np.max(relative_error(actual, desired, absolute=True))


def min_and_max(a):
    """
    Returns the pair (min(a), max(a)).
    """
    return np.min(a), np.max(a)


def slice_to_support(p):
    """
    Returns slice excluding zeros, if any, in the tails of distribution p.
    """
    positive = p > 0
    a = np.argmax(positive)
    b = len(positive) - np.argmax(positive[::-1])
    return slice(a, b, None)
        
    
def gamma_density(x, alpha=0.5, beta=500, dps=mp.dps):
    """
    Returns result of high-precision calculation of Gamma probability density.
    
    The intended use is in numerical integration, to check other calculations.
    Integrate with an expression like `mp.quad(gamma_density, [a, b])`, where
    `a` and `b` are the limits of the integral.
    """
    with mp.workdps(dps):
        alpha = mp.mpf(alpha)
        beta = mp.mpf(beta)
        result = beta ** alpha / mp.gamma(alpha) * x ** (alpha - 1)
        exponent = -beta * x
        try:
            result *= mp.exp(exponent)
        except:
            result *= exp_vector(exponent)
        return result

        
def regularized_lower_incomplete_gamma(x, alpha=0.5, beta=500, dps=mp.dps):
    """
    Returns array of values of the regularized lower incomplete gamma function.
    
    This is the cumulative distribution function of the Gamma distribution with
    shape parameter alpha and rate parameter beta. Parameter `x` is expected to
    be iterable, and the function returns a 1-by-1 array if it is scalar. The
    calculations are done with `dps` digits of of precision. The returned values
    are at the same precision.
    
    We don't expect to experiment with values of `alpha` other than 0.5, which
    is a very nice case: the standard `erf()` is applied to the square root of
    `beta * x`. In other cases, the calculations take some seconds to complete.
    """
    with mp.workdps(dps):
        z = np.multiply(x, mp.mpf(beta))
        if alpha == 0.5:
            return erf(z ** 0.5)
        return mp.rgamma(alpha) * hypergeometric_incomplete_gamma(alpha, z, dps)

    
def hypergeometric_lower_incomplete_gamma(s, z, dps=mp.dps):
    """
    TO DO: FIX vectorized 3-argument function
    Returns array of values of the lower incomplete gamma function.
    
    Parameter `z` is expected to be iterable, and an array is returned even if
    it is scalar. The result is not regularized. Kummer's confluent hyper-
    geometric function is used in the calculation. (See "Incomplete gamma
    function" in Wikipedia for details.) The calculations are done with `dps`
    digits of precision. The returned values are at the same precision.

    The intended use is for testing. To calculate the cumulative distribution
    function of the Gamma distribution, set `s` equal to alpha, and `z` equal
    to the product of beta and the array of x values for which results are
    desired. Multiply the result by `mp.rgamma(alpha)`, the reciprocal Gamma
    function, to regularize.
    """
    with mp.workdps(dps):
        s = mp.mpf(s)
        return z ** s / s * hyp1f1_vector(s, s + 1, -z)         


################################################################################
#                  Generate command to run Basener's JavaScript
################################################################################


def bs_command(percentage_of_mutations_that_are_beneficial=0.001,
               mutation_distribution_type='Gaussian',
               population_size='Finite',
               number_of_years=N_YEARS['Gaussian'],
               number_of_discrete_population_fitness_values=N_TYPES['Gaussian'],
               script_path='BS.js',
               output_path='bs5_3.json'):
    """
    Returns command for running Basener script (for Sect 5.3, by default).
    
    Adds 1 to the given number of years, and subtracts 1 from the given number
    of discrete fitness values.
    """
    return 'node {0} {1} {2} {3} {4} {5} {6}'.format(
                  script_path,
                  percentage_of_mutations_that_are_beneficial,
                  mutation_distribution_type,
                  population_size,
                  number_of_years + 1,
                  number_of_discrete_population_fitness_values,
                  output_path)


def json_to_zipped_pickle(path_without_extension):
    """
    with open('strings.json') as json_data:
    d = json.load(json_data)
    print(d)
    """
    with open(path_without_extension + '.json') as f:
        bs = json.load(f)
    try:
        bs['PctBeneficial'] = float(bs['PctBeneficial'])
    except ValueError:
        bs['PctBeneficial'] = None
    bs['numYears'] = int(bs['numYears'])
    bs['numIncrements'] = int(bs['numIncrements'])
    bs['b'] = np.array(bs['b'], dtype=float)
    bs['Psolution'] = np.array(bs['Psolution'], dtype=float)
    bs['P'] = np.array(bs['P'], dtype=float)
    bs['m'] = np.array(bs['m'], dtype=float)
    bs['meanFitness'] = np.array(bs['meanFitness'], dtype=float)
    bs['varianceFitness'] = np.array(bs['varianceFitness'], dtype=float)
    bs['mutation_probs'] = np.array(bs['mutation_probs'], dtype=float)
    bs['MP'] = np.array(bs['MP'], dtype=float)
    bs['mDelta'] = float(bs['mDelta'])
    with gzip.open(path_without_extension + '.pickled.gz', 'wb') as f:
        pickle.dump(bs, f, -1)
