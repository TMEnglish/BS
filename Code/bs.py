import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gzip
import pickle
import warnings
from matplotlib import animation, rc
from math import fsum


# Generate JS/HTML animations.
# HTML5 animations require (sometimes tricky) FFmpeg installation on the host.
plt.rcParams['animation.html'] = 'jshtml'


# Use the Seaborn package to generate plots.
sns.set() 
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
sns.set_style("darkgrid", {"axes.facecolor": ".92"})
sns.set_palette(sns.color_palette("Set2", 4))



class Factors(object):
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
    fitnesses (alternatively, growth factors) in the population after n years.
    """
    def __init__(self, population, n_years=0):
        """
        Record the trajectory of `population` over `n_years` of evolution.
        """
        self.p = population
        self.trajectory = np.array([self.p[:]])
        if n_years > 0:
            self(n_years)
   
    def __call__(self, n_years=1):
        """
        Extend the evolutionary trajectory by the given number of years.
        """
        n = len(self.trajectory)
        new_trajectory = np.empty((n_years + n, len(self.p)))
        new_trajectory[:n] = self.trajectory
        self.trajectory = new_trajectory
        for i in range(n, n + n_years):
            self.p.annual_update()
            self.trajectory[i] = self.p[:]

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
        Return the string that labels the population/process.
        """
        return self.p.label
    
    def normalized(self):
        """
        Returns the trajectory with each point normalized.
        """
        t = self.trajectory
        # TO DO: Can this be expressed as an outer product?
        return (t.T / np.sum(t, axis=1).T).T
    
    def mean_and_variance(self, effective=False):
        """
        Returns mean and variance of fitnesses at each point in the trajectory.
        """
        return mean_and_variance(self.growth_factors(effective), self.normalized())


class BS_Evolution(Evolution):
    def __init__(self, bs):
        self.p = BS_Population(bs)
        self.trajectory = bs['Psolution']        


class XWrappedBS(Evolution):
    """
    `Evolution` instance constructed from the results of running Basener's code.
    """
    def __init__(self, bs_dict, label=''):
        """
        Construct `Evolution` instance according to dictionary `bs_dict`.
        
        The vector of annual growth factors (fitnesses) is stored under the key
        'm'. The sequence of frequency distributions over the annual growth
        factors is stored under the key 'Psolution'.
        """
        self.trajectory = bs_dict['Psolution']
        self.annual_growth_factors = bs_dict['m']
        self.annual_birth_factors = bs_dict['b']
        self.annual_death_factor = 0.1
        self.birth_gain = fsum(bs_dict['mutation_probs'])
        self.label = label
       
    def __call__(self, n=None):
        raise Exception('WrappedBS instances cannot be extended.')
    

class XPopulation(object):
    """
    TO DO: doc
    """    
    def __init__(self, initial_distribution, births_redistribution,
                       label='', n_updates_per_year=1):
        """
        TO DO: doc
        
        The initial distribution is normalized.
        """
        assert initial_distribution.delta == births_redistribution.delta
        self.initializer = initial_distribution
        self.redistribution = births_redistribution
        self.birth_gain = births_redistribution.norm()
        self.label = label
        self.n_updates_per_year = n_updates_per_year
        self.freqs = initial_distribution[:] / initial_distribution.norm()
        self.annual_birth_factors = births_redistribution.birth_factors()
        self.annual_growth_factors = births_redistribution.growth_factors()
        self.annual_death_factor = births_redistribution.factors.death
        self.birth_factors = self.annual_birth_factors / n_updates_per_year
        self.exp_birth_factors = np.exp(self.birth_factors) - 1
        self.death_factor = self.annual_death_factor / n_updates_per_year
        self.exp_death_factor = np.exp(-self.death_factor)
        self.year = 0

    def annual_update(self):
        """
        Do one year's worth of updates to the frequencies.
        
        Births distributed outside the range of fitnesses are lumped with births
        at the endpoints of the interval.
        
        If offspring are always identical to their parents in fitness (either
        because there are no mutations or because mutations have no effect on
        fitness), and a particular fitness is initially of frequency f0, then
        the frequency of that is f0 * exp(n * growth_factors) after n years
        """
        for _ in range(self.n_updates_per_year):
            self.freqs *= self.exp_death_factor
            births = self.freqs * self.exp_birth_factors
            self.freqs += self.redistribution(births)
        self.year += 1

        
class Population(object):
    def __init__(self, initial_freqs, mutations, updates_per_year=1,
                       norm=np.max, lossy=True, label=''):
        self.freqs = np.array(initial_freqs)
        self.births = np.empty_like(self.freqs)
        self.annual_factors = initial_freqs.factors
        self.death_factor = self.annual_factors.death / updates_per_year
        self.updates_per_year = updates_per_year
        n = len(initial_freqs)
        birth_factors = self.annual_factors.birth / updates_per_year
        self.birthing = np.repeat([birth_factors], n, axis=0)
        self.birthing *= mutations.convolution_matrix(lossy)
        self.norm = norm
        self.lossy = lossy
        self.label = label
        self.zero = self.freqs[0] * 0
    
    def annual_update(self):
        for _ in range(self.updates_per_year):
            np.dot(self.birthing, self.freqs, out=self.births)
            self.freqs *= 1 - self.death_factor
            self.freqs += self.births
        if not self.norm is None:
            relatively_small = self.freqs <= 1e-9 * self.norm(self.freqs)
            self.freqs[relatively_small] = self.zero  
        return self.freqs

    def growth_factors(self, effective=False):
        if effective:
            g = np.array([fsum(col) for col in self.birthing.T])
            g += 1 - self.death_factor
            return np.log(g) * self.updates_per_year
        return self.annual_factors.growth

    def get_frequencies(self, normed=False):
        """
        Returns the frequencies of fitnesses in the current population.
        
        If `normalize` is true, proportions are returned instead of frequencies.
        """
        result = np.array(self.freqs)
        if normed:
            result /= self.size()
        return result
    
    def size(self):
        """
        Returns the size of the population, relative to its size in year 0.
        
        The size of the population is the sum of the frequencies of the discrete
        fitnesses. The size of the initial population is equated with 1. Thus if
        the current frequences sum to F, then the size of the population has
        have changed by a factor of F since year 0.
        """
        return fsum(self.freqs)
    
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
    def __init__(self, bs):
        self.freqs = np.array(bs['Psolution'][-1])
        self.births = np.empty_like(self.freqs)
        self.annual_factors = Factors(bs['numIncrements'])
        self.death_factor = self.annual_factors.death
        self.updates_per_year = 1
        self.birthing = bs['MP']
        self.norm = np.max
        self.lossy = True
        self.label = ''
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
        self.p = np.zeros_like(self.domain)
        self.p[self.zero_index] = 1
        
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
        self.p[:] = self.masses(stats.norm(mean, std), self.domain, approximate)
        if self.zero_centered and mean == 0:
            self.p[:] = (self.p + self.p[::-1]) / 2
    
    def symmetrized_gamma(self, alpha, beta, approximate=False):
        """
        In all cases, the Gamma CDF is used to set the probability of 0.
        """
        if not self.zero_centered:
            raise Exception('Zero is not at the center of the domain')
        gamma = stats.gamma(alpha, scale=1/beta)
        x = self.domain
        self.p[x > 0] = self.masses(gamma, x[x > 0], approximate) / 2
        self.p[x < 0] = self.p[x > 0][::-1]
        self.p[x == 0] = gamma.cdf(self.delta / 2)
    
    def norm(self):
        """
        Returns an accurate sum of the probabilities in the distribution.
        """
        return fsum(np.sort(self.p))
    
    def normalize(self):
        """
        Divides all probabilities by their (accurate) sum.
        """
        self.p /= self.norm()
    
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
        Returns the n-th raw moment of the distribution.
        """
        product = self.p * self.domain ** n
        i = np.argsort(np.abs(product))
        # pos = fsum(np.sort(product[product > 0]))
        # neg = fsum(np.sort(product[product < 0])[::-1])
        return fsum(product[i])
    
    def mean_and_variance(self):
        """
        Returns (mean, variance) of the distribution.
        """
        mean = self.moment(1)
        variance = self.moment(2) - mean ** 2
        return mean, variance
            
    def vlines(self, axes, x_offset=0, label=None, color='k', lw=3):
        """
        Plot the distribution on the axes as vlines, return the vlines object.
        
        When plotting vertical lines for two distributions on the same axes,
        shift one plot slightly to the left with a negative `x_offset`, and the
        other slightly to the right with a positive `x_offset`.
        """
        if label is None:
            label = self.get_label()
        axes.set_xlabel('Change in Nominal Growth Factor')
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
        self.factors = factors
        super().__init__(factors.growth, factors.delta)
    
    def growth_factors(self):
        return self.factors.growth
    
    def birth_factors(self):
        return self.factors.birth
        

class GaussianFrequencies(Frequencies):
    """
    A discretized Gaussian distribution of probability over growth factors.
    """
    def __init__(self, factors, mean=0.044, std=0.005, crop=11.2, density=True):
        """
        Set distribution to discretized Normal(mean, std) over growth factors.
        
        The `factors` parameter is an instance of the `Factors` class. With the
        default settings of the other parameters, the result is close to the
        initial frequency distribution generated by Basener's code. Here the
        sum used in normalizing the distribution is accurate. In Basener's code,
        the sum is inaccurate.
        
        Growth factors differing from the given `mean` by more than `crop`
        standard deviations are assign zero frequency. Set `density=False` to
        assign probability masses instead of probability densities to the
        growth factors, prior to normalization.
        """
        super().__init__(factors)
        self.given_mean = mean
        self.given_std = std
        z = (self.domain - mean) / std
        if density:
            np.exp(-0.5 * z ** 2, out=self.p)
        else:
            self.gaussian(mean, std)
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
    def __init__(self, factors, rv, density=False, normed=True):
        """
        Sets a symmetric distribution of probability over mutation effects.
        
        Parameter `factors` is an instance of class `Factors`. Parameter `rv`
        is a scipy.stats "frozen rv," e.g., stats.norm(0, 0.002) for the
        Gaussian case of the article. The Boolean `density` determines whether
        probability densities are used instead of probability masses in
        construction of the distribution. The Boolean `normed` determines
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
        x = self.domain
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
    
    def convolution_matrix(self, lossy=True):
        n = (len(self.p) + 1) // 2
        def row(i):
            r = np.array(self.p[i:i+n])
            if not lossy:
                r[0] += fsum(self.p[:i])
                r[-1] += fsum(self.p[i+n:][::-1])
            return r[::-1]
        return [row(i) for i in range(n)]
    
    def gimmick(self):
        """
        Assign the probability of minimally delerious effect to zero effect.
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
        x = self.domain
        self.p[x > 0] *= percent_beneficial / fsum(self.p[x > 0][::-1])
        self.p[x <= 0] *= (1 - percent_beneficial) / fsum(self.p[x <= 0])
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
    def __init__(self, processes, subtitle='[set_subtitle]', n_years=0):
        """
        Store the evolutionary processes (identical to one another in length).
        
        The given processes are extended by the given number of years. The given
        subtitle is used in plots and animations.
        """
        assert np.array([len(p) == len(processes[0]) for p in processes]).all()
        self.processes = list(processes)
        self.subtitle = subtitle
        if n_years > 0:
            self(n_years)

    def __call__(self, n_years):
        """
        Extend the evolutionary processes by the specified number of years.
        """
        for ev in self.processes:
            ev(n_years)
    
    def __len__(self):
        """
        Return the length of the evolutionary processes (including year 0).
        """
        return len(self.processes[0])

    def set_subtitle(self, subtitle):
        self.subtitle = subtitle
    
    def mean_variance_plots(self, line_styles=None):
        return mean_variance_plots(self.processes, line_styles=line_styles,
                                   subtitle=self.subtitle)
    
    def animate(self, nframes=100, duration=10000, n_years=None,
                      line_styles=None, effective=True):
        """
        Return animation of one or more evolutionary processes.
        
        If the number of frames, `nframes`, is 0, then a static figure is
        returned instead of an animation.
        """
        if n_years is None:
            n_years = len(self) - 1
            length = len(self)
        else:
            length = n_years + 1
        if nframes is 0:
            stride = None
        else:
            if nframes > length:
                nframes = length
            stride = length // nframes
            if duration < nframes:
                duration = nframes
            interval = round(duration / nframes)
        labels = [str(p.p) for p in self.processes]
        if line_styles is None:
            line_styles = ['-' for p in self.processes]
        g = [p.p.growth_factors(effective) for p in self.processes]
        procs = [p[:length] for p in self.processes]
        normed = [p.normalized()[:length] for p in self.processes]
        p = [procs[:], normed[:]]
        n = length
        if not stride is None:
            for i in range(2):
                p[i] = [np.concatenate((y[:n:stride], [y[n-1]])) for y in p[i]]
        n_frames = len(p[0][0])
        lines = np.empty((2, len(p[0])), dtype=object)
        is_interactive = plt.isinteractive()
        plt.interactive(False)
        fig, ax = plt.subplots(2, sharex=True)
        # Construct lines by plotting them for the final year
        for n in range(2):
            for i in range(len(g)):
                w = p[n][i][-1] > 0
                lines[n][i], = ax[n].plot(g[i][w], p[n][i][-1][w],
                                          label=labels[i], ls=line_styles[i],
                                          lw=4, zorder=10)
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
            ax[1].set_xlabel('Effective Growth Factor')
        else:
            ax[1].set_xlabel('Nominal Growth Factor')
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Frequency')
        ax[1].set_ylabel('Proportion')
        ax[1].legend(loc='best')
        y_max = np.max(procs)
        y_lim = ax[0].get_ylim()
        if y_lim[1] < y_max:
            ax[0].set_ylim(y_lim[0], y_max)
        plt.interactive(is_interactive)

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
            out = animation.FuncAnimation(fig, animator, init_func=initializer,
                                          frames=n_frames, interval=interval,
                                          blit=True, repeat_delay=2000)
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
    Returns actual/desired - 1, with 0/0 defined equal to 1.
    
    Raises `ValueError` if the arguments do not agree everywhere in sign.
    """
    if np.shape(actual) != np.shape(desired):
        raise ValueError('Arguments are not identical in shape')
    if not np.array_equal(np.sign(actual), np.sign(desired)):
        raise ValueError('Arguments do not agree everywhere in sign')
    result = np.zeros_like(actual)
    i = desired[:] != 0
    result[i] = actual[i] / desired[i] - 1
    if absolute:
        result = np.abs(result)
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
