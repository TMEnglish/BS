import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gzip
import warnings
from matplotlib import animation, rc
from math import fsum


# Generate JS/HTML animations.
# HTML5 animations require (sometimes tricky) FFmpeg installation on the host.
plt.rcParams['animation.html'] = 'jshtml'


# Use the Seaborn package to generate plots.
sns.set() 
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set_palette(sns.color_palette("Set2", 4))


DEATH_RATE = 0.1
MAX_BIRTH_RATE = 0.25
BIRTH_RATE_INTERVAL = np.array([0, MAX_BIRTH_RATE])
GROWTH_RATE_INTERVAL = BIRTH_RATE_INTERVAL - DEATH_RATE
MUTATION_EFFECT_INTERVAL = np.array([-MAX_BIRTH_RATE, MAX_BIRTH_RATE])


################################################################################
#             Parameters of the Basener-Sanford experiments
################################################################################

# The spacing of points in the intervals of birth rates and growth rates
DELTA = {
            'NoneExact' : 0.0004,   # Sect 5.2
            'Gaussian'  : 0.0010,   # Sect 5.3
            'Gamma'     : 0.0005    # Sect 5.4
        }
DELTA['None'] = DELTA['NoneExact']


# I don't include year 0 in the count of years as Basener does.
N_YEARS = {
            'NoneExact' : 3500,
            'Gaussian'  : 300,
            'Gamma'     : 2500
          }
N_YEARS['None'] = N_YEARS['NoneExact']


# I don't exclude the greatest rates from the intervals as Basener does.
N_RATES = dict() 
for case, delta in DELTA.items():
    N_RATES[case] = round(MAX_BIRTH_RATE / delta) + 1



################################################################################
#          Classes for populations and evolutionary trajectories
################################################################################


class Evolution(object):
    """
    Record the evolution of a Population instance.
    
    The n-th element of the evolutionary trajectory gives the frequencies of
    fitnesses (alternatively, growth rates) in the population after n years.
    """
    def __init__(self, population, n_years=0):
        """
        Record the trajectory of `population` over `n_years` of evolution.
        """
        self.p = population
        self.annual_growth_rates = population.annual_growth_rates
        self.trajectory = np.array([self.p.get_frequencies()])
        self.label = str(population)
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
    
    def __len__(self):
        """
        Return the length of the evolutionary trajectory.
        """
        return len(self.trajectory)

    def __getitem__(self, index_or_slice):
        """
        Index or slice the evolutionary trajectory.
        """
        return self.trajectory[index_or_slice]
    
    def __str__(self):
        """
        Return the string that labels the population/process.
        """
        return self.label
    
    def normalized(self):
        """
        Returns the trajectory with each point normalized.
        """
        t = self.trajectory
        return (t.T / np.sum(t, axis=1).T).T
    
    def mean_and_variance(self):
        """
        Returns mean and variance of fitnesses at each point in the trajectory.
        """
        return mean_and_variance(self.annual_growth_rates, self.normalized())
    
    def growth_rates(self):
        """
        Returns the discretized range of growth rates.
        """
        return self.annual_growth_rates
    
    def set_label(self, label):
        """
        Change the label used in plotting.
        """
        self.label = str(label)


class WrappedTrajectory(Evolution):
    def __init__(self, trajectory, growth_rates, label=''):
        self.trajectory = np.array(trajectory, dtype=float)
        self.annual_growth_rates = np.array(growth_rates, dtype=float)
        self.label = label
       
    def __call__(self, n=None):
        raise Exception('WrappedTrajectory instances cannot be extended.')
    

class Population(object):
    """
    TO DO: doc
    """    
    def __init__(self, initial_distribution, births_redistribution,
                       label='', n_updates_per_year=1):
        """
        TO DO: doc
        """
        assert initial_distribution.n_rates == births_redistribution.n_rates
        self.initializer = initial_distribution
        self.redistribution = births_redistribution
        self.label = label
        self.n_updates_per_year = n_updates_per_year
        self._freqs = np.array(initial_distribution[:])
        self.annual_birth_rates = initial_distribution.birth_rates
        self.annual_growth_rates = self.annual_birth_rates - DEATH_RATE
        self.birth_rates = self.annual_birth_rates / n_updates_per_year
        self.birth_factors = np.exp(self.birth_rates) - 1
        self.death_rate = DEATH_RATE / n_updates_per_year
        self.death_factor = np.exp(-self.death_rate)
        self.year = 0

    def annual_update(self):
        """
        Do one year's worth of updates to the frequencies.
        
        Births distributed outside the range of fitnesses are lumped with births
        at the endpoints of the interval.
        
        If offspring are always identical to their parents in fitness (either
        because there are no mutations or because mutations have no effect on
        fitness), and a particular fitness is initially of frequency f0, then
        the frequency of that is f0 * exp(n * growth_rates) after n years
        """
        for _ in range(self.n_updates_per_year):
            self._freqs *= self.death_factor
            births = self._freqs * self.birth_factors
            self._freqs += self.redistribution(births)
        self.year += 1
    
    def get_frequencies(self, normalize=False):
        """
        Returns the frequencies of fitnesses in the current population.
        
        If `normalize` is true, proportions are returned instead of frequencies.
        """
        result = np.array(self._freqs)
        if normalize:
            result /= np.sum(result)
        return result
    
    def size(self):
        """
        Returns the size of the population, relative to its size in year 0.
        
        The size of the population is the sum of the frequencies of the discrete
        fitnesses. The size of the initial population is equated with 1. Thus if
        the current frequences sum to F, then the size of the population has
        have changed by a factor of F since year 0.
        """
        return np.sum(self._freqs)
    
    def age(self):
        """
        Returns the number of annual updates that have occurred.
        """
        return self.year
    
    def __getitem__(self, index_or_slice):
        """
        Index or slice the  frequencies of the fitnesses.
        """
        return self._freqs[index_or_slice]
        
    def __len__(self):
        """
        Returns the number of discrete growth rates (and frequencies).
        """
        return len(self._freqs)
    
    def __str__(self):
        """
        Returns the label of the population.
        """
        return self.label


class BS_Population(Population):
    """
    Override the `annual_update` method of the superclass.
    """
    BS_THRESHOLD = 1e-9
    
    def __init__(self, initial_distribution, births_redistribution, label='',
                       n_updates_per_year=1, threshold_norm=None):
        """
        Register the threshold norm and invoke the superclass initializer.
        """
        self.threshold_norm = threshold_norm
        super().__init__(initial_distribution, births_redistribution, 
                         label=label, n_updates_per_year=n_updates_per_year)

    def annual_update(self):
        """
        Treat rates as linear factors, set subthreshold frequencies to zero.
        
        As in Basener's script, births that are distributed outside the range
        of growth rates are discarded. Also, the death rate and birth rates are
        treated as linear rather than logarithmic. The error is small when the
        rates are close to zero. A question here is whether increasing the
        number of updates per year, and scaling the rates inversely, increases
        the accuracy (when there is no thresholding).
        """
        for _ in range(self.n_updates_per_year):
            births = self._freqs * self.birth_rates
            births = np.convolve(births, self.redistribution[:], mode='valid')
            self._freqs *= 1.0 - self.death_rate
            self._freqs += births
            if not self.threshold_norm is None:
                norm = self.threshold_norm(self._freqs)
                above_threshold = self._freqs >= self.BS_THRESHOLD * norm
                self._freqs *= above_threshold
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
    def __init__(self, interval, n_points, label=None):
        """
        Initialize with all probability mass on the point closest to zero.
        """
        self.label = label
        self.zero_centered = interval[0] == -interval[1] and n_points % 2 == 1
        if self.zero_centered:
            self.zero_index = n_points // 2
            d = np.linspace(0, interval[1], self.zero_index + 1)
            self.domain = np.concatenate((-d[::-1], d[1:]))
            assert len(self.domain) == n_points
        else:
            self.domain = np.linspace(interval[0], interval[1], n_points)
            self.zero_index = np.argmin(np.abs(self.domain))
            if self.domain[self.zero_index] != 0:
                warnings.warning('Zero is not in the domain')
        self.delta = (self.domain[-1] - self.domain[0]) / (len(self) - 1)
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
        if self.zero_centered:
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
    
    def normalize(self):
        n = np.argmax(self.p)
        center = self.p[n]
        left = self.p[:n]
        right_reversed = self.p[n+1:][::-1]
        total = fsum([fsum(left), fsum(right_reversed), center])
        self.p /= total
    
    def set_label(self, label):
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
        Returns n-th moment (calculated slowly, but accurately).
        """
        product = self.p * self.domain ** n
        i = np.argsort(np.abs(product))
        neg = fsum(np.where(product[i] < 0, product[i], 0))
        pos = fsum(np.where(product[i] > 0, product[i], 0))
        return neg + pos
    
    def mean_and_variance(self):
        """
        Returns (mean, variance) of rv with this distribution over the domain.
        """
        mean = self.moment(1)
        variance = self.moment(2) - mean ** 2
        return mean, variance
            
    def vlines(self, axes, x_offset=0, label=None):
        """
        Plot the distribution on the axes as vlines, return the vlines object.
        
        When plotting vertical lines for two distributions on the same axes,
        shift one plot slightly to the left with a negative `x_offset`, and the
        other slightly to the right with a positive `x_offset`.
        """
        if label is None:
            label = self.get_label()
        return axes.vlines(self.domain + x_offset, 0, self.p, label=label)
    
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
#          Initial distributions of the population over growth rates
################################################################################


class RatesDistribution(Distribution):
    """
    Base class for probability distributions over discrete growth rates.
    
    Although the primary use of this class is as a base, it can be instantiated
    directly, in which case all of the probability mass is associated with the
    growth rate 0.
    """
    def __init__(self, n_rates):
        self.n_rates = n_rates
        super().__init__(GROWTH_RATE_INTERVAL, n_rates)
        self.growth_rates = self.domain
        self.birth_rates = self.growth_rates + DEATH_RATE
        

class GaussianRates(RatesDistribution):
    """
    A discretized Gaussian distribution of probability over growth rates.
    """
    def __init__(self, n_rates, mean=0.044, std=0.005, crop=None,
                       approximate=False):
        """
        Set distribution to discretized Normal(mean, std) over growth rates.
        
        For BS replication, set `crop=11.2` and `approximate=True`.
             
        The number of discrete growth rates is `n_rates`. Contrary to what BS
        say in the article, the number is not the same for all experiments. The
        probabilities are 0 for growth rates differing from the mean by more
        than `crop` standard deviations. To suppress cropping, set `crop=None`.
        
        To calculate probabilities as Basener does, set `approximate` to true.
        The default values of the other parameters come from the BS Section 5.
        Contrary to what BS say in the article, the number of discrete growth
        rates varies from experiment to experiment. 
        """
        super().__init__(n_rates)
        self.given_mean = mean
        self.given_std = std
        self.gaussian(mean, std, approximate)
        if not crop is None:
            self.p *= np.abs(self.domain - mean) <= crop * std
        self.normalize()


################################################################################
#          Distributions of births over mutation effects on growth rate
################################################################################


class EffectsDistribution(Distribution):
    """
    Base class for probability distributions over mutation effects on fitness.
    
    The subclass initializer calls the initializer, `__init__`, sets the
    probability distribution, and then calls `finish` to complete
    initialization.
    
    Although the primary use of this class is as a base, it can be instantiated
    directly, in which case the probability of zero mutation effect is 1. This
    is useful when addressing BS Sections 5.1 and 5.2.
    """
    def __init__(self, n_rates=N_RATES['NoneExact']):
        """
        Deposits all probability mass on zero mutation effect.
        """
        self.n_rates = n_rates
        super().__init__(MUTATION_EFFECT_INTERVAL, 2 * n_rates - 1)
        
    def finish(self, gimmick=False, percent_beneficial=None, normed=True,
                     number_of_mutations=1, log_number_of_loci=0):
        """
        Finish initialization of the instance.

        1. Rebalance the distribution unless `percent_beneficial` is `None`. If
           `normed`, the result is a normalized distribution. Otherwise, the
           result is that of Basener's script.
        2. If `gimmick`, then the probability that mutation has no effect on
           fitness is set to the probability that mutation has a minimally
           deleterious effect on fitness, as in Basener's script.
        3. If `normed`, then normalize the distribution.
        4. Adjust the distribution for i.i.d. mutation effects at the loci. The
           default settings `number_of_mutations=1` and `log_number_of_loci=0`
           result in no change to the distribution.
        """
        if not percent_beneficial is None:
            self.rebalance(percent_beneficial, normed)
        if gimmick:
            self.p[self.zero_index] = self.p[self.zero_index - 1]
        if normed:
            self.normalize()
        self.iid_effects(number_of_mutations, log_number_of_loci)        
    
    def rebalance(self, percent_beneficial, normed=True):
        """
        If `normed` is false, then it is assumed, as in Basener's script, that
        the probability that mutation is beneficial and the probability that
        mutation is non-beneficial are both .5 prior to adjustment.
        """
        x = self.domain
        if normed:
            positive_mass = np.sum(self.p[x > 0])
            non_positive_mass = np.sum(self.p[x <= 0])
        else:
            positive_mass = .5
            non_positive_mass = .5
        self.p[x > 0] *= percent_beneficial / positive_mass
        self.p[x <= 0] *= (1 - percent_beneficial) / non_positive_mass
    
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
        fig = plt.figure()
        ax = fig.gca()
        vlines = self.vlines(ax, x_offset=x_offset, label=label)
        mean, variance = self.mean_and_variance()
        std = np.sqrt(variance)
        subtitle = '\nMean {0}, Standard Deviation {1}'.format(mean, std)
        ax.set_title(title + subtitle)
        ax.set_xlabel('Difference in Growth Rate of Offspring from Parent')
        ax.set_ylabel('Probability')
        return fig 
    
    def __call__(self, other, discard_excess=False):
        return self.convolve(other, discard_excess)

        
class GaussianEffects(EffectsDistribution):
    """
    
    """
    def __init__(self, n_rates=N_RATES['Gaussian'], 
                       mean=0, std=0.002, approximate=False, 
                       percent_beneficial=None, gimmick=False, normed=True, 
                       number_of_mutations=1, log_number_of_loci=0):
        """
        To replicate the experiment of BS Section 5.3, set `approximate=True`
        and `normed=False`. For more information, see the documentation of
        `EffectsDistribution.finish`.
        """
        super().__init__(n_rates)
        self.mean = mean
        self.std = std
        self.approximate = approximate
        self.gaussian(mean, std, approximate=approximate)
        self.finish(percent_beneficial=percent_beneficial,
                    normed=normed,
                    gimmick=gimmick,
                    number_of_mutations=number_of_mutations,
                    log_number_of_loci=log_number_of_loci)
    

class GammaEffects(EffectsDistribution):
    """
    Symmetrized, perhaps rebalanced Gamma distribution over mutation effects.
    """
    def __init__(self, n_rates=N_RATES['Gamma'],
                       alpha=0.5, beta=0.5/0.001, approximate=False,
                       percent_beneficial=0.001, gimmick=False, normed=True,
                       number_of_mutations=1, log_number_of_loci=0):
        """
        Sets the symmetrized, and perhaps rebalanced, Gamma distribution.
        
        To replicate the experiment of BS Section 5.4, set `approximate=True`,
        `gimmick=True` and `normed=False`. For more information, see the
        documentation of `EffectsDistribution.finish`.
        """
        super().__init__(n_rates)
        self.alpha = alpha
        self.beta = beta
        self.approximate = approximate
        self.symmetrized_gamma(alpha, beta, approximate=approximate)
        self.finish(percent_beneficial=percent_beneficial,
                    normed=normed,
                    gimmick=gimmick,
                    number_of_mutations=number_of_mutations,
                    log_number_of_loci=log_number_of_loci)



################################################################################
#                            Plots and animations
################################################################################


class CompareProcesses(object):
    """
    Container of multiple evolutionary processes under comparison.
    """
    def __init__(self, processes, subtitle='[set_subtitle]', n_years=0):
        """
        Store the evolutionary processes (identical to one another in length).
        
        The given processes are extended by the given number of years. The given
        subtitle is used in plots and animations.
        """
        assert (np.array([len(p) for p in processes]) == len(processes[0])).all()
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
    
    def animate(self, nframes=100, duration=10000, line_styles=None):
        """
        Return animation of one or more evolutionary processes.
        
        If the `nframes` is 0, then a static figure is returned instead of
        an animation.
        """
        if nframes is 0:
            stride = None
        else:
            if nframes > len(self):
                nframes = len(self)
            stride = len(self) // nframes
            if duration < nframes:
                duration = nframes
            interval = round(duration / nframes)
        processes = self.processes
        labels = [str(p) for p in processes]
        if line_styles is None:
            line_styles = ['-' for p in processes]
        g = [p.growth_rates() for p in processes]
        p = [processes[:], [p.normalized() for p in processes]]
        n = len(self)
        if not stride is None:
            for i in range(2):
                p[i] = [np.concatenate((y[:n:stride], [y[n-1]])) for y in p[i]]
                # for y in p[i]:
                    # y[y == 0] = np.nan
        n_frames = len(p[0][0])
        lines = np.empty((2, len(p[0])), dtype=object)
        is_interactive = plt.isinteractive()
        plt.interactive(False)
        fig, ax = plt.subplots(2, sharex=True)
        for n in range(2):
            for i in range(len(g)):
                w = p[n][i][-1] > 0
                lines[n][i], = ax[n].plot(g[i][w], p[n][i][-1][w], label=labels[i],
                                          ls=line_styles[i], lw=2, zorder=10)
################################################################################
        for n in range(2):
            for i in range(len(g)):
                w = p[n][i][0] > 0
                ax[n].plot(g[i][w], p[n][i][0][w], c='black', lw=0.5, alpha=0.5, zorder=3)
                w = p[n][i][-1] > 0
                ax[n].plot(g[i][w], p[n][i][-1][w], c=lines[n][i].get_c(), lw=1, alpha=0.7)
        fig.suptitle('Evolution for {0} Years{1}'.format(len(self)-1, self.subtitle))
        ax[1].set_xlabel('Malthusian Growth Factor')
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Frequency')
        ax[1].set_ylabel('Proportion')
        ax[0].legend(loc='best')
        plt.interactive(is_interactive)

        def initializer():
            for n in range(2):
                for line, x, y in zip(lines[n], g, p[n]):
                    line.set_xdata(x[y[0] > 0])
                    line.set_ydata(y[0][y[0] > 0])
                    line.set_zorder(2)
            return lines.flatten()

        def animator(i):
            for n in range(2):
                for line, x, y in zip(lines[n], g, p[n]):
                    line.set_xdata(x[y[i] > 0])
                    line.set_ydata(y[i][y[i] > 0])
                    if i == 1:
                        line.set_zorder(10)
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
        labels = np.array([str(ev) for ev in evs])
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

    
def animate(processes, labels=None, stride=10, subtitle=''):
    """
    Return animation of one or more evolutionary processes.
    
    The `evs` and the `labels` are corresponding sequences of Evolution
    instances and labels to associate with them in the animation.
    """
    if labels is None:
        labels = [str(p) for p in processes]
    assert len(processes) == len(labels)
    x = [p.growth_rates() for p in processes]
    p = [p.normalized() for p in processes]
    n = np.min([len(p) for p in processes])
    p = [np.concatenate((y[:n:stride], [y[n-1]])) for y in p]
    n_frames = len(p[0])
    lines = np.empty(len(p), dtype=object)
    is_interactive = plt.isinteractive()
    plt.interactive(False)
    fig = plt.figure();
    ax = fig.gca()
    for i in range(len(x)):
        lines[i], = ax.plot(x[i], np.zeros_like(x[i]), label=labels[i])
    for i in range(len(x)):
        ax.plot(x[i], p[i][0], c=lines[i].get_c(), lw=1, alpha=.5)
        ax.plot(x[i], p[i][-1], c=lines[i].get_c(), lw=1, alpha=.5)
    ax.set_title('Evolution for {0} Years{1}'.format(n-1, subtitle))
    ax.set_xlabel('Growth Rate = Birth Rate - Death Rate')
    ax.set_ylabel('Proportion of the Population')
    ax.legend(loc='best')
    
    def initializer():
        for line, y in zip(lines, p):
            line.set_ydata(y[0])
        return lines

    def animator(i):
        for line, y in zip(lines, p):
            line.set_ydata(y[i])
        return lines

    # Suppress warning due to Matplotlib mishandling of nan.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = animation.FuncAnimation(fig, animator, init_func=initializer,
                                    frames=n_frames, interval=20, blit=True,
                                    repeat_delay=2000)
    plt.close(fig)
    plt.interactive(is_interactive)
    return a



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
               number_of_discrete_population_fitness_values=N_RATES['Gaussian'],
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
                  number_of_discrete_population_fitness_values - 1,
                  output_path)

def bs_data(basename):
    """
    Load gzipped JSON data output by my modification of Basener's script.
    
    The `basename` excludes the .json.gz extension.
    """
    with gzip.open(basename + '.json.gz', "rb") as f:
        bs_data = json.loads(f.read().decode("ascii"))
    return bs_data
