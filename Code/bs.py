import numpy as np
import numpy.linalg as la
import numpy.random as rand
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import math
import json
import gzip
import bz2
import pickle
import warnings
from matplotlib import animation, rc
from mpmath import mp
from scipy.sparse.linalg import eigs
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from IPython.display import Image, display, HTML


# Set the default number of digits of precision in mpmath multiprecision
# operations.
mp.dps = 60


# Make some mpmath functions into quasi-ufuncs taking either scalar or
# array arguments.

mp_float = np.frompyfunc(mp.mpf, 1, 1)  # type conversion
mp_exp = np.frompyfunc(mp.exp, 1, 1)
erf = np.frompyfunc(mp.erf, 1, 1)
erfc = np.frompyfunc(mp.erfc, 1, 1)
gamma = np.frompyfunc(mp.gamma, 1, 1)
rgamma = np.frompyfunc(mp.rgamma, 1, 1) # reciprocal gamma
hyp1f1 = np.frompyfunc(mp.hyp1f1, 3, 1) # confluent hypergeometric 1_F_1
hyperu = np.frompyfunc(mp.hyperu, 3, 1) # confluent hypergeometric 2_F_2


# Provide alternatives for some NumPy functions that do not handle
# mpmath's multiprecision floating-point type.

def linspace(a, b, nsteps):
    """
    Tries to return result of NumPy linspace, falls back on mpmath version.
    """
    try:
        return np.linspace(a, b, nsteps)
    except:
        return np.array(mp.linspace(a, b, nsteps))

def exp(x):
    """
    Tries to return result of NumPy exp, falls back on mpmath version.
    """
    try:
        return np.exp(x)
    except:
        return mp_exp(x)

    
# Generalized fsum and frexp work with float and multiprecision float.

def fsum(a):
    """
    Return either mp.fsum(a) or math.fsum(a), depending on type of `a`.
    """
    if type(a[0]) is mp.mpf:
        return mp.fsum(a)
    else:
        return math.fsum(a)

def frexp(x):
    """
    Return either mp.frexp(x) or math.frexp(x), depending on type of `x`.
    """
    if type(x) is mp.mpf:
        return mp.frexp(x)
    else:
        return math.frexp(x)
    

# Select type of animation.
# HTML5 animations require FFmpeg installation with non-default settings.
# plt.rcParams['animation.html'] = 'jshtml'
plt.rcParams['animation.html'] = 'html5'


# Use the Seaborn package to generate plots.
sns.set()
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
sns.set_style("darkgrid", {"axes.facecolor": ".92"})
sns.set_palette(sns.color_palette("Set2", 4))


class Factors(object):
    """
    REPLACED by class Rates. Perhaps useful in replication of BS results.
    
    Stores death, birth, and growth factors, as well as mutational effects.
    """
    @classmethod
    def construct(cls, min_fitness=-0.1, max_fitness=0.1, bin_width=5e-4):
        """
        Alternative constructor.
        """
        basetype = type((max_fitness - min_fitness) / bin_width)
        max_fitness = basetype(max_fitness)
        n_classes = round((max_fitness - min_fitness) / bin_width) + 1
        return Factors(n_classes, death=-min_fitness, max_growth=max_fitness)

    def __init__(self, n_classes, death=0.1, max_growth=0.15, exclude_max=False):
        """
        Set growth and birth factors, as well as mutational effects.
        
        With default arguments, the calculated values are exactly equal to
        those calculated in Basener's code. The supplied value of `n_classes` is
        the number of fitness classes. Set `exclude_max` to true in order to
        follow Basener in excluding the upper endpoint of the fitness interval.
        When including the upper endpoint, add one to the number of types
        (e.g., the 500 fitness classes of Basener and Sanford become 501).
        
        The type of all floating point numbers of this object is determined by
        `type(max_growth + death)`. Put simply, to work in multiprecision
        floating point, supply a number of that type as one of the arguments.
        """
        basetype = type(max_growth + death)
        death = basetype(death)
        max_growth = basetype(max_growth)
        self.n_classes = n_classes
        self.death = death
        self.max_growth = max_growth
        self.exclude_max = exclude_max
        if exclude_max:
            self.delta = (max_growth + death) / n_classes
            self.growth = linspace(-death, max_growth, n_classes+1)[:-1]
        else:
            self.delta = (max_growth + death) / (n_classes - 1)
            self.growth = linspace(-death, max_growth, n_classes)
        self.growth = convert(self.growth, basetype)
        self.birth = self.growth + death
        assert self.birth[0] == 0
        self.effects = np.concatenate((-self.birth[::-1], self.birth[1:]))
        
    def convert(self, basetype):
        """
        Converts all floating-point members to given type.
        """
        self.death = basetype(self.death)
        self.max_growth = basetype(self.max_growth)
        self.delta = basetype(self.delta)
        self.growth = convert(self.growth, basetype)
        self.birth = convert(self.birth, basetype)
        self.effects = convert(self.effects, basetype)

        

########################################################################
#             Parameters of the Basener-Sanford experiments
########################################################################

n_classes = {
            'NoneExact' : 626,   # Sects 5.1, 5.2
            'Gaussian'  : 251,   # Sect 5.3
            'Gamma'     : 501    # Sect 5.4
        }
n_classes['None'] = n_classes['NoneExact']


# I don't include year 0 in the count of years as Basener does.
N_YEARS = {
            'NoneExact' : 3500,
            'Gaussian'  : 300,
            'Gamma'     : 2500
          }
N_YEARS['None'] = N_YEARS['NoneExact']



########################################################################
#                     Class for matrix of derivatives
########################################################################


class Derivative(object):
    
    def __init__(self, factors, mutation_probs, norm=True):
        self.factors = factors
        n = len(factors.growth)
        self.W = np.empty((n, n), dtype=type(factors.birth[0]))
        cols = [mutation_probs[j:j+n] for j in range(n)][::-1]
        sums = [math.fsum(col) for col in cols]
        for j in range(n):
            if norm:
                self.W[:, j] = factors.birth[j] / sums[j] * cols[j]
            else:
                self.W[:, j] = factors.birth[j] * cols[j]
            self.W[j, j] -= factors.death
    
    def __call__(self, ignored_time, state):
        """
        Returns vector of derivatives; for use by ODE solvers.
        """
        return np.dot(self.W, state)
        

class AltDerivative(object):
    
    def __init__(self, factors, mutation_probs, norm=True):
        self.factors = factors
        n = len(factors.growth)
        self.W = np.empty((n, n), dtype=type(factors.birth[0]))
        cols = [mutation_probs[j:j+n] for j in range(n)]
        sums = [math.fsum(col) for col in cols]
        for j in range(n):
            if norm:
                self.W[:, j] = factors.birth[j] / sums[j] * cols[j]
            else:
                self.W[:, j] = factors.birth[j] * cols[j]
            self.W[j, j] -= factors.death
        self.cols = cols
        self.sums = sums
    
    def __call__(self, ignored_time, state):
        """
        Returns vector of derivatives; for use by ODE solvers.
        """
        return np.dot(self.W, state)
        



########################################################################
#          Classes for populations and evolutionary trajectories
########################################################################

class Evolution(object):
    """
    Record the evolution of a Population instance.
    
    The n-th element of the evolutionary trajectory gives the frequencies
    of fitnesses (alternatively, growth factors) in the population after
    n epochs.
    """
    def __init__(self, population, n_epochs=0, years_per_epoch=1,
                 x_stride=1, use_ode_solver=False):
        """
        Records trajectory of `population` over `n_epochs` of evolution.
        """
        if use_ode_solver and not population.norm is None:
            raise Exception('Cannot use ODE solver when thresholding')
        self.use_ode_solver = use_ode_solver
        self.p = population
        self.trajectory = np.array([self.p[::x_stride]])
        self.sums = np.array([np.sum(self.p[:])])
        self.years_per_epoch = years_per_epoch
        self.xstride = x_stride
        self(n_epochs)
   
    def __call__(self, n_epochs=1):
        """
        Extends evolutionary trajectory by the given number of epochs.
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
        if self.use_ode_solver:
            extension, bias = self.p.ode_solver(n_epochs, self.years_per_epoch)
            extension *= 2 ** -bias                    # TO DO: Avoid overflow
            self.trajectory[n:] = extension[:, ::self.xstride]
            self.sums[n:] = np.sum(extension, axis=1)
        else:
            for i in range(n, n + n_epochs):
                for _ in range(self.years_per_epoch):
                    self.p.annual_update()
                self.trajectory[i] = self.p[::self.xstride]
                self.sums[i] = np.sum(self.p[:])
                
    def set_trajectory(self, trajectory):
        """
        Sets trajectory and associated sums, with downsampling.
        """
        t_stride = self.years_per_epoch
        self.trajectory = np.array(trajectory[::t_stride, ::self.xstride])
        self.sums = np.sum(trajectory[::t_stride], axis=1)
        
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
        t = self.trajectory[begin:end:stride]
        return (t.T / self.sums[begin:end:stride]).T
    
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
        raise Exception('BS_Evolution needs to set new members of Evolution')
        self.p = BS_Population(bs, label)
        self.trajectory = bs['Psolution']

        
class Population(object):
    def __init__(self, initial_freqs, mutations, steps_per_year=1,
                       norm=None, threshold=1e-9, zero_forever=False,
                       lossy=False, label='', matrix=True, bias=0):
        if zero_forever and norm is None:
            raise Exception('Must set `norm` for `zero_forever` condition')
        self.initial_freqs = initial_freqs
        self.freqs = np.array(initial_freqs)
        self.births = np.empty_like(self.freqs)
        self.annual_factors = initial_freqs.factors
        self.death_factor = (self.annual_factors.death + bias) / steps_per_year
        self.birth_factors = (self.annual_factors.birth + bias) / steps_per_year
        self.steps_per_year = steps_per_year
        self.mutations = mutations
        if matrix:
            self.birthing = mutations.matrix(lossy) * self.birth_factors
            m = [mp.log(mp.fsum(col) + 1 - self.death_factor)
                     for col in self.birthing.T]
            self.effective_growth = np.multiply(m, self.steps_per_year)
            self.effective_birth = self.effective_growth - self.effective_growth[0]
        else:
            """HACK HACK HACK"""
            self.effective_growth = self.annual_factors.growth
            self.effective_birth = self.annual_factors.birth
            """HACK HACK HACK"""
        self.norm = norm
        self.threshold = threshold
        self.zero_forever = zero_forever
        self.lossy = lossy
        self.label = label
        self.zero = self.freqs[0] * 0
        self.log_scalar = 0
        
    def eigen(self):
        """
        Returns the eigenvalues and eigenvectors of matrix W (BS Section 4).
        
        The calculation is not redone if it's been done previously.
        """
        try:
            return self.e_values, self.e_vectors
        except:
            pass
        n = len(self)
        W = self.birthing
        W[np.diag_indices(n)] -= self.death_factor
        try:
            self.e_values, self.e_vectors = la.eig(W)
        except:
            A = mp.matrix(W.tolist())
            e_values, e_vectors = mp.eig(A)
            self.e_values = np.array(e_values)
            self.e_vectors = np.array(e_vectors.tolist())
        W[np.diag_indices(n)] += self.death_factor
        return self.e_values, self.e_vectors
        
    def equilibrium(self, v0=None, maxiter=None, npower=1000):
        """
        Returns the equilibrium distribution of the population over fitnesses.
        
        The equilibrium distribution is the real part of the eigenvector
        corresponding to the greatest real eigenvalue, scaled to unit length.
        
        The initial vector `v0` and and the maximum number of iterations 
        `maxiter` are passed under the same names to the `eigs` function of
        scipy.sparse.linalg. In limited testing, it appears that the solution
        returned by `eigs` can be improved slightly by subsequent application
        of the power method for `npower` iterations.
        """
        n = len(self)
        W = self.birthing
        W[np.diag_indices(n)] -= self.death_factor
        _, e_vectors = eigs(W, 1, which='LR', v0=v0, maxiter=maxiter)
        v = e_vectors[:, 0].real
        v /= math.fsum(v)
        for _ in range(npower):
            v += np.dot(W, v)
            v /= math.fsum(v)
        W[np.diag_indices(n)] += self.death_factor
        return v
        

    def solve_for_constraints(self):
        """
        TO DO: This needs high-precision arithemetic.
        """
        try:
            return self.c
        except:
            pass
        _, e_vectors = self.eigen()
        self.c = la.solve(e_vectors, self.initial_freqs)
        return self.c
        
    def solution_using_eigenvectors(self, t, normed=False):
        """
        TO DO: This may need high-precision arithmetic.
        """
        e_values, e_vectors = self.eigen()
        c = self.solve_for_constraints()
        P_t = np.dot(e_vectors, c * np.exp(t * e_values)).real
        if normed:
            P_t /= float(mp.fsum(P_t))
        return P_t
    
    def birthTo_rates(self, P):
        """
        Returns the rates of birth TO organisms in fitness classes.
        
        The birth-TO rates correspond to B^out of BS Equation 3.4.
        """
        # Multiply vectors of frequencies and birth factors pointwise.
        return np.multiply(P, self.birth_factors)
    
    def birthOf_rates(self, P):
        """
        Returns the rates of birth OF organisms in fitness classes.

        The birth-OF rates correspond to B^in of BS Equation 3.4.
        """
        try:
            # Multiply `birthing` matrix [b_j f_ij] by frequency vector.
            births_of = np.dot(self.birthing, P)
        except:
            # Convolve the birth-TO rates of the classes with the mutational
            # effects distribution in order to obtain the rates of birth OF
            # organisms of the fitness classes.
            births_to = self.birthTo_rates(P)
            births_of = self.mutations(births_to, self.lossy)
        return births_of
    
    def evaluate_theorem(self, P=None):
        """
        Returns values of the two right-hand-side terms Eq. 3.4 in BS's theorem.
        
        The sum of the two values is the rate of change in mean fitness for a
        population with frequency distribution `P` over fitnesses. The first of
        the two is the rate of change due to selection, and the second is the
        rate of change due to mutation.
                        
        The default value of `P` is the frequency distribution stored by this
        `Population` instance.
        """
        if P is None:
            P = self.freqs
        m = self.birth_factors - self.death_factor
        mean_m, var_m = mean_and_variance(m, P / math.fsum(P))
        B_in = self.birthOf_rates(P)
        B_out = self.birthTo_rates(P)
        return var_m, np.dot(B_in - B_out, m - mean_m) / math.fsum(P)
        
            
    def __call__(self, t, P):
        """
        Returns dP/dt of BS Equation 3.2. For use by numerical IVP solver.
        """
        rates = self.birthOf_rates(P)
        rates -= self.death_factor * P
        return rates
    
    def jacobian(self, t, P):
        """
        Returns matrix of gradients for BS Equation 3.2 Used by IVP solver.
        """
        J = np.multiply(self.birthing, P)
        J[np.diag_indices(len(self))] = self.death_factor * P
        return J
    
    def ode_solver(self, n_epochs, years_per_epoch):
        """
        Returns scaled frequencies at ends of epochs, along with log-scalar.
        
        To obtain the correct frequencies, divide the returned frequencies
        by 2 raised to the power of the returned log-scalar.
        """
        # Scale frequencies by power of 2 to make the maximum exponent
        # about 512. This improves accuracy.
        log_scalar = 512 - round(np.log2(max(self.freqs)))
        self.log_scalar += log_scalar
        self.freqs[:] *= 2 ** log_scalar
        # The time unit is 1 /`steps_per_year`
        duration = n_epochs * years_per_epoch * self.steps_per_year
        times = np.linspace(0, duration, duration + 1)
        solution = odeint(self, self.freqs, times, rtol=1e-13, atol=1e-11)
        self.freqs[:] = solution[-1]
        stride = years_per_epoch* self.steps_per_year
        return solution[1::stride], self.log_scalar
            
    def update(self):
        """
        Update frequencies for one time step.
        """
        births = self.birthOf_rates(self.freqs)
        self.freqs *= 1 - self.death_factor
        self.freqs += births
        
    def annual_update(self):
        """
        Use Euler's method to solve for the next year's frequencies.
        """
        if self.zero_forever:
            zeroed = self.freqs <= self.threshold * self.norm(self.freqs)
            self.freqs[zeroed] = self.zero
        for _ in range(self.steps_per_year):
            self.update()
            if not self.norm is None:
                small = self.freqs <= self.threshold * self.norm(self.freqs)
                if self.zero_forever:
                    np.logical_or(small, zeroed, out=zeroed)
                    self.freqs[zeroed] = self.zero
                else:
                    self.freqs[small] = self.zero
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
        
        If `normed` is true, proportions are returned instead of frequencies.
        """
        if normed:
            return self.freqs / self.size()
        return self.freqs
    
    def size(self):
        """
        Returns an accurate sum of the frequencies.
        """
        return mp.fsum(self.freqs)
    
    def normalize(self):
        """
        Divides the frequencies by their sum.
        """
        self.freqs[:] = self.freqs / mp.fsum(self.freqs)
    
    def shift(self, n):
        """
        Shifts the frequency distribution by `n` bins.
        
        If `n` is positive, the shift is to the right, and if `n` is negative,
        the shift is to the left. Vacated bins have their frequencies set to 0.
        """
        if n > 0:
            self.freqs[n:] = self.freqs[:-n]
            self.freqs[:n] = self.zero
        else:
            self.freqs[:n] = self.freqs[-n:]
            self.freqs[n:] = self.zero
    
    def mean(self, effective=False):
        return moment(self.freqs, self.growth_factors(effective), 1)
    
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
        self.steps_per_year = 1
        self.birthing = bs['MP']
        self.norm = np.max
        self.lossy = True
        self.label = label
        self.zero = 0



########################################################################
#                  Base class for discrete distributions
########################################################################


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
        basetype = type(domain[0])
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
        self.p = np.array(self.domain)
        self.p[:] = basetype(0)
        self.p[self.zero_index] = basetype(1)
        self.endpoints = linspace(domain[0] - delta / 2,
                                  domain[-1] + delta / 2, n_points + 1)
        
    def convert(self, newtype):
        """
        Convert type of all floating-point components to `newtype`.
        """
        self.delta = newtype(self.delta)
        self.domain = convert(self.domain, newtype)
        self.p = convert(self.p, newtype)
        
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
        
    def gaussian(self, mean, std, density=False):
        """
        Set distribution to discretized Gaussian.
        
        The result is not necessarily normalized. If the standard deviation
        `std` is zero, then unit mass is associated with the domain point
        closest to the given `mean`.
        
        Calculations are done in multiprecision floating point. The base type
        of the distribution is preserved. If the base type is ordinary float,
        then it may be that small probabilities are rounded to zero.
        """
        if std == 0:
            i = np.argmin(np.abs(self.domain - mean))
            self.p[i] = 1
            return
        
        # Define standard-normal probability density function.
        def pdf(z):
            return exp(-0.5 * z ** 2)
        
        if density:
            # Follow BS in setting probabilities proportional to densities
            # at domain points.
            self.p[:] = pdf((self.domain - mean) / std) * self.delta
        else:
            # Difference the cumulative distribution function at endpoints
            # of subintervals centered on equispaced points in domain. Work
            # with double the precision `mp.dps`.
            with mp.workdps(2 * mp.dps):
                args = (self.endpoints - mp.mpf(mean)) / (std * mp.sqrt(2))
                cdf = 0.5 * (1 + erf(args))
                self.p[:] = cdf[1:] - cdf[:-1]
                    
        if self.zero_centered and mean == 0:
            # Make the distribution perfectly symmetric.
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
        Divides all probabilities by their sum.
        """
        self.p[:] = self.p / self.norm()  # preserve base type
    
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

    def __setitem__(self, index_or_slice, value):
        """
        Assign value to element or slice of the distribution.
        """
        self.p[index_or_slice] = value



########################################################################
#          Initial distributions of the population over growth factors
########################################################################


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
    
    def convert(self, newtype):
        """
        Convert type of floating-point components to `newtype`.
        """
        super().convert(newtype)
        self.factors.convert(newtype)

        

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
        
        If the standard deviation `std` is zero, then unit probability mass is
        associated with the fitness closest to the mean.
        
        The resulting discrete distribution is normalized in all cases.
        """
        super().__init__(factors)
        self.given_mean = mean
        self.given_std = std
        self.gaussian(mean, std, density)
        if std > 0 and crop < np.inf:
            z = (self.domain - mean) / std
            self.p[np.abs(z) > crop] = 0
        self.normalize()


########################################################################
#          Distributions of births over mutation effects on growth rate
########################################################################


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
    
    def convert(self, newtype):
        """
        Convert type of floating-point components to `newtype`.
        """
        super().convert(newtype)
        self.factors.convert(newtype)
    
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
            for j in range(n):
                c[:, j] = c[:, j] / mp.fsum(c[:, j])
        return c
    
    def gimmick(self):
        """
        Assigns the probability of minimally delerious effect to zero effect.
        """
        zero = len(self) // 2
        self.p[zero] = self.p[zero - 1]
    
    def reweight(self, percent_beneficial):
        """
        TO DO: ERRONEOUS
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

    def probability_neutral(self):
        """
        Returns the probability that mutation has zero effect on fitness.
        """
        return self.p[self.zero_index]
    
    def probability_deleterious(self):
        """
        Returns the probability that mutation has a negative effect on fitness.
        """
        return mp.fsum(self.p[:self.zero_index])

    def probability_advantageous(self):
        """
        Returns the probability that mutation has a positive effect on fitness.
        """
        return mp.fsum(self.p[:self.zero_index:-1])
        
    def deleterious_to_advantageous(self):
        """
        Returns ratio of probabilities of deleterious and advantageous effects.
        """
        advantageous = self.probability_advantageous()
        if advantageous > 0.0:
            return self.probability_deleterious() / advantageous
        return np.inf
    
    def iid_effects(self, number_of_mutations=1, log_number_of_loci=0,
                          truncate_self_convolution=False):
        """
        TO DO: One-line description.
        
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



########################################################################
#                            Plots and animations
########################################################################


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
    
    def Xanimate(self, nframes=100, duration=10000, effective=False, dpi=600):
        """
        OBSOLETE: Returns animation of evolutionary processes.
        
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
        normed = [p.normalized(end=length, stride=stride) 
                      for p in self.processes]
        normed_last = [p.normalized(begin=length-1, end=length) 
                           for p in self.processes]
        normed = [np.concatenate((a, b)) for a, b in zip(normed, normed_last)]
        procs = [np.concatenate((a, b)) for a, b in zip(procs, procs_last)]
        p = [normed[:], procs[:]]
        n_frames = len(p[0][0])
        lines = np.empty((2, len(p[0])), dtype=object)
        is_interactive = plt.isinteractive()
        plt.interactive(False)
        
        # Assume that animation will be scaled down elsewhere.
        # Set aspect ratio to 16:9, preserving default width of figures.
        # Note that the figure size is expressed in inches, and that font
        # sizes and and line widths are expressed in points (72 per inch).
        
        size = (6.4, 3.6)
        fig, ax = plt.subplots(2, sharex=True, dpi=dpi, figsize=size)
        
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
                           lw=1, alpha=1, ls=lines[n][i].get_ls())
        
        title = 'Evolution for {0} Years'.format(n_years)
        fig.suptitle(title)
        ax[0].set_title(self.subtitle)
        if effective:
            ax[1].set_xlabel('Effective Fitness')
        else:
            ax[1].set_xlabel('Fitness')
        ax[1].set_yscale('log')
        ax[1].set_ylabel('Scaled Frequency')
        ax[0].set_ylabel('Relative Frequency')
        ax[0].legend(loc='best')
        plt.interactive(is_interactive)
        
        def adjust_ylim(ax, data):
            for d in data:
                raveled = np.ravel(d)
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
    
    def animate(self, nframes=100, duration=10000, effective=False, dpi=600):
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
        # Introduce stride in views of unnormalized
        procs = [p[:length:stride] for p in self.processes] 
        # Views of last unnormalized frames
        procs_last = [p[length-1:length] for p in self.processes]
        normed = [p.normalized(end=length, stride=stride) 
                      for p in self.processes]
        normed_last = [p.normalized(begin=length-1, end=length) 
                           for p in self.processes]
        normed = [np.concatenate((a, b)) for a, b in zip(normed, normed_last)]
        procs = [np.concatenate((a, b)) for a, b in zip(procs, procs_last)]
        p = [normed[:], procs[:]]
        n_frames = len(p[0][0])
        lines = np.empty((2, len(p[0])), dtype=object)
        is_interactive = plt.isinteractive()
        plt.interactive(False)
        
        # Assume that animation will be scaled down elsewhere.
        # Set aspect ratio to 16:9, preserving default width of figures.
        # Note that the figure size is expressed in inches, and that font
        # sizes and and line widths are expressed in points (72 per inch).
        
        size = (6.4, 3.6)
        fig, ax = plt.subplots(2, sharex=True, dpi=dpi, figsize=size)
        
        # Construct lines by plotting them for the INITIAL time
        for n in range(2):
            for i in range(len(g)):
                w = p[n][i][0] > 0  # Boolean indices of support
                lines[n][i], = ax[n].plot(g[i][w], p[n][i][0][w],
                                          label=labels[i], lw=4, zorder=10,
                                          ls=self.linestyles[i],
                                          c=self.colors[i],)
        
        # Plot thin black lines for initial distributions
        # Plot transparent lines for final distributions. This is a trick to
        # get good legend placement.
        for n in range(2):
            for i in range(len(g)):
                w = p[n][i][0] > 0
                ax[n].plot(g[i][w], p[n][i][0][w], c='black', lw=1, alpha=0.5,
                           zorder=11)
                w = p[n][i][-1] > 0
                ax[n].plot(g[i][w], p[n][i][-1][w], c=lines[n][i].get_c(),
                           lw=1, alpha=0, ls=lines[n][i].get_ls())
        
        
        title = 'Evolution for {0} Years'.format(n_years)
        fig.suptitle(title)
        ax[0].set_title(self.subtitle)
        if effective:
            ax[1].set_xlabel('Effective Fitness')
        else:
            ax[1].set_xlabel('Fitness')
        ax[1].set_yscale('log')
        ax[1].set_ylabel('Scaled Frequency')
        ax[0].set_ylabel('Relative Frequency')
        ax[0].legend(loc='best')
        plt.interactive(is_interactive)
        
        def adjust_ylim(ax, data):
            for d in data:
                raveled = np.ravel(d)
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
            """
            for n in range(2):
                for line, x, y in zip(lines[n], g, p[n]):
                    line.set_xdata(x[y[0] > 0])
                    line.set_ydata(y[0][y[0] > 0])
                    line.set_lw(1)
            """
            return lines.flatten()

        def animator(i):
            for n in range(2):
                for line, x, y in zip(lines[n], g, p[n]):
                    line.set_xdata(x[y[i] > 0])
                    line.set_ydata(y[i][y[i] > 0])
                    """
                    if i == 1:
                        line.set_lw(4)
                    """
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



########################################################################
#                              Utility functions
########################################################################


def ivp_solution(derivative, initial_frequencies, times, max_step=1/128):
    """
    Use the Runge-Kutta 4(5) method to solve for frequencies at specified
    `times`, given the frequencies at the first of those times.
    """
    interval = (times[0], times[-1] + max_step)
    result = solve_ivp(derivative, interval, initial_frequencies,
                       method='RK45', t_eval=times, max_step=max_step,
                       rtol=1e-13, atol=1e-11)
    return result.y.T, result


def save_and_display(figure, filename, format='png', dpi=600, close=True):
    """
    Displays figure after saving it with the given attributes.
    
    If `close` is true, then the plot of the figure is closed.
    """
    figure.savefig(filename, format=format, dpi=dpi)
    display(Image(filename=filename))
    if close:
        plt.close(figure)

    
def save_video(anim, filename, fps=None):
    """
    Write animation `anim` to file, setting frames per second as specified.
    
    Tested only for `filename` with extension `.mp4`.
    """
    if fps is None:
        fps = round(1000/anim._interval)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps)
    anim.save(filename, writer)

def display_video(filename):
    html = """
           <video width="100%" controls autoplay loop>
               <source src="{0}" type="video/mp4">
           </video>
           """.format(filename)
    display(HTML(html))

def save_and_display_video(anim, filename, fps=None):
    save_video(anim, filename, fps)
    display_video(filename)

    
def convert(iterable, new_basetype):
    """
    Returns Numpy array with objects in iterable converted to new type.
    """
    return np.array([new_basetype(x) for x in iterable])


def accurate_sum(a):
    """
    Returns sum of elements in array `a` as multiprecision float.
    """
    return mp.fsum(a)


def moment(frequency, x, n):
    """
    Returns n-th raw moment as multiprecision float.
    """
    x = mp_float(x)
    x **= n
    return mp.fsum(np.multiply(frequency, x)) / mp.fsum(frequency)


def mean(frequency, x):
    """
    Returns mean for `frequency` distribution over `x`.
    
    Sums are calculated accurately using `fsum`.
    """
    return fsum(frequency * x) / fsum(frequency)


def mean_var(frequency, x):
    """
    Returns mean and variance for `frequency` distribution over `x`.
    
    Sums are calculated accurately using `fsum`.
    """
    norm = fsum(frequency)
    mom1 = fsum(frequency * x) 
    mom2 = fsum(frequency * x**2)
    var = (mom2 - mom1**2 / norm) / norm
    mean = mom1 / norm
    return mean, var

   
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


def regress_through_origin(x, y):
    """
    Return slope obtained by regression through the origin.
    """
    sum_x = fsum(x)
    sum_y = fsum(y)
    n = len(x)
    cov = fsum(x * y) - sum_x * sum_y / n
    var = fsum(x * x) - sum_x * sum_x / n
    return cov / var 
    

def relative_error(actual, desired, absolute=False):
    """
    Returns (actual - desired) / desired, with 0/0 defined equal to 0.
    """
    if np.shape(actual) != np.shape(desired):
        raise ValueError('Arguments are not identical in shape')
    result = np.zeros_like(desired)
    key = actual != desired
    if np.any(desired[key] == 0):
        return np.inf
    result[key] = (actual[key] - desired[key]) / desired[key]
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
    Returns slice excluding zeros in the tails of distribution p.
    """
    positive = p > 0
    a = np.argmax(positive)
    b = len(positive) - np.argmax(positive[::-1])
    return slice(a, b, None)


def normal_pdf(x, mean='0', std='1'):
    """
    Returns result of multiprecision calculation of Gaussian density.
    """
    x = mp_float(x)
    return exp(-0.5 * ((x - mp_float(mean)) / mp_float(std)) ** 2)


def normal_cdf(x, mean='0', std='1'):
    """
    Returns result of multiprecision calculation of Gaussian CDF.
    
    The argument `x` may be an array.
    """
    x = mp_float(x)
    arg = (x - mp_float(mean)) / (mp_float(std) * 2)
    return 0.5 * (1 + erf(arg))


def normal_ccdf(x, mean='0', std='1'):
    """
    Returns Gaussian complementary CDF value as multiprecision float.
    
    The argument `x` may be an array.
    """
    x = mp_float(x)
    arg = (x - mp_float(mean)) / (mp_float(std) * 2)
    return 0.5 * erfc(arg)


def binned_normal(bin_walls, mean='0', std='1', normed=True):
    """
    Returns normal masses for bins with given `bin_walls`.
    
    The probability masses are multiprecision floating point numbers.
    """
    # Calculate bin masses by differencing the CDF.
    cdf = normal_cdf(bin_walls, mean, std)
    per_cdf = cdf[1:] - cdf[:-1]
    #
    # Calculate bin masses by differencing the complementary CDF.
    ccdf = normal_ccdf(bin_walls, mean, std)
    per_ccdf = ccdf[:-1] - ccdf[1:]
    #
    # In the lower tail, differencing the CDF results in greater, and
    # more accurate, masses than differencing the complementary CDF. The
    # opposite holds in the upper tail. The maximum of the calculated
    # masses for a bin is the more accurate of the two. 
    p = np.maximum(per_cdf, per_ccdf)
    if normed:
        p /= fsum(p)
    return p


def gamma_pdf(x, alpha='0.5', beta='500'):
    """
    Returns Gamma probability density as multiprecision float.
    
    The intended use is in numerical integration, to check other
    calculations. Integrate with an expression like
    
        `mp.quad(gamma_density, [a, b])`,
        
    where `a` and `b` are the limits of the integral.
    """
    x = mp_float(x)
    alpha = mp_float(alpha)
    beta = mp_float(beta)
    result = beta ** alpha / gamma(alpha) * x ** (alpha - 1)
    result *= mp_exp(-beta * mp_float(x))
    return result

    
def gamma_cdf(x, alpha='0.5', beta='500'):
    """
    Returns multiprecision value of the Gamma CDF.
    """
    return regularized_lower_incomplete_gamma(x, alpha, beta)

    
def gamma_ccdf(x, alpha='0.5', beta='500'):
    """
    Returns multiprecision value of the Gamma complementary CDF.
    """
    return regularized_upper_incomplete_gamma(x, alpha, beta)


def regularized_upper_incomplete_gamma(x, alpha=0.5, beta=500, 
                                       allow_special=True):
    """
    Returns multiprecision value of the Gamma complementary CDF.

    The result is the value of the regularized (upper) incomplete gamma
    function with arguments alpha and z = beta * x. The Boolean value
    of `allow_special` determines whether special cases get special
    handling.
    """
    a = mp_float(alpha)
    z = mp_float(beta) * mp_float(x)
    #
    # Use of erfc when alpha is 0.5 improves speed and accuracy.
    if a == 0.5 and allow_special:
        return erfc(z ** 0.5)
    #
    # http://functions.wolfram.com/GammaBetaErf/Gamma2/26/01/03/0001/
    # gives the formula for the incomplete gamma function (with U
    # denoting the the Tricomi confluent hypergeometric function).
    return rgamma(a) * exp(-z) * hyperu(1 - a, 1 - a, z)

        
def regularized_lower_incomplete_gamma(x, alpha=0.5, beta=500,
                                       allow_special=True):
    """
    Returns multiprecision value of the Gamma CDF.
    
    We don't expect to experiment with values of `alpha` other than
    0.5, which is a very nice case: the standard `erf()` is applied to
    the square root of `beta * x`. In other cases, the calculations
    take some seconds to complete.
    """
    x = mp_float(x)
    alpha = mp_float(alpha)
    beta = mp_float(beta)
    z = np.multiply(x, beta)
    #
    # Use of erfc when alpha is 0.5 improves speed and accuracy.
    if alpha == 0.5 and allow_special:
        return erf(z ** 0.5)
    unregularized = hypergeometric_lower_incomplete_gamma(alpha, z)
    return mp.rgamma(alpha) * unregularized

    
def hypergeometric_lower_incomplete_gamma(s, z):
    """
    Returns value of the lower incomplete gamma function.
    
    The result is not regularized. Kummer's confluent hypergeometric
    function is used in the calculation. (See "Incomplete gamma
    function" in Wikipedia for details.)

    To calculate the cumulative distribution function of the Gamma
    distribution, set `s` equal to alpha, and `z` equal to the product
    of beta and the array of x values for which results are desired.
    Multiply the result by `mp.rgamma(alpha)`, the reciprocal Gamma
    function, to regularize.
    """
    s = mp_float(s)
    z = mp_float(z)
    return z ** s / s * hyp1f1(s, s + 1, -z)


########################################################################
#               Generate command to run Basener's JavaScript
########################################################################


def bs_command(percentage_of_mutations_that_are_beneficial=0.001,
               mutation_distribution_type='Gaussian',
               population_size='Finite',
               number_of_years=N_YEARS['Gaussian'],
               number_of_discrete_population_fitness_values=n_classes['Gaussian'],
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
