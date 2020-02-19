class FiniteSim(object):
    """
    Base class.
    
    Requires that `capacity' method be defined by subclass.
    """
    def __init__(self, rates, mutation_matrix, init_dist, steps_per_year=1,
                       max_size=1e9, init_size=None):
                       # init_mean=0.044, init_std=0.005):
        self.rates = rates
        self.n = len(rates.fitness)
        self.u = np.array(mutation_matrix.T)
        self.init_dist = init_dist
        self.steps_per_year = steps_per_year
        self.birth_rates = self.rates.birth / steps_per_year
        self.death_rate = self.rates.death / steps_per_year
        self.birthing = self.birth_rates * mutation_matrix
        self.max_size = max_size
        if init_size is None:
            init_size = max_size
        elif init_size > max_size:
            raise Exception('Initial pop size greater than max size')
        # self.init_mean = init_mean
        # self.init_std = init_std
        # p = normal_pdf(rates.fitness, init_mean, init_std)
        # p = (p / fsum(p)).astype(float)
        self.freqs = rand.multinomial(init_size, init_dist)
        self.n_years = 0
        self.results = np.empty((1, len(self.freqs)), dtype=int)
        self.results[self.n_years] = self.freqs


    def run(self, n_years):
        """
        SLOW!
        """
        self._allocate_memory(n_years)
        freqs = self.freqs
        for _ in range(n_years):
            for _ in range(self.steps_per_year):
                pop_size = np.sum(freqs)
                if pop_size == 0:
                    return
                expect_deaths = self.death_rate * pop_size
                #
                # Numbers of unmutated births are Poisson distributed.
                lambdas = self.birth_rates * freqs
                expect_births = fsum(lambdas)
                p = lambdas / expect_births
                #
                # Adjust expected numbers of births and deaths to keep
                # expected pop size from exceeding carrying capacity.
                expect_deaths, expect_births = \
                    self.capacity(pop_size, expect_deaths, expect_births)
                #
                # Numbers of deaths are binomially distributed.
                d = expect_deaths / pop_size
                n_deaths = [rand.binomial(n, d) for n in freqs]
                #
                # Numbers of unmutated births are Poisson distributed.
                total_births = stats.poisson.rvs(expect_births)
                n_births = rand.multinomial(total_births, p)
                #
                # Add numbers of mutated births to frequencies.
                for j in range(self.n):
                    if n_births[j] > 0:
                        freqs += rand.multinomial(n_births[j], self.u[j])
                #
                # Subtract numbers of deaths from frequencies.
                np.subtract(freqs, n_deaths, out=freqs)
            self.n_years += 1
            self.results[self.n_years] = freqs


    def __call__(self, n_years):
        """
        FAST!
        """
        self._allocate_memory(n_years)
        freqs = self.freqs
        for _ in range(n_years):
            for _ in range(self.steps_per_year):
                pop_size = np.sum(freqs)
                if pop_size == 0:
                    return
                expect_deaths = self.death_rate * pop_size
                #
                # Numbers of mutant births are Poisson distributed.
                lambdas = np.dot(self.birthing, freqs)
                expect_births = fsum(lambdas)
                p = lambdas / expect_births
                #
                # Adjust expected numbers of births and/or deaths to keep
                # expected pop size from exceeding carrying capacity.
                expect_deaths, expect_births = \
                    self.capacity(pop_size, expect_deaths, expect_births)
                #
                # Numbers of deaths are binomially distributed.
                d = expect_deaths / pop_size
                n_deaths = rand.binomial(freqs, d)
                #
                # Numbers of mutant births are Poisson distributed.
                total_births = stats.poisson.rvs(expect_births)
                n_mutants = rand.multinomial(total_births, p)
                #
                # Update the frequencies.
                np.add(freqs, n_mutants, out=freqs)
                np.subtract(freqs, n_deaths, out=freqs)
            self.n_years += 1
            self.results[self.n_years] = freqs

    def _allocate_memory(self, n):
        rows, cols = self.results.shape
        new = np.zeros((rows+n, cols), dtype=int)
        new[:rows] = self.results
        self.results = new

    def _excess(self, pop_size, expected_deaths, expected_births):
        expected_size = pop_size - expected_deaths + expected_births
        return max(0, expected_size - self.max_size)

    def __getitem__(self, which):
        return self.results[which]
    
    def __len__(self):
        return len(self.results)
    
class VariableBirthRates(FiniteSim):
    def capacity(self, pop_size, expected_deaths, expected_births):
        excess = self._excess(pop_size, expected_deaths, expected_births)
        return expected_deaths, max(0, expected_births - excess)

class VariableDeathRate(FiniteSim):
    def capacity(self, pop_size, expected_deaths, expected_births):
        """excess = self._excess(pop_size, expected_deaths, expected_births)
        return expected_deaths + excess, expected_births"""
        expected_deaths = max(expected_deaths, pop_size - self.max_size)
        return expected_deaths, expected_births
    
class LogisticBirthRates(FiniteSim):
    def capacity(self, pop_size, expected_deaths, expected_births):
        scale = max(0, 1 - (pop_size - expected_deaths) / self.max_size)
        return expected_deaths, scale * expected_births
        """scale = 1 - pop_size / self.max_size
        expected_births = expected_deaths + scale * (expected_births
                                                     - expected_deaths)
        return expected_deaths, expected_births"""