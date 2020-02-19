class Equilibria(object):
    """
    NEED
    
    ... rows corresponding to probability distributions over mutational
    effects, and columns corresponding to upper limits on fitness.
    """
    bin_width = mp_float('5e-4')
    
    def __init__(self, comp_cdf, weights, weight_labels, fitness_limits,
                       death_rate='0.1'):
        """
        NEED
        """
        w = Equilibria.bin_width
        d = mp_float(death_rate)
        self.death_rate = d
        self.weight_labels = np.array(weight_labels)
        self.limits = np.array(fitness_limits)
        birth_rate_limits = mp_float(fitness_limits) + d
        self.rates = [Rates(limit, d, w) for limit in birth_rate_limits]
        nbins = np.max([len(r.effects) for r in self.rates])
        self.probs = [binned_mixture(comp_cdf, nbins, weight, w)
                          for weight in weights]
        #
        # Create arrays to hold results, with rows corresponding to
        # probability distributions over mutational effects, and columns
        # corresponding to upper limits on fitness.
        m = len(self.probs)
        n = len(self.rates)
        self.eq = np.empty((m, n), dtype=object)
        self.mean = np.empty((m, n))
        self.var = np.empty((m, n))
        #
        # There is one slope for each row, obtained by regression of
        # means on variances (through the origin).
        self.slope = np.empty(m)
        #
        # Now fill the arrays with results.
        self._calculate()
        
    def _calculate(self):
        # Calculate equilibria along with means/variances of fitnesses.
        mean = self.mean
        var = self.var
        eq = self.eq
        for i, probs in enumerate(self.probs):
            for j, rates in enumerate(self.rates):
                trim = (len(probs) - len(rates.effects)) // 2
                p = probs[trim : len(probs)-trim]
                d = Derivative(p / fsum(p), rates, basetype=float)
                eq[i,j] = d.equilibrium()
                mean[i,j], var[i,j] = mean_var(eq[i,j], rates.fitness)
            self.slope[i] = regress_through_origin(var[i], mean[i])

    def plot_stats(self, supertitle='SUPER', title='TITLE'):
        """
        NEED
        """
        # Set up two axis systems, one for means, the other for variances.
        sns.set_palette(sns.color_palette("Set2", len(self.mean)))
        fig, ax_mean = plt.subplots()
        ax_var = ax_mean.twinx()
        ax_var.grid(False)
        ax_mean.grid(True, zorder=0)
        #
        # Plot the means and scaled variances as functions of the upper
        # limit on fitness, row by row. Each row corresponds to a
        # distribution of probability over mutational effects. For each
        # row, the variances are scaled by the slope obtained by
        # regression of the means on the variances (through the origin).
        limits = [r.fitness[-1] for r in self.rates]
        z = zip(self.mean, self.var, self.slope, self.weight_labels)
        for m, v, s, label in z:
            ax_mean.plot(limits, m, label=label)
            ax_var.plot(limits, s*v, label=label, marker='o', ls='none')
        #
        # Add titles, axis labels, and legends.
        fig.suptitle(supertitle)
        ax_mean.set_title(title)
        ax_mean.set_xlabel('Upper Limit on Fitness')
        ax_mean.set_ylabel('Mean Fitness at Equilibrium')
        ax_var.set_ylabel('Scaled Variance in Fitness at Equilibrium')
        ax_mean.legend(title='Mean', loc='upper left')
        ax_var.legend(title='Variance', loc='lower right')
        #
        self.fig, self.ax_mean, self.ax_var = fig, ax_mean, ax_var
    
    def _begin_plot_curves(self, eq):
        sns.set_palette(sns.color_palette("Set2", len(eq)))
        self.fig, self.ax = plt.subplots()
        self.lines = np.empty(len(eq), dtype=object)
        max_y = max([e.max() for e in eq])
        self.ax.set_ylim([-0.0003, 1.05 * max_y])
        
    def plot_column(self, j):
        eq = self.eq[:, j]
        fitness = self.rates[j].fitness
        variable_name = 'Beneficial Weight'
        constant_name = 'Upper Limit on Fitness {}'.format(fitness[-1])
        self._begin_plot_curves(eq)
        for i in range(len(eq)):
            label = '{}'.format(self.weight_labels[i])
            self.lines[i], = plt.plot(fitness, eq[i], label=label)
        self._finish_plot_curves(variable_name, constant_name)

    def plot_row(self, i):
        eq = self.eq[i, :]
        variable_name = 'Fitness Limit'
        constant_name = 'Beneficial Effects Weight {}'
        constant_name = constant_name.format(self.weight_labels[i])
        self._begin_plot_curves(eq)
        for j in range(len(eq)):
            label = '{}'.format(self.limits[j])
            fitness = self.rates[j].fitness
            self.lines[j], = plt.plot(fitness, eq[j], label=label)
            self.lines[j].set_zorder(100 - j)
        self._finish_plot_curves(variable_name, constant_name)

    def _finish_plot_curves(self, variable_name, constant_name):
        self.ax.set_ylabel('Relative Frequency')
        self.ax.set_xlabel('Fitness')
        supertitle = 'Dependence of Equilibrium Distribution on '
        self.fig.suptitle(supertitle + variable_name)
        self.ax.set_title('Infinite Population, ' + constant_name)
        self.ax.legend(title=variable_name, loc='best')
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    def save_and_display(self, filename):
        """
        Save figure to disk, display it, and close the plot.
        """
        save_and_display(self.fig, filename)

    def __getitem__(self, which):
        """
        Returns equilibrium along with mean and variance of fitness.
        """
        return self.eq[which], self.mean[which], self.var[which]