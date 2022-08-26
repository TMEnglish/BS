from scipy import stats


class OldEquilibria(object):
    """
    Container for an array of equilibria and associated statistics.
    
    Each row of the array corresponds to a weighting of beneficial
    mutational effects, and each column corresponds to an upper limit
    on fitness.
    
    Equilibria are calculated with death parameter `d` equal to zero. 
    The plotting methods adjust for nonzero `d`.
    """    
    def __init__(self, gammas, b_maxes, mu='1', Params=Parameters):
        """
        Create array of equilibria, and calculate associated statistics.
        
        Parameters
        * `gammas` : weightings of beneficial mutational effects
        * `b_maxes`: upper limits on the birth parameter
        * `mu`     : mutation rate in all cases
        * `Params` : subclass of `Parameters`
        
        The `gammas` must be integer powers of 2 or 10.
        """
        # Create arrays to hold results, with rows corresponding to
        # probability distributions over mutational effects, and columns
        # corresponding to upper limits on birth parameter.
        m, n = len(gammas), len(b_maxes)
        self.params = np.empty((m, n), dtype=object)
        self.eq = np.empty((m, n), dtype=object)
        self.e_value = np.empty((m, n))
        self.eigen_error = np.empty((m, n))
        self.mean = np.empty((m, n))
        self.var = np.empty((m, n))
        #
        # Calculate equilibria along with means/variances of fitnesses,
        # with death parameter `d` set to zero.
        for i in range(m):
            for j in range(n):
                params = Params(b_maxes[j], d='0', mu=mu, gamma=gammas[i])
                self.params[i,j] = params
                b = self.params[i,j].b
                B = self.params[i,j].B()
                e_value, e_vector, error = equilibrium(B)
                self.e_value[i,j] = e_value
                self.eq[i,j] = e_vector
                self.eigen_error[i,j] = error
                self.mean[i,j], self.var[i,j] = mean_var(self.eq[i,j], b)

    def plot_stats(self, d='0.1'):
        """
        Plot the means and variances in fitness for all equilibria.
        """
        # Set up two axis systems, one for means, the other for variances.
        sns.set_palette(sns.color_palette("Set2", len(self.mean)))
        fig, ax_m = plt.subplots()
        ax_v = ax_m.twinx()
        ax_v.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax_v.grid(False)
        #
        # Plot the means and variances as functions of the upper limit
        # on fitness, row by row. Each row corresponds to a weighting 
        # `gamma` of beneficial mutational effects.
        lims = [float(p.b[-1] - Fraction(d)) for p in self.params[0,:]]
        gammas = [exp_latex(p.gamma, '\gamma=') for p in self.params[:,0]]
        mean = self.mean - float(d)
        for m, v, g in zip(mean, self.var, gammas):
            ax_m.plot(lims, m, label=g, marker='+', ls='none')
            ax_v.plot(lims, v, label=g, marker='o', mfc='none', ls='none')
        #
        # Fit lines to means by linear regression, and plot them.
        xlim = ax_m.get_xlim()
        ylim = ax_m.get_ylim()
        ends = np.array(xlim)
        for means in mean:
            r = stats.linregress(lims, means)
            fitted_line = r.slope*ends + r.intercept
            ax_m.plot(ends, fitted_line, color='k', lw=0.5, zorder=1)
        #
        # Keep the x and y limits where they were prior to plotting
        # the fitted lines.
        ax_m.set_xlim(*xlim)
        ax_m.set_ylim(*ylim)
        #
        # Add title, axis labels, and legends.
        ax_m.xaxis.set_major_locator(plt.MaxNLocator(len(lims)))
        t = 'Linear Relations of Fitness Limit and Equilibrium Moments\n'
        t = t + 'Infinite Population, Beneficial Effects Weight $\gamma$'
        ax_m.set_title(t, size='large', pad=12)
        ax_m.set_xlabel('Upper Limit on Fitness')
        ax_m.set_ylabel('Mean Fitness at Equilibrium')
        ax_v.set_ylabel('Variance in Fitness at Equilibrium')
        ax_m.legend(title='Mean', loc='upper left')
        ax_v.legend(title='Variance', loc='lower right')
        #
        self.fig, self.ax_m, self.ax_v = fig, ax_m, ax_v
    
    def _begin_plot_curves(self, eq):
        # Set up an axis system and a color palette for a plot of
        # equlibria in one row or one column of the array.
        sns.set_palette(sns.color_palette("Set2", len(eq)))
        self.fig, self.ax = plt.subplots()
        self.lines = np.empty(len(eq), dtype=object)
        max_y = max([e.max() for e in eq])
        self.ax.set_ylim([-0.0003, 1.05 * max_y])
        
    def plot_column(self, j, d='0.1'):
        """
        Plot equilibria for `j`-th upper limit on fitness.
        """
        eq = self.eq[:,j]
        params = self.params[:,j]
        fitness = (params[0].b - Fraction(d)).astype(float)
        variable_name = 'Beneficial Effects Weight'
        abbrev_var_name = 'Weight $\gamma$'
        constant_name = 'Fitness Upper Limit {:5.3f}'.format(fitness[-1])
        self._begin_plot_curves(eq)
        for i in range(len(eq)):
            gamma = exp_latex(params[i].gamma, '\gamma=')
            self.lines[i], = plt.plot(fitness, eq[i], label=gamma)
        self._finish_plot_curves(variable_name, abbrev_var_name,
                                 constant_name)

    def plot_row(self, i, d='0.1'):
        """
        Plot equilibria for `i`-th weighting of beneficial effects.
        """
        eq = self.eq[i,:]
        params = self.params[i,:]
        gamma = exp_latex(params[0].gamma, '\gamma=')
        variable_name = 'Fitness Upper Limit'
        abbrev_var_name = 'Fitness Limit'
        constant_name = 'Beneficial Effects Weight {}'.format(gamma)
        self._begin_plot_curves(eq)
        for j in range(len(eq)):
            fitness = (params[j].b - Fraction(d)).astype(float)
            fitness_limit = '{:5.3f}'.format(fitness[-1])
            self.lines[j], = plt.plot(fitness, eq[j], label=fitness_limit)
            self.lines[j].set_zorder(100 - j)
        self._finish_plot_curves(variable_name, abbrev_var_name,
                                 constant_name)

    def alt_plot_row(self, i):
        """
        Plot equilibria for `i`-th weighting of beneficial effects.

        HACK: This is with scaling.
        """
        eq = self.eq[i,:]
        params = self.params[i,:]
        gamma = exp_latex(params[0].gamma, '\gamma=')
        variable_name = 'Number of Classes ($n$)'
        abbrev_var_name = ''
        constant_name = 'Beneficial Effects Weight {}'.format(gamma)
        self._begin_plot_curves(eq)
        max_y = 0
        for j in range(len(eq)):
            b = params[j].b
            scaled_eq = eq[j] * b[-1]
            max_y = max(max_y, scaled_eq.max())
            fitness = (b/b[-1]).astype(float)
            fitness_limit = '$b_n$ = {:5.3f}'.format(float(b[-1]))
            self.lines[j], = plt.plot(fitness, scaled_eq, label=fitness_limit)
            #self.lines[j].set_zorder(100 - j)
        self.ax.set_ylim([-0.0003, 1.05 * max_y])
        self._finish_plot_curves(variable_name, abbrev_var_name,
                                 constant_name)
        self.ax.set_xlabel('Birth Rate Parameter ($\\times b_n^{-1}$)')
        self.ax.set_ylabel('Relative Frequency ($\\times b_n$)')
    
    def _finish_plot_curves(self, variable_name, abbrev_var_name,
                                  constant_name):
        # Set titles and label axes.
        self.ax.set_ylabel('Relative Frequency')
        self.ax.set_xlabel('Fitness')
        title = 'Dependence of Equilibrium on {}\nInfinite Population, {}'
        title = title.format(variable_name, constant_name)
        self.ax.set_title(title, size='large', pad=12)
        self.ax.legend(title=abbrev_var_name, loc='best')
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    def save_and_display(self, filename):
        """
        Saves the figure to disk, displays it, and closes the plot.
        """
        save_and_display(self.fig, filename)

    def __getitem__(self, key):
        """
        Returns an equilibrium along with mean and variance.
        """
        return self.eq[key], self.mean[key], self.var[key]