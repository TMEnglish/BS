class Equilibria(object):
    """
    Container for a 3-D array of equilibrium distributions.
    
    The dimensions correspond to
    * the weighting of beneficial mutational effects,
    * the upper limit on the birth rate parameter, and
    * the genomic mutation rate.
    
    Equilibria are calculated with death rate parameter `d` set to zero.
    """    
    def __init__(self, gammas=['1e-3', '1e-6', '1e-9'],
                       b_maxes=['0.15', '0.2', '0.25'],
                       mus=['1', '0.1'],
                       log_L=14,
                       Params=Parameters):
        """
        Create array of equilibria, and calculate associated statistics.
        
        Parameters
        * `gammas` : weightings of beneficial mutational effects
        * `b_maxes`: upper limits on the birth rate parameter
        * `mus`    : genomic mutation rates
        * `log_L`  : base-2 logarithm (an integer) of the number of loci
        * `Params` : subclass of `Parameters`
        
        The `gammas` must be integer powers of 2 or 10. Setting `log_L`
        greater than 14 has very little effect on the results.
        """
        # Create arrays to hold results, with rows corresponding to
        # probability distributions over mutational effects, and columns
        # corresponding to upper limits on birth parameter.
        m, n, k = len(gammas), len(b_maxes), len(mus)
        self.params = np.empty((m, n, k), dtype=object)
        self.eq = np.empty((m, n, k), dtype=object)
        self.e_value = np.empty((m, n, k))
        self.eigen_error = np.empty((m, n, k))
        self.mean = np.empty((m, n, k))
        self.var = np.empty((m, n, k))
        #
        # Calculate equilibria along with means/variances of fitnesses,
        # with death parameter `d` set to zero.
        for i, gamma in enumerate(gammas):
            for j, b_max in enumerate(b_maxes):
                for k, mu in enumerate(mus):
                    params = Params(b_max, d='0', mu=mu, gamma=gamma,
                                    log_L=log_L)
                    self.params[i,j,k] = params
                    e_value, e_vector, error = equilibrium(params.B())
                    self.e_value[i,j,k] = e_value
                    self.eq[i,j,k] = e_vector
                    self.eigen_error[i,j,k] = error
                    mean, var = mean_var(self.eq[i,j,k], params.b)
                    self.mean[i,j,k], self.var[i,j,k] = mean, var

    def plot(self, text_loc=(0.05, 0.25), legend_loc=(0.05, 0.61), 
                   fontsize=9, **kwargs):
        """
        Plots 2-D grid of equilibrium distributions.
        
        Rows of subplots correspond to mixture weights, and columns
        correspond to upper limits on the birth rate parameter. In each
        subplot, there is a curve for each of the mutation rates.

        Parameters
        * `text_loc`   : location of gamma (mixture weight) text
        * `legend_loc` : legend location within the upper right plot
        * `fontsize`   : font size of text within the plots
        * `kwargs`     : arguments passed to `subplots` initializer

        Instance members set
        * `fig`        : Matplotlib figure
        * `ax`         : 2-D array of axes of subplots
        * `xlabel`     : x-axis label attached to one of the subplots
        * `ylabel`     : y-axis label attached to one of the subplots

        Call the `save_and_display` instance method to display the
        figure `fig`. Instance members can be used to tweak elements of
        the plot.
        """
        m, n, k = self.eq.shape
        self.fig, self.ax = plt.subplots(m, n, subplot_kw=kwargs,
                                         sharex='col', sharey='row')
        for i in range(m):
            for j in range(n):
                for params, eq in zip(self.params[i,j], self.eq[i,j]):
                    label = '$\\mu={:3.1f}$'.format(float(params.mu))
                    self.ax[i,j].plot(params.b, eq, label=label, lw=1)
            gamma = exp_latex(self.params[i,0,0].gamma, '\gamma=')
            ax = self.ax[i,0]
            ax.text(*text_loc, gamma, transform=ax.transAxes, bbox=None, 
                    fontsize=fontsize, verticalalignment='top')
        self.ax[-1,n//2].set_xlabel('Birth Rate Parameter')
        self.xlabel = self.ax[-1,n//2].get_xlabel()
        self.ax[m//2,0].set_ylabel('Relative Frequency')
        self.ylabel = self.ax[m//2,0].get_ylabel()
        self.ax[0,-1].legend(loc=legend_loc, fontsize=fontsize,
                             framealpha=0.0)
        t = 'Equilibrium Distributions for Various Parameter Settings'
        self.ax[0,1].set_title(t, fontsize='large', pad=12)

    def save_and_display(self, filename):
        """
        Saves previously created figure to disk, and displays it.
        """
        save_and_display(self.fig, filename)

    def __getitem__(self, key):
        """
        Returns tuple of equilibrium distribution and associated stats.
        
        The tuple is a triple comprising 
        * the equilibrium distribution (array of relative frequencies),
        * the corresponding mean value of the birth rate parameter, and
        * the corresponding variance in birth rate parameter. 
        """
        return self.eq[key], self.mean[key], self.var[key]