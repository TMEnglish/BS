##### WRONG INDENT!!!!
class EqPlot(object):
"""
    Figure comprising a 2-D grid of plots of equilibrium distributions.
    
    Call the `save_and_display` instance method to display the figure.
    The title and axis labels are attached to subplots. Their placement
    must be adjusted manually if the number of rows (or columns) is not
    odd. The subplots are accessed by indexing the instance.
"""
    def __init__(self, eqs, d=0.1, lw=2, text_loc=(0.05, 0.25), 
                 legend_loc=(0.05, 0.61), fontsize=9, **kwargs):
        """
        Plots 2-D grid of equilibrium distributions.

        Parameters
        * `eqs`       : an instance of `Equilibria` (wrapping 3-D array)
        * `d`         : death-rate parameter
        * `lw`        : width of lines in plots
        * `text_loc`  : location of mixture-weight text in left subplots
        * `legend_loc`: legend location within the upper right subplot
        * `fontsize`  : font size of text within the subplots
        * `kwargs`    : arguments passed to `subplots` initializer
        
        Rows of subplots correspond to mixture weights, and columns
        correspond to upper limits on the birth rate parameter. In each
        subplot, there is a curve for each of the mutation rates.
        """
        # Set up an m-by-n grid of subplots corresponding to first two
        # dimensions of the array of equilibrium distributions.
        m, n, k = eqs.shape
        self.L = 2**eqs.log_L
        self.delta = eqs.delta
        self.ax = np.empty((m, n), dtype=object)
        self.fig, self.ax[:] = plt.subplots(m, n, subplot_kw=kwargs,
                                            sharex='col', sharey='row')
        for i in range(m):
            for j in range(n):
                for eq in eqs[i,j]:
                    label = '$U={:3.1f}$'.format(float(eq.q.U))
                    self.ax[i,j].plot(eq.b - d, eq, lw=lw, label=label)
            # Display the beneficial effects weight in the leftmost of
            # the subplots in the row.
            gamma = exp_latex(eq[i,0,0].q.gamma, '\gamma=')
            ax = self.ax[i,0]
            ax.text(*text_loc, gamma, transform=ax.transAxes, bbox=None, 
                    fontsize=fontsize, verticalalignment='top')
        #
        # Place the label of the x-axis (y-axis) with the middle column
        # (row) of the grid. Place the title above the middle column.
        label = '$d=${}'.format(d)
        self.ax[-1,n//2].set_xlabel('Fitness ({})'.format(label))
        self.ax[m//2,0].set_ylabel('Frequency')
        t = 'Equilibrium Distributions for the Mutation-Selection Model'
        self.ax[0,n//2].set_title(t, fontsize=14, pad=11)
        #
        # Place the legend within the upper right subplot.
        self.ax[0,-1].legend(loc=legend_loc, fontsize=fontsize,
                             frameon=FRAMEON)

    def save_and_display(self, filename):
        """
        Saves the figure to a file, and then displays it.
        """
        print('L={}, delta={}'.format(self.L, self.delta))
        save_and_display(self.fig, filename)

    def __getitem__(self, key):
        return self.ax[key]