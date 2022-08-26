class SanfordPlot(object):
    def __init__(self, q, n=13, total_width=0.75, labels=None):
        # Obtain multilocus distributions from `dfe` for each (log) number of
        # loci in `log_L`. Produce bar plots for the distributions over the
        # interval [(1-n)w, (n-1)w], where w is the bin width. The total
        # width of each group of bars is `total_width * w`. Default labeling
        # of bar plots can be overridden by supplying `labels`.
        #
        delta = q[0].delta
        gamma = q[0].gamma
        nd = len(q)
        x = (delta * np.arange(1-n, n)).astype(float)
        total_width *= delta
        bar_w = total_width / nd
        first_offset = -total_width/2 + bar_w/2
        last_offset = total_width/2 - bar_w/2
        offsets = np.linspace(first_offset, last_offset, nd)
        c = sns.color_palette("tab10")[:2*nd]
        if labels is None:
            labels = [exp_latex(d.L, prefix='L=', base=2) for d in q]
        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        ax1.grid(False)
        ax1.yaxis.set_visible(False)
        ax0.grid(axis='x')
        U_str = '$U = {}$'.format(float(q[0].U))
        gamma_str = exp_latex(q[0].gamma, prefix='\\gamma=', base=10)
        t = 'Distribution of the Sum of IID Fitness Effects at $L$ Loci\n{}'
        sub = '({}, {})'.format(U_str, gamma_str)
        ax0.set_title(t.format(sub))
        ax0.set_xlabel('Overall Change in Fitness')
        ax0.set_ylabel('Scaled Probability')
        # Plot unscaled probabilities of non-positive effects
        mid = q[0].k
        for i, (off, l) in enumerate(zip(offsets, labels)):
            d = q[i][mid-(n-1):mid+n].astype(float)
            ax0.bar(x[:n]+off, d[:n], color=c[i], label=l, width=bar_w)
        # Plot scaled probabilities of positive effects.
        for i, off, lab in zip(reversed(range(len(labels))), 
                               offsets, 
                               reversed(labels)):
            d = q[i][mid-(n-1):mid+n].astype(float)
            dpos = float((1 - gamma) / gamma) * d[n:]
            ax1.bar(x[n:]+off, dpos, color=c[i+nd], label=lab, width=bar_w)
        ax1.set_ylim(ax0.get_ylim())
        ax0.legend(title='$1 \\cdot q$', loc='upper left', frameon=FRAMEON)
        ax1.legend(title='$(1-\\gamma)/\\gamma \\cdot q$', 
                   loc='upper right', frameon=FRAMEON)
        self.fig = fig
        self.ax = (ax0, ax1)
        
    def set_ylim(self, *args):
        for ax in self.ax:
            ax.set_ylim(*args)
            
    def save_and_display(self, filename):
        save_and_display(self.fig, filename)