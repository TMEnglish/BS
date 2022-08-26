from IPython.display import Image, display, HTML

# Use the Seaborn package to generate plots.
import seaborn as sns
sns.set()
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 3})
sns.set_style("darkgrid", {"axes.facecolor": ".92"})
FRAMEON=False
# Or sns.set_style("whitegrid"); FRAMEON=True
sns.set_palette(sns.color_palette("Set2", 8))


########################################################################
def save_and_display(figure, filename, form='png', dpi=600, close=True,
                     directory='./Figures/'):
    """
    Displays figure after saving it with the given attributes.
    
    If `close` is true, then the plot of the figure is closed.
    """
    path = directory + filename
    figure.savefig(path, format=form, dpi=dpi)
    display(Image(filename=path))
    if close:
        plt.close(figure)


def exp_latex(value, prefix='', form='{:3.1f}', base=10):
    """
    Returns LaTeX expression of `value` in exponential notation.
    
    Parameters
      * `value` is convertible to multiprecision float
      * string `prefix` goes at the beginning of the expression
      * string `form` is the format of the multiplier, if not 1
      * integer `base` is the base of the exponential expression
      
    The returned string begins and ends with single-$ delimiters. The
    intended use of `prefix` is to insert something like 'x='. That is,
    the returned string might be '$x= 1.23 \\times 10^{5}$'.
    """
    log = mp.log(to_mpf(value), base)
    n = int(log)
    a = base**float(log-n)
    if a < 1:
        a *= 10
        n -= 1
    exponential = '{}^{}{}{}'.format(base, '{', n, '}')
    if a == 1:
        a = ''
    else:
        a = form.format(a)
        a = '{}\\times '.format(a)
    return'${} {} {}$'.format(prefix, a, exponential)


def add_curve(m, p, threshold=0.0, ax=None, lw=3, **kwargs):
    """
    Plots thresholded frequency curve, and returns axis system.
    
    A figure is created if no axis system is supplied.
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('Fitness')
        ax.set_ylabel('Frequency')
    p = p / math.fsum(p)
    p[p < threshold] = 0.0
    p /= fsum(p)
    ax.plot(m[p > 0], p[p > 0], lw=lw, **kwargs)
    return ax


def bs_plot(solutions, m, mid, ls=['-', '-', '-'], lw=3, ax=None):
    """
    Plots normalized `solutions` indexed 0, `mid`, and -1.
    
    Returns the figure and the axis system for the plot.
    
    It is assumed that `solutions` is a 2-dimensional array of non-
    negative numbers, that the sum of elements for each row is positive,
    and that `solutions[n]` corresponds to year `n`.
    
    The linestyles for the three plotted solutions are given by `ls`.
    If an axis system `ax` is supplied, then the plot is on that axis
    system. Otherwise, a new figure and axis system are created.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_xlabel('Pseudo-Fitness')
    ax.set_ylabel('Frequency')
    years = [0, mid, len(solutions) - 1]
    for year, style in zip(years, ls):
        s = solutions[year] / fsum(solutions[year])
        label = 'Year {}'.format(year)
        ax.plot(m[s>0], s[s>0], label=label, ls=style, lw=lw)
    ax.legend(loc='best', fontsize='small', frameon=FRAMEON)
    return ax.get_figure(), ax