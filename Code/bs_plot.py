def bs_plot(frequencies, m, intermediate, ls=['-', '-', '-'], ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_xlabel('Fitness')
    ax.set_ylabel('Relative Frequency')
    years = [0, intermediate, len(frequencies) - 1]
    #colors = ['r', 'b', 'limegreen']
    for year, style in zip(years, ls):
        p = frequencies[year] / math.fsum(frequencies[year])
        mean = math.fsum(p * m)
        label = 'Year {}'.format(year)
        ax.plot(m[p>0], p[p>0], label=label, ls=style)
    ax.legend(loc='best', fontsize='small')
    return fig, ax

def Xbs_plot(frequencies, m, years, ls=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if ls is None:
        ls = ['-']*len(years)
    ax.set_xlabel('Fitness')
    ax.set_ylabel('Relative Frequency')
    p = frequencies[0] / fsum(frequencies[0])
    ax.plot(m[p>0], p[p>0], c='black')
    for year, style in zip(years, ls):
        p = frequencies[year] / math.fsum(frequencies[year])
        label = 'Year {0}'.format(year)
        ax.plot(m[p>0], p[p>0], label=label, ls=style)
    ax.legend(loc='best', fontsize='small')
    return fig, ax