def bs_plot(frequencies, m, intermediate, ls=['-', '-', '-']):
    fig, ax = plt.subplots()
    ax.set_xlabel('Fitness')
    ax.set_ylabel('Relative Frequency')
    years = [0, intermediate, len(frequencies) - 1]
    colors = ['r', 'b', 'limegreen']
    for year, color, style in zip(years, colors, ls):
        p = frequencies[year] / math.fsum(frequencies[year])
        mean = math.fsum(p * m)
        label = 'Year {0} (mean {1:.3f})'.format(year, mean)
        ax.plot(m[p>0], p[p>0], color=color, label=label, ls=style)
    ax.legend(loc='best')
    return fig, ax