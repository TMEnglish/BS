def bs_plot(frequencies, m, intermediate):
    fig, ax = plt.subplots()
    ax.set_xlabel('Malthusian Growth Factor (Fitness)')
    ax.set_ylabel('Proportion')
    years = [0, intermediate, len(frequencies) - 1]
    colors = ['r', 'b', 'limegreen']
    for year, color in zip(years, colors):
        p = frequencies[year] / fsum(frequencies[year])
        mean = fsum(p * m)
        label = 'Year {0} (mean {1:.3f})'.format(year, mean)
        ax.plot(m[p>0], p[p>0], color=color, label=label)
    ax.legend(loc='upper left')
    return fig, ax