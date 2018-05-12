def plot_errors(P, data, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Relative Error')
    e = [np.max(relative_error(p[d>0], d[d>0], absolute=True)) for p, d in zip(P, data)]
    ax.plot(e, label='max absolute')
    e = [np.mean(relative_error(p[d>0], d[d>0], absolute=True)) for p, d in zip(P, data)]
    ax.plot(e, label='mean absolute')
    e = [np.mean(relative_error(p[d>0], d[d>0], absolute=False)) for p, d in zip(P, data)]
    ax.plot(e, label='mean signed')
    ax.legend(loc='upper left')
    return fig, ax