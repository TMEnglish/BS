from matplotlib import animation                  ###### , rc
from IPython.display import Image, display, HTML
import seaborn as sns
# ASSUME: import matplotlib.pyplot as plt


# Select the type of animation.
# HTML5 animations require FFmpeg installation with non-default settings.
# plt.rcParams['animation.html'] = 'jshtml'
plt.rcParams['animation.html'] = 'html5'


# Use the Seaborn package to generate plots.
sns.set()
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
sns.set_style("darkgrid", {"axes.facecolor": ".92"})
sns.set_palette(sns.color_palette("Set2", 4))


def save_and_display(figure, filename, format='png', dpi=600, close=True):
    """
    Displays figure after saving it with the given attributes.
    
    If `close` is true, then the plot of the figure is closed.
    """
    figure.savefig(filename, format=format, dpi=dpi)
    display(Image(filename=filename))
    if close:
        plt.close(figure)

    
def save_video(anim, filename, fps=None):
    """
    Write animation `anim` to file, setting frames per second as specified.
    
    Tested only for `filename` with extension `.mp4`.
    """
    if fps is None:
        fps = round(1000/anim._interval)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps)
    anim.save(filename, writer)

    
def display_video(filename):
    html = """
           <video width="100%" controls autoplay loop>
               <source src="{0}" type="video/mp4">
           </video>
           """.format(filename)
    display(HTML(html))

    
def save_and_display_video(anim, filename, fps=None):
    save_video(anim, filename, fps)
    display_video(filename)


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


def exp_latex(value, prefix=''):
    """
    Returns LaTeX expression of `value` as integer power of 2 or 10.
    """
    value = float(value)
    base = 2
    log = round(mp.log(value, base))
    if value != base**log:
        base = 10
        log = round(mp.log(value, base))
        if value != base**log:
            message = '`value` is not an integer power of 2 or 10'
            raise ValueError(message)
    string = '${}{}^{}{}{}$'.format(prefix, base, '{', log, '}')
    return string