"""
Command-line script to make a figure that shows the locations of various
poles in the complex energy and complex lambda planes.
"""
import numpy as np
import pylab as plt
import seaborn as sns

plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amsfonts}')
plt.rcParams.update({"text.usetex": True, "font.family": "cm"})

def main():
    """Makes the plot."""
    with sns.plotting_context("paper", font_scale=12/9.6):

        fig, axes = plt.subplot_mosaic([['a', 'b', 'c', 'd'],
                                        ['e', 'e', 'e', 'e'],
                                        ['e', 'e', 'e', 'e']],
                                        layout='constrained',
                                        figsize=(8, 4))

        x = np.linspace(-1, 1, num=100)
        y = np.sqrt(1-x**2)
        axes['e'].plot(x, +y, color='k', ls='--')
        axes['e'].plot(x, -y, color='k', ls='--')

        colors = sns.color_palette()

        energy = 0.2*np.arange(1, 11)
        y = np.zeros(len(energy))

        axes['a'].plot(energy, y, 'o', color=colors[0])
        axes['e'].plot(np.exp(-energy), y, 'o', color=colors[0],
                    label='Decaying states')

        axes['b'].plot(energy, y, '>', color=colors[1])
        axes['e'].plot(np.exp(+energy), y, '>', color=colors[1],
                    label='Thermal states')

        axes['c'].plot(+energy, y, 'd', color=colors[2])
        axes['e'].plot(-np.exp(-energy), y, 'd', color=colors[2],
                    label='Staggered states')

        axes['d'].plot(+energy, y, '<', color=colors[3])
        axes['e'].plot(-np.exp(+energy), y, '<', color=colors[3],
                    label='Thermal staggered states')

        for key in ['a', 'b', 'c', 'd', 'e']:
            axes[key].axhline(y=0, color='k')
            axes[key].axvline(x=0, color='k')

        for key in ['a', 'b', 'c', 'd', 'e']:
            axes[key].axis(False)

        for key in ['a', 'b', 'c', 'd']:
            axes[key].set_ylim(-0.1, 1)
            axes[key].text(0.15, 0.9,
                    "$E$",
                    horizontalalignment='left',
                    verticalalignment='center',
                    bbox=dict(facecolor='none', edgecolor='k'),
                    transform=axes[key].transAxes)

        axes['a'].text(0.5, 0.5,
                    "Decaying states\n $\\sim e^{-E t}$",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes['a'].transAxes)
        axes['b'].text(0.5, 0.5,
                    "Thermal states\n $\\sim e^{+E t}$",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes['b'].transAxes)
        axes['c'].text(0.55, 0.5,
                    "Staggered states\n $\\sim (-1)^t e^{-E t}$",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes['c'].transAxes)
        axes['d'].text(0.55, 0.5,
                    "Thermal staggered states\n $\\sim (-1)^t e^{+E t}$",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes['d'].transAxes)
        lim = 1.1
        axes['e'].set_xlim(-4*lim, +4*lim)
        axes['e'].set_ylim(-1.25*lim, +1.25*lim)

        axes['e'].text(0.02, 0.9,
                    r"$\lambda$",
                    horizontalalignment='left',
                    verticalalignment='center',
                    bbox=dict(facecolor='none', edgecolor='k'),
                    transform=axes['e'].transAxes)

        axes['e'].text(0.33, 0.75,
                r"$|\lambda|=1$",
                horizontalalignment='left',
                verticalalignment='center',
                alpha=0.65,
                transform=axes['e'].transAxes)

        axes['e'].legend()
        fig.savefig("Figures/Hamburger/eigenvalue_plane.pdf")

if __name__ == '__main__':
    main()