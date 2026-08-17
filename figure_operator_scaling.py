"""
Command-line executable script to plot scaling of bounds as operator basis changes.
"""
import numpy as np
import pandas as pd
import hamburger_mp as ham
import seaborn as sns
from collections import namedtuple
from tqdm import tqdm
from mpmath import mp, mpf, mpc

import pylab as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amsfonts}')
plt.rcParams.update({"text.usetex": True, "font.family": "cm"})

Bounds = namedtuple("Bounds", ["upper", "lower", "keys"])

def main():
    """
    Compute the bounds and make the plot.
    """
    mp.dps = 200

    eps = mpf("0.05")
    energies = mp.linspace(0, 1, 100)
    nt = 8

    results = []
    for nops in range(1, 11):
        print(f"Generating bounds with {nops} operator(s).")
        toy = ham.BesselModel(nt=nt, nops=nops)
        bounds = compute_bounds(toy, eps, energies)
        exact = compute_exact_data(toy, eps, energies)
        results.append({
            'nops': nops,
            'toy': toy,
            'bounds': bounds,
            'exact': exact,
            'eps': eps,
        })
    results = pd.DataFrame(results)

    print("Plotting results")
    colors = sns.color_palette("viridis", n_colors=len(results)+1)
    with sns.plotting_context("paper", font_scale=12/9.6):
        fig, _ = plot_varying_operator_basis(results, colors=colors)
        fig.savefig("Figures/Hamburger/varying_operator_basis.pdf")

        fig, _ = plot_colorbar(colors=colors)
        fig.savefig("Figures/Hamburger/colorbar.pdf")


def compute_bounds(toy, eps, energies):
    """
    Compute bounds for the smeared spectral function using given correlator
    input at fixed distance from the real-lambda line, z=lmbda_x+eps

    Parameters
    ----------
    toy : BesselModel
    eps : mpf
        Distance from the real-lambda line
    lmbda_x : (N, ) array_like
        Locations on the real-energy line defining where to evaluate the bounds

    Returns
    -------
    Bounds / namedtuple
        * bounds.upper, bounds.lower are dicts containing the upper and lower
            bounds from projecting the Weyl matrix balls.
            Keys identify the relevant component of the matrix correlator.
            Values are arrays with the bounds.
        * bounds.keys are the keys used to index bounds.upper/bounds.lower
    """
    corr = toy.correlator
    nops = toy.nops

    # Initialize basis vectors for projection of Weyl ball onto Wertevorrat
    basis = []
    for n in range(nops):
        vec = np.zeros(nops)
        vec[n] = 1
        basis.append(vec)

    bases = {}
    for a in range(nops):
        for b in range(nops):
            if a <= b:
                bases[(a,b)] = [basis[a], basis[b]]

    # Initialize storage for smeared spectral functions
    yp = {key : [] for key in bases.keys()}
    ym = {key : [] for key in bases.keys()}

    # Compute spectral reconstruction at each point
    for Ex in tqdm(energies):
        # z is the location in the complex transfer-matrix-eigenvalue plane
        # lambda = exp(-aE), i.e., aE = -log(lambda)
        z = mpc(mp.exp(-Ex), eps)

        try:
            w = ham.compute_weyl_matrix(z, corr)
            R, S, Sdagger, T = ham.unpack_weyl(w)

        except Exception as err:
            print(repr(err))
            print(f"Failure for Ex={Ex}")
            assert False

        #########################################
        # Compute parameters defining Weyl ball #
        #########################################
        # rho_g = left radius g from French "gauche"
        # rho_d = right radius d from French "droite"
        C = ham.inv(R) @ S
        rho_g = ham.inv(R)
        rho_d = Sdagger @ ham.inv(R) @ S - T
        rho_g_half = ham.sqrtm(rho_g)
        rho_d_half = ham.sqrtm(rho_d)

        #########################################
        # Apply Lemmma 1 to isolate Wertevorrat #
        #########################################
        for key, (u, v) in bases.items():

            c = u.T @ C @ v
            r = ham.norm(rho_g_half.conjugate().T @ u) * ham.norm(rho_d_half @ v)
            # Isolate bounds on spectral function: rho(x) = (1/pi)*Im G(x)
            # Note the Jacobian factor of lambda = exp(-energy)
            bounds = (c.imag + r, c.imag - r)
            yp[key].append(max(bounds)/np.pi * mp.exp(-Ex))
            ym[key].append(min(bounds)/np.pi * mp.exp(-Ex))

    return Bounds(yp, ym, bases.keys())


def compute_exact_data(toy, eps, energies):
    """
    Compute the exact smeared spectral function for operator combinations
    (0,0), (0,1), and (1,1).
    """
    exact = {}
    tmp = np.stack([toy.rho_smeared(mp.exp(-Ex) + eps*1j) for Ex in energies])
    for key in [(0,0),(0,1),(1,1)]:
        exact[key] = np.array([-val.imag/np.pi for val in tmp[:, key[0], key[1]]])
    return exact


def plot_varying_operator_basis(results, colors=None):
    """
    Plot a comparison of the bounds and exact result for varying number of operators.

    Parameters
    ----------
    results : pandas.DataFrame
        Each row contains results for a given number of operators, including
        both the exact results as well as the upper and lower bounds.
    colors : sequence of matplotlib colors
        colors, e.g., as returned by seaborn.color_palette()

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the generated plot.

    axes : matplotlib.axes.Axes
        Axes on which the data were plotted.
    """
    operator_key = (0,0)  # operator combination to examine
    idx = 20              # energy index at which to consider convergence
    eps = results['eps'].unique().item()
    nt = results['toy'].values[0].nt
    if colors is None:
        colors = sns.color_palette("viridis", n_colors=len(results))
    assert len(colors) == len(results)+1

    fig, _axes = plt.subplot_mosaic(
        [["density", "fixed_energy"],
        ["density", "scaling"]],
        layout="constrained",
        figsize=(6, 3)
    )
    axes = (_axes['density'], _axes['fixed_energy'], _axes['scaling'])

    for _, row in results.iterrows():
        nops = row['nops']
        upper = row['bounds'].upper
        lower = row['bounds'].lower
        color = colors[nops]

        ################################################
        # Plot smeared spectral function versus energy #
        ################################################

        x = np.linspace(0, 1, 100)
        yp = {key: np.array([float(vn) for vn in val]) for key, val in upper.items()}
        ym = {key: np.array([float(vn) for vn in val]) for key, val in lower.items()}
        for key in [operator_key]:
            label = f"{nops}"+r"$\times$"+f"{nops}"
            axes[0].fill_between(x=x, y1=ym[key], y2=yp[key], alpha=1.0,
                                 color=color, label=label)

        #########################################
        # Plot scaling behavior at fixed energy #
        #########################################

        # First, plot rho_epsilon(omega) as an errorbar
        for key in [operator_key]:
            ymean = (yp[key][idx] + ym[key][idx])/2
            yerr = (yp[key][idx] - ym[key][idx])/2
            axes[1].errorbar(nops, y=ymean, yerr=yerr, capsize=5, color=color)

        # Second, plot the relative uncertainy
        for key in [operator_key]:
            ymean = (yp[key][idx] + ym[key][idx])/2
            yerr = (yp[key][idx] - ym[key][idx])/2
            axes[2].errorbar(nops, y=yerr/ymean, fmt='o', color=color)

    #########################
    # Plot the exact result #
    #########################
    exact = results['exact'][0] # Exact results are always the same
    jacobian = np.exp(-x[idx])
    axes[1].axhline(y=float(exact[operator_key][idx]*jacobian),
                    color='k', ls='-', alpha=0.5)

    ###############
    # Text labels #
    ###############

    axes[0].text(0.95, 0.95,
                ("(0,0) component\n"
                r"Fixed "
                r"$\epsilon$="
                f"{float(eps):.2f}\n"
                f"{nt} timeslices"
                ),
                horizontalalignment='right',
                verticalalignment='top',
                transform=axes[0].transAxes)

    axes[1].text(0.95, 0.95,
                (r"Fixed $\omega$="
                f"{float(x[idx]):.2f}\n"
                r"$\epsilon$="
                f"{float(eps):.2f}\n"
                ),
                horizontalalignment='right',
                verticalalignment='top',
                transform=axes[1].transAxes)

    axes[2].text(0.95, 0.95,
                (r"Fixed $\omega$="
                f"{float(x[idx]):.2f}\n"
                r"$\epsilon$="
                f"{float(eps):.2f}\n"
                ),
                horizontalalignment='right',
                verticalalignment='top',
                transform=axes[2].transAxes)

    ##############
    # Formatting #
    ##############

    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel(r"$\omega$")
    axes[0].set_ylabel(r"$\rho_\epsilon(\omega)$")

    axes[1].set_xticks([])
    axes[1].set_ylabel(r"$\rho_\epsilon(\omega)$")

    axes[2].axhline(0.01, ls='--', color='k', label=r'1\%')
    axes[2].axhline(0.001, ls='-.', color='k', label=r'0.1\%')
    axes[2].set_xticks(range(1,11))
    axes[2].set_xlabel("Number of operators")
    axes[2].set_ylabel(r"Rel. Uncertainty")
    axes[2].set_yscale("log")
    axes[2].set_yticklabels([])
    axes[2].legend(loc='lower left', framealpha=1)

    return fig, axes


def my_palplot(pal, size=1., ax=None):
    """
    Plot the values in a color palette as a horizontal array.

    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, e.g., as returned by seaborn.color_palette()
    size : float
        Scaling factor for size of plot
    ax : matplotlib.axes.Axes` or array of Axes
        Existing axis to use

    Returns
    -------
    ax : matplotlib.axes.Axes
        The primary axes associated with the figure.
    """
    n = len(pal)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(n,1),
            cmap=mpl.colors.ListedColormap(list(pal)),
            interpolation="nearest", aspect="auto")
    # The proper way to set no ticks
    ax.yaxis.set_major_locator(ticker.NullLocator())
    return ax

def plot_colorbar(colors):
    """
    Plots a colorbar.

    Parameters
    ----------
    colors : sequence of matplotlib colors
        colors, e.g., as returned by seaborn.color_palette()

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The primary axes associated with the figure.
    """
    fig, ax = plt.subplots(1, figsize=(0.5, 3))
    _ = my_palplot(colors[1:11], ax=ax)
    ax.set_xticks([])
    ax.yaxis.tick_right()
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(1,11))
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Number of operators", rotation=-90)
    fig.subplots_adjust(bottom=0.02, top=0.98, right=0.4)
    return fig, ax



if __name__ == "__main__":
    main()