"""
Command-line executable script to plot scaling of bounds as smearing varies.
"""

import numpy as np
import pandas as pd
import hamburger_mp as ham
import seaborn as sns
from collections import namedtuple
from tqdm import tqdm
from mpmath import mp, mpf, mpc
import pylab as plt

plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amsfonts}')
plt.rcParams.update({"text.usetex": True, "font.family": "cm"})

Bounds = namedtuple("Bounds", ["upper", "lower", "keys"])

def main():
    """
    Compute the bounds and make the plot.
    """
    mp.dps = 500
    nt = 8
    epsilons = np.logspace(-10, 1, num=50)
    dE = mpf("1e-2")

    results = []
    for nops in [2,4,6,8,10]:
        toy = ham.BesselModel(nt=nt, nops=nops, dE=dE)
        result = compute_bounds(toy, epsilons)
        results.append(result)
    results = pd.DataFrame(results)

    with sns.plotting_context("paper", font_scale=12/9.6):
        fig, _ = plot_bound_scaling(results, epsilons, dE, nt)
        fig.savefig("Figures/Hamburger/epsilon_scaling_bessel_model.pdf")

def compute_bounds(toy, epsilons, lmbda_x=mpf("1")):

    """
    Compute bounds for the smeared spectral function using given correlator
    input at various values of the smearing width

    Parameters
    ----------
    toy : BesselModel
    epsilons : array_like
        Distances from the real-lambda line
    Ex : mpf
        Location along the real-lambda line to evalute the bounds

    Returns
    -------
    Bounds / namedtuple
        * bounds.upper, bounds.lower are dicts containing the upper and lower
            bounds from projecting the Weyl matrix balls.
            Keys identify the relevant component of the matrix correlator.
            Values are arrays with the bounds.
        * bounds.keys are the keys used to index bounds.upper/bounds.lower
    """
    nops = toy.nops
    corr = toy.correlator

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
    yp = { key : [] for key in bases.keys()}
    ym = { key : [] for key in bases.keys()}
    radii = {key : [] for key in bases.keys()}
    for eps in tqdm(epsilons):
        # z is the location in the complex transfer-matrix-eigenvalue plane
        # lambda = exp(-aE), i.e., aE = -log(lambda)

        z = mpc(lmbda_x, mpf(eps))
        try:
            w = ham.compute_weyl_matrix(z, corr)
            R, S, Sdagger, T = ham.unpack_weyl(w)

        except Exception as err:
            print(repr(err))
            print(f"Failure for Ex={Ex}")
            assert False
            continue

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

        r_operator =\
            np.linalg.norm(np.array(rho_g_half, dtype=complex), ord=2)\
            * np.linalg.norm(np.array(rho_d_half, dtype=complex), ord=2)

        #########################################
        # Apply Lemmma 1 to isolate Wertevorrat #
        #########################################
        for key, (u, v) in bases.items():
            c = u.T @ C @ v
            r = ham.norm(rho_g_half.conjugate().T @ u) * ham.norm(rho_d_half @ v)
            bounds = (c.imag + r, c.imag - r)
            yp[key].append(max(bounds))
            ym[key].append(min(bounds))
            radii[key].append(r_operator)

    yp = {key: np.array([vn for vn in val]) for key, val in yp.items()}
    ym = {key: np.array([vn for vn in val]) for key, val in ym.items()}
    radii = {key: np.array([vn for vn in val]) for key, val in radii.items()}
    result = {
        'nops': nops,
        'lmbda_x': lmbda_x,
        'eps': eps,
        'y+': yp,
        'y-': ym,
        'r': radii,
    }
    return result


def plot_bound_scaling(df, epsilons, dE, nt):
    """
    Plot a comparison of the bounds in terms of the linear extent of the
    Wertevorrat as the smearing changes for fixed number of operators.

    Parameters
    ----------
    df : pandas.DataFrame
        Each row contains results for a given number of operators, including
        both the exact results as well as the upper and lower bounds.
    epsilons : array_like
        The smearing widths
    dE : float
        The parameter E0 appearing in the model spectral function
    nt : int
        Number of Euclidean times

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the generated plot.

    axes : matplotlib.axes.Axes
        Axes on which the data were plotted.
    """
    operator_key = (0, 0)
    lmbda_x = df['lmbda_x'].unique().item()

    fig, ax = plt.subplots(1, figsize=(4, 3))
    colors = sns.color_palette("viridis", n_colors=5)

    ###########################################################################
    # Plot linear extent of Wertevorrat as smearing changes for each operator #
    ###########################################################################
    for idx, (nops, subdf) in enumerate(df.groupby('nops')):
        r = subdf['r'].item()
        ax.errorbar(x=epsilons, y=r[operator_key],
                    label=f"{nops} operators", color=colors[idx])

    # Plot location of parameter E0 in model
    ax.axvline(x=dE, color='k', ls='--', alpha=0.25)

    ###########################
    # Formatting and labeling #
    ###########################
    ax.text(0.96, 0.92,
        r"$E_0=\tfrac{1}{100}$",
        horizontalalignment='right',
        verticalalignment='top',
        alpha=0.5,
        fontsize=10,
        transform=ax.transAxes)

    ax.text(0.5, 0.95,
        (r"Fixed $\omega$="
        f"{float(mp.log(lmbda_x)):.0f}\n"
        f"{nt} timeslices"
        ),
        horizontalalignment='center',
        verticalalignment='top',
        fontsize=10,
        transform=ax.transAxes)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=11, loc='lower left')
    ax.set_ylim(bottom=1e-22, top=10e14)
    ax.set_xlim(right=10)
    ax.set_xlabel(r"Smearing $\epsilon$")
    ax.set_ylabel(r"$\begin{Vmatrix}\sqrt{\rho_d}\end{Vmatrix} \, \begin{Vmatrix}\sqrt{\rho_g} \end{Vmatrix}$")
    fig.tight_layout()
    return fig, ax


if __name__ == '__main__':
    main()