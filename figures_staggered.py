"""
Command-line executable script to plot a proof-of-concept comparsion to exact
results for a toy staggered correlator.
"""
import numpy as np
import pylab as plt
import hamburger_mp as ham
import seaborn as sns
from tqdm import tqdm
from collections import namedtuple
from mpmath import mp, mpf, mpc

plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amsfonts}')
plt.rcParams.update({"text.usetex": True, "font.family": "cm"})

Bounds = namedtuple("Bounds", ["upper", "lower", "keys"])

def main():
    """
    Compute the bounds and make the plots.
    """
    mp.dps = 100

    # Generate data from a toy model for a matrix-valued staggered correlator
    print("Generating toy data.")
    nt = 48
    t = np.arange(nt)
    nstates = 20
    toy = ham.StaggeredModel(nt=nt, nstates=nstates, decaying=True, thermal=True)

    # Compute bounds on the smeared spectral function
    print("Computing bounds.")
    eps = mpf("0.05")
    lmbda_x = mp.linspace(-1.5, 1.5, 1000)
    bounds = compute_bounds(toy.correlator, eps, lmbda_x)

    # Compute exact smeared spectral function for comparison
    print("Computing exact smeared spectral function")
    exact = {}
    tmp = np.stack([toy.rho_smeared(xn + eps*1j) for xn in lmbda_x])
    for key in bounds.upper:
        # Unpack array --> dict to match structure of bounds
        exact[key] = np.array([-val.imag/np.pi for val in tmp[:, key[0], key[1]]])

    print("Generating plots.")
    with sns.plotting_context("paper", font_scale=12/9.6):
        fig, _ = plot_correlator(t, toy.correlator, toy.nops)
        fig.tight_layout()
        fig.savefig("Figures/Hamburger/staggered_correlator.pdf")

    with sns.plotting_context("paper", font_scale=1.5):
        fig, _ = plot_comparison_lambda(lmbda_x, bounds, exact)
        fig.savefig("Figures/Hamburger/staggered_reconstruction_lambda.pdf")

    with sns.plotting_context("paper", font_scale=1.5):
        fig, _ = plot_comparison_energy(lmbda_x, eps, bounds, exact, toy)
        fig.savefig("Figures/Hamburger/staggered_reconstruction_energy.pdf")

def plot_correlator(t, corr, nops):
    """
    Plot the three independent components of a 2x2 matrix of correlators.

    Parameters
    ----------
    t : (nt) array_like
        Temporal argument for correlator array
    corr : (nt, nops, nops) array_like
        The correlator array
    nops : int
        Number of operators

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the generated plot.

    ax : matplotlib.axes.Axes
        Axes on which the data were plotted.
    """
    assert nops==2, "Labels only support for nops==2."
    fig, ax = plt.subplots(1, figsize=(4, 3))
    labels = {
        (0,0): r'$C_{00}(t)$',
        (0,1): r'$C_{01}(t)$',
        (1,1): r'$C_{11}(t)$',
    }
    for a in range(nops):
        for b in range(nops):
            if a <= b:
                pos = corr[:,a,b] > 0
                neg = ~pos
                ax.plot(t[pos], corr[:,a,b][pos], 'o', label=labels[(a,b)])
                # Mirror negative values to be positive
                ax.plot(t[neg], -corr[:,a,b][neg], 'o',
                        markerfacecolor='None',
                        color=ax.lines[-1].get_color())
    ax.legend(ncols=1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$C_{ab}(t)$')
    return fig, ax

def compute_bounds(corr, eps, lmbda_x):
    """
    Compute bounds for the smeared spectral function using given correlator
    input at fixed distance from the real-lambda line, z=lmbda_x+eps

    Parameters
    ----------
    corr : (nt, nop, nop) array_like
        The input correlator data to be interpolated
    eps : mpf
        Distance from the real-lambda line
    lmbda_x : (N, ) array_like
        Locations on the real-lambda line defining where to evaluate the bounds

    Returns
    -------
    Bounds / namedtuple
        * bounds.upper, bounds.lower are dicts containing the upper and lower
          bounds from projecting the Weyl matrix balls.
          Keys identify the relevant component of the matrix correlator.
          Values are arrays with the bounds.
        * bounds.keys are the keys used to index bounds.upper/bounds.lower

    """
    yp = { key : [] for key in [(0,0), (0,1), (1,1)]}
    ym = { key : [] for key in [(0,0), (0,1), (1,1)]}

    e0 = np.array([1,0])
    e1 = np.array([0,1])
    bases = {
        (0,0): [e0, e0],
        (0,1): [e0, e1],
        (1,1): [e1, e1]
    }

    for xn in tqdm(lmbda_x):
        # z is the location in the complex transfer-matrix-eigenvalue plane
        # lambda = exp(-aE), i.e., aE = -log(lambda)
        z = mpc(xn, eps)
        try:
            w = ham.compute_weyl_matrix(z, corr)
            R, S, Sdagger, T = ham.unpack_weyl(w)
        except Exception as err:
            print(repr(err))
            print(f"Failure for Ex={lmbda_x}")
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
            bounds = (c.imag + r, c.imag - r)
            yp[key].append(max(bounds)/np.pi)
            ym[key].append(min(bounds)/np.pi)

    yp = {key: np.array([vn for vn in val]) for key, val in yp.items()}
    ym = {key: np.array([vn for vn in val]) for key, val in ym.items()}
    return Bounds(yp, ym, bases.keys())


def plot_comparison_lambda(lmbda_x, bounds, exact):
    """
    Plot a comparison of the bounds and exact result as a function of lambda.

    Parameters
    ----------
    lmbda_x : (npts, ) list or array_like
        The independent variable for plotting.
        Assumed to be the real part of the complex variable lambda.
    bounds : namedtuple / Bounds
        The bounds
    exact : dict
        Exact data, assumed to have the same structure as upper/lower bounds

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the generated plot.
    axes : matplotlib.axes.Axes
        Axes on which the data were plotted.
    """
    yp, ym, keys = bounds.upper, bounds.lower, bounds.keys
    x = np.array(lmbda_x, dtype=float)

    colors = sns.color_palette()
    fig, axes = plt.subplots(nrows=2, ncols=1,
                             sharex=True,
                             gridspec_kw=dict(width_ratios=[1], height_ratios=[3, 1]))
    ax1, ax2 = axes

    #######################################################
    # Plot allowed regions between upper and lower bounds #
    #######################################################
    for key, color in zip(keys, colors):
        y1 = np.array(ym[key], dtype=float)
        y2 = np.array(yp[key], dtype=float)
        ax1.fill_between(x=x, y1=y1, y2=y2, alpha=0.5,
                        color=color, label=f"{key} Wertevorrat")

    ###################
    # Plot exact data #
    ###################
    ax1.plot(x, exact[(0,0)], color=colors[0], ls='-.', label='Exact')
    ax1.plot(x, exact[(0,1)], color=colors[1], ls='-.', label='Exact')
    ax1.plot(x, exact[(1,1)], color=colors[2], ls='-.', label='Exact')

    ####################################################
    # Plot location of exact result within Wertevorrat #
    ####################################################
    for key in exact.keys():
        center = (yp[key] + ym[key])/2
        width = (yp[key] - ym[key])/2
        diff = (exact[key] - center)/width
        ax2.plot(x, diff)

    ax1.set_xticks([])
    ax2.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    fig.subplots_adjust(wspace=0, hspace=0, left=0.15, right=0.95, bottom=0.15, top=0.95)
    ax1.legend(ncols=2)
    ax2.set_xlabel(r"Re$(\lambda)$")
    ax1.set_ylabel(r"$\tilde{\rho}_\epsilon(\lambda)$")
    ax2.set_ylabel(r"$\frac{\rm{exact-center}}{\rm{width}}$")
    ax1.set_xlim(-1.5, 1.5)
    return fig, axes

def plot_comparison_energy(lmbda_x, eps, bounds, exact, toy):
    """
    Plot a comparison of the bounds and exact result as a function of energy.

    Parameters
    ----------
    lmbda_x : (npts, ) list or array_like
        The independent variable for plotting.
        Assumed to be the real part of the complex variable lambda.
    bounds : namedtuple / Bounds
        The bounds
    exact : dict
        Exact data, assumed to have the same structure as upper/lower bounds
    toy : StaggeredModel
        The model object containing the locations of the exact energy levels

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the generated plot.
    axes : matplotlib.axes.Axes
        Axes on which the data were plotted.
    """
    yp, ym, keys = bounds.upper, bounds.lower, bounds.keys
    lmbda_x = np.array(lmbda_x, dtype=float)
    z = np.array(lmbda_x) + 1j*float(eps)

    mask_minus = (lmbda_x <= 0)
    x_minus = -np.log(-z[mask_minus]).real

    mask_plus = (lmbda_x > 0)
    x_plus = -np.log(z[mask_plus]).real

    colors = sns.color_palette()
    fig, axes = plt.subplots(nrows=2, ncols=1,
                             sharex=True,
                             gridspec_kw=dict(width_ratios=[1], height_ratios=[1, 1]))
    ax1, ax2 = axes

    #######################################################
    # Plot allowed regions between upper and lower bounds #
    #######################################################
    for key, color in zip(keys, colors):
        y1 = np.array(ym[key], dtype=float)
        y2 = np.array(yp[key], dtype=float)

        # Decaying states
        # Note the Jacobian factor of lambda = exp(-energy) for decaying states
        ax1.fill_between(x=x_plus,
                         y1=y1[mask_plus] * lmbda_x[mask_plus],
                         y2=y2[mask_plus] * lmbda_x[mask_plus],
                         alpha=0.5,
                         color=color, label=f"{key}")

        # Oscillating states
        # Note the Jacobian factor of lambda = -exp(-energy) for oscillating states
        ax2.fill_between(x=x_minus,
                         y1=y1[mask_minus] * (-lmbda_x[mask_minus]),
                         y2=y2[mask_minus] * (-lmbda_x[mask_minus]),
                         alpha=0.5,
                         color=color, label=f"{key}")

    ###################
    # Plot exact data #
    ###################

    # Locations of exact energy levels
    for k in range(toy.nstates):
        ax1.axvline(toy._energy_decay(k), color='k', ls='--', alpha=0.05)
    for k in range(toy.nstates):
        ax2.axvline(toy._energy_osc(k), color='k', ls='-.', alpha=0.05)

    # Smeared spectral functions - decaying states
    # Note the Jacobian factor of lambda = exp(-energy) for decaying states
    ax1.plot(x_plus, exact[(0,0)][mask_plus] * lmbda_x[mask_plus],
             color=colors[0], ls='-.', label='Exact')
    ax1.plot(x_plus, exact[(0,1)][mask_plus] * lmbda_x[mask_plus],
             color=colors[1], ls='-.', label='Exact')
    ax1.plot(x_plus, exact[(1,1)][mask_plus] * lmbda_x[mask_plus],
             color=colors[2], ls='-.', label='Exact')

    # Smeared spectral functions - oscillating states
    # Note the Jacobian factor of lambda = exp(-energy) for oscillating states
    ax2.plot(x_minus, exact[(0,0)][mask_minus] * (-lmbda_x[mask_minus]),
             color=colors[0], ls='-.', label='Exact')
    ax2.plot(x_minus, exact[(0,1)][mask_minus] * (-lmbda_x[mask_minus]),
             color=colors[1], ls='-.', label='Exact')
    ax2.plot(x_minus, exact[(1,1)][mask_minus] * (-lmbda_x[mask_minus]),
             color=colors[2], ls='-.', label='Exact')

    ##############
    # Formatting #
    ##############
    fig.subplots_adjust(wspace=0, hspace=0)
    ax1.set_xlim(left=0, right=1.25)
    ax1.legend(ncols=2, loc='upper right')
    ax2.set_xlabel(r"$\omega $")
    ax1.set_ylabel(r"$\rho_\epsilon(\omega)$")
    ax2.set_ylabel(r"$\rho_\epsilon(\omega)$")
    ax1.text(0.30, 0.75,
           "Decaying states\n$\\propto e^{-Et}$",
           horizontalalignment='center',
           verticalalignment='center',
           alpha=0.65,
           transform=ax1.transAxes)
    ax2.text(0.35, 0.75,
           "Oscillating states\n $\\propto (-1)^t e^{-Et}$",
           horizontalalignment='center',
           verticalalignment='center',
           alpha=0.65,
           transform=ax2.transAxes)
    return fig, axes

if __name__ == '__main__':
    main()