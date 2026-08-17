This repository contains a reference implementation of code supporting the
scientific paper:

>Moment problems and bounds for matrix-valued smeared spectral functions
>R. Abbott, W.I. Jay, and P.R. Oare
>https://arxiv.org/pdf/2508.01377.

This paper computes tight, rigorous bounds on certain smeared spectral functions
using Euclidean-time inputs viewed as truncated moment sequences.
These bounds are computed using the general solution to the truncated moment problem, which has been derived in

>Analytic theory of a class of interpolation problems
>I. V. Kovalishina
>Mathematics of the USSR-Izvestiya 22, 419 (1984).

The reference implementation is contained in `hamburger_mp.py`.
The repository also contains stand-alone scripts that reproduce all of the
figures in the paper.
  * figure_poles - Figure 1
  * figures_staggered - Figures 2 and 3
  * figure_opeator_scaling - Figure 4
  * figure_smearing_scaling - Figure 5

Assuming [uv](https://docs.astral.sh/uv/getting-started/installation/) is installed, running `./generate_figures.sh` will generate all of the figures by running the four scripts.

A key mathematical hypothesis that enables the underlying methods is that of
spectral positivity. This assumption amounts to requiring a positive-definite
Hankel matrix.
Since Hankel matrices are notoriously ill-conditioned, this requirement is often
violated for noisy data or data computed to finite (say, double) numerical precision.
These methods are not intended for out-of-the-box use with noisy data.

Instead, the code is intended for theoretical exploration of a crucial question
of principle: how much information about smeared spectral observables exists in
a given truncated moment sequence.
For a given truncated moment sequence, the resulting bounds for the Hamburger
moment problem are tight in the sense that tighter bounds are only possible by:
1. including more data (more moments/Euclidean times) or
2. incorporating additional mathematical hypothesis appropriate for a given
    problem for example with some notion of smoothness or regularity or asymptotic scaling.