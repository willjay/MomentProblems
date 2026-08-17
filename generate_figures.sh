#!/bin/bash -l

mkdir -p Figures/Hamburger
uv run figure_poles.py
uv run figures_staggered.py
uv run figure_operator_scaling.py
uv run figure_smearing_scaling.py
