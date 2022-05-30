#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Contour of 2D TGV Re=100 at T=32, using Modulus/PINN
"""
import pathlib
import numpy
import h5py
from matplotlib import pyplot
from matplotlib import cm
from matplotlib import colors

# matplotlib configuration
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["P052", "Pagella", "Palatino", "Palatino Linotype", "Times New Roman"],
    "figure.constrained_layout.use": True,
})

# directories
rootdir = pathlib.Path(__file__).resolve().parents[1]
modulusdir = rootdir.joinpath("modulus", "taylor-green-vortex-2d-re100", "output")
rootdir.joinpath("figures").mkdir(exist_ok=True)

# read in data
with h5py.File(modulusdir.joinpath("snapshots.h5"), "r") as dset:
    print("reading x")
    x = dset["a100_8/x"][...]
    print("reading yu")
    y = dset["a100_8/y"][...]
    print("reading u")
    u = dset["a100_8/32/u"][...]
    print("reading v")
    v = dset["a100_8/32/v"][...]

# contour lines' levels
levels = numpy.linspace(-0.6, 0.6, 13)

# plot
fig, axs = pyplot.subplots(1, 2, sharex=False, sharey=True, figsize=(6, 3), dpi=166)
fig.suptitle(r"Taylor-Green vortex, Re=100, $t=32$, PINN")

# left figure
axs[0].contourf(x, y, u, 128, vmin=-0.6, vmax=0.6, cmap="cividis")
axs[0].contour(x, y, u, levels, colors="k", linewidths=0.5)
axs[0].set_aspect("equal", "box")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].set_title("u-velocity")

# right figure
axs[1].contourf(x, y, v, 128, vmin=-0.6, vmax=0.6, cmap="cividis")
axs[1].contour(x, y, v, levels, colors="k", linewidths=0.5)
axs[1].set_aspect("equal", "box")
axs[1].set_xlabel("x")
axs[1].set_title("v-velocity")

# colorbar
cbar = fig.colorbar(cm.ScalarMappable(colors.Normalize(-0.6, 0.6), "cividis"), ax=axs)
cbar.add_lines(levels=levels, colors=["k"]*levels.size, linewidths=0.5)

# save
fig.savefig(rootdir.joinpath("figures", "tgv-pinn-contour-t32.png"), bbox_inches="tight", dpi=166)
