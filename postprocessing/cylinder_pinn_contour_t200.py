#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""2D cylinder flow Re200, T=200, Modulus/PINN
"""
import pathlib
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
modulusdir = rootdir.joinpath("modulus", "cylinder-2d-re200")
petibmdir = rootdir.joinpath("petibm", "cylinder-2d-re200")
rootdir.joinpath("figures").mkdir(exist_ok=True)

# read data
with h5py.File(modulusdir.joinpath("output", "snapshots.h5"), "r") as dset:
    x = dset["nn_256/x"][...]
    y = dset["nn_256/y"][...]
    u = dset["nn_256/200.0/u"][...]
    wz = dset["nn_256/200.0/vorticity_z"][...]

# normalization for colormaps
norms = {
    "u": colors.Normalize(-0.32, 1.4),
    "wz": colors.CenteredNorm(0., 5)
}

# plot
fig, axs = pyplot.subplots(2, 1, sharex=True, sharey=False, figsize=(6, 4), dpi=166)
fig.suptitle(r"Flow distribution, $Re=200$ at $t=200$, PINN")

axs[0].contourf(x, y, u, 256, norm=norms["u"], cmap="cividis")
axs[0].add_artist(pyplot.Circle((0., 0.), 0.5, color="w"))
axs[0].set_xlim(-3, 14)
axs[0].set_ylim(-3, 3)
axs[0].set_aspect("equal", "box")
axs[0].set_ylabel("y")
axs[0].set_title(r"$u$ velocity")

axs[1].contourf(x, y, wz, 512, norm=norms["wz"], cmap="cividis")
axs[1].add_artist(pyplot.Circle((0., 0.), 0.5, color="w"))
axs[1].set_xlim(-3, 14)
axs[1].set_ylim(-3, 3)
axs[1].set_aspect("equal", "box")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].set_title(r"Vorticity")

fig.colorbar(cm.ScalarMappable(norms["u"], "cividis"), ax=axs[0])
fig.colorbar(cm.ScalarMappable(norms["wz"], "cividis"), ax=axs[1])
fig.savefig(rootdir.joinpath("figures", "cylinder-pinn-contour-t200.png"), bbox_inches="tight", dpi=166)
