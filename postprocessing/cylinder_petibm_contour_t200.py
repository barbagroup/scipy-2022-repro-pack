#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""2D cylinder flow Re200, T=200, PetIBM
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
modulusdir = rootdir.joinpath("modulus", "cylinder-2d-re200-zero-ic")
petibmdir = rootdir.joinpath("petibm", "cylinder-2d-re200")
rootdir.joinpath("figures").mkdir(exist_ok=True)

# read data
with h5py.File(petibmdir.joinpath("output", "grid.h5"), "r") as dset:
    x = {"u": dset["u"]["x"][...], "wz": dset["wz"]["x"][...]}
    y = {"u": dset["u"]["y"][...], "wz": dset["wz"]["y"][...]}

with h5py.File(petibmdir.joinpath("output", "0040000.h5"), "r") as dset:
    u = dset["u"][...]
    wz = dset["wz"][...]

# normalization for colormaps
norms = {
    "u": colors.Normalize(-0.32, 1.4),
    "wz": colors.CenteredNorm(0., 5)
}

# plot
fig, axs = pyplot.subplots(2, 1, sharex=True, sharey=False, figsize=(6, 4), dpi=166)
fig.suptitle(r"Flow distribution, $Re=200$ at $t=200$, PetIBM")

axs[0].contourf(x["u"], y["u"], u, 256, norm=norms["u"], cmap="cividis")
axs[0].add_artist(pyplot.Circle((0., 0.), 0.5, color="w"))
axs[0].set_xlim(-3, 14)
axs[0].set_ylim(-3, 3)
axs[0].set_aspect("equal", "box")
axs[0].set_ylabel("y")
axs[0].set_title(r"$u$ velocity")

axs[1].contourf(x["wz"], y["wz"], wz, 512, norm=norms["wz"], cmap="cividis")
axs[1].add_artist(pyplot.Circle((0., 0.), 0.5, color="w"))
axs[1].set_xlim(-3, 14)
axs[1].set_ylim(-3, 3)
axs[1].set_aspect("equal", "box")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].set_title(r"Vorticity")

fig.colorbar(cm.ScalarMappable(norms["u"], "cividis"), ax=axs[0])
fig.colorbar(cm.ScalarMappable(norms["wz"], "cividis"), ax=axs[1])
fig.savefig(rootdir.joinpath("figures", "cylinder-petibm-contour-t200.png"), bbox_inches="tight", dpi=166)
