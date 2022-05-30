#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Drag and lift coefficients

"""
import pathlib
import numpy
import h5py
from matplotlib import pyplot

# matplotlib configuration
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["P052", "Pagella", "Palatino", "Palatino Linotype", "Times New Roman"],
    "figure.constrained_layout.use": True,
})


# directories
rootdir = pathlib.Path(__file__).resolve().parents[1]
modulusdir = rootdir.joinpath("modulus", "cylinder-2d-re200", "output")
petibmdir = rootdir.joinpath("petibm", "cylinder-2d-re200", "output")
rootdir.joinpath("figures").mkdir(exist_ok=True)

# read data
pinn = numpy.zeros((201, 3), dtype=float)
pinn[:, 0] = numpy.linspace(0., 200, 201)
with h5py.File(modulusdir.joinpath("snapshots.h5"), "r") as dset:
    pinn[:, 1] = dset["nn_256/cd"][...]
    pinn[:, 2] = dset["nn_256/cl"][...]

petibm = numpy.loadtxt(petibmdir.joinpath("forces-0.txt"), dtype=float)

# plot
fig, ax = pyplot.subplots(1, 1, sharex=True, sharey=False, figsize=(6, 4), dpi=166)
fig.suptitle(r"Lift and drag coefficients, $Re=200$")

# petibm
ax.plot(petibm[:, 0], petibm[:, 1]*2, "k-", lw=1, alpha=0.8, label=r"$C_D$, PetIBM")
ax.plot(petibm[:, 0], petibm[:, 2]*2, "k-.", lw=1, alpha=0.8, label=r"$C_L$, PetIBM")

# pinn
ax.plot(pinn[:, 0], pinn[:, 1], c="tab:red", ls="-", lw=1, alpha=0.8, label=r"$C_D$, PINN")
ax.plot(pinn[:, 0], pinn[:, 2], c="tab:red", ls="-.", lw=1, alpha=0.8, label=r"$C_L$, PINN")

ax.set_ylim(-1, 2)
ax.set_xlabel(r"$T_{sim}$")
ax.set_ylabel("$C_D$ and $C_L$")
ax.legend(loc=0)
fig.savefig(rootdir.joinpath("figures", "cylinder-cd-cl.png"), bbox_inches="tight", dpi=166)
