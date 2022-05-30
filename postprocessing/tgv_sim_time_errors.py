#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""L2 error v.s. simulation time.
"""
import pathlib
import numpy
import pandas
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
modulusdir = rootdir.joinpath("modulus", "taylor-green-vortex-2d-re100", "output")
petibmdir = rootdir.joinpath("petibm", "taylor-green-vortex-2d-re100", "output")
rootdir.joinpath("figures").mkdir(exist_ok=True)

# read data
pinn = pandas.read_csv(modulusdir.joinpath("sim-time-errors.csv"), index_col=0, header=[0, 1, 2])
petibm = pandas.read_csv(petibmdir.joinpath("perf.csv"), index_col=0, header=[0, 1, 2])
petibm = petibm.rename(columns=lambda inp: "" if "Unnamed" in inp else inp, level=1)
petibm = petibm.rename(columns=lambda inp: "" if "Unnamed" in inp else inp, level=2)

# plot
fig, ax = pyplot.subplots(1, 1, figsize=(6, 3), dpi=166)
fig.suptitle(r"$L_2$ error in $u$ v.s. simulation time")
ax.semilogy(pinn.index, pinn[("a100_1", "l2norm", "u")], label="PINN, 1 A100", alpha=0.6)
ax.semilogy(pinn.index, pinn[("a100_8", "l2norm", "u")], label="PINN, 8 A100", alpha=0.6)
ax.semilogy(petibm.index, petibm[("16x16", "l2norm", "u")], label="PetIBM, 16x16", ls="--", alpha=0.6)
ax.semilogy(petibm.index, petibm[("32x32", "l2norm", "u")], label="PetIBM, 32x32", ls="--", alpha=0.6)
ax.semilogy(petibm.index, petibm[("1024x1024", "l2norm", "u")], label="PetIBM, 1024x12024", ls="--", alpha=0.6)
ax.set_xlabel(r"t")
ax.set_ylabel(r"$L_2$-norm")
ax.set_ylim(1e-7, 2)
ax.set_yticks(numpy.power(10., numpy.arange(-7, 1)))
ax.legend(loc=0, ncol=3)
fig.savefig(rootdir.joinpath("figures", "tgv-sim-time-errors.png"), bbox_inches="tight", dpi=166)
