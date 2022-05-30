#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Total losses versus training iterations.
"""
import sys
import pathlib
from matplotlib import pyplot

# find helpers
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].joinpath("modulus")))
from helpers.tbreader import read_tensorboard_data  # pylint: disable=import-error  # noqa: E402

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
rootdir.joinpath("figures").mkdir(exist_ok=True)

# read data
data1 = read_tensorboard_data(modulusdir.joinpath("nn_256", "outputs"))
data2 = read_tensorboard_data(modulusdir.joinpath("nn_512", "outputs"))

# plot
fig, ax = pyplot.subplots(1, 1, figsize=(6, 3), dpi=166)
fig.suptitle("Convergence history: total loss v.s. iteration")
ax.semilogy(data1["step"], data1["loss"].ewm(span=10).mean(), label="256 neurons per layer")
ax.semilogy(data2["step"], data2["loss"].ewm(span=10).mean(), label="512 neurons per layer")
ax.set_xlabel("Iteration")
ax.set_ylabel("Total loss")
ax.legend(loc=0)
fig.savefig(rootdir.joinpath("figures", "cylinder-pinn-training-convergence.png"), bbox_inches="tight", dpi=166)
