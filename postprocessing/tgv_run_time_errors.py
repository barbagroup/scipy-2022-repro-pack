#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Error norms of velocity v.s. run time (wall time)
"""
import pathlib
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
pinn = pandas.read_csv(modulusdir.joinpath("wall-time-errors.csv"), index_col=0, header=[0, 1, 2])
pinn = pinn.rename(columns=lambda inp: "" if "Unnamed" in inp else inp, level=1)
pinn = pinn.rename(columns=lambda inp: "" if "Unnamed" in inp else inp, level=2)
petibm = pandas.read_csv(petibmdir.joinpath("perf.csv"), index_col=0, header=[0, 1, 2])
petibm = petibm.rename(columns=lambda inp: "" if "Unnamed" in inp else inp, level=1)
petibm = petibm.rename(columns=lambda inp: "" if "Unnamed" in inp else inp, level=2)

time2, time8, time32 = [], [], []
err2, err8, err32 = [], [], []
for n in [16, 32, 64, 128, 256, 512, 1024]:
    time2.append(petibm.loc[2.0, (f"{n}x{n}", "walltime")])
    time8.append(petibm.loc[8.0, (f"{n}x{n}", "walltime")])
    time32.append(petibm.loc[32.0, (f"{n}x{n}", "walltime")])
    err2.append(petibm.loc[2.0, (f"{n}x{n}", "l2norm", "u")])
    err8.append(petibm.loc[8.0, (f"{n}x{n}", "l2norm", "u")])
    err32.append(petibm.loc[32.0, (f"{n}x{n}", "l2norm", "u")])

# plot
fig, ax = pyplot.subplots(1, 1, figsize=(6, 3), dpi=166)
fig.suptitle(r"$L_2$ error in $u$ v.s. wall time")

lines1 = [
    ax.loglog(pinn["runtime"], pinn[("l2norm", "u", "2")].ewm(span=10).mean(), label=r"$t=2$", alpha=0.8)[0],
    ax.loglog(pinn["runtime"], pinn[("l2norm", "u", "8")].ewm(span=10).mean(), label=r"$t=8$", alpha=0.8)[0],
    ax.loglog(pinn["runtime"], pinn[("l2norm", "u", "32")].ewm(span=10).mean(), label=r"$t=32$", alpha=0.8)[0]
]

legend1 = ax.legend(handles=lines1, loc=2, bbox_to_anchor=(0.16, 1), ncol=1, fontsize=9, title="PINN")

lines2 = [
    ax.loglog(time2, err2, marker="x", label=r"$t=2$", alpha=0.8, ls="--")[0],
    ax.loglog(time8, err8, marker="x", label=r"$t=8$", alpha=0.8, ls="--")[0],
    ax.loglog(time32, err32, marker="x", label=r"$t=32$", alpha=0.8, ls="--")[0]
]

legend2 = ax.legend(handles=lines2, loc=2, bbox_to_anchor=(0, 1), ncol=1, fontsize=9, title="PetIBM")

ax.set_xlabel("Wall time (seconds)")
ax.set_ylabel(r"$L_2$-norm")
ax.add_artist(legend1)
ax.add_artist(legend2)
fig.savefig(rootdir.joinpath("figures", "tgv-run-time-errors.png"), bbox_inches="tight", dpi=166)
