#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Process PetIBM 2D TGV Re100.
"""
import re
import itertools
import pathlib
import numpy
import pandas
import h5py


def analytical_solution(x, y, t, nu, field, V0=1., L=1., rho=1.):
    """Get analytical solution of 2D TGV.
    """
    if field == "u":
        return V0 * numpy.cos(x/L) * numpy.sin(y/L) * numpy.exp(-2.*nu*t/L**2)
    elif field == "v":
        return - V0 * numpy.sin(x/L) * numpy.cos(y/L) * numpy.exp(-2.*nu*t/L**2)
    elif field == "p":
        return - rho * V0**2 * numpy.exp(-4.*nu*t/L**2) * (numpy.cos(2.*x/L) + numpy.cos(2.*y/L)) / 4.
    elif field == "wz":
        return - 2. * V0 * numpy.cos(x/L) * numpy.cos(y/L) * numpy.exp(-2.*nu*t/L**2) / L
    elif field == "KE":  # kinetic energy
        return numpy.pi**2 * L**2 * V0**2 * rho * numpy.exp(-4.*nu*t/L**2)
    elif field == "KEDR":  # kinetic energy dissipation rate
        return 4. * numpy.pi**2 * V0**2 * nu * rho * numpy.exp(-4.*nu*t/L**2)
    elif field == "enstrophy":  # enstrophy
        return 2. * numpy.pi**2 * V0**2 * nu * rho * numpy.exp(-4.*nu*t/L**2)
    else:
        raise ValueError


def get_iter_wall_time(file):
    """Get wall time from a PETSc log file.
    """

    with open(file, "r", encoding="utf-8") as fobj:
        for line in fobj:
            result = re.search(r"Time \(sec\):\s+?([\de+-\.]+)", line)
            if result is not None:
                perf = float(result.group(1))
                break
        else:
            raise RuntimeError(f"Couldn't find run time ({file}).")

    return perf


def get_case_data(workdir, fields=["u", "v", "p"]):
    """Get data from a single case.
    """

    # read gridlines
    with h5py.File(workdir.joinpath("output", "grid.h5"), "r") as dsets:
        x = {key: dsets[f"{key}/x"][...] for key in fields}
        y = {key: dsets[f"{key}/y"][...] for key in fields}

    # padding the nodes not used in simulation due to periodic BC
    x["u"] = numpy.concatenate((numpy.full((1,), -numpy.pi, float), x["u"]))
    y["v"] = numpy.concatenate((numpy.full((1,), -numpy.pi, float), y["v"]))

    # convert gridlines to mesh grids
    xy = {key: numpy.meshgrid(x[key], y[key]) for key in fields}
    x = {key: val[0] for key, val in xy.items()}
    y = {key: val[1] for key, val in xy.items()}

    # error data holder
    data = pandas.DataFrame(
        data=None,
        index=pandas.Index([], dtype=float, name="time"),
        columns=pandas.MultiIndex.from_product((["l1norm", "l2norm"], fields)).append(
            pandas.Index((("walltime", ""),))),
    )

    # snapshot data holder (for contour plotting)
    snapshots = {"x": x, "y": y}

    for fname in workdir.joinpath("output").glob("*.h5"):
        if fname.name == "grid.h5":
            continue

        with h5py.File(fname, "r") as dsets:
            time = round(dsets["p"].attrs["time"])
            preds = {key: dsets[key][...] for key in fields}

        # padding the ignored nodes due to periodic BCs
        preds["u"] = numpy.concatenate((preds["u"][:, -1:], preds["u"]), axis=1)
        preds["v"] = numpy.concatenate((preds["v"][-1:, :], preds["v"]), axis=0)

        for key in fields:
            ans = analytical_solution(x[key], y[key], time, 0.01, key)
            err = abs(preds[key]-ans)
            data.loc[time, ("l1norm", key)] = 4 * numpy.pi**2 * err.sum() / err.size
            data.loc[time, ("l2norm", key)] = 2 * numpy.pi * numpy.sqrt((err**2).sum()/err.size)

        # get the wall time
        if int(fname.stem) != 0:
            data.loc[time, "walltime"] = get_iter_wall_time(workdir.joinpath("output", "logs", f"{fname.stem}.log"))
        else:
            data.loc[time, "walltime"] = 0

        # save the prediction data
        snapshots[time] = preds

    return data, snapshots


def main(workdir):
    """Main function.
    """

    # save all post-processed data here
    workdir.joinpath("output").mkdir(exist_ok=True)

    # cases' names
    cases = [f"{n}x{n}" for n in [16, 32, 64, 128, 256, 512, 1024]]

    # target fields
    fields = ["u", "v", "p"]

    # initialize a data holder for errors
    data = pandas.DataFrame(
        data=None, dtype=float,
        index=pandas.Index([], dtype=float, name="time"),
        columns=pandas.MultiIndex.from_product(
            (cases, ["l1norm", "l2norm"], fields)).append(
                pandas.MultiIndex.from_product((cases, ["walltime"], [""]))
            )
    )

    # hdf5 file
    h5file = h5py.File(workdir.joinpath("output", "snapshots.h5"), "w")

    # read and process data case-by-case
    for job in cases:
        print(f"Handling {job}")
        data[job], snapshots = get_case_data(workdir.joinpath(job), fields)

        for time, field in itertools.product(snapshots.keys(), fields):
            h5file.create_dataset(f"{job}/{time}/{field}", data=snapshots[time][field], compression="gzip")

    h5file.close()

    data.sort_index(axis=0)
    data.sort_index(axis=1)
    data.to_csv(workdir.joinpath("output", "perf.csv"))


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[1]
    root = root.joinpath("petibm", "taylor-green-vortex-2d-re100")
    main(root)
