#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Process Modulus 2D Cylinder Re=200
"""
import io
import sys
import pathlib
import lzma
import itertools
import numpy
import h5py
import torch
import modulus.key
import modulus.graph
import modulus.architecture.fully_connected
import omegaconf
import sympy

# find helpers
for parent in pathlib.Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        sys.path.insert(0, str(parent))
        from helpers.pdes import IncompNavierStokes  # pylint: disable=import-error
        from helpers.pdes import Vorticity  # pylint: disable=import-error
        from helpers.pdes import QCriterion  # pylint: disable=import-error
        from helpers.pdes import VelocityGradients  # pylint: disable=import-error
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


def create_graph(cfg, params=None):
    """Create a computational graph without proper parameters in the model.
    """

    # network
    net = modulus.architecture.fully_connected.FullyConnectedArch(
        input_keys=[modulus.key.Key("x"), modulus.key.Key("y"), modulus.key.Key("t")],
        output_keys=[modulus.key.Key("u"), modulus.key.Key("v"), modulus.key.Key("p")],
        **{k: v for k, v in cfg.arch.fully_connected.items() if k != "_target_"}
    )

    # update parameters
    if params is not None:
        net.load_state_dict(params)

    # navier-stokes equation
    nseq = IncompNavierStokes(cfg.custom.nu, cfg.custom.rho, 2, True)

    # vorticity
    vorticity = Vorticity(dim=2)

    # q-criterion
    qcriterion = QCriterion(dim=2)

    # velocity gradients
    velgradient = VelocityGradients(dim=2)

    nodes = \
        nseq.make_nodes() + vorticity.make_nodes() + qcriterion.make_nodes() + velgradient.make_nodes() + \
        [net.make_node(name="flow-net", jit=cfg.jit)]

    dtype = next(net.parameters()).dtype

    return nodes, dtype


def get_model_from_file(cfg, filename):
    """Read snapshot data.
    """

    # load the snatshot data
    with lzma.open(filename, "rb") as obj:  # open the file
        snapshot = torch.load(obj, map_location=cfg.device)

    # load the model parameters form the snapshot data
    with io.BytesIO(snapshot["model"]) as obj:  # load the model
        params = torch.jit.load(obj, map_location=cfg.device).state_dict()

    # get a computational graph
    graph, dtype = create_graph(cfg, params)

    # timestamp and training iteration
    timestamp = snapshot["time"]
    step = snapshot["step"]

    return step, timestamp, graph, dtype


def get_case_data(cfg, workdir, fields=["u", "v", "p"]):
    """Get data from a single case.
    """

    # identify the last iteration
    mxstep = max([
        fname.stem.replace("flow-net-", "") for
        fname in workdir.joinpath("inferencers").glob("flow-net-*.pth")
    ], key=int)

    # get the computational graph
    _, _, graph, dtype = get_model_from_file(cfg, workdir.joinpath("inferencers", f"flow-net-{mxstep}.pth"))

    # get a subset in the computational graph that gives us desired quantities
    model = modulus.graph.Graph(
        graph, modulus.key.Key.convert_list(["x", "y", "t"]), modulus.key.Key.convert_list(fields))

    # gridlines
    npx = numpy.linspace(cfg.custom.xbg, cfg.custom.xed, 743)  # vertices
    npx = (npx[1:] + npx[:-1]) / 2  # cell centers
    npy = numpy.linspace(cfg.custom.ybg, cfg.custom.yed, 361)  # vertices
    npy = (npy[1:] + npy[:-1]) / 2  # cell centers
    npx, npy = numpy.meshgrid(npx, npy)
    shape = npx.shape

    # torch version of gridlines; reshape to N by 1
    torchx = torch.tensor(npx.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=True)
    torchy = torch.tensor(npy.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=True)

    # snapshot data holder (for contour plotting)
    snapshots = {"x": npx, "y": npy}

    for time in cfg.eval_times:

        preds = model({"x": torchx, "y": torchy, "t": torch.full_like(torchx, time)})
        preds = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}

        # save the prediction data
        snapshots[time] = preds

    return snapshots


def get_drag_lift_coefficients(cfg, workdir):
    """Get drag and lift coefficients.
    """

    # identify the last iteration
    mxstep = max([
        fname.stem.replace("flow-net-", "") for
        fname in workdir.joinpath("inferencers").glob("flow-net-*.pth")
    ], key=int)

    # get the computational graph
    _, _, graph, dtype = get_model_from_file(cfg, workdir.joinpath("inferencers", f"flow-net-{mxstep}.pth"))

    # required fields for calculating forces
    fields = ["u", "v", "p", "u_x", "u_y", "v_x", "v_y"]

    # get a subset in the computational graph that gives us desired quantities
    model = modulus.graph.Graph(
        graph, modulus.key.Key.convert_list(["x", "y", "t"]), modulus.key.Key.convert_list(fields))

    # coordinates to infer (on the cylinder surface)
    nr = 720
    theta = numpy.linspace(0., 2*numpy.pi, 720, False)
    nx = numpy.cos(theta)  # x component of normal vector
    ny = numpy.sin(theta)  # y component of normal vector
    npx = numpy.cos(theta) * float(cfg.custom.radius)
    npy = numpy.sin(theta) * float(cfg.custom.radius)

    # time
    nt = 201
    times = numpy.linspace(0., 200., nt)

    # reshape to N by 1 vectors and create torch vectors (sharing the same memory space)
    torchx = torch.tensor(npx.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=True)
    torchy = torch.tensor(npy.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=True)

    # plot time frame by time frame
    cd = numpy.zeros_like(times)
    cl = numpy.zeros_like(times)
    for i, time in enumerate(times):

        preds = model({"x": torchx, "y": torchy, "t": torch.full_like(torchx, time)})
        preds = {k: v.detach().cpu().numpy().flatten() for k, v in preds.items()}

        fd = cfg.custom.nu * (
            nx * ny**2 * preds["u_x"] + ny**3 * preds["u_y"] - nx**2 * ny * preds["v_x"] - nx * ny**2 * preds["v_y"]
        )

        pd = preds["p"] * nx

        cd[i] = 2 * 2 * numpy.pi * cfg.custom.radius * numpy.sum(fd - pd) / nr

        fl = cfg.custom.nu * (
            nx**2 * ny * preds["u_x"] + nx * ny**2 * preds["u_y"] - nx**3 * preds["v_x"] - nx**2 * ny * preds["v_y"]
        )

        pl = preds["p"] * ny

        cl[i] = - 2 * 2 * numpy.pi * cfg.custom.radius * numpy.sum(fl - pl) / nr

    return cd, cl


def main(workdir):
    """Main function.
    """

    # save all post-processed data here
    workdir.joinpath("output").mkdir(exist_ok=True)

    # cases' names
    cases = [f"nn_{n}" for n in [256, 512]]

    # target fields
    fields = ["u", "v", "p", "vorticity_z"]

    # hdf5 file
    h5file = h5py.File(workdir.joinpath("output", "snapshots.h5"), "w")

    # read and process data case-by-case
    for job in cases:
        print(f"Handling {job}")

        jobdir = workdir.joinpath(job, "outputs")

        cfg = omegaconf.OmegaConf.load(jobdir.joinpath(".hydra", "config.yaml"))
        cfg.device = "cpu"
        cfg.custom.scale = float(sympy.sympify(cfg.custom.scale).evalf())
        cfg.eval_times = [200.0]

        snapshots = get_case_data(cfg, jobdir, fields)

        h5file.create_dataset(f"{job}/x", data=snapshots["x"], compression="gzip")
        h5file.create_dataset(f"{job}/y", data=snapshots["y"], compression="gzip")
        for time, field in itertools.product(cfg.eval_times, fields):
            h5file.create_dataset(f"{job}/{time}/{field}", data=snapshots[time][field], compression="gzip")

        cd, cl = get_drag_lift_coefficients(cfg, jobdir)
        h5file.create_dataset(f"{job}/cd", data=cd, compression="gzip")
        h5file.create_dataset(f"{job}/cl", data=cl, compression="gzip")

    h5file.close()


if __name__ == "__main__":

    # find the root of the folder `modulus`
    for root in pathlib.Path(__file__).resolve().parents:
        if root.joinpath("modulus").is_dir():
            break
    else:
        raise FileNotFoundError("Couldn't locate the path to the folder `cases`.")

    root = root.joinpath("modulus", "cylinder-2d-re200")

    main(root)
