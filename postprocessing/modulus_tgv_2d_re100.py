#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Process Modulus 2D TGV Re=100
"""
import datetime
import io
import sys
import pathlib
import lzma
import multiprocessing
import itertools
import numpy
import pandas
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
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


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


def create_graph(cfg, params=None):
    """Create a computational graph without proper parameters in the model.
    """

    xbg = ybg = - cfg.custom.scale * numpy.pi
    xed = yed = cfg.custom.scale * numpy.pi

    # network
    net = modulus.architecture.fully_connected.FullyConnectedArch(
        input_keys=[modulus.key.Key("x"), modulus.key.Key("y"), modulus.key.Key("t")],
        output_keys=[modulus.key.Key("u"), modulus.key.Key("v"), modulus.key.Key("p")],
        periodicity={"x": (float(xbg), float(xed)), "y": (float(ybg), float(yed))},
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

    nodes = \
        nseq.make_nodes() + vorticity.make_nodes() + qcriterion.make_nodes() + \
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


def get_run_time(timestamps):
    """Process timestamps, eliminate gaps betwenn Slurm job submissions, and return accumulated run time.

    The returned run times are in seconds.
    """

    timestamps = numpy.array(timestamps)
    diff = timestamps[1:] - timestamps[:-1]
    truncated = numpy.sort(diff)[5:-5]

    avg = truncated.mean()
    std = truncated.std(ddof=1)
    diff[numpy.logical_or(diff < avg-2*std, diff > avg+2*std)] = avg
    diff = numpy.concatenate((numpy.full((1,), avg), diff))  # add back the time for the first result

    return numpy.cumsum(diff)


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
    npx = numpy.linspace(-cfg.custom.scale*numpy.pi, cfg.custom.scale*numpy.pi, 513)  # vertices
    npx = (npx[1:] + npx[:-1]) / 2  # cell centers
    npy = numpy.linspace(-cfg.custom.scale*numpy.pi, cfg.custom.scale*numpy.pi, 513)  # vertices
    npy = (npy[1:] + npy[:-1]) / 2  # cell centers
    npx, npy = numpy.meshgrid(npx, npy)
    shape = npx.shape

    # torch version of gridlines; reshape to N by 1
    torchx = torch.tensor(npx.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=False)
    torchy = torch.tensor(npy.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=False)

    # error data holder
    data = pandas.DataFrame(
        data=None,
        index=pandas.Index([], dtype=float, name="time"),
        columns=pandas.MultiIndex.from_product((["l1norm", "l2norm"], fields)),
    )

    # snapshot data holder (for contour plotting)
    snapshots = {"x": npx, "y": npy}

    for time in cfg.eval_times:

        preds = model({"x": torchx, "y": torchy, "t": torch.full_like(torchx, time)})
        preds = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}

        for key in fields:
            ans = analytical_solution(npx, npy, time, 0.01, key)
            err = abs(preds[key]-ans)
            data.loc[time, ("l1norm", key)] = 4 * numpy.pi**2 * err.sum() / err.size
            data.loc[time, ("l2norm", key)] = 2 * numpy.pi * numpy.sqrt((err**2).sum()/err.size)

        # save the prediction data
        snapshots[time] = preds

    return data, snapshots


def get_error_vs_walltime(cfg, workdir, fields):
    """Get error v.s. walltime
    """

    # gridlines
    npx = numpy.linspace(-cfg.custom.scale*numpy.pi, cfg.custom.scale*numpy.pi, 513, dtype=numpy.float32)  # vertices
    npx = (npx[1:] + npx[:-1]) / 2  # cell centers
    npy = numpy.linspace(-cfg.custom.scale*numpy.pi, cfg.custom.scale*numpy.pi, 513, dtype=numpy.float32)  # vertices
    npy = (npy[1:] + npy[:-1]) / 2  # cell centers
    npx, npy = [val.reshape(-1, 1) for val in numpy.meshgrid(npx, npy)]

    # a copy of torch version
    x, y = torch.from_numpy(npx), torch.from_numpy(npy)

    # initialize data holders
    data = pandas.DataFrame(
        data=None,
        index=pandas.Index([], dtype=int, name="iteration"),
        columns=pandas.MultiIndex.from_product(
            [["l1norm", "l2norm"], fields, cfg.eval_times]
        ).append(pandas.Index([("timestamp", "", "")])),
    )

    def single_process(rank, inputs, outputs):
        """A single workder in multi-processing setting.
        """
        while True:
            try:
                fname = inputs.get(True, 2)
            except multiprocessing.queues.Empty:
                inputs.close()
                outputs.close()
                return

            print(f"[Rank {rank}] processing {fname.name}")

            # initialize data holders
            temp = pandas.Series(
                data=None, dtype=float,
                index=pandas.MultiIndex.from_product(
                    [["l1norm", "l2norm"], fields, cfg.eval_times]
                ).append(pandas.Index([("timestamp", "", "")])),
            )

            # get the computational graph
            step, timestamp, graph, _ = get_model_from_file(cfg, fname)

            # convert to epoch time
            temp.loc["timestamp"] = datetime.datetime.fromisoformat(timestamp).timestamp()

            # get a subset in the computational graph that gives us desired quantities
            model = modulus.graph.Graph(
                graph, modulus.key.Key.convert_list(["x", "y", "t"]), modulus.key.Key.convert_list(fields))

            for time in cfg.eval_times:
                preds = model({"x": x, "y": y, "t": torch.full_like(x, time)})
                preds = {k: v.detach().cpu().numpy() for k, v in preds.items()}

                for key in fields:
                    ans = analytical_solution(npx, npy, time, 0.01, key)
                    err = abs(preds[key]-ans)
                    temp.loc[("l1norm", key, time)] = float(4 * numpy.pi**2 * err.sum() / err.size)
                    temp.loc[("l2norm", key, time)] = float(2 * numpy.pi * numpy.sqrt((err**2).sum()/err.size))

            outputs.put((step, temp))
            inputs.task_done()
            print(f"[Rank {rank}] done processing {fname.name}")

    # collect all model snapshots
    files = multiprocessing.JoinableQueue()
    for i, file in enumerate(workdir.joinpath("inferencers").glob("flow-net-*.pth")):
        files.put(file)

    # initialize a queue for outputs
    results = multiprocessing.Queue()

    # workers
    procs = []
    for rank in range(multiprocessing.cpu_count()//2):
        proc = multiprocessing.Process(target=single_process, args=(rank, files, results))
        proc.start()
        procs.append(proc)

    # block until the queue is empty
    files.join()

    # extra from result queue
    while not results.empty():
        step, result = results.get(False)
        data.loc[step] = result

    # sort with iteration numbers
    data = data.sort_index()

    # get wall time using timestamps
    data["runtime"] = get_run_time(data["timestamp"])

    return data


def main(workdir):
    """Main function.
    """

    # save all post-processed data here
    workdir.joinpath("output").mkdir(exist_ok=True)

    # cases' names
    cases = [f"a100_{n}" for n in [1, 2, 4, 8]]

    # target fields
    fields = ["u", "v", "p"]

    # initialize a data holder for errors
    data = pandas.DataFrame(
        data=None, dtype=float,
        index=pandas.Index([], dtype=float, name="time"),
        columns=pandas.MultiIndex.from_product((cases, ["l1norm", "l2norm"], fields)),
    )

    # hdf5 file
    h5file = h5py.File(workdir.joinpath("output", "snapshots.h5"), "w")

    # read and process data case-by-case
    for job in cases:
        print(f"Handling {job}")

        jobdir = workdir.joinpath(job, "outputs")

        cfg = omegaconf.OmegaConf.load(jobdir.joinpath(".hydra", "config.yaml"))
        cfg.device = "cpu"
        cfg.custom.scale = float(sympy.sympify(cfg.custom.scale).evalf())
        cfg.eval_times = list(range(0, 101, 2))

        data[job], snapshots = get_case_data(cfg, jobdir, fields)

        h5file.create_dataset(f"{job}/x", data=snapshots["x"], compression="gzip")
        h5file.create_dataset(f"{job}/y", data=snapshots["y"], compression="gzip")
        for time, field in itertools.product(cfg.eval_times, fields):
            h5file.create_dataset(f"{job}/{time}/{field}", data=snapshots[time][field], compression="gzip")

    h5file.close()
    data.to_csv(workdir.joinpath("output", "sim-time-errors.csv"))

    # get error versus wall time from a100_8
    jobdir = workdir.joinpath("a100_8", "outputs")
    cfg = omegaconf.OmegaConf.load(jobdir.joinpath(".hydra", "config.yaml"))
    cfg.device = "cpu"
    cfg.custom.scale = float(sympy.sympify(cfg.custom.scale).evalf())
    cfg.eval_times = [2, 8, 32]
    data = get_error_vs_walltime(cfg, workdir.joinpath("a100_8", "outputs"), fields)
    data.to_csv(workdir.joinpath("output", "wall-time-errors.csv"))


if __name__ == "__main__":

    # find the root of the folder `modulus`
    for root in pathlib.Path(__file__).resolve().parents:
        if root.joinpath("modulus").is_dir():
            break
    else:
        raise FileNotFoundError("Couldn't locate the path to the folder `cases`.")

    root = root.joinpath("modulus", "taylor-green-vortex-2d-re100")

    main(root)
