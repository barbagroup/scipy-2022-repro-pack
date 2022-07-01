#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""2D Cylinder, Re200
"""
import sys
import pathlib
import sympy
from torch import sum as _torchsum  # pylint: disable=no-name-in-module
from torch import abs as _torchabs  # pylint: disable=no-name-in-module
from modulus.key import Key as _Key
from modulus.hydra.config import ModulusConfig as _ModulusConfig
from modulus.geometry.csg.csg_2d import Rectangle as _Rectangle
from modulus.geometry.csg.csg_2d import Circle as _Circle
from modulus.architecture.fully_connected import FullyConnectedArch as _FullyConnectedArch
from modulus.continuous.constraints.constraint import PointwiseInteriorConstraint as _PointwiseInteriorConstraint
from modulus.continuous.constraints.constraint import PointwiseBoundaryConstraint as _PointwiseBoundaryConstraint
from modulus.continuous.domain.domain import Domain as _Domain
from modulus.continuous.solvers.solver import Solver as _Solver
from modulus.continuous.monitor.monitor import PointwiseMonitor as _PointwiseMonitor

# find helpers
for parent in pathlib.Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        sys.path.insert(0, str(parent))
        from helpers.pdes import IncompNavierStokes  # pylint: disable=import-error
        from helpers.inferencers import SaveModelInferencer  # pylint: disable=import-error
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


def get_computational_graph(cfg: _ModulusConfig):
    """Returns the computational graph as a list of nodes.
    """

    netconf = {k: v for k, v in cfg.arch.fully_connected.items() if k != "_target_"}

    pde = IncompNavierStokes(cfg.custom.nu, cfg.custom.rho, 2, True)

    net = _FullyConnectedArch(
        input_keys=[_Key("x"), _Key("y"), _Key("t")],
        output_keys=[_Key("u"), _Key("v"), _Key("p")],
        **netconf
    )

    nodes = pde.make_nodes() + [net.make_node(name="flow-net", jit=cfg.jit)]

    return nodes, net


def get_computational_domain(cfg: _ModulusConfig):
    """Get a geometry object representing the computational domain.
    """

    xbg = sympy.sympify(cfg.custom.xbg).evalf()  # -8
    xed = sympy.sympify(cfg.custom.xed).evalf()  # 25
    ybg = sympy.sympify(cfg.custom.ybg).evalf()  # -8
    yed = sympy.sympify(cfg.custom.yed).evalf()  # 8
    radius = sympy.sympify(cfg.custom.radius).evalf()

    geo = _Rectangle((xbg, ybg), (xed, yed)) - _Circle((0, 0), radius)

    return geo


def get_initial_constraint(nodes, geo, cfg):
    """Get an interior pointwise constraint for initial conditions (i.e., t=0)
    """
    xbg = sympy.sympify(cfg.custom.xbg).evalf()
    xed = sympy.sympify(cfg.custom.xed).evalf()
    ybg = sympy.sympify(cfg.custom.ybg).evalf()
    yed = sympy.sympify(cfg.custom.yed).evalf()

    x, y, t = sympy.symbols("x, y, t")

    constraint = _PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 1.0, "v": 0.0, "p": 0.0},
        batch_size=cfg.batch_size.ic,
        bounds={x: (xbg, xed), y: (ybg, yed)},
        param_ranges={t: 0.0}
    )

    return constraint


def get_boundary_constraints(nodes, geo, cfg):
    """Get BCs.
    """

    xbg = sympy.sympify(cfg.custom.xbg).evalf()
    xed = sympy.sympify(cfg.custom.xed).evalf()
    ybg = sympy.sympify(cfg.custom.ybg).evalf()
    yed = sympy.sympify(cfg.custom.yed).evalf()
    radius = sympy.sympify(cfg.custom.radius).evalf()  # 0.5

    x, y, t = sympy.symbols("x, y, t")

    # inlet
    inlet = _PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 1.0, "v": 0.0},
        batch_size=cfg.batch_size.inlet,
        criteria=sympy.Eq(x, xbg),
        param_ranges={t: (0., cfg.custom.tend)}
    )

    # outlet
    outlet = _PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"p": 0.0},
        batch_size=cfg.batch_size.outlet,
        criteria=sympy.Eq(x, xed),
        param_ranges={t: (0., cfg.custom.tend)}
    )

    # bottom
    bottom = _PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 1.0, "v": 0.0},
        batch_size=cfg.batch_size.bottom,
        criteria=sympy.Eq(y, ybg),
        param_ranges={t: (0., cfg.custom.tend)}
    )

    # top
    top = _PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 1.0, "v": 0.0},
        batch_size=cfg.batch_size.top,
        criteria=sympy.Eq(y, yed),
        param_ranges={t: (0., cfg.custom.tend)}
    )

    # noslip
    noslip = _PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0.0, "v": 0.0},
        batch_size=cfg.batch_size.cylinder,
        criteria=sympy.And(sympy.Ge(x, -radius), sympy.Le(x, radius), sympy.Ge(y, -radius), sympy.Le(y, radius)),
        param_ranges={t: (0., cfg.custom.tend)}
    )

    return {"inlet": inlet, "outlet": outlet, "top": top, "bottom": bottom, "noslip": noslip}


def get_pde_constraint(nodes, geo, cfg):
    """Get an interior constraint for PDE residuals.
    """

    xbg = sympy.sympify(cfg.custom.xbg).evalf()
    xed = sympy.sympify(cfg.custom.xed).evalf()
    ybg = sympy.sympify(cfg.custom.ybg).evalf()
    yed = sympy.sympify(cfg.custom.yed).evalf()

    x, y, t = sympy.symbols("x, y, t")

    constraint = _PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0.0, "momentum_x": 0.0, "momentum_y": 0.0},
        batch_size=cfg.batch_size.npts,
        bounds={x: (xbg, xed), y: (ybg, yed)},
        param_ranges={t: (0., cfg.custom.tend)}
    )

    return constraint


def get_residual_monitors(nodes, geo, cfg):
    """Get a monitor to record PDE residuals.
    """

    xbg = sympy.sympify(cfg.custom.xbg).evalf()
    xed = sympy.sympify(cfg.custom.xed).evalf()
    ybg = sympy.sympify(cfg.custom.ybg).evalf()
    yed = sympy.sympify(cfg.custom.yed).evalf()
    radius = sympy.sympify(cfg.custom.radius).evalf()  # 0.5

    x, y, t = sympy.symbols("x, y, t")

    residual = _PointwiseMonitor(
        invar=geo.sample_interior(
            nr_points=cfg.batch_size.npts,
            bounds={x: (xbg, xed), y: (ybg, yed)},
            param_ranges={t: (0., cfg.custom.tend)}
        ),
        output_names=["continuity", "momentum_x", "momentum_y"],
        metrics={
            "continuity_res": lambda var: _torchsum(var["area"] * _torchabs(var["continuity"])),
            "momentum_x_res": lambda var: _torchsum(var["area"] * _torchabs(var["momentum_x"])),
            "momentum_y_res": lambda var: _torchsum(var["area"] * _torchabs(var["momentum_y"])),
        },
        nodes=nodes,
    )

    noslip = _PointwiseMonitor(
        invar=geo.sample_boundary(
            nr_points=cfg.batch_size.cylinder,
            criteria=sympy.And(sympy.Ge(x, -radius), sympy.Le(x, radius), sympy.Ge(y, -radius), sympy.Le(y, radius)),
            param_ranges={t: (0., cfg.custom.tend)},
        ),
        output_names=["u", "v"],
        metrics={
            "u_l1norm": lambda var: _torchsum(var["area"] * _torchabs(var["u"])),
            "v_l1norm": lambda var: _torchsum(var["area"] * _torchabs(var["v"])),
        },
        nodes=nodes,
    )

    freestream = _PointwiseMonitor(
        invar=geo.sample_boundary(
            nr_points=cfg.batch_size.inlet+cfg.batch_size.top+cfg.batch_size.bottom,
            criteria=sympy.Or(sympy.Eq(x, xbg), sympy.Eq(y, ybg), sympy.Eq(y, yed)),
            param_ranges={t: (0., cfg.custom.tend)},
        ),
        output_names=["u", "v"],
        metrics={
            "u_l1norm": lambda var: _torchsum(var["area"] * _torchabs(var["u"]-1.0)),
            "v_l1norm": lambda var: _torchsum(var["area"] * _torchabs(var["v"])),
        },
        nodes=nodes,
    )

    return {"pde-residual": residual, "cylinder-noslip": noslip, "freestream": freestream}


def get_solver_domains(nodes, geo, cfg):
    """Get a modules' Domain for the first time window.

    No BC constraints needed because they are periodic BCs and are handled by the neural network.
    """

    domain = _Domain()

    # true IC, i.e., at t = 0 sec, for the first window
    domain.add_constraint(get_initial_constraint(nodes, geo, cfg), name="ic")

    # boundary conditions
    for name, bc in get_boundary_constraints(nodes, geo, cfg).items():  # pylint: disable=invalid-name
        domain.add_constraint(bc, name=name)

    # interior PDE residuals (same for both)
    domain.add_constraint(get_pde_constraint(nodes, geo, cfg), name="residual")

    # inferencers (save the model params and timestamps)
    domain.add_inferencer(inferencer=SaveModelInferencer(nodes, "flow-net"))

    # add monitored quanties to tensorboard
    for name, monitor in get_residual_monitors(nodes, geo, cfg).items():
        domain.add_monitor(monitor, name=name)

    return domain


def main(cfg: _ModulusConfig):
    """The main function.
    """
    nodes, _ = get_computational_graph(cfg)
    geo = get_computational_domain(cfg)
    domain = get_solver_domains(nodes, geo, cfg)
    solver = _Solver(cfg=cfg, domain=domain)
    solver.solve()
    return 0


if __name__ == "__main__":
    import modulus
    root = pathlib.Path(__file__).resolve().parent

    # re-make the cmd arguments with the program name and desired workdir
    sys.argv = sys.argv[:1] + [r"hydra.run.dir=outputs"]

    # modulus.main is a function wrapper to help loading configs
    sys.exit(modulus.main(str(root), "config")(main)())
