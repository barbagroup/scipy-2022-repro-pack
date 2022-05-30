#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Custom PDEs for Modulus.
"""
from numbers import Number as _ScalarType
from sympy import symbols as _symbols
from sympy import Expr as _Expr
from sympy import Number as _Number
from sympy import Function as _Function
from modulus.pdes import PDES as _PDES


class IncompNavierStokes(_PDES):
    """Incompressible Navier-Stokes equation for all dimensions.

    Inconmpressible means D rho / D t = 0 (the material derivative; not regular time derivative).
    The density may still be a function of coordinates and time.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, nu, rho, dim=2, time=True):  # pylint: disable=super-init-not-called
        # pylint: disable=not-callable
        assert isinstance(nu, (_ScalarType, _Expr, str)), f"Wrong type. Got {type(nu)}"
        assert isinstance(rho, (_ScalarType, _Expr, str)), f"Wrong type. Got {type(rho)}"
        assert dim in [1, 2, 3], f"Illegal dimension: {dim}"

        # initialize input symbols for later use
        xyz = list(_symbols("x, y, z"))
        time = _symbols("t")

        # disable the gravity term for now
        grav = [_Number(0), _Number(0), _Number(0)]

        # actual independent variables depend on problem's dimension
        invars = xyz[:dim]

        # independent variables also depend on whether it's a steady or unsteady problem
        if time:
            invars.append(time)

        # process kinematic viscosity depending on the type of nu
        if isinstance(nu, str):
            self.kvis = _Function(nu)(*invars)
        elif isinstance(nu, _ScalarType):
            self.kvis = _Number(nu)
        else:
            self.kvis = nu

        # process density depending on the type of rho
        if isinstance(rho, str):
            self.rho = _Function(rho)(*invars)
        elif isinstance(rho, _ScalarType):
            self.rho = _Number(rho)
        else:
            self.rho = rho

        # dependent variables (using constant 0 makes derivatives trivial when not needed)
        uvw = [
            _Function("u")(*invars),
            _Function("v")(*invars) if dim > 1 else _Number(0),
            _Function("w")(*invars) if dim > 2 else _Number(0),
        ]
        pres = _Function("p")(*invars)

        # sub-terms
        divergence = sum([(self.rho * vel).diff(var) for var, vel in zip(xyz, uvw)])
        convection = list(sum([wave * vel.diff(var) for var, wave in zip(xyz, uvw)]) for vel in uvw)
        diffusion = list(sum([vel.diff(var, 2) for var in xyz]) for vel in uvw)

        # continuity equation
        self.equations = {}
        self.equations["continuity"] = self.rho.diff(time) + divergence

        # momentums
        for i, orientaton in enumerate(["x", "y", "z"]):
            self.equations[f"momentum_{orientaton}"] = (
                uvw[i].diff(time) + convection[i] +
                pres.diff(xyz[i]) / self.rho - self.kvis * diffusion[i] - grav[i]
            )


class Vorticity(_PDES):
    """Vorticity calculations.
    """

    def __init__(self, dim=2):  # pylint: disable=super-init-not-called
        assert dim in [1, 2, 3], f"Illegal dimension: {dim}"

        # all possible independent variables
        xyz = list(_symbols("x, y, z"))

        # actual independent variables depend on problem's dimension
        invars = xyz[:dim]

        # dependent variables (using constant 0 makes derivatives trivial when not needed)
        uvw = [
            _Function("u")(*invars),  # pylint: disable=not-callable
            _Function("v")(*invars) if dim > 1 else _Number(0),  # pylint: disable=not-callable
            _Function("w")(*invars) if dim > 2 else _Number(0),  # pylint: disable=not-callable
        ]

        self.equations = {}
        self.equations["vorticity_x"] = uvw[2].diff(xyz[1]) - uvw[1].diff(xyz[2])
        self.equations["vorticity_y"] = uvw[0].diff(xyz[2]) - uvw[2].diff(xyz[0])
        self.equations["vorticity_z"] = uvw[1].diff(xyz[0]) - uvw[0].diff(xyz[1])


class QCriterion(_PDES):
    """Q criterion.
    """

    def __init__(self, dim=2):  # pylint: disable=super-init-not-called
        assert dim in [1, 2, 3], f"Illegal dimension: {dim}"

        # all possible independent variables
        xyz = list(_symbols("x, y, z"))

        # actual independent variables depend on problem's dimension
        invars = xyz[:dim]

        # dependent variables (using constant 0 makes derivatives trivial when not needed)
        uvw = [
            _Function("u")(*invars),  # pylint: disable=not-callable
            _Function("v")(*invars) if dim > 1 else _Number(0),  # pylint: disable=not-callable
            _Function("w")(*invars) if dim > 2 else _Number(0),  # pylint: disable=not-callable
        ]

        self.equations = {
            "qcriterion": (
                - (uvw[0].diff(xyz[0])**2 + uvw[1].diff(xyz[1])**2 + uvw[2].diff(xyz[2])**2) / 2
                - uvw[0].diff(xyz[1]) * uvw[1].diff(xyz[0])
                - uvw[0].diff(xyz[2]) * uvw[2].diff(xyz[0])
                - uvw[1].diff(xyz[2]) * uvw[2].diff(xyz[1])
            )
        }


class VelocityGradients(_PDES):
    """Gradients of velocity.
    """

    def __init__(self, dim=2):  # pylint: disable=super-init-not-called
        assert dim in [1, 2, 3], f"Illegal dimension: {dim}"

        # all possible independent variables
        xyz = list(_symbols("x, y, z"))

        # actual independent variables depend on problem's dimension
        invars = xyz[:dim]

        # dependent variables (using constant 0 makes derivatives trivial when not needed)
        uvw = [
            _Function("u")(*invars),  # pylint: disable=not-callable
            _Function("v")(*invars) if dim > 1 else _Number(0),  # pylint: disable=not-callable
            _Function("w")(*invars) if dim > 2 else _Number(0),  # pylint: disable=not-callable
        ]

        self.equations = {
            "u_x": uvw[0].diff(xyz[0]),
            "u_y": uvw[0].diff(xyz[1]),
            "u_z": uvw[0].diff(xyz[2]),
            "v_x": uvw[1].diff(xyz[0]),
            "v_y": uvw[1].diff(xyz[1]),
            "v_z": uvw[1].diff(xyz[2]),
            "w_x": uvw[2].diff(xyz[0]),
            "w_y": uvw[2].diff(xyz[1]),
            "w_z": uvw[2].diff(xyz[2]),
        }
