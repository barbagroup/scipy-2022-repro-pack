#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Custom inferencers for Modulus.
"""
import io as _io
import logging as _logging
import lzma as _lzma
import pathlib as _pathlib
from datetime import datetime as _datetime
from datetime import timezone as _timezone

import torch as _torch
from modulus.continuous.inferencer.inferencer import Inferencer as _Inferencer
from modulus.distributed.manager import DistributedManager as _DistributedManager


class SaveModelInferencer(_Inferencer):
    """An inferencer to only save the model parameters.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, nodes, name):
        self.manager = _DistributedManager()
        self.device = self.manager.device
        self.nodes = nodes
        self.name = name
        self.log = _logging.getLogger(__name__)

        for node in nodes:
            if hasattr(node, "name") and node.name == self.name:
                self.model = node.evaluate
                break
        else:
            raise ValueError(f"Couldn't find network {name} in the nodes")


    def save_results(self, name, prefix, writer, save_filetypes, step):
        """Save the model parameters to a file.
        """
        # pylint: disable=unused-argument, arguments-differ, too-many-arguments

        filename = _pathlib.Path(prefix).joinpath(f"{self.name}-{step:07d}.pth")
        time = _datetime.utcnow().replace(tzinfo=_timezone.utc).isoformat()

        with _io.BytesIO() as mem:
            if hasattr(self.model, "module"):
                _torch.jit.save(self.model.module, mem)
            else:
                _torch.jit.save(self.model, mem)
            mem.seek(0)

            with _lzma.open(filename, "wb") as fobj:
                _torch.save({"step": step, "time": time, "model": mem.read()}, fobj)

        self.log.info("Done saving model for step %d to %s", step, str(filename))
