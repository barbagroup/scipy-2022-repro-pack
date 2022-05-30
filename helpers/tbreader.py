#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Read tensorboard data

"""
import pathlib
import pandas
from tensorboard.backend.event_processing import event_accumulator


def read_tensorboard_data(workdir: pathlib.Path):
    """Read tensorboard events.
    """

    # get a list of all event files
    filenames = workdir.glob("**/events.out.tfevents.*")

    # data holder
    data = []

    # read events, one dataframe per event file
    for filename in filenames:
        reader = event_accumulator.EventAccumulator(
            path=str(filename),
            size_guidance={event_accumulator.TENSORS: 0}
        )
        reader.Reload()

        keymap = {
            "Train/loss_aggregated": "loss",
            "Monitors/pde_residual/continuity_res": "cont_res",
            "Monitors/pde_residual/momentum_x_res": "momem_x_res",
            "Monitors/pde_residual/momentum_y_res": "momem_y_res",
            "Monitors/pde-residual/continuity_res": "cont_res",  # underscore was replaced by hyphen at some point
            "Monitors/pde-residual/momentum_x_res": "momem_x_res",  # underscore was replaced by hyphen at some point
            "Monitors/pde-residual/momentum_y_res": "momem_y_res",  # underscore was replaced by hyphen at some point
        }

        frame = pandas.DataFrame()

        for key, name in keymap.items():
            try:
                temp = pandas.DataFrame(reader.Tensors(key)).set_index("step")
            except KeyError as err:
                if "not found in Reservoir" in str(err):
                    continue
                raise

            temp[name] = temp["tensor_proto"].apply(lambda inp: inp.float_val[0])

            if "wall_time" in frame.columns:
                temp = temp.drop(columns=["tensor_proto", "wall_time"])
            else:
                temp = temp.drop(columns=["tensor_proto"])

            frame = frame.join(temp, how="outer")

        # add to collection
        data.append(frame)

    # concatenate (and sort) all partial individual dataframes
    data = pandas.concat(data).sort_index()

    return data.reset_index(drop=False)


# a test
if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[1]

    data = read_tensorboard_data(
        root.joinpath("cases", "taylor-green-vortex-2d", "one-net", "re100", "a100_8", "outputs")
    )

    print(data)
