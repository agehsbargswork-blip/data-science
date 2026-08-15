# SPDX-FileCopyrightText: 2026-present Aleksandrs Gehsbargs <agehsbargs.work@gmail.com>
#
# SPDX-License-Identifier: MIT
from .generator import Generator
from .models import Models


def fit_exp_normal(data, control=None, start=None):
    """Fit an exponential-normal mixture model."""
    return Models(
        data=data,
        control=control,
        start=start,
    ).fit_exp_norm()


def fit_multinomial(data, n_components, control=None, start=None):
    """Fit a multinomial mixture model."""
    return Models(
        data=data,
        control=control,
        start=start,
    ).fit_multinom(n_components=n_components)


def simulate_exp_normal(size, parameters=None):
    """Simulate observations from an exponential-normal mixture."""
    return Generator(
        size=size,
        parameters=parameters,
    ).gen_exp_norm()


def simulate_multinomial(size, parameters=None):
    """Simulate observations from a multinomial mixture."""
    return Generator(
        size=size,
        parameters=parameters,
    ).gen_multinom()
