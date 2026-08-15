# SPDX-FileCopyrightText: 2026-present Aleksandrs Gehsbargs <agehsbargs.work@gmail.com>
#
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(eq=False)
class ExpNormalFit:
    """Structured result returned by an exponential-normal fit."""

    parameters: dict
    weights: dict
    converged: bool
    n_iter: int
    responsibilities: np.ndarray
    log_likelihood_history: list
    deltas: list
    total_delta: float

    @property
    def log_likelihood(self):
        if not self.log_likelihood_history:
            return None
        return self.log_likelihood_history[-1]


@dataclass(eq=False)
class MultinomialFit:
    """Structured result returned by a multinomial mixture fit."""

    parameters: dict
    weights: np.ndarray
    mixture_profiles: np.ndarray
    converged: bool
    n_iter: int
    responsibilities: np.ndarray
    profiles_delta: float
    log_likelihood_history: Optional[list] = None

    @property
    def log_likelihood(self):
        if not self.log_likelihood_history:
            return None
        return self.log_likelihood_history[-1]


@dataclass(eq=False)
class ExpNormalSimulation:
    """Structured result returned by an exponential-normal simulation."""

    data: np.ndarray
    parameters: dict


@dataclass(eq=False)
class MultinomialSimulation:
    """Structured result returned by a multinomial mixture simulation."""

    data: np.ndarray
    labels: np.ndarray
    mixture_profiles: np.ndarray
    parameters: dict
