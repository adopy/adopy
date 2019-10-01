from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple

import numpy as np
import pandas as pd

from adopy.functions import (
    get_nearest_grid_index,
    get_random_design_index,
)
from adopy.types import array_like

from ._task import Task
from ._model import Model
from ._engine import Engine

__all__ = ['EngineHR']


class EngineHR(Engine):
    def __init__(self,
                 task: Task,
                 model: Model,
                 grid_design: Dict[str, Any],
                 grid_param: Dict[str, Any],
                 lambda_et: Optional[float] = None,
                 dtype: Optional[Any] = np.float32):
        super(EngineHR, self).__init__(task, model, grid_design, grid_param,
                                       dtype)

        if model.task != task:
            raise ValueError('Given task and model are not matched.')

        self.reset()

        self.lambda_et = lambda_et

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def lambda_et(self):
        return self._lambda_et

    @lambda_et.setter
    def lambda_et(self, v):
        if v and not (0 <= v <= 1):
            raise ValueError('Invalid value for lambda_et')

        self._lambda_et = v

    ###########################################################################
    # Methods
    ###########################################################################

    def reset(self):
        """
        Reset the engine as in the initial state.
        """
        super(EngineHR, self).reset()

        self.eligibility_trace = np.zeros(self.grid_design.shape[0],
                                          dtype=self.dtype)

    def get_design(self, kind='optimal'):
        # type: (str) -> pd.Series
        r"""
        Choose a design with a given type.

        * ``optimal``: an optimal design :math:`d^*` that maximizes the mutual
          information.
        * ``random``: a design randomly chosen.

        Parameters
        ----------
        kind : {'optimal', 'random'}, optional
            Type of a design to choose

        Returns
        -------
        design : array_like
            A chosen design vector
        """

        if kind == 'optimal':
            self._update_mutual_info()
            idx_design = np.argmax(
                self.mutual_info * (1 - self.eligibility_trace))

        elif kind == 'random':
            idx_design = get_random_design_index(self.grid_design)

        else:
            raise ValueError(
                'The argument kind should be "optimal" or "random".')

        return self.grid_design.iloc[idx_design]

    def update(self, design, response):
        r"""
        Update the posterior :math:`p(\theta | y_\text{obs}(t), d^*)` for
        all discretized values of :math:`\theta`.

        .. math::
            p(\theta | y_\text{obs}(t), d^*) =
                \frac{ p( y_\text{obs}(t) | \theta, d^*) p_t(\theta) }
                    { p( y_\text{obs}(t) | d^* ) }

        Parameters
        ----------
        design
            Design vector for given response
        response
            Any kinds of observed response
        """
        super(EngineHR, self).update(design, response)

        idx_design = get_nearest_grid_index(design, self.grid_design)

        if self.lambda_et:
            self.eligibility_trace *= self.lambda_et
            self.eligibility_trace[idx_design] += 1
