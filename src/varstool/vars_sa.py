import pandas as pd
import numpy  as np
import sampling.starvars as sv
import sa.SA as sens_analysis # this is vague - correct it if you can
import matplotlib.pyplot as plt

from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
    Callable,
    Any,
)

from typing_extensions import (
    Protocol,
)

from collections.abc import (
    Iterable,
)


__all__ = ["VARS", "GVARS", "DVARS", "Sampler", "Model"]


class Sampler(Protocol):
    __doc__ = """A sampling class the returns random numbers based
    on **kwargs variables"""

    def __init__(
        self,
        callback: Callable = None,
        seed: Optional[int] = None,
        args: Any = None,
        kwargs: Dict[str, Any] = None,
    ) -> None:

        # initialize instance values
        self.callback = callback
        self.kwargs   = size_kwargs

    def __call__(self, ) -> Union[Iterable, float]:
        return self.callback(**self.kwargs)


class Model(Protocol):
    __doc__ = """A model class that runs the model in a iteration
    (if needed) and returns values"""

    def __init__(self, ): ...

    def __iter__(self, ): ...

    def __next__(self, ): ...


class VARS(object):
    __doc__ = """VARS object"""

    #-------------------------------------------
    # Constructors

    def __init__(
        self,
        parameters: Dict[str, Tuple[float, float]] = None,
        delta_h: Optional[float] = 0.1,
        num_stars: Optional[int] = 100,
        ivars_scales: Optional[Tuple[float, ...]] = (0.1, 0.3, 0.5),
        sampler: Sampler = None,
        model: Model = None,
        seed: Optional[int] = 123456789,
        bootstrap_flag: Optional[bool] = False,
        bootstrap_size: Optional[int]  = 1000,
        bootstrap_ci: Optional[int] = 0.9,
        report_verbose: Optional[bool] = False,
        report_freq: int = 10,
    ) -> None:

        # initialize values
        self.parameters = parameters
        self.delta_h = delta_h
        self.num_stars = num_stars
        self.star_centres = 5
        self.star_points  = 7

    @classmethod
    def from_dict(cls, input_dict):

        return cls()

    @classmethod
    def from_text(cls, input_text_file):

        return cls()


    #-------------------------------------------
    # Representators
    def __repr__(self, ):

        return "test"

    def _repr_html(self, ):

        pass

    def __str__(self, ):

        return


    #-------------------------------------------
    # Core properties
    @property
    def centres(self, ):
        return self.__star_centres

    @centres.setter
    def centres(self, new_centres):
        self.__star_centres = new_centres

    @property
    def points(self, ):
        return self.__star_points

    @points.setter
    def points(self, new_points):
        self.__star_points = new_points


    #-------------------------------------------
    # Core functions
    @staticmethod
    def generate_star(star_centres):

        # generate star points using star.py functions
        return sv.star_vars(star_centres, delta_h=self.delta_h, rettype='DataFrame')


    def _plot(self, ):

        # make similar plots as the matlab plots showing the important
        # SA results from analysis

        pass


    def run_online(self, ):
        # create instance of sampler to get star centers

        # generate star points

        # do analysis on model using variogram, IVARS, etc.

        # figure out a way to return results
        return


    def run_offline(star_points,):

        # do analysis on offline formatted star points

        # figure out a way to return results

        return


class GVARS(VARS):
    def __init__(self, ):
        super().__init__()


class DVARS(VARS):
    def __init__(self, ):
        super().__init__()

