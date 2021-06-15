import warnings
import decimal

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

from .sampling import starvars
from .sa import vars_funcs

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
    runtime_checkable,
)

from collections.abc import (
    Iterable,
)


@runtime_checkable
class Sampler(Protocol):
    __doc__ = """A sampling class the returns random numbers based
    on **kwargs arguments"""

    def __init__(
        self,
        func: Callable = None,
        seed: Optional[int] = None,
        unknown_options: Dict[str, Any] = None,
    ) -> None:

        # initialize instance values
        self.callback = callback
        self.kwargs   = kwargs

    def __call__(self, ) -> Union[Iterable, float]:
        return self.callback(**self.kwargs)


@runtime_checkable
class Model(Protocol):
    __doc__ = """A wrapper class for a function of interest"""

    def __init__(
        self, 
        func: Callable = None,
        unkown_options: Dict = None,
    ) -> None:

        # check whether the input is a callable
        assert callable(func)
        self.func = func

        # unkown_options must be a dict
        assert isinstance(unkown_options, dict)
        if unknown_options:
            self.unknown_options = unkown_options
        else:
            self.unknown_options = {}

    def __repr__(self, ) -> str:
        """official representation"""
        return "wrapped function: "+self.func.__name__

    def __str__(self, ) -> str:
        """the name of the wrapper function"""
        return self.func.__name__

    def __call__(
        self,
        params,
        options: Dict = None,
    ) -> Union[Iterable, float, int]:

        # check if params is array-like object
        assert isinstance(params, \
            (pd.DataFrame, pd.Series, np.array, list, tuple))

        if options:
            self.unknown_options = options

        return self.func(params, **self.unknown_options)


class VARS(object):
    __doc__ = """VARS object"""

    #-------------------------------------------
    # Constructors

    def __init__(
        self,
        parameters: Dict[Union[str, int], Tuple[float, float]] = {}, # name and bounds
        delta_h: Optional[float] = 0.1, # delta_h for star sampling
        num_stars: Optional[int] = 100, # number of star points
        ivars_scales: Optional[Tuple[float, ...]] = (0.1, 0.3, 0.5), # ivars scales
        sampler: Sampler = None, # sampling method for star centres
        model: Model = None, # model (function) to run for each star point
        seed: Optional[int] = 123456789, # randomization state
        bootstrap_flag: Optional[bool] = False, # bootstrapping flag
        bootstrap_size: Optional[int]  = 1000, # bootstrapping size
        bootstrap_ci: Optional[int] = 0.9, # bootstrap confidence interval
        report_verbose: Optional[bool] = False, # reporting - using tqdm??
        report_freq: int = 10, # not sure whether we should include this?
    ) -> None:

        # initialize values
        self.parameters = parameters
        self.delta_h = delta_h
        self.num_stars = num_stars
        self.ivars_scales = ivars_scales
        self.__star_centres = None # no default value required
        self.__star_points  = None # no default value required
        self.seed = seed
        self.bootstrap_flag = bootstrap_flag
        self.bootstrap_size = bootstrap_size
        self.bootstrap_ci = bootstrap_ci
        self.report_verbose = report_verbose


        # Check input arguments
        ## default value for the IVARS scales are 0.1, 0.3, and 0.5
        if not self.ivars_scales:
            warnings.warn(
                "IVARS scales are not valid, default values of (0.1, 0.3, 0.5) \n"
                "will be considered.",
                UserWarning,
                stacklevel=1
            )
            self.ivars_scales = (0.1, 0.3, 0.5)

        ## if there is any value in IVARS scales that is greater than 0.5
        if any(i for i in self.ivars_scales if i > 0.5):
            warnings.warn(
                "IVARS scales greater than 0.5 are not recommended.",
                UserWarning,
                stacklevel=1
            )

        ## if delta_h is not a factor of 1, NaNs or ZeroDivison might happen
        if (decimal.Decimal(str(1)) % decimal.Decimal(str(self.delta_h))) != 0:
            warnings.warn(
                "If delta_h is not a factor of 1, NaNs and ZeroDivisionError are probable. \n"
                "It is recommended to change `delta_h` to a divisible number by 1.",
                RuntimeWarning,
                stacklevel=1
            )

        ## if delta_h is not between 0 and 1
        if ((delta_h <= 0) or (delta_h >=1 )):
            raise ValueError(
                "`delta_h` must be greater than 0 and less than 1."
            )

        ## check if num_stars is a positive integer
        if ((not isinstance(num_stars, (int, np.int32, np.int64))) or \
            (num_stars < 0)):
            raise ValueError(
                "`num_stars` must be a positive integer."
            )

        ## check seed dtype and sign
        if ((not isinstance(seed, int)) or (seed < 0)):
            warnings.warn(
                "`seed` must be an integer greater than zero."
                " value is set to default, i.e., 123456789"
            )
            self.seed = 123456789

        ## check bootstrap dtype and sign
        if ((not isinstance(bootstrap_size, int)) or (bootstrap_size < 1)):
            raise ValueError(
                "`bootstrap_size` must be an integer greater than one."
                " value is set to default, i.e., 1000"
            )

        ## check bootstrap ci dtype, value and sign
        if ((not isinstance(bootstrap_ci, float)) or \
            (bootstrap_ci <= 0) or \
            (bootstrap_ci >= 1)):
            warnings.warn(
                "bootstrap condifence interval (CI) must be a float"
                " within (0, 1) interval. Setting `boostrap_ci` value"
                " to default, i.e., 0.9"
            )
            self.bootstrap_ci = 0.9

        ## check the dtypes and instances
        ### `parameters`
        if not isinstance(parameters, dict):
            raise ValueError(
                "`parameters` should be of type `dict`; the keys must be"
                "their names, and values must be the lower and upper bounds"
                "of their factor space."
            )

        ### `sampler`
        if sampler:
            if not isinstance(sampler, Sampler):
                raise ValueError(
                    "`sampler` algorithm must be of type varstool.Sampler."
                )
        self.sampler = sampler

        ### `model`
        if model:
            if not isinstance(model, Model):
                raise ValueError(
                    "`model` must be of type varstool.Model."
                )
        self.model = model

        # adding anything else here?!

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

    ## using dunder variables for avoiding confusion with
    ## D-/GVARS sublcasses.

    @property
    def centres(self, ):
        return self.__star_centres

    @centres.setter
    def centres(self, new_centres):
        if not isinstance(new_centres, \
            (pd.DataFrame, pd.Series, np.array, list, tuple)):
            raise TypeError(
                "new_centres must be an array-like object: "
                "pandas.Dataframe, pandas.Series, numpy.array, list, tuple"
            )
        self.__star_centres = new_centres

    @property
    def points(self, ):
        return self.__star_points

    @points.setter
    def points(self, new_points):
        if not isinstance(new_points, \
            (pd.DataFrame, pd.Series, np.array, list, tuple)):
            raise TypeError(
                "new_points must be an array-like object: "
                "pandas.Dataframe, pandas.Series, numpy.array, list, tuple"
            )
        self.__star_points = new_points


    #-------------------------------------------
    # Core functions
    @staticmethod
    def generate_star(star_centres, delta_h, param_names=[]):

        # generate star points using star.py functions
        star_points = starvars(star_centres, delta_h=delta_h, parameters=param_names, rettype='DataFrame')

        # figure out way to return this?
        return star_points  # for now will just do this


    def _plot(self, ):

        # make similar plots as the matlab plots showing the important
        # SA results from analysis

        pass


    # param_names is the name of the model parameters might need a better name
    def run_online(self, param_names=[]):
        # call sampler to get star centres
        self.centres(self.sampler())

        # generate star points
        self.points(starvars(star_centres=self.centres(), delta_h=self.delta_h, parameters=param_names, rettype='DataFrame'))

        # apply model to the generated star points
        df = vars_funcs.apply_unique(self.model, self.points())
        df.index.names = ['centre', 'param', 'points']

        # get paired values for each section based on 'h'
        pair_df = df[self.model.__name__].groupby(level=[0,1]).apply(vars_funcs.section_df)
        pair_df.index.names = ['centre', 'param', 'h', 'pair_ind']

        # get mu_star value
        mu_star_df = df[self.model.__name__].groupby(level=[0,1]).mean()
        mu_star_df.index.names = ['centre', 'param']

        # overall mean of the unique evaluated function value over all star points
        mu_overall = df[self.model.__name__].unique().mean()

        # overall variance of the unique evaluated function over all star points
        var_overall = df[self.model.__name__].unique().var(ddof=1)

        # sectional covariogram calculation
        cov_section_all = vars_funcs.cov_section(pair_df, mu_star_df)

        # variogram calculation
        self.variogram_value = vars_funcs.variogram(pair_df)

        # morris calculation
        self.morris_values = vars_funcs.morris_eq(pair_df)

        # overall covariogram calculation
        self.covariogram_value = vars_funcs.covariogram(pair_df, mu_overall)

        # expected value of the overall covariogram calculation
        self.e_covariogram_value = vars_funcs.e_covariogram(cov_section_all)

        # sobol calculation
        self.sobol_value = vars_funcs.sobol_eq(self.variogram_value, self.e_covariogram_value, var_overall)

        # IVARS calculation
        self.ivars_df = pd.DataFrame.from_dict({scale: self.variogram_value.groupby(level=0).apply(vars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
             for scale in self.ivars_scales}, 'index')

        if self.bootstrap_flag:
            # create result dataframes/series if bootstrapping is chosen to be done
            result_bs_variogram = pd.Series(dtype='float64')
            result_bs_sobol = pd.Series(dtype='float64')
            result_bs_ivars_df = pd.DataFrame()

            for i in range(0, self.bootstrap_size):
                # bootstrapping to get CIs
                # specify random sequence by sampling with replacement
                bootstrap_rand = np.random.choice(list(range(0, 10)), size=len(range(0, 10)), replace=True).tolist()
                bootstrapped_pairdf = pd.concat([pair_df.loc[pd.IndexSlice[i, :, :, :], :] for i in bootstrap_rand])
                bootstrapped_df = pd.concat([df.loc[pd.IndexSlice[i, :, :], :] for i in bootstrap_rand])

                # calculating sectional covariograms
                bootstrapped_cov_section_all = pd.concat([cov_section_all.loc[pd.IndexSlice[i, :]] for i in bootstrap_rand])

                # calculating variogram, ecovariogram, variance, mean, Sobol, and IVARS values
                bootstrapped_variogram = vars_funcs.variogram(bootstrapped_pairdf)

                bootstrapped_ecovariogram = vars_funcs.e_covariogram(bootstrapped_cov_section_all)

                bootstrapped_var = bootstrapped_df[self.model.__name__].unique().var(ddof=1)

                bootstrapped_sobol = vars_funcs.sobol_eq(bootstrapped_variogram, bootstrapped_ecovariogram, bootstrapped_var)

                bootstrapped_ivars_df = pd.DataFrame.from_dict(
                    {scale: bootstrapped_variogram.groupby(level=0).apply(vars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
                     for scale in self.ivars_scales}, 'index')

                # unstack variogram for sorting later on
                bootstrapped_variogram_df = bootstrapped_variogram.unstack(level=0)

                # attatch new results to previous results (order does not matter here)
                result_bs_variogram = pd.concat([bootstrapped_variogram_df, result_bs_variogram])
                result_bs_sobol = pd.concat([bootstrapped_sobol, result_bs_sobol])
                result_bs_ivars_df = pd.concat([bootstrapped_ivars_df, result_bs_ivars_df])

            # need to do data manipulation to variogram and sobol results
            # in order sort all the parameters independently
            #TODO


            # result_bs_variogram.sort_values(param_names) # not finished
            # result_bs_sobol.sort_values(param_names) # not finished

            # calculating the confidence interval limits for variogram/sobol results
            # this part needs to be fixed also
            # self.variogram_low = result_bs_variogram.iloc[self.bootstrap_size*(1-self.bootstrap_ci)/2]
            # self.variogram_upp = result_bs_variogram.iloc[self.bootstrap_size*(1 - ((1 - self.bootstrap_ci)/2))]

            # self.sobol_low = result_bs_sobol.iloc[self.bootstrap_size*(1-self.bootstrap_ci)/2]
            # self.sobol_upp = result_bs_sobol.iloc[self.bootstrap_size*(1 - ((1 - self.bootstrap_ci)/2))]

            # calculate upper and lower confidence interval limits of the ivars values
            self.ivars_low = pd.DataFrame()
            self.ivars_upp = pd.DataFrame()
            for scale in self.ivars_scales:
                self.ivars_low = pd.concat(
                    [self.ivars_low, result_bs_ivars_df.loc[scale].quantile((1 - 0.9) / 2).rename(scale).to_frame()], axis=1)
                self.ivars_upp = pd.concat(
                    [self.ivars_upp, result_bs_ivars_df.loc[scale].quantile(1 - ((1 - 0.9) / 2)).rename(scale).to_frame()],
                    axis=1)

            # transpose the results to get them in the right format
            self.ivars_low = self.ivars_low.transpose()
            self.ivars_upp = self.ivars_upp.transpose()


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
