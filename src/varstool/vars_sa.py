import warnings
import decimal

import pandas as pd
import numpy  as np
import sampling.starvars as sv
import sa.vars_funcs as vf
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
    on **kwargs arguments"""

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
        parameters: Dict[Union[str, int], Tuple[float, float]] = None, # name and bounds
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
        self.star_centres = None # no default value required
        self.star_points  = None # no default value required
        self.seed = seed
        self.bootstrap_size = bootstrap_size
        self.bootstrap_ci = bootstrap_ci
        self.report_verbose = report_verbose


        # Check input arguments
        ## default value for the IVARS scales are 0.1, 0.3, and 0.5
        if not self.ivars_scales:
            warnings.warn(
                "IVARS scales are not valid, default values of (0.1, 0.3, 0.5)"
                "will be considered.",
                UserWarning,
                stacklevel=1
            )
            self.ivars_scales = (0.1, 0.3, 0.5)
        ## if there is any value in IVARS scales that is greater than 0.5
        if any(i for i in self.ivars_scales if i > 0.5):
            warning.warn(
                "IVARS scales greater than 0.5 are not recommended.",
                UserWarning,
                stacklevel=1
            )
        ## if delta_h is not a factor of 1, NaNs or ZeroDivison might happen
        if (decimal.Decimal(str(1)) % decimal.Decimal(str(self.delta_h))) != 0:
            warning.warn(
                "If delta_h is not a factor of 1, NaNs and ZeroDivisionError are probable. "
                "It is recommended to change `delta_h` to a divisible number by 1.",
                RuntimeWarning,
                stacklevel=1
            )
        ## if delta_h is not between 0 and 1
        if ((delta_h <= 0) or (delta_h >=1 )):
            raise ValueError(
                "`delta_h` must be greater than 0 and less than 1."
            )


        ## check the dtypes and instances
        ### `parameters`
        if not isinstance(parameters, dict):
            raise ValueError(
                "`parameters` should be of type `dict`; the keys must be"
                "their names, and values must be the lower and upper bounds"
                "of their factor space."
            )
        ### `sampler`
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "`sampler` algorithm must be of type varstool.Sampler."
            )
        ### `model`
        if not isinstance(model, Model):
            raise ValueError(
                "`model` must be of type varstool.Model."
            )












        # default value for seed is 123456789
        if seed is None:
            self.seed = 123456789
        else:
            self.seed = seed

        # default value for bootstrap flag is false
        if bootstrap_flag is None:
            self.bootstrap_flag = False
        else:
            self.bootstrap_flag = bootstrap_flag

        # default value for bootstrap size is 1000
        if bootstrap_size is None:
            self.bootstrap_size = 1000
        else:
            self.bootstrap_size = bootstrap_size

        # default bootstrap confidence interval is 0.9
        if bootstrap_ci is None:
            self.bootstrap_ci = 0.9
        else:
            self.bootstrap_ci = bootstrap_ci

        # default value for report_verbose is false
        if report_verbose is None:
            self.report_verbose = False
        else:
            self.report_verbose = report_verbose


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
        star_points = sv.star_vars(star_centres, delta_h=delta_h, parameters=parameters, rettype='DataFrame')

        # figure out way to return this?
        return star_points  # for now will just do this


    def _plot(self, ):

        # make similar plots as the matlab plots showing the important
        # SA results from analysis

        pass


    def run_online(self, ):
        # create instance of sampler to get star centers (random numbers)
        sample = Sampler(self.sampler, self.parameters)
        self.centres(sample())

        # generate star points
        self.points(sv.star_vars(star_centres=self.centres(), delta_h=self.delta_h, parameters=param_names, rettype='DataFrame'))

        # apply model to the generated star points
        df = vf.apply_unique(self.model, self.points())

        # possibly do some data manipulation here not sure yet

        # get paired values for each section based on 'h'
        pair_df = df[self.model.__name__].groupby(level=[0,1]).apply(vf.section_df)

        # get mu_star value?
        mu_star_df = df[self.model.__name__].groupby(level=[0,1]).mean()

        # overall mean of the unique evaluated function value over all star points
        mu_overall = df[self.model.__name__].unique().mean()

        # overall variance of the unique evaluated function over all star points
        var_overall = df[self.model.__name__].unique().var(ddof=1)

        # also possibly add data manipulation to pair_df and mu_star_df not sure

        # sectional covariogram calculation
        cov_section_all = vf.cov_section(pair_df, mu_star_df)

        # variogram calculation
        variogram_value = vf.variogram(pair_df)

        # morris calculation
        morris_values = vf.morris_eq(pair_df)

        # overall covariogram calculation
        covariogram_value = vf.covariogram(pair_df, mu_overall)

        # expected value of the overall covariogram calculation
        e_covariogram_value = vf.e_covariogram(cov_section_all)

        # sobol calculation
        sobol_value = vf.sobol_eq(variogram_value, e_covariogram_value, var_overall)

        # IVARS calculation
        ivars_df = pd.DataFrame.from_dict({scale: variogram_value.groupby(level=0).apply(vf.ivars, scale=scale, delta_h=self.delta_h) \
             for scale in self.ivars_scales}, 'index')

        # gotta make sure this is in a for loop and fix the note book displays to work in .py maybe?
        if self.bootstrap_flag:
            # bootstrapping to get CIs
            # specify random sequence by sampling with replacement
            bootstrap_rand = np.random.choice(list(range(0, 10)), size=len(range(0, 10)), replace=True).tolist()
            bootstrapped_pairdf = pd.concat([pair_df.loc[pd.IndexSlice[i, :, :, :], :] for i in bootstrap_rand])
            bootstrapped_df = pd.concat([df.loc[pd.IndexSlice[i, :, :], :] for i in bootstrap_rand])
            display(bootstrapped_pairdf)
            display(bootstrap_rand)

            # calculating sectional covariograms
            bootstrapped_cov_section_all = pd.concat([cov_section_all.loc[pd.IndexSlice[i, :]] for i in bootstrap_rand])
            display('sectional variogram:')
            display(bootstrapped_cov_section_all)
            display(bootstrap_rand)

            # calculating variogram, ecovariogram, variance, mean, Sobol, and IVARS values
            bootstrapped_variogram = vf.variogram(bootstrapped_pairdf)
            display('variogram:')
            display(bootstrapped_variogram.unstack(level=0))

            bootstrapped_ecovariogram = vf.e_covariogram(bootstrapped_cov_section_all)
            display('E(covariogram):')
            display(bootstrapped_ecovariogram.unstack(level=0))

            bootstrapped_var = bootstrapped_df[self.model.__name__].unique().var(ddof=1)
            display('variance:', bootstrapped_var)

            bootstrapped_sobol = vf.sobol_eq(bootstrapped_variogram, bootstrapped_ecovariogram, bootstrapped_var)
            display('sobol:', bootstrapped_sobol)

            ivars_values = [0.1, 0.3, 0.5]
            delta_h = 0.1
            boostrapped_ivars_df = pd.DataFrame.from_dict(
                {scale: bootstrapped_variogram.groupby(level=0).apply(vf.ivars, scale=scale, delta_h=delta_h) \
                 for scale in ivars_values}, 'index')
            display('ivars:', boostrapped_ivars_df)

            # return results in a dictionary ***not finished yet
            return {
                'Directional_Variogram': 1,
                'Directional_Covariogram': 1,
                'Directional_Expected_Covariogram': 1,
                'Integrated_Variogram': 1,
                'Factor_Rankings(IVARS)': 1,
                'VARS-TO': 1,
                'Factor_Rankings(VARS_TO)': 1,
                'VARS-ABE': 1,
                'VARS-ACE': 1,
                'gamma_LL': 1,
                'gamma_UL': 1,
                'IVARS_LL': 1,
                'IVARS_UL': 1,
                'VARS-TO_LL': 1,
                'VARS-TO_UL': 1,
                'Reliability_Estimates(IVARS)': 1,
                'Factor_Grouping(VARS-TO/IVARS50)': 1,
                'Reliability_Estimates(Factor_Grouping)': 1
            }


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

