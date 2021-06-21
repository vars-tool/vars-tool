import warnings
import decimal

import pandas as pd
import numpy  as np
#import numpy.typing as npt # let's further investigate this
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
    List,
)

from typing_extensions import (
    runtime_checkable,
)

from collections.abc import (
    Iterable,
)


class Model():
    __doc__ = """A wrapper class for a function of interest"""

    def __init__(
        self, 
        func: Callable = None,
        unknown_options: Dict[str, Any] = {},
    ) -> None:

        # check whether the input is a callable
        assert callable(func)
        self.func = func

        # unkown_options must be a dict
        assert isinstance(unknown_options, dict)
        self.unknown_options = {}
        
        if unknown_options:
            self.unknown_options = unknown_options

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

        # check if params is an array-like object
        assert isinstance(params, 
            (pd.DataFrame, pd.Series, np.ndarray, List, Tuple))

        if options:
            self.unknown_options = options

        return self.func(params, **self.unknown_options)


class VARS(object):
    __doc__ = """VARS object"""

    #-------------------------------------------
    # Constructors

    def __init__(
        self,
        star_centres = [],  # sampled star centres (random numbers) used to create star points
        parameters: Dict[Union[str, int], Tuple[float, float]] = {}, # name and bounds
        delta_h: Optional[float] = 0.1, # delta_h for star sampling
        ivars_scales: Optional[Tuple[float, ...]] = (0.1, 0.3, 0.5), # ivars scales
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
        self.ivars_scales = ivars_scales
        self.__star_centres = star_centres
        self.__star_points  = [] # an empty list works for now - but needs to be changed
        self.seed = seed
        self.bootstrap_flag = bootstrap_flag
        self.bootstrap_size = bootstrap_size
        self.bootstrap_ci = bootstrap_ci
        self.report_verbose = report_verbose

        # analysis stage is set to False before running anything
        self.run_status = False

        # Check input arguments
        # ***add error checking, and possibily default value for star centres?


        ## default value for the IVARS scales are 0.1, 0.3, and 0.5
        if not self.ivars_scales:
            warnings.warn(
                "IVARS scales are not valid, default values of (0.1, 0.3, 0.5) "
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

        ### `model`
        if model:
            if not isinstance(model, Model):
                raise TypeError(
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
    def __repr__(self, ) -> str:
        """show the status of VARS analysis"""

        status_star_centres = "Star Centres: " + ("Loaded" if len(self.__star_centres) != 0 else "Not Loaded")
        status_star_points = "Star Points: " + ("Loaded" if len(self.__star_points) != 0 else "Not Loaded")
        status_parameters = "Parameters: " + (str(len(self.parameters))+" paremeters set" if self.parameters else "None")
        status_delta_h = "Delta h: " + (str(self.delta_h)+"" if self.delta_h else "None")
        status_model = "Model: " + (str(self.model)+"" if self.model else "None")
        status_seed = "Seed Number: " + (str(self.seed)+"" if self.seed else "None")
        status_bstrap = "Bootstrap: " + ("On" if self.bootstrap_flag else "Off")
        status_bstrap_size = "Bootstrap Size: " + (str(self.bootstrap_size)+"" if self.bootstrap_flag else "N/A")
        status_bstrap_ci = "Bootstrap CI: " + (str(self.bootstrap_ci)+"" if self.bootstrap_flag else "N/A")
        status_analysis = "VARS Analysis: " + ("Done" if self.run_status else "Not Done")

        status_report_list = [status_star_centres, status_star_points, status_parameters, \
                              status_delta_h, status_model, status_seed, status_bstrap, \
                              status_bstrap_size, status_bstrap_ci, status_analysis]

        return "\n".join(status_report_list)


    def _repr_html(self, ):

        pass


    def __str__(self, ) -> str:

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
        if not isinstance(new_centres, 
              (pd.DataFrame, pd.Series, np.ndarray, List, Tuple)):
            raise TypeError(
                "new_centres must be an array-like object: "
                "pandas.Dataframe, pandas.Series, numpy.array, List, Tuple"
            )
        self.__star_centres = new_centres

    @property
    def points(self, ):
        return self.__star_points

    @points.setter
    def points(self, new_points):
        if not isinstance(new_points, 
              (pd.DataFrame, pd.Series, np.ndarray, List, Tuple)):
            raise TypeError(
                "new_points must be an array-like object: "
                "pandas.Dataframe, pandas.Series, numpy.array, List, Tuple"
            )
        self.__star_points = new_points


    #-------------------------------------------
    # Core functions
    @staticmethod
    def generate_star(star_centres, delta_h, param_names):

        # generate star points using star.py functions
        star_points = starvars.star(star_centres, delta_h=delta_h, parameters=param_names, rettype='DataFrame')

        # figure out way to return this?
        return star_points  # for now will just do this


    def _plot(self, ):

        # make similar plots as the matlab plots showing the important
        # SA results from analysis

        pass


    # param_names is the name of the model parameters might need a better name
    def run_online(self, ):

        # generate star points
        self.__star_points = starvars.star(self.__star_centres, # star centres
                                           delta_h=self.delta_h, # delta_h
                                           parameters=[*self.parameters], # parameters dictionary keys
                                           rettype='DataFrame') # return type is a dataframe

        # apply model to the generated star points
        df = vars_funcs.apply_unique(self.model, self.__star_points)
        df.index.names = ['centre', 'param', 'points']

        # get paired values for each section based on 'h'
        pair_df = df[str(self.model)].groupby(level=[0,1]).apply(vars_funcs.section_df)
        pair_df.index.names = ['centre', 'param', 'h', 'pair_ind']

        # get mu_star value
        mu_star_df = df[str(self.model)].groupby(level=[0,1]).mean()
        mu_star_df.index.names = ['centre', 'param']

        # overall mean of the unique evaluated function value over all star points
        mu_overall = df[str(self.model)].unique().mean()

        # overall variance of the unique evaluated function over all star points
        var_overall = df[str(self.model)].unique().var(ddof=1)

        # sectional covariogram calculation
        cov_section_all = vars_funcs.cov_section(pair_df, mu_star_df)

        # variogram calculation
        self.variogram_value = vars_funcs.variogram(pair_df)

        # morris calculation
        self.morris_value = vars_funcs.morris_eq(pair_df)

        # overall covariogram calculation
        self.covariogram_value = vars_funcs.covariogram(pair_df, mu_overall)

        # expected value of the overall covariogram calculation
        self.e_covariogram_value = vars_funcs.e_covariogram(cov_section_all)

        # sobol calculation
        self.sobol_value = vars_funcs.sobol_eq(self.variogram_value, self.e_covariogram_value, var_overall)

        # do factor ranking on sobol results
        self.sobol_factor_ranking = self._factor_ranking(self.sobol_value)

        # IVARS calculation
        self.ivars_df = pd.DataFrame.from_dict({scale: self.variogram_value.groupby(level=0).apply(vars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
             for scale in self.ivars_scales}, 'index')

        # do factor ranking on IVARS results
        self.ivars_factor_ranking = self._factor_ranking(self.ivars_df)

        if self.bootstrap_flag:
            # create result dataframes if bootstrapping is chosen to be done
            result_bs_variogram = pd.DataFrame()
            result_bs_sobol = pd.DataFrame()
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

                bootstrapped_var = bootstrapped_df[self.model.func.__name__].unique().var(ddof=1)

                bootstrapped_sobol = vars_funcs.sobol_eq(bootstrapped_variogram, bootstrapped_ecovariogram, bootstrapped_var)

                bootstrapped_ivars_df = pd.DataFrame.from_dict(
                    {scale: bootstrapped_variogram.groupby(level=0).apply(vars_funcs.ivars, scale=scale, delta_h=self.delta_h) \
                     for scale in self.ivars_scales}, 'index')

                # unstack variogram so that results concat nicely
                bootstrapped_variogram_df = bootstrapped_variogram.unstack(level=0)

                # swap sobol results rows and columns so that results concat nicely
                bootstrapped_sobol_df = bootstrapped_sobol.to_frame().transpose()

                # attatch new results to previous results (order does not matter here)
                result_bs_variogram = pd.concat([bootstrapped_variogram_df, result_bs_variogram])
                result_bs_sobol = pd.concat([bootstrapped_sobol_df, result_bs_sobol])
                result_bs_ivars_df = pd.concat([bootstrapped_ivars_df, result_bs_ivars_df])

            # calculate upper and lower confidence interval limits for variogram results
            self.variogram_low = pd.DataFrame()
            self.variogram_upp = pd.DataFrame()
            # iterate through each h value
            for h in np.unique(result_bs_variogram.index.values).tolist():
                # find all confidence interval limits for each h value
                self.variogram_low = pd.concat(
                    [self.variogram_low, result_bs_variogram.loc[h].quantile((1 - self.bootstrap_ci) / 2).rename(h).to_frame()], axis=1)
                self.variogram_upp = pd.concat(
                    [self.variogram_upp, result_bs_variogram.loc[h].quantile(1 - ((1 - self.bootstrap_ci) / 2)).rename(h).to_frame()],
                    axis=1)

            # index value name is h?? not sure if this should be changed later
            self.variogram_low.index.names = ['h']
            self.variogram_upp.index.names = ['h']

            # transpose to get into correct format
            self.variogram_low = self.variogram_low.transpose()
            self.variogram_upp = self.variogram_upp.transpose()

            # calculate upper and lower confidence interval limits for sobol results in a nice looking format
            self.sobol_low = result_bs_sobol.quantile((1 - self.bootstrap_ci) / 2).rename('').to_frame().transpose()
            self.sobol_upp = result_bs_sobol.quantile(1 - ((1 - self.bootstrap_ci) / 2)).rename('').to_frame().transpose()

            # calculate upper and lower confidence interval limits of the ivars values
            self.ivars_low = pd.DataFrame()
            self.ivars_upp = pd.DataFrame()
            # iterate through each IVARS scale
            for scale in self.ivars_scales:
                # find confidence interval limits for each scale
                self.ivars_low = pd.concat(
                    [self.ivars_low, result_bs_ivars_df.loc[scale].quantile((1 - self.bootstrap_ci) / 2).rename(scale).to_frame()], axis=1)
                self.ivars_upp = pd.concat(
                    [self.ivars_upp, result_bs_ivars_df.loc[scale].quantile(1 - ((1 - self.bootstrap_ci) / 2)).rename(scale).to_frame()],
                    axis=1)

            # transpose the results to get them in the right format
            self.ivars_low = self.ivars_low.transpose()
            self.ivars_upp = self.ivars_upp.transpose()

        self.run_status = True


    def run_offline(star_points,):

        # do analysis on offline formatted star points

        # figure out a way to return results

        return


    def _factor_ranking(self, factors):
        """ Ranks factors based on their influence (how large or small results are)
            The lowest rank corresponds to the most influential (larger) factor

            parameters:
            factors: an array like input that contains factors that are to be ranked

            returns:
            a numpy array containing the ranks of each factor in their corresponding index
        """
        # check the factors is array like
        if not isinstance(factors,
              (pd.DataFrame, pd.Series, np.ndarray, List, Tuple)):
            raise TypeError(
                "factors must be an array-like object: "
                "pandas.Dataframe, pandas.Series, numpy.array, List, Tuple"
            )

        # gather indices for sorting factor in descending order
        temp = np.argsort(factors)[::-1]
        # create an array the same shape and type as temp
        ranks = np.empty_like(temp)
        # rank factors with highest value being the lowest rank
        ranks[temp] = np.arange(len(factors))
        
        return ranks


class GVARS(VARS):
    def __init__(self, ):
        super().__init__()


class DVARS(VARS):
    def __init__(self, ):
        super().__init__()
