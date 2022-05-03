# -*- coding: utf-8 -*-

"""
This module contains 6 different sampling methods, that are:
1. halton sequence
2. lating hypercube sampling (lhs)
3. progressive lating hypercube sampling (plhs)
4. sobol sequence
5. symetrical latin hypercube sampling (symlhs)
6. Generalized Star‐Based (gSTAR) Sampling
7. STAR sampling (starvars)

"""
from .g_starvars import g_star
from .lhs import lhs
from .plhs import plhs
from .sobol_sequence import sobol_sequence
from .halton import halton
from .symlhs import symlhs
from .starvars import star

__all__ = ["g_starvars", "halton", "lhs", "plhs", "sobol_sequence", "starvars", "symlhs"]
