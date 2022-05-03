# -*- coding: utf-8 -*-

"""
This module contains 6 different sampling methods, that are:
1. Generalized Star‐Based (gSTAR) Sampling
2. halton sequence
3. lating hypercube sampling (lhs)
4. progressive lating hypercube sampling (plhs)
5. sobol sequence
6. STAR sampling (starvars)
7. symetrical latin hypercube sampling (symlhs)

"""
from .g_starvars import g_star
from .lhs import lhs
from .plhs import plhs
from .sobol_sequence import sobol_sequence
from .halton import halton
from .symlhs import symlhs
from .starvars import star

__all__ = ["g_starvars", "halton", "lhs", "plhs", "sobol_sequence", "starvars", "symlhs"]
