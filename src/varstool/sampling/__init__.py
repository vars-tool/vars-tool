# -*- coding: utf-8 -*-

"""
This module contains 6 different sampling methods, that are:
1. halton sequence
2. lating hypercube sampling (lhs)
3. progressive lating hypercube sampling (plhs)
4. sobol sequence
5. STAR sampling
6. symetrical latin hypercube sampling (symlhs)

"""
from .lhs import lhs
from .plhs import plhs
from .sobol_sequence import sobol_sequence
from .halton import halton
from .symlhs import symlhs
from .starvars import star

__all__ = ["halton", "lhs", "plhs", "sobol_sequence", "starvars", "symlhs"]
