"""Organized numerical methods package."""

from functions.spd_precondioners import RidgewPartialChol
from .iterative_inverse import (
    Gauss_Seidel,
    Newton_Shulz,
    compute_errors_per_iteration,
    direct,
    split,
    validate,
)
from .preconditioners import (
    back_substitution,
    back_substitution_matrix,
    forward_substitution,
    forward_substitution_matrix,
    incomplete_lu,
    precondition_matrix,
    prrlu,
    recover_inverse,
    solve_with_ilu,
)
from .haar import givensRot, givensRotVec, randOrth
from .stoch_trace_est import STE, STE_kry
from .cg import ConjugateGradient
from .spd_precondioners import (
    RandomlyPivotedCholesky,
    RidgewPartialChol
)
