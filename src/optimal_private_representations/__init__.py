"""Reference algorithms for private representation-based task learning.

The public API uses capped ordered calibration. Recorded experiments use the
explicit legacy subpackage. Mathematical privacy guarantees concern the ideal
mechanisms, not a certified finite-precision deployment.
"""
from .calibration import fit_ordered_private_dp, output_count_cap
from .learner import FinitePrivateTaskModel, fit_finite_private_task_learner

__version__ = "0.1.0"
__all__ = [
    "FinitePrivateTaskModel", "fit_finite_private_task_learner",
    "fit_ordered_private_dp", "output_count_cap",
]

