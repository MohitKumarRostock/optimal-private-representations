"""Small deterministic check of imports, exact oracles, and randomized queries."""
from fractions import Fraction as F
import math

import numpy as np

from .. import fit_finite_private_task_learner, fit_ordered_private_dp
from ..legacy.calibration import fit_ordered_private_dp as uncapped
from ..experiments.population import exhaustive_rational_oracle, coarsen


def run():
    p,means=(F(1,3),)*3,(F(0),F(1,2),F(1))
    base=exhaustive_rational_oracle(p,means,F(2),F(5,12))
    if base.risk != F(37,240):
        raise AssertionError("Exact three-point oracle changed")
    good_p,good_m,*_=coarsen(p,means,(0,1,1))
    good=exhaustive_rational_oracle(good_p,good_m,F(2),F(5,12))
    cross_p,cross_m,*_=coarsen(p,means,(0,1,0))
    cross=exhaustive_rational_oracle(cross_p,cross_m,F(2),F(5,12))
    if good.risk != base.risk or cross.risk-base.risk != F(1,80):
        raise AssertionError("Exact deficiency examples changed")
    p=np.array([.3,.2,.2,.3]);means=np.array([0,.5,.5,1.])
    capped=fit_ordered_private_dp(p,p*means,means,math.log(4))
    full=uncapped(p,p*means,means,math.log(4))
    if capped["k"] != 3 or abs(capped["explained_second_moment"]-full["explained_second_moment"])>1e-13:
        raise AssertionError("Three-output optimizer check failed")
    states=np.tile(np.arange(4),32)
    labels=np.random.default_rng(83).binomial(1,means[states])
    model=fit_finite_private_task_learner(states,labels,cells=5,epsilon_query=math.log(2),
                                          alpha_train=1.,rng=np.random.default_rng(84))
    rows=model.probabilities(np.arange(5))
    if not np.allclose(rows.sum(axis=1),1) or np.max(rows/rows.min(axis=0))>2+1e-12:
        raise AssertionError("Full-alphabet query rows are inconsistent")
    indices=model.sample_representations(np.arange(5),rng=np.random.default_rng(85))
    decoded=model.decode(indices)
    direct=model.predict(np.arange(5),rng=np.random.default_rng(85))
    np.testing.assert_array_equal(decoded,direct)
    return {"status":"passed","exact_three_point_risk":"37/240",
            "exact_crossing_deficiency":"1/80","three_output_case":True,
            "declared_states_including_unseen":5,"private_prediction_shape":list(direct.shape),
            "scope":"Finite implementation checks only; not a numerical privacy certificate."}

