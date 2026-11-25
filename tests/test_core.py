"""Small scientific regressions for the packaged capped API."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from optimal_private_representations import (
    fit_finite_private_task_learner, fit_ordered_private_dp, output_count_cap,
)
from optimal_private_representations.io import prepare_output
from optimal_private_representations.validation.output_cap import exhaustive_objective


class CoreChecks(unittest.TestCase):
    def test_three_outputs_are_not_silently_restricted_to_binary(self):
        p=np.array([.3,.2,.2,.3]);m=np.array([0.,.5,.5,1.])
        best=fit_ordered_private_dp(p,p*m,m,math.log(4))
        binary=fit_ordered_private_dp(p,p*m,m,math.log(4),max_outputs=2)
        self.assertEqual(best["k"],3)
        self.assertAlmostEqual(best["explained_second_moment"]-binary["explained_second_moment"],27/23560,places=14)
        self.assertLessEqual(output_count_cap(math.log(4),100),9)

    def test_ordered_optimizer_with_ties_zero_mass_and_nonmonotone_means(self):
        p=np.array([.2,0,.3,.1,.4]);m=np.array([.8,.2,.1,.9,.4])
        scores=np.array([0.,0.,1.,2.,3.])
        for epsilon in [0.,math.log(2),math.log(4)]:
            fitted=fit_ordered_private_dp(p,p*m,scores,epsilon)
            exact=exhaustive_objective(p,p*m,scores,epsilon)
            self.assertAlmostEqual(fitted["explained_second_moment"],exact,places=13)
            self.assertEqual(fitted["q"].shape[0],5)

    def test_zero_training_budget_is_data_independent(self):
        settings=dict(cells=4,epsilon_query=math.log(2),alpha_train=0.)
        first=fit_finite_private_task_learner(np.zeros(16,dtype=int),np.zeros(16),**settings)
        second=fit_finite_private_task_learner(np.full(16,3,dtype=int),np.ones(16),**settings)
        np.testing.assert_array_equal(first.decoder,second.decoder)
        np.testing.assert_array_equal(first.decoder,[.5])
        np.testing.assert_array_equal(first.probabilities(np.arange(4)),np.ones((4,1)))

    def test_private_query_sampling_and_unseen_state(self):
        states=np.tile(np.arange(3),64)
        labels=np.tile([0.,.3,1.],64)
        model=fit_finite_private_task_learner(states,labels,cells=4,
                epsilon_query=math.log(2),alpha_train=1.,rng=np.random.default_rng(9))
        q=model.probabilities(np.arange(4))
        np.testing.assert_allclose(q.sum(axis=1),1)
        self.assertLessEqual(float(np.max(q/q.min(axis=0))),2+1e-12)
        indices=model.sample_representations(np.arange(4),rng=np.random.default_rng(3))
        np.testing.assert_array_equal(model.decode(indices),model.predict(np.arange(4),rng=np.random.default_rng(3)))
        self.assertEqual(model.sample_representations(np.array([],dtype=int)).size,0)

    def test_outputs_cannot_overwrite_reference_or_source(self):
        root=Path(__file__).resolve().parents[1]
        for target in [root,root/"data/reference",root/"data/reference/generated",root/"results/reference"]:
            with self.assertRaises(ValueError):
                prepare_output(target)
        with tempfile.TemporaryDirectory() as temp:
            self.assertTrue(prepare_output(Path(temp)/"generated").is_dir())


if __name__=="__main__":
    unittest.main()

