"""Check the recorded-data contract without a full experiment rerun."""
from pathlib import Path
import unittest

from optimal_private_representations.validation.reference import (
    verify_hashes, replay_selected_fits,
)


class ReferenceReplayChecks(unittest.TestCase):
    reference=Path(__file__).resolve().parents[1]/"data/reference"

    def test_frozen_reference_hashes(self):
        self.assertEqual(verify_hashes(self.reference)["files_checked"],20)

    def test_selected_recorded_fits_after_import_migration(self):
        result=replay_selected_fits(self.reference)
        self.assertEqual(result["replayed_fits"],48)
        self.assertLess(result["max_risk_error"],5e-13)


if __name__=="__main__":
    unittest.main()

