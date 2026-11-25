"""Connect original recorded source hashes to the importable legacy modules.

The original hashes are preserved in the frozen protocol. Imports and module
paths changed during packaging, so both original and packaged byte hashes are
recorded. No outside manuscript directory is needed for replay validation.
"""
import hashlib
from pathlib import Path

from .legacy import learner, calibration

ORIGINAL_LEGACY_HASHES = {'finite_private_task_learner.py': '6238d26656f16ebf38a2df468243163d7c4c4f657a69e7e25c4f966bb76ede53', 'ordered_private_calibration.py': 'ac57ca2f5a596260e8e4f559269776ce3c630468f1b5fbd4e0b5f1617e381302'}
PACKAGED_LEGACY_HASHES = {'finite_private_task_learner.py': 'c34557e95fb8489c1d68206e5488bb28f4aec9da9c54e96c732a19bcab9b2043', 'ordered_private_calibration.py': '5011d093b5ccf49c76dc57f6a77b106c3b740a589cc15cf3b3b008b6266d899f'}


def legacy_source_hashes():
    return {
        "finite_private_task_learner.py": hashlib.sha256(Path(learner.__file__).read_bytes()).hexdigest(),
        "ordered_private_calibration.py": hashlib.sha256(Path(calibration.__file__).read_bytes()).hexdigest(),
    }


def validate_protocol_sources(protocol):
    current = legacy_source_hashes()
    recorded = protocol.get("source_hashes", {})
    if recorded == current:
        return "packaged sources match this run's protocol"
    if recorded == ORIGINAL_LEGACY_HASHES and current == PACKAGED_LEGACY_HASHES:
        return "original recorded sources match the documented package migration"
    raise RuntimeError("Protocol source hashes do not match the preserved legacy implementation")
