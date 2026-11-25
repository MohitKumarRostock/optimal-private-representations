"""Separate immutable reference artifacts from generated working outputs."""
from pathlib import Path


def repository_root():
    """Locate a checkout for defaults; explicit reference paths also work."""
    source_root = Path(__file__).resolve().parents[2]
    for candidate in [Path.cwd(), *Path.cwd().parents, source_root]:
        if (candidate / "pyproject.toml").is_file() and (
                candidate / "src/optimal_private_representations").is_dir():
            return candidate
    return None


def find_reference_dir(path=None):
    if path is not None:
        candidate = Path(path).expanduser().resolve()
    else:
        root = repository_root()
        candidate = (root or Path.cwd()) / "data/reference"
    if not (candidate / "learning_protocol.json").is_file():
        raise FileNotFoundError(
            f"Reference data not found at {candidate}. Run from the repository "
            "root or pass --reference-dir /path/to/data/reference.")
    return candidate.resolve()


def prepare_output(path, reference_dir=None):
    """Create an output directory, rejecting overlap with frozen data/source."""
    target = Path(path).expanduser().resolve()
    protected = [Path(__file__).resolve().parent]
    root = repository_root()
    if root is not None:
        protected += [root / "data/reference", root / "results/reference"]
    if reference_dir is not None:
        protected.append(Path(reference_dir).expanduser().resolve())
    for frozen in protected:
        frozen = frozen.resolve()
        if target == frozen or target in frozen.parents or frozen in target.parents:
            raise ValueError(f"Output directory overlaps protected inputs: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target

