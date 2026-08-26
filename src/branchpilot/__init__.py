"""BranchPilot: budget-conditioned adaptive inference control."""

from importlib.metadata import PackageNotFoundError, version

from branchpilot.calibration import OperatingPoint, select_operating_point
from branchpilot.policy import BranchPilotPolicy, Decision
from branchpilot.runtime import PilotResult, PilotSession
from branchpilot.schema import Rollout, Sample

__all__ = [
    "BranchPilotPolicy",
    "Decision",
    "OperatingPoint",
    "PilotResult",
    "PilotSession",
    "select_operating_point",
    "Rollout",
    "Sample",
]

try:
    __version__ = version("branchpilot")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"
