"""BranchPilot: budget-conditioned adaptive inference control."""

from importlib.metadata import PackageNotFoundError, version

from branchpilot.calibration import (
    DeploymentPlan,
    OperatingPoint,
    select_deployment_plan,
    select_operating_point,
)
from branchpilot.policy import BranchPilotPolicy, Decision
from branchpilot.runtime import PilotResult, PilotSession
from branchpilot.schema import Rollout, Sample
from branchpilot.strategies import (
    ConsecutiveAgreementStrategy,
    FixedStrategy,
    StoppingStrategy,
    VoteConfidenceStrategy,
    strategy_from_spec,
)

__all__ = [
    "BranchPilotPolicy",
    "ConsecutiveAgreementStrategy",
    "Decision",
    "DeploymentPlan",
    "FixedStrategy",
    "OperatingPoint",
    "PilotResult",
    "PilotSession",
    "select_operating_point",
    "select_deployment_plan",
    "StoppingStrategy",
    "VoteConfidenceStrategy",
    "strategy_from_spec",
    "Rollout",
    "Sample",
]

try:
    __version__ = version("branchpilot")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"
