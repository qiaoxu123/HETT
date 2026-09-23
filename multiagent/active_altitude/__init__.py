"""Active-altitude navigation building blocks.

The package is deliberately independent from the legacy rollout.  The rollout
adapter can consume these components behind one feature flag without changing
the released HETT path, which keeps all planned ablations reproducible.
"""

from .belief import BeliefDiagnostics, MultiScaleBelief
from .controller import (
    AltitudeAction,
    ControllerConfig,
    ControllerDecision,
    ControllerSignals,
    ControllerState,
    NavigationPhase,
    UncertaintyAltitudeController,
)
from .teacher import TeacherTargets, build_teacher_targets

__all__ = [
    "AltitudeAction",
    "BeliefDiagnostics",
    "ControllerConfig",
    "ControllerDecision",
    "ControllerSignals",
    "ControllerState",
    "MultiScaleBelief",
    "NavigationPhase",
    "TeacherTargets",
    "UncertaintyAltitudeController",
    "build_teacher_targets",
]
