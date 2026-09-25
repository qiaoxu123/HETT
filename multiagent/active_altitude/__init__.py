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
from .fine_refinement import (
    FineAltitudeConfig,
    FineAltitudeController,
    FineAltitudeDecision,
    FineAltitudeSignals,
    FineAltitudeState,
    FineNavigationPhase,
    canonical_map_pose,
)

__all__ = [
    "AltitudeAction",
    "BeliefDiagnostics",
    "ControllerConfig",
    "ControllerDecision",
    "ControllerSignals",
    "ControllerState",
    "FineAltitudeConfig",
    "FineAltitudeController",
    "FineAltitudeDecision",
    "FineAltitudeSignals",
    "FineAltitudeState",
    "FineNavigationPhase",
    "MultiScaleBelief",
    "NavigationPhase",
    "TeacherTargets",
    "UncertaintyAltitudeController",
    "build_teacher_targets",
    "canonical_map_pose",
]
