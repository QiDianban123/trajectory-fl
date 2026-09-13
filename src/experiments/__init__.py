"""Experiment execution and reproducibility helpers."""

from src.experiments.run_context import MANIFEST_SCHEMA_VERSION, RunContext, file_checksum

__all__ = [
    "CentralizedExperiment",
    "CentralizedExperimentOutput",
    "CentralizedExperimentRequest",
    "ExperimentInputError",
    "FederatedExperiment",
    "FederatedRunRequest",
    "LocalOnlyExperiment",
    "LocalOnlyRunRequest",
    "MANIFEST_SCHEMA_VERSION",
    "ModeRunResult",
    "RunContext",
    "ThreeModeMatrixResult",
    "file_checksum",
    "stable_config_digest",
    "validate_three_mode_matrix",
]


def __getattr__(name: str) -> object:
    if name in {
        "CentralizedExperiment",
        "CentralizedExperimentOutput",
        "CentralizedExperimentRequest",
    }:
        from src.experiments import centralized

        return getattr(centralized, name)
    if name in {
        "ExperimentInputError",
        "FederatedExperiment",
        "FederatedRunRequest",
        "LocalOnlyExperiment",
        "LocalOnlyRunRequest",
        "ModeRunResult",
        "ThreeModeMatrixResult",
        "stable_config_digest",
        "validate_three_mode_matrix",
    }:
        from src.experiments import three_mode

        return getattr(three_mode, name)
    raise AttributeError(name)
