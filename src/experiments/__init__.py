"""Experiment execution and reproducibility helpers."""

from src.experiments.run_context import MANIFEST_SCHEMA_VERSION, RunContext, file_checksum

__all__ = [
    "CentralizedExperiment",
    "CentralizedExperimentOutput",
    "CentralizedExperimentRequest",
    "MANIFEST_SCHEMA_VERSION",
    "RunContext",
    "file_checksum",
]


def __getattr__(name: str) -> object:
    if name in {
        "CentralizedExperiment",
        "CentralizedExperimentOutput",
        "CentralizedExperimentRequest",
    }:
        from src.experiments import centralized

        return getattr(centralized, name)
    raise AttributeError(name)
