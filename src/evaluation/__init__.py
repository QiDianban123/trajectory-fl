"""Evaluation metrics, visualization, and result serialization helpers."""

from src.evaluation.centralized import (
    CentralizedEvaluationOutput,
    CentralizedEvaluationRequest,
    InverseTransformScaler,
    LossHistory,
    PhysicalEvaluation,
    PhysicalTrajectoryBatch,
    evaluate_centralized,
    evaluate_prediction_arrays,
    inverse_transform_batch,
)
from src.evaluation.data_diagnostics import (
    diagnostic_file_stem,
    generate_data_diagnostic_figures,
    plot_anomaly_counts,
    plot_inverse_transform_check,
    plot_raw_cleaned_trajectories,
    plot_truth_trajectory,
)
from src.evaluation.federated_results import (
    ClientResultRecord,
    RoundRecord,
    compare_modes,
    plot_round_metrics,
    summarize_client_results,
)
from src.evaluation.metrics import ade, compute_metrics, fde
from src.evaluation.prediction import PredictionCollection, collect_predictions
from src.evaluation.result_store import ResultRecord, ResultStore, write_csv, write_json
from src.evaluation.visualization import (
    plot_convergence,
    plot_loss_curve,
    plot_mode_comparison,
    plot_prediction_trajectory,
    plot_trajectory,
)

__all__ = [
    "CentralizedEvaluationOutput",
    "CentralizedEvaluationRequest",
    "InverseTransformScaler",
    "LossHistory",
    "PhysicalEvaluation",
    "PhysicalTrajectoryBatch",
    "PredictionCollection",
    "ResultRecord",
    "ClientResultRecord",
    "RoundRecord",
    "ResultStore",
    "ade",
    "compute_metrics",
    "collect_predictions",
    "diagnostic_file_stem",
    "evaluate_centralized",
    "evaluate_prediction_arrays",
    "fde",
    "generate_data_diagnostic_figures",
    "inverse_transform_batch",
    "plot_anomaly_counts",
    "plot_convergence",
    "plot_inverse_transform_check",
    "plot_loss_curve",
    "plot_mode_comparison",
    "plot_prediction_trajectory",
    "plot_raw_cleaned_trajectories",
    "plot_trajectory",
    "plot_truth_trajectory",
    "write_csv",
    "write_json",
    "compare_modes",
    "plot_round_metrics",
    "summarize_client_results",
]
