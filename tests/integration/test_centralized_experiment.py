"""S2-G centralized orchestration and real smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.data.adapters import TrajectorySample
from src.data.dataset import TrajectoryDataset
from src.data.loading import ProcessedDataBundle, create_dataloaders
from src.data.preprocess import TrainingCoordinateScaler, WindowSpec
from src.experiments.centralized import (
    CentralizedExperiment,
    CentralizedExperimentRequest,
)
from src.models.lstm_seq2seq import LSTMSeq2Seq
from src.training.trainer import EpochStats, EvaluationResult, FitResult

CODE_SHA = "a" * 40
SPLIT_ID = "highd-s2-g-split"


def _config(*, epochs: int = 1) -> dict[str, dict[str, Any]]:
    return {
        "data": {"schema_version": 1},
        "model": {
            "schema_version": 1,
            "model": {
                "name": "lstm_encoder_decoder",
                "history_steps": 3,
                "future_steps": 2,
                "input_size": 2,
                "output_size": 2,
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "dtype": "float32",
                "coordinate_representation": "absolute_position",
                "device_policy": "trainer_managed",
            },
            "training": {
                "loss": "mse",
                "optimizer": "adam",
                "learning_rate": 0.02,
                "gradient_clip_norm": 1.0,
                "batch_size": 2,
                "epochs": epochs,
            },
        },
        "experiment": {
            "schema_version": 1,
            "run": {"name": "s2-g-smoke", "mode": "centralized", "seed": 42},
            "execution": {"device": "cpu", "num_workers": 0},
        },
    }


def _sample(split: str, index: int) -> TrajectorySample:
    history = np.array(
        [[index, 0.0], [index + 1.0, 0.0], [index + 2.0, 0.0]], dtype=np.float32
    )
    future = np.array(
        [[index + 3.0, 0.0], [index + 4.0, 0.0]], dtype=np.float32
    )
    return TrajectorySample(
        history=history,
        future=future,
        meta={
            "dataset_name": "highd",
            "data_version": "sample-v1",
            "recording_id": 1,
            "vehicle_id": index,
            "history_start_frame": 0,
            "history_end_frame": 2,
            "future_start_frame": 3,
            "future_end_frame": 4,
            "split_id": SPLIT_ID,
            "split": split,
        },
    )


def _bundle(tmp_path: Path) -> ProcessedDataBundle:
    window = WindowSpec(history_steps=3, future_steps=2, stride=1)
    datasets = {
        split: TrajectoryDataset(
            [_sample(split, index) for index in range(4)],
            split=split,
            split_id=SPLIT_ID,
            window_spec=window,
        )
        for split in ("train", "validation", "test")
    }
    scaler = TrainingCoordinateScaler()
    scaler.mean_ = np.zeros(2, dtype=np.float32)
    scaler.scale_ = np.ones(2, dtype=np.float32)
    scaler.fitted_split = "train"
    source = tmp_path / "processed"
    source.mkdir()
    (source / "split_manifest.json").write_text("{}", encoding="utf-8")
    return ProcessedDataBundle(
        datasets=datasets,
        scaler=scaler,
        stats={"sample_count": 12},
        data_version="sample-v1",
        split_id=SPLIT_ID,
        cache_key="cache-key",
        source=source,
    )


class _Reader:
    def __init__(self, bundle: ProcessedDataBundle, events: list[str]) -> None:
        self.bundle = bundle
        self.events = events

    def load(self, *args: object, **kwargs: object) -> ProcessedDataBundle:
        del args, kwargs
        self.events.append("load")
        return self.bundle


class _FakeTrainer:
    def __init__(
        self, contract: object, config: object, events: list[str], fail: bool = False
    ) -> None:
        del contract
        self.config = config
        self.device = torch.device("cpu")
        self.events = events
        self.fail = fail

    def fit(
        self,
        model: LSTMSeq2Seq,
        train_batches: object,
        validation_batches: object,
        *,
        initial_state: object,
    ) -> FitResult:
        del train_batches, validation_batches, initial_state
        self.events.append("fit")
        if self.fail:
            raise RuntimeError("fake training failure")
        return FitResult(
            epoch_stats=(EpochStats(0, 4, 0.5, 0.6),),
            best_epoch=0,
            checkpoint_payload={
                "schema_version": 1,
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "model_config": model.to_config(),
                "seed": self.config.seed,
                "epoch": 0,
                "split_id": self.config.split_id,
                "metrics": {"loss": 0.6},
            },
        )

    def evaluate(self, model: LSTMSeq2Seq, batches: object) -> EvaluationResult:
        del model, batches
        self.events.append("evaluate")
        return EvaluationResult(sample_count=4, loss=0.6)

    def save_checkpoint(self, payload: object, path: Path) -> Path:
        self.events.append("checkpoint")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(payload), path)
        return path


def _request(
    tmp_path: Path, bundle: ProcessedDataBundle, run_id: str
) -> CentralizedExperimentRequest:
    return CentralizedExperimentRequest(
        config_bundle=_config(),
        processed_dir=bundle.source,
        project_root=tmp_path,
        run_id=run_id,
        output_root=tmp_path / "outputs",
        code_sha=CODE_SHA,
    )


def test_fake_trainer_workflow_order_and_complete_artifacts(tmp_path: Path) -> None:
    events: list[str] = []
    bundle = _bundle(tmp_path)
    reader = _Reader(bundle, events)

    def loader_factory(data: object, *, contract: object, config: object) -> object:
        events.append("loaders")
        return create_dataloaders(data, contract=contract, config=config)  # type: ignore[arg-type]

    def model_factory(config: object) -> LSTMSeq2Seq:
        events.append("model")
        return LSTMSeq2Seq.from_model_config(config)  # type: ignore[arg-type]

    def trainer_factory(contract: object, config: object) -> _FakeTrainer:
        events.append("trainer")
        return _FakeTrainer(contract, config, events)

    output = CentralizedExperiment(
        reader=reader,
        loader_factory=loader_factory,
        model_factory=model_factory,
        trainer_factory=trainer_factory,
    ).run(_request(tmp_path, bundle, "fake-run"))

    assert events[:7] == ["load", "loaders", "model", "trainer", "fit", "checkpoint", "evaluate"]
    assert output.record.sample_count == 4
    assert output.baseline_record.model == "constant_velocity"
    required = {
        "config_snapshot.json",
        "metadata.json",
        "train.log",
        "metrics.json",
        "metrics.csv",
        "predictions.npz",
        "manifest.json",
        "checkpoints/best.pt",
        "baseline/metrics.json",
        "baseline/metrics.csv",
        "figures/loss_curve.png",
        "figures/prediction_trajectory.png",
        "figures/baseline_trajectory.png",
    }
    actual = {
        path.relative_to(output.output_dir).as_posix()
        for path in output.output_dir.rglob("*")
        if path.is_file()
    }
    assert required <= actual
    manifest = json.loads(output.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) >= {
        "config",
        "metadata",
        "log",
        "checkpoint",
        "metrics_json",
        "metrics_csv",
        "manifest",
    }
    with pytest.raises(FileExistsError):
        CentralizedExperiment(reader=reader).run(_request(tmp_path, bundle, "fake-run"))


def test_failed_training_keeps_failure_record_and_manifest(tmp_path: Path) -> None:
    events: list[str] = []
    bundle = _bundle(tmp_path)

    def trainer_factory(contract: object, config: object) -> _FakeTrainer:
        return _FakeTrainer(contract, config, events, fail=True)

    experiment = CentralizedExperiment(
        reader=_Reader(bundle, events),
        loader_factory=lambda data, *, contract, config: create_dataloaders(
            data, contract=contract, config=config
        ),
        trainer_factory=trainer_factory,
    )
    with pytest.raises(RuntimeError, match="fake training failure"):
        experiment.run(_request(tmp_path, bundle, "failed-run"))

    output_dir = tmp_path / "outputs" / "failed-run"
    record = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert "fake training failure" in record["error"]
    assert (output_dir / "manifest.json").is_file()


def test_real_small_centralized_smoke_uses_lstm_and_baseline(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    output = CentralizedExperiment(reader=_Reader(bundle, [])).run(
        _request(tmp_path, bundle, "real-smoke")
    )

    assert np.isfinite(output.loss)
    assert np.isfinite(output.record.ade)
    assert np.isfinite(output.record.fde)
    assert output.baseline_record.ade == pytest.approx(0.0)
    assert output.baseline_record.fde == pytest.approx(0.0)
