"""Create a deterministic D1 example of metric and trajectory visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import ade, fde


def main(output: Path = Path("outputs/d1-metric-example.png")) -> None:
    history = np.array([[-3.0, -0.2], [-2.0, -0.1], [-1.0, 0.0], [0.0, 0.0]])
    future = np.array([[1.0, 0.2], [2.0, 0.7], [3.0, 1.3]])
    prediction = np.array([[1.0, 0.1], [2.05, 0.55], [3.15, 1.1]])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(*history.T, "o-", label="history")
    ax.plot(*future.T, "o-", label="future truth")
    ax.plot(*prediction.T, "o--", label="prediction")
    ax.set(title=f"D1 visualization example | ADE={ade(prediction, future):.3f}, FDE={fde(prediction, future):.3f}",
           xlabel="x (m)", ylabel="y (m)")
    ax.legend()
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
