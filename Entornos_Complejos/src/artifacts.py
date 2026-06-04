"""Localiza y reconstruye resultados experimentales cuando faltan."""
from __future__ import annotations

from pathlib import Path


TABULAR_FILES = {
    "tabular_training_metrics.csv",
    "tabular_evaluation_by_seed.csv",
    "tabular_summary.csv",
    "tabular_learning_fingerprint.png",
    "tabular_robustness.png",
    "episode_length_indicator.png",
    "river_policy_maps.png",
}

APPROX_FILES = {
    "approx_training_metrics.csv",
    "approx_evaluation_by_seed.csv",
    "approx_summary.csv",
    "approx_learning_fingerprint.png",
    "approx_robustness.png",
    "continuous_trajectories.png",
}


def _summary(evaluation):
    return evaluation.groupby(["study", "agent", "condition"], as_index=False).agg(
        mean_return=("mean_return", "mean"),
        mean_length=("mean_length", "mean"),
        success_rate=("success_rate", "mean"),
        risk_rate=("risk_rate", "mean"),
    )


def _ensure_tabular(output_dir: Path, requested: str) -> None:
    import pandas as pd

    from .plotting import (
        plot_episode_length_indicator,
        plot_learning_fingerprint,
        plot_robustness,
    )

    training_path = output_dir / "tabular_training_metrics.csv"
    evaluation_path = output_dir / "tabular_evaluation_by_seed.csv"
    summary_path = output_dir / "tabular_summary.csv"

    if not training_path.exists() or not evaluation_path.exists() or requested == "river_policy_maps.png":
        from .experiments import run_tabular_study

        print("Faltan resultados tabulares base; se regenerara el estudio RiverCourier.")
        run_tabular_study(output_dir)
        return

    training = pd.read_csv(training_path)
    if not summary_path.exists():
        _summary(pd.read_csv(evaluation_path)).to_csv(summary_path, index=False)
    summary = pd.read_csv(summary_path)

    if requested == "tabular_learning_fingerprint.png":
        plot_learning_fingerprint(
            training,
            output_dir / requested,
            "RiverCourier: huella de aprendizaje",
            100,
        )
    elif requested == "tabular_robustness.png":
        plot_robustness(
            summary,
            output_dir / requested,
            "RiverCourier: robustez ante una crecida",
        )
    elif requested == "episode_length_indicator.png":
        plot_episode_length_indicator(training, output_dir / requested)


def _ensure_approx(output_dir: Path, requested: str) -> None:
    import pandas as pd

    from .plotting import plot_learning_fingerprint, plot_robustness

    training_path = output_dir / "approx_training_metrics.csv"
    evaluation_path = output_dir / "approx_evaluation_by_seed.csv"
    summary_path = output_dir / "approx_summary.csv"

    if not training_path.exists() or not evaluation_path.exists() or requested == "continuous_trajectories.png":
        from .experiments import run_approximation_study

        print("Faltan resultados aproximados base; se regenerara el estudio ThermalGlider.")
        run_approximation_study(output_dir)
        return

    training = pd.read_csv(training_path)
    if not summary_path.exists():
        _summary(pd.read_csv(evaluation_path)).to_csv(summary_path, index=False)
    summary = pd.read_csv(summary_path)

    if requested == "approx_learning_fingerprint.png":
        plot_learning_fingerprint(
            training,
            output_dir / requested,
            "ThermalGlider: huella de aprendizaje",
            70,
        )
    elif requested == "approx_robustness.png":
        plot_robustness(
            summary,
            output_dir / requested,
            "ThermalGlider: robustez ante turbulencia",
        )


def _ensure_all_summary(output_dir: Path) -> None:
    import pandas as pd

    tabular = pd.read_csv(result_path(output_dir, "tabular_summary.csv"))
    approx = pd.read_csv(result_path(output_dir, "approx_summary.csv"))
    pd.concat([tabular, approx], ignore_index=True).to_csv(
        output_dir / "all_summary.csv",
        index=False,
    )


def result_path(output_dir: str | Path, filename: str) -> Path:
    """Devuelve un resultado existente o lo reconstruye antes de devolverlo."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    if path.exists():
        return path

    if filename in TABULAR_FILES:
        _ensure_tabular(output_dir, filename)
    elif filename in APPROX_FILES:
        _ensure_approx(output_dir, filename)
    elif filename == "all_summary.csv":
        _ensure_all_summary(output_dir)
    else:
        raise ValueError(f"Resultado desconocido: {filename}")

    if not path.exists():
        raise RuntimeError(f"No se pudo reconstruir el resultado requerido: {path}")
    return path
