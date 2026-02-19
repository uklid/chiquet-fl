from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class SynthModelSpec:
    name: str
    input_dim: int = 2
    hidden_dim: int = 16
    output_dim: int = 1


def resolve_synth_model_spec(model_cfg: Any) -> SynthModelSpec:
    # Backward-compatible default if cfg.model is missing.
    if not isinstance(model_cfg, Mapping):
        return SynthModelSpec(name="linear_logreg_2d", output_dim=1)

    name = str(model_cfg.get("name", "")).strip().lower()
    params = model_cfg.get("params", {})
    if not isinstance(params, Mapping):
        params = {}

    # Keep old names/semantics as aliases to linear logistic regression.
    if name in {"", "none", "logreg", "linear_logreg_2d"}:
        input_dim = int(params.get("input_dim", 2))
        if input_dim <= 0:
            raise ValueError("logreg input_dim must be > 0.")
        return SynthModelSpec(name="linear_logreg_2d", input_dim=input_dim, output_dim=1)

    if name == "mlp_small":
        input_dim = int(params.get("input_dim", 2))
        hidden_dim = int(params.get("hidden_dim", 16))
        output_dim = int(params.get("output_dim", 2))
        if input_dim <= 0:
            raise ValueError("mlp_small input_dim must be > 0.")
        if hidden_dim <= 0:
            raise ValueError("mlp_small hidden_dim must be > 0.")
        if output_dim not in {1, 2}:
            raise ValueError("mlp_small output_dim must be 1 or 2 for binary classification.")
        return SynthModelSpec(
            name="mlp_small",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    raise ValueError(f"Unsupported synth2d model: {name}")


def build_synth_model(spec: SynthModelSpec) -> torch.nn.Module:
    if spec.name == "linear_logreg_2d":
        if spec.output_dim != 1:
            raise ValueError("linear_logreg_2d only supports output_dim=1.")
        return torch.nn.Linear(spec.input_dim, 1)

    if spec.name == "mlp_small":
        return torch.nn.Sequential(
            torch.nn.Linear(spec.input_dim, spec.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(spec.hidden_dim, spec.output_dim),
        )

    raise ValueError(f"Unsupported synth2d model: {spec.name}")
