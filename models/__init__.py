from __future__ import annotations

from typing import Dict, Type
import torch.nn as nn

from .tepnet import TEPNet
from .tcn import TCN
from .lstm_net import LSTMNet
from .transformer_net import TransformerNet
from .patchtst import PatchTST

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "tepnet": TEPNet,
    "tcn": TCN,
    "lstm": LSTMNet,
    "transformer": TransformerNet,
    "patchtst": PatchTST,
}


def get_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())
