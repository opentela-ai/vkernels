"""Model-specific capture choices, resolved once before planning."""

from dataclasses import dataclass
from typing import Callable

from .model_gpt2 import GPT2Config, SymbolicModelArgs, build_forward, random_weights
from .model_qwen3 import Qwen3Config, Qwen3ModelArgs, build_qwen3_forward, random_qwen3_weights
from .model_qwen35 import Qwen35ModelArgs, build_qwen35_forward
from .model_deepseek_v4 import DeepseekV4ModelArgs, build_deepseek_v4_forward
from .qwen35_arch import Qwen35Config, random_qwen35_weights
from .deepseek_v4_arch import DeepseekV4Config, random_deepseek_weights


@dataclass(frozen=True)
class ModelAdapter:
    body: Callable
    weights: Callable
    arguments: Callable
    row_positions: bool = False


ADAPTERS = {
    GPT2Config: ModelAdapter(build_forward, random_weights, SymbolicModelArgs),
    Qwen3Config: ModelAdapter(build_qwen3_forward, random_qwen3_weights, lambda r, c, w: Qwen3ModelArgs(r, c)),
    Qwen35Config: ModelAdapter(
        build_qwen35_forward, random_qwen35_weights, lambda r, c, w: Qwen35ModelArgs(r, c), True
    ),
    DeepseekV4Config: ModelAdapter(
        build_deepseek_v4_forward, random_deepseek_weights, lambda r, c, w: DeepseekV4ModelArgs(r, c), True
    ),
}


def model_adapter(config) -> ModelAdapter:
    try:
        return ADAPTERS[type(config)]
    except KeyError:
        raise TypeError(f"no compiler adapter for {type(config).__name__}") from None
