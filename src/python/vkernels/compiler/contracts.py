"""Canonical operator contracts shared by capture, lowering, and execution.

Lowering owns tile/region rules; this registry names that rule once alongside
its reference and emitted-device implementation. Attribute types are checked
before lowering, so malformed IR cannot silently select a different policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class OperatorContract:
    kind: str
    lowering: str
    task_kind: str
    device_template: str
    attribute_types: Mapping[str, type]

    @property
    def reference_body(self) -> str:
        return "_body_" + self.task_kind

    def validate_attributes(self, attributes: Mapping) -> None:
        for name, expected in self.attribute_types.items():
            if name in attributes and not isinstance(attributes[name], expected):
                raise TypeError(f"{self.kind}.{name} must be {expected.__name__}")


CONTRACTS = MappingProxyType(
    {
        "linear": OperatorContract("linear", "lower_linear", "gemm", "linear_task", MappingProxyType({})),
        "linear_fp8": OperatorContract(
            "linear_fp8", "lower_linear_fp8", "gemv_fp8", "linear_fp8_task", MappingProxyType({})
        ),
        "indexer_scores": OperatorContract(
            "indexer_scores", "lower_indexer_scores", "indexer_scores", "indexer_scores_task", MappingProxyType({})
        ),
        "index_topk": OperatorContract(
            "index_topk", "lower_index_topk", "index_topk", "index_topk_task", MappingProxyType({})
        ),
        "layer_norm": OperatorContract(
            "layer_norm", "lower_layer_norm", "layernorm", "layernorm_task", MappingProxyType({})
        ),
        "rms_norm": OperatorContract("rms_norm", "lower_rms_norm", "rms_norm", "rms_norm_task", MappingProxyType({})),
        "rms_norm_gated": OperatorContract(
            "rms_norm_gated",
            "lower_rms_norm_gated",
            "rms_norm_gated",
            "rms_norm_gated_task",
            MappingProxyType({"activation": str}),
        ),
        "rope": OperatorContract(
            "rope",
            "lower_rope",
            "rope",
            "rope_task",
            MappingProxyType({"which": str, "position": str, "position_form": str, "convention": str}),
        ),
        "gelu": OperatorContract("gelu", "lower_gelu", "elementwise", "elementwise_task", MappingProxyType({})),
        "swiglu": OperatorContract("swiglu", "lower_swiglu", "elementwise", "elementwise_task", MappingProxyType({})),
        "add": OperatorContract("add", "lower_add", "elementwise", "elementwise_task", MappingProxyType({})),
        "embedding": OperatorContract(
            "embedding",
            "lower_embedding",
            "embedding",
            "embedding_task",
            MappingProxyType({"position": str, "position_form": str}),
        ),
        "cache_append": OperatorContract(
            "cache_append",
            "lower_cache_append",
            "cache_append",
            "cache_append_task",
            MappingProxyType({"position": str, "position_form": str}),
        ),
        "gdn_conv": OperatorContract("gdn_conv", "lower_gdn_conv", "gdn_conv", "gdn_conv_task", MappingProxyType({})),
        "gdn_delta": OperatorContract(
            "gdn_delta", "lower_gdn_delta", "gdn_delta", "gdn_delta_task", MappingProxyType({})
        ),
        "kda_delta": OperatorContract(
            "kda_delta", "lower_kda_delta", "kda_delta", "task_kda_delta", MappingProxyType({})
        ),
        "kda_fused_decode": OperatorContract(
            "kda_fused_decode",
            "lower_kda_fused_decode",
            "kda_fused_decode",
            "task_kda_fused",
            MappingProxyType({}),
        ),
        "cache_append_paged": OperatorContract(
            "cache_append_paged",
            "lower_cache_append_paged",
            "cache_append_paged",
            "cache_append_paged_task",
            MappingProxyType({"position": str, "position_form": str}),
        ),
        "mhc_pre": OperatorContract("mhc_pre", "lower_mhc_pre", "mhc_pre", "mhc_pre_task", MappingProxyType({})),
        "mhc_post": OperatorContract("mhc_post", "lower_mhc_post", "mhc_post", "mhc_post_task", MappingProxyType({})),
        "attention_scores": OperatorContract(
            "attention_scores",
            "lower_attention_scores",
            "attention_scores",
            "attention_scores_task",
            MappingProxyType({"position": str, "position_form": str}),
        ),
        "attention_scores_paged": OperatorContract(
            "attention_scores_paged",
            "lower_attention_scores_paged",
            "attention_scores_paged",
            "attention_scores_paged_task",
            MappingProxyType({"position": str, "position_form": str}),
        ),
        "mla_scores": OperatorContract(
            "mla_scores", "lower_mla_scores", "mla_scores", "mla_scores_task", MappingProxyType({"position_form": str})
        ),
        "mla_values": OperatorContract(
            "mla_values", "lower_mla_values", "mla_values", "mla_values_task", MappingProxyType({"position_form": str})
        ),
        "conjugate_rope": OperatorContract(
            "conjugate_rope",
            "lower_conjugate_rope",
            "conjugate_rope",
            "conjugate_rope_task",
            MappingProxyType({"position_form": str}),
        ),
        "softmax": OperatorContract(
            "softmax",
            "lower_softmax",
            "softmax",
            "softmax_task",
            MappingProxyType({"position": str, "position_form": str}),
        ),
        "attention_values": OperatorContract(
            "attention_values",
            "lower_attention_values",
            "attention_values",
            "attention_values_task",
            MappingProxyType({"gated": bool, "position": str, "position_form": str}),
        ),
        "attention_values_paged": OperatorContract(
            "attention_values_paged",
            "lower_attention_values_paged",
            "attention_values_paged",
            "attention_values_paged_task",
            MappingProxyType({"position": str, "position_form": str}),
        ),
        "moe_route": OperatorContract(
            "moe_route", "lower_moe_route", "moe_route", "moe_route_task", MappingProxyType({})
        ),
        "moe_expert": OperatorContract(
            "moe_expert", "lower_moe_expert", "moe_expert", "moe_expert_task", MappingProxyType({})
        ),
        "moe_combine": OperatorContract(
            "moe_combine", "lower_moe_combine", "moe_combine", "moe_combine_task", MappingProxyType({})
        ),
        "compressor_append": OperatorContract(
            "compressor_append",
            "lower_compressor_append",
            "compressor_append",
            "compressor_append_task",
            MappingProxyType({"position": str, "position_form": str}),
        ),
    }
)
