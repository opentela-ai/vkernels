"""Lower operators using the shared contract registry."""

from __future__ import annotations
from ..contracts import CONTRACTS
from ..operator_ir import Operator, OperatorGraph
from ..task_ir import TaskFamily
from .dense import (
    lower_linear as lower_linear,
    lower_linear_fp8 as lower_linear_fp8,
    lower_layer_norm as lower_layer_norm,
    lower_rms_norm as lower_rms_norm,
    lower_rms_norm_gated as lower_rms_norm_gated,
    lower_rope as lower_rope,
    lower_gelu as lower_gelu,
    lower_swiglu as lower_swiglu,
    lower_add as lower_add,
    lower_embedding as lower_embedding,
    lower_moe_route as lower_moe_route,
    lower_moe_expert as lower_moe_expert,
    lower_moe_combine as lower_moe_combine,
)
from .attention import (
    lower_indexer_scores as lower_indexer_scores,
    lower_index_topk as lower_index_topk,
    lower_cache_append as lower_cache_append,
    lower_cache_append_paged as lower_cache_append_paged,
    lower_attention_scores as lower_attention_scores,
    lower_attention_scores_paged as lower_attention_scores_paged,
    lower_mla_scores as lower_mla_scores,
    lower_mla_values as lower_mla_values,
    lower_conjugate_rope as lower_conjugate_rope,
    lower_softmax as lower_softmax,
    lower_attention_values as lower_attention_values,
    lower_attention_values_paged as lower_attention_values_paged,
)
from .recurrent import (
    lower_gdn_conv as lower_gdn_conv,
    lower_gdn_delta as lower_gdn_delta,
    lower_kda_delta as lower_kda_delta,
    lower_kda_fused_decode as lower_kda_fused_decode,
    lower_mhc_pre as lower_mhc_pre,
    lower_mhc_post as lower_mhc_post,
    lower_compressor_append as lower_compressor_append,
)
from .common import (
    GEMM_TILE_M as GEMM_TILE_M,
    GEMM_TILE_N as GEMM_TILE_N,
    ELEM_TILE as ELEM_TILE,
    EMBED_TILE_C as EMBED_TILE_C,
    THREADS_PER_WORKER as THREADS_PER_WORKER,
    GEMV_FP8_TILE_N as GEMV_FP8_TILE_N,
    INDEXER_TILE_M as INDEXER_TILE_M,
)

LOWERINGS_VERSION = "0.2.0"
LOWERINGS = {name: globals()[spec.lowering] for name, spec in CONTRACTS.items()}


def lower_op(op: Operator, graph: OperatorGraph) -> TaskFamily:
    if op.kind in CONTRACTS:
        CONTRACTS[op.kind].validate_attributes(op.attributes)
    factory = LOWERINGS.get(op.kind)
    if factory is None:
        raise KeyError(f"no task lowering registered for operator kind {op.kind!r}")
    return factory(op, graph)


def lower_graph(graph: OperatorGraph) -> list[TaskFamily]:
    """One family per recorded operator, in program (phase) order."""
    return [lower_op(op, graph) for op in graph.ops]


def check_thread_contract(families: list[TaskFamily]) -> list[str]:
    """All families must agree on the common thread count (§7.1)."""
    threads = {f.threads for f in families}
    if len(threads) != 1:
        return [
            f"thread-group contract mismatch: families use {sorted(threads)}; a persistent worker cannot change its physical block size between tasks"
        ]
    return []
