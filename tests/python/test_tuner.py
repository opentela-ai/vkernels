"""The single tuner (torch_ops.tuner): registry, native sidecar I/O, driver.

The native-store tests mirror tests/core/test_tuning.cpp rule for rule —
a sidecar written here must be byte-compatible with what the C++ reader
persists, and vice versa. No GPU is needed: the arch token is pinned via
VKERNELS_TUNING_ARCH and every artifact lives in a temp store.
"""

import json

import pytest

from vkernels.torch_ops.tuner import (
    NativeStore,
    REGISTRY,
    clear,
    parse_tune_arch,
    parse_tune_body,
    status,
    tune,
)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A temp store + pinned arch; resets the env on exit."""
    monkeypatch.setenv("VKERNELS_TUNING_CACHE", str(tmp_path))
    monkeypatch.setenv("VKERNELS_TUNING_ARCH", "sm000")
    return tmp_path


# -- the sidecar format (parity with core/tuning.cpp) ------------------------


def test_parse_body_comments_blanks_orphans(store):
    text = (
        "# vk-native-tuning/1\n"
        "# arch=sm000 cu_count=48 written_by=bench\n"
        "key=1,4096,64\n"
        "split=64\n"
        "\n"
        "garbage line\n"
        "key=not-an-int\n"
        "split=5\n"          # orphan: malformed key anchored nothing
        "key=2,64\n"
        "split=abc\n"        # malformed value: skipped
        "width=128\n"
    )
    records = parse_tune_body(text)
    assert records == {(1, 4096, 64): {"split": 64}, (2, 64): {"width": 128}}


def test_parse_arch_header_then_filename(store):
    assert parse_tune_arch("# vk-native-tuning/1\n# arch=gfx942 cu_count=228\n",
                           "k.gfx942.tune") == "gfx942"
    assert parse_tune_arch("key=1\nsplit=2\n", "k.sm121.tune") == "sm121"


def test_upsert_replaces_and_keeps_other_records(store):
    native = NativeStore("k", store_dir=store)
    native.upsert((1, 2, 3), {"split": 64}, written_by="test")
    native.upsert((4, 5), {"split": 8}, written_by="test")
    native.upsert((1, 2, 3), {"split": 16}, written_by="test")  # replaces
    assert native.records() == {
        (1, 2, 3): {"split": 16},
        (4, 5): {"split": 8},
    }
    # The file is named <kernel>.<arch>.tune and carries the standard header.
    path = store / "k.sm000.tune"
    text = path.read_text()
    assert text.startswith("# vk-native-tuning/1\n# arch=sm000")
    assert "written_by=test" in text


def test_arch_selection_rules(store):
    # Exact arch wins.
    NativeStore("one", store_dir=store).upsert((1, 2, 3), {"split": 16})
    assert NativeStore("one", store_dir=store).records()[(1, 2, 3)] == {"split": 16}

    # Single foreign-arch file: honored (one-machine rule).
    (store / "two.gfx942.tune").write_text(
        "# vk-native-tuning/1\n# arch=gfx942\nkey=1,2,3\nsplit=64\n")
    assert NativeStore("two", store_dir=store).records()[(1, 2, 3)] == {"split": 64}

    # Two foreign files, none matching: ambiguous -> empty (lenient).
    # (A single foreign file would be honored by the one-machine rule.)
    (store / "three.gfx90a.tune").write_text("key=1\nsplit=32\n")
    (store / "three.gfx942.tune").write_text("key=1\nsplit=64\n")
    assert NativeStore("three", store_dir=store).records() == {}


def test_upsert_without_arch_raises(store):
    # arch="" models a host-only box (no query result, no pin).
    with pytest.raises(Exception, match="arch"):
        NativeStore("k", store_dir=store, arch="").upsert((1,), {"split": 2})


# -- the registry + driver ---------------------------------------------------


def test_registry_covers_both_tiers():
    tiers = {t.tier for t in REGISTRY}
    assert tiers == {"triton", "native"}
    by_name = {t.name: t for t in REGISTRY}
    assert by_name["mhc_projection"].sweep
    assert by_name["dsa_topk_logits_split_for"].bench == "dsa_topk_logits_bench_cuda"
    # The HIP sweep harness cannot persist yet: cataloged, never mis-run.
    assert by_name["dsa_sparse_fwd_split_for"].persists is False


def test_tune_unknown_name(store):
    with pytest.raises(Exception, match="unknown tunable"):
        tune(["nope"], store_dir=store)


def test_tune_non_persisting_entry_is_a_clean_skip(store):
    (report,) = tune(["dsa_sparse_fwd_split_for"], store_dir=store)
    assert report["ok"] is False
    assert "does not write the store" in report["detail"]


def test_tune_native_missing_bench(store, monkeypatch):
    import vkernels.torch_ops.tuner as tuner_mod

    monkeypatch.setattr(tuner_mod, "find_bench_binary", lambda name: None)
    (report,) = tune(["dsa_topk_logits_split_for"], store_dir=store)
    assert report["ok"] is False
    assert "bench not built" in report["detail"]


def test_status_reports_both_tiers(store):
    NativeStore("native_k", store_dir=store).upsert((1,), {"split": 4})
    doc = {
        "schema": "vk-tuning-store/1", "kernel": "triton_k",
        "device": {}, "records": {"[1]": {}}, "updated": "now",
    }
    (store / "triton_k.sm000.json").write_text(json.dumps(doc))

    report = status(store_dir=store)
    assert report["enabled"] and report["arch"] == "sm000"
    by_kernel = {s["kernel"]: s for s in report["stores"]}
    assert by_kernel["native_k"]["tier"] == "native"
    assert by_kernel["native_k"]["records"] == 1
    assert by_kernel["triton_k"]["tier"] == "triton"
    assert by_kernel["triton_k"]["records"] == 1
    # Registry coverage flags the untuned entries.
    tuned = {r["name"]: r["tuned"] for r in report["registry"]}
    assert tuned["mhc_projection"] is False


def test_status_ignores_unknown_schema_lenient(store):
    (store / "weird.sm000.json").write_text('{"schema": "other", "records": {}}')
    by_kernel = {s["kernel"]: s for s in status(store_dir=store)["stores"]}
    assert by_kernel["weird"]["records"] == 0


def test_clear_removes_both_tiers(store):
    NativeStore("k", store_dir=store).upsert((1,), {"split": 4})
    (store / "k.sm000.json").write_text("{}")
    (store / "unrelated.sm000.tune").write_text("key=1\nsplit=2\n")

    (row,) = clear(["k"], store_dir=store)
    assert row == {"name": "k", "cleared": True, "tiers": ["triton", "native"]}
    assert list(store.glob("k.*")) == []
    assert (store / "unrelated.sm000.tune").exists()  # untouched
