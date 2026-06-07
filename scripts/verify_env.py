#!/usr/bin/env python
"""verify_env.py — Phase 1 environment diagnostic for Microglia_foundation_model.

Runs a series of independent checks against the local environment and prints a
PASS/WARN/FAIL line for each, then a summary. Designed to run on the target box
(Linux + RTX A5000) after `uv sync`.

Usage:
    uv run python verify_env.py
    uv run python verify_env.py --ckpt tests/small-v2.ckpt
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[status]
    line = f"[{icon}] {status:<4} {name}"
    if detail:
        line += f" — {detail}"
    print(line)


def check_python() -> None:
    v = sys.version_info
    ok = (3, 10) <= (v.major, v.minor) < (3, 13)
    record(
        "python version",
        PASS if ok else FAIL,
        f"{v.major}.{v.minor}.{v.micro} (need >=3.10,<3.13)",
    )


def check_torch_cuda() -> None:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        record("torch import", FAIL, str(e))
        return
    record("torch import", PASS, f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        record("cuda available", FAIL, "torch.cuda.is_available() == False")
        return
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    record("cuda available", PASS, f"{name}, {total:.1f} GB")
    if total < 20:
        record(
            "gpu memory",
            WARN,
            f"{total:.1f} GB < 24 GB — reduce max_len/batch_size when fine-tuning",
        )
    else:
        record("gpu memory", PASS, f"{total:.1f} GB")


def check_import(module: str, label: str | None = None, required: bool = True) -> bool:
    label = label or module
    try:
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "")
        record(f"import {label}", PASS, ver)
        return True
    except Exception as e:  # noqa: BLE001
        record(f"import {label}", FAIL if required else WARN, str(e).splitlines()[0])
        return False


def check_flash_attention() -> None:
    # The `flash` extra installs simpler_flash[flash]; the FlashTransformer is
    # imported by tasks/finetune.py, so it must work for Phase 4.
    try:
        from simpler_flash import FlashTransformer  # noqa: F401
        record("flash attention", PASS, "simpler_flash.FlashTransformer import ok")
    except Exception as e:  # noqa: BLE001
        record("flash attention", WARN, f"{str(e).splitlines()[0]} (needed for Phase 4)")


def check_lamin() -> None:
    if not check_import("lamindb", "lamindb", required=True):
        return
    # Ontology population is what model code actually needs; probe lightly.
    try:
        import bionty as bt  # noqa: F401
        record("bionty (ontologies)", PASS, "")
    except Exception as e:  # noqa: BLE001
        record("bionty (ontologies)", WARN, str(e).splitlines()[0])
    record(
        "lamin initialized",
        WARN,
        "run `lamin init` + populate_my_ontology() before model code (see tests/test_base.py)",
    )


def check_scprint2() -> None:
    try:
        import scprint2
        from scprint2 import scPRINT2  # noqa: F401
        record("import scprint2", PASS, f"v{getattr(scprint2, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        record("import scprint2", FAIL, str(e).splitlines()[0])
        return
    try:
        from scprint2.tasks import Denoiser, Embedder, GNInfer  # noqa: F401
        record("import tasks", PASS, "Embedder / Denoiser / GNInfer")
    except Exception as e:  # noqa: BLE001
        record("import tasks", FAIL, str(e).splitlines()[0])


def check_checkpoint(ckpt: str | None) -> None:
    if not ckpt:
        record("checkpoint", WARN, "no --ckpt given; supply a verified small/medium .ckpt")
        return
    if not os.path.exists(ckpt):
        record("checkpoint", FAIL, f"not found: {ckpt}")
        return
    size = os.path.getsize(ckpt) / 1024**2
    if size < 1:
        record("checkpoint", FAIL, f"{ckpt} is {size:.2f} MB — likely a corrupt stub, do not use")
        return
    try:
        from scprint2 import scPRINT2
        scPRINT2.load_from_checkpoint(ckpt, map_location="cpu")
        record("checkpoint load", PASS, f"{ckpt} ({size:.0f} MB)")
    except Exception as e:  # noqa: BLE001
        record("checkpoint load", FAIL, str(e).splitlines()[-1])
        traceback.print_exc()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="path to a scPRINT-2 checkpoint to test-load")
    args = ap.parse_args()

    print("=" * 60)
    print(" Microglia_foundation_model — environment diagnostic")
    print("=" * 60)

    check_python()
    check_torch_cuda()
    check_flash_attention()
    check_import("scdataloader", "scdataloader")
    check_lamin()
    check_scprint2()
    check_checkpoint(args.ckpt)

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    print("-" * 60)
    print(f"summary: {len(results)} checks, {n_fail} FAIL, {n_warn} WARN")
    if n_fail:
        print("→ resolve FAILs before Phase 3. Paste this output to debug.")
    elif n_warn:
        print("→ usable; address WARNs (esp. lamin init + flash attn) before Phase 4.")
    else:
        print("→ environment green. Proceed to Phase 2/3.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
