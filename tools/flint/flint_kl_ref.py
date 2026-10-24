"""tools/flint/flint_kl_ref.py - quantization quality of Flint GGUFs as strata-flint runs them: on the parity prompts'
token sequences (the prompt + a forced continuation, from a flint_kl.py run's <tag>.ids.json), the next-token
distributions of a reference (the BF16 file through llama.cpp's CPU backend, made once and kept) against each
candidate file through strata-flint.  Reported per file: mean KL(ref || file), its median and p99, and the top-1
agreement over all steps.

    python tools/flint/flint_kl_ref.py --dump LLAMA_DUMP --ref-model BF16.gguf --strata STRATA_FLINT --ids DIR --out DIR
        MODEL.gguf [MODEL.gguf ...] [--strata-args "..."] [--ref-ngl 0]
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(cmd, log: Path):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed (rc {r.returncode}), see {log}")


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ref-model", required=True)
    ap.add_argument("--strata", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strata-args", default="")
    ap.add_argument("--ref-ngl", default="0")
    ap.add_argument("--vocab", type=int, default=150144)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tags = sorted(p.name[: -len(".ids.json")] for p in Path(a.ids).glob("*.ids.json"))
    V = a.vocab
    res = {m: {"kl": [], "top1": []} for m in a.models}
    for tag in tags:
        meta = json.loads((Path(a.ids) / f"{tag}.ids.json").read_text())
        toks, force = ",".join(map(str, meta["prompt"])), ",".join(map(str, meta["gen"]))
        ref = out / f"{tag}.ref.full"
        if not ref.exists():
            run([a.dump, "--model", a.ref_model, "--tokens", toks, "--force", force, "--ngl", a.ref_ngl,
                 "--dump-full-logits", str(ref)], out / f"{tag}.ref.log")
        P = log_softmax(np.fromfile(ref, dtype=np.float32).reshape(-1, V))
        line = f"{tag}: {len(P)} steps"
        for m in a.models:
            f = out / f"{tag}.cand.full"
            run([a.strata, "--model", m, "--tokens", toks, "--force", force, "--dump-full-logits", str(f),
                 *shlex.split(a.strata_args)], out / f"{tag}.{Path(m).stem}.log")
            Q = log_softmax(np.fromfile(f, dtype=np.float32).reshape(-1, V))
            n = min(len(P), len(Q))
            d = (np.exp(P[:n]) * (P[:n] - Q[:n])).sum(axis=1)
            t1 = P[:n].argmax(axis=1) == Q[:n].argmax(axis=1)
            res[m]["kl"].extend(d.tolist())
            res[m]["top1"].extend(t1.tolist())
            f.unlink(missing_ok=True)
            line += f"; {Path(m).stem}: KL {d.mean():.4f} top-1 {t1.mean():.3f}"
        print(line, flush=True)
    for m in a.models:
        k, t = np.array(res[m]["kl"]), np.array(res[m]["top1"])
        print(f"{Path(m).name:48s} mean KL {k.mean():.5f}  median {np.median(k):.5f}  p99 {np.percentile(k, 99):.4f}  "
              f"top-1 {t.mean():.4f}  ({len(k)} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
