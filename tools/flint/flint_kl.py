"""tools/flint/flint_kl.py - teacher-forced parity of strata-flint against llama.cpp on the same GGUF (the method of
tools/qwen35/kl.py): per prompt, llama.cpp GPU's greedy answer is the forced continuation; the next-token
distributions of llama.cpp GPU (the reference P), llama.cpp CPU (the noise floor between two correct engines) and
strata-flint are compared step by step.  Reported per prompt and overall: mean KL(P || Q), top-1 agreement, and how
many greedy tokens strata-flint's own (unforced) answer shares with llama.cpp's before they part.

    python tools/flint/flint_kl.py --dump LLAMA_DUMP --strata STRATA_FLINT --model GGUF --prompts FILE.json --out DIR
        [--n 128] [--no-cpu] [--strata-args "..."]

FILE.json: a list of {"tag": ..., "text": ...}; each text is wrapped in Flint's chat template (<|user|>text<|assistant|>).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

EOS = {150001, 150003, 150005, 150018}


def run(cmd: list[str], log: Path) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed (rc {r.returncode}), see {log}")
    return r.stdout


def ids(out: str, key: str) -> list[int]:
    for line in out.splitlines():
        if line.startswith(key + ":"):
            return [int(x) for x in line.split(":", 1)[1].split()]
    raise SystemExit(f"no {key}: in the output")


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--strata", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--strata-args", default="")
    ap.add_argument("--vocab", type=int, default=150144)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    engines = ["strata"] + ([] if a.no_cpu else ["cpu"])
    kl = {k: [] for k in engines}
    top1 = {k: [] for k in engines}
    agree = []
    V = a.vocab
    for p in prompts:
        tag, text = p["tag"], "<|user|>" + p["text"] + "<|assistant|>"
        f = {k: out / f"{tag}.{k}.full" for k in ("gpu", "cpu", "strata")}
        meta = out / f"{tag}.ids.json"
        if not meta.exists() or not f["gpu"].exists():
            o = run([a.dump, "--model", a.model, "--prompt", text, "--n", str(a.n), "--ngl", "99", "--dump-full-logits",
                     str(f["gpu"])], out / f"{tag}.gpu.log")
            meta.write_text(json.dumps({"prompt": ids(o, "prompt"), "gen": ids(o, "gen")}))
        m = json.loads(meta.read_text())
        prompt, gen = m["prompt"], m["gen"]
        toks, force = ",".join(map(str, prompt)), ",".join(map(str, gen))
        if "cpu" in engines and not f["cpu"].exists():
            run([a.dump, "--model", a.model, "--tokens", toks, "--force", force, "--ngl", "0", "--dump-full-logits",
                 str(f["cpu"])], out / f"{tag}.cpu.log")
        run([a.strata, "--model", a.model, "--tokens", toks, "--force", force, "--dump-full-logits", str(f["strata"]),
             *shlex.split(a.strata_args)], out / f"{tag}.strata.log")
        o = run([a.strata, "--model", a.model, "--tokens", toks, "--n", str(len(gen)), *shlex.split(a.strata_args)],
                out / f"{tag}.strata_greedy.log")
        sg = ids(o, "gen")
        same = 0
        while same < min(len(sg), len(gen)) and sg[same] == gen[same]:
            same += 1
        agree.append((same, len(gen)))
        P = log_softmax(np.fromfile(f["gpu"], dtype=np.float32).reshape(-1, V))
        line = f"{tag}: prompt {len(prompt)} tokens, {len(P)} steps; greedy same for {same}/{len(gen)}"
        for k in engines:
            Q = log_softmax(np.fromfile(f[k], dtype=np.float32).reshape(-1, V))
            n = min(len(P), len(Q))
            d = (np.exp(P[:n]) * (P[:n] - Q[:n])).sum(axis=1)
            t1 = P[:n].argmax(axis=1) == Q[:n].argmax(axis=1)
            kl[k].extend(d.tolist())
            top1[k].extend(t1.tolist())
            line += f"; {k}: KL {d.mean():.5f} top-1 {t1.mean():.3f}"
        f["strata"].unlink(missing_ok=True)
        print(line, flush=True)
    for k in engines:
        print(f"{k:6s} vs llama.cpp GPU: mean KL {np.mean(kl[k]):.5f} (median {np.median(kl[k]):.6f}, p99 "
              f"{np.percentile(kl[k], 99):.4f}), top-1 {np.mean(top1[k]):.4f} over {len(kl[k])} steps")
    if "cpu" in engines:
        print(f"ratio strata / floor: {np.mean(kl['strata']) / max(np.mean(kl['cpu']), 1e-12):.2f}")
    print("greedy tokens equal to llama.cpp's before the first difference: " +
          ", ".join(f"{s}/{n}" for s, n in agree))
    return 0


if __name__ == "__main__":
    sys.exit(main())
