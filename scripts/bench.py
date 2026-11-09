# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Measure TTFT, prefill speed and decode tok/s for an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from uuid import uuid4

FILLER = (
    "The quick brown fox jumps over the lazy dog while the satellite maps the coastline. "
)
DECODE_PROMPT = "List the numbers from 1 to 500 in order, separated by commas, with no other text."


class StreamResult:
    def __init__(self) -> None:
        self.ttft: float = float("nan")
        self.elapsed: float = float("nan")
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.chars: int = 0
        self.error: str = ""
        self.content_events: int = 0
        self.blockwise: bool = False


SAMPLING_SUPPORTED = True


def api_request(base: str, path: str, api_key: str | None, payload: dict | None = None, timeout: int = 600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    return urllib.request.urlopen(req, timeout=timeout)


def list_models(base: str, api_key: str | None) -> list[str]:
    with api_request(base, "/models", api_key, timeout=30) as resp:
        ids = [m["id"] for m in json.load(resp)["data"]]
    return sorted(ids)


def pick_model(base: str, api_key: str | None, chosen: str | None) -> str:
    models = list_models(base, api_key)
    if not models:
        sys.exit("endpoint reported no models")
    if chosen:
        if chosen not in models:
            sys.exit(f"'{chosen}' is not served here; available: {', '.join(models)}")
        return chosen

    print("models:")
    for i, m in enumerate(models, 1):
        print(f"  [{i}] {m}")
    while True:
        try:
            raw = input(f"model to benchmark [1-{len(models)}]: ").strip()
        except EOFError:
            sys.exit("no model selected (stdin is not interactive): pass --model")
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        if raw in models:
            return raw
        print("please enter a number from the list (or an exact model id)")


def stream_chat(base: str, api_key: str | None, model: str, prompt: str, max_tokens: int, retries: int = 1) -> StreamResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if SAMPLING_SUPPORTED:
        payload["temperature"] = 0
    res = StreamResult()
    t0 = time.perf_counter()
    first = None
    first_content = None
    chunks = 0
    with api_request(base, "/chat/completions", api_key, payload) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            event = json.loads(body)
            err = event.get("error") or {}
            if err:
                res.error = str(err.get("message") or err)[:200]
                break
            usage = event.get("usage")
            if usage:
                res.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                res.completion_tokens = int(usage.get("completion_tokens") or 0)
            for choice in event.get("choices", []):
                if first is None and (choice.get("delta") is not None or choice.get("text") is not None):
                    first = time.perf_counter()
                delta = choice.get("delta") or {}
                piece = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or choice.get("text")
                    or ""
                )
                ids = delta.get("token_ids") or choice.get("token_ids") or []
                extra = len(ids) if isinstance(ids, list) else 0
                if piece or extra:
                    if first_content is None:
                        first_content = time.perf_counter()
                    res.content_events += 1
                    chunks += extra or 1
                    res.chars += len(piece)
    res.elapsed = time.perf_counter() - t0
    if res.error and retries and "sampling parameters" in res.error and SAMPLING_SUPPORTED:
        globals()["SAMPLING_SUPPORTED"] = False
        print(f"  endpoint rejects sampling params, continuing without temperature")
        return stream_chat(base, api_key, model, prompt, max_tokens, retries - 1)
    anchor = first_content if first_content is not None else first
    if anchor is None:
        return res
    res.ttft = anchor - t0
    if res.content_events <= 1 and res.completion_tokens > 8:
        res.blockwise = True
    if not res.completion_tokens:
        res.completion_tokens = max(chunks, 1)
    if not res.prompt_tokens:
        res.prompt_tokens = max(len(prompt) // 4, 1)
    return res


def long_prompt(target_tokens: int, salt: str = "") -> str:
    per_repeat = max(len(FILLER) // 4, 1)
    repeats = max(target_tokens // per_repeat, 1)
    header = f"Case file {salt}: " if salt else ""
    return header + FILLER * repeats + "How many sentences did I just write? Answer with a single number."


def report(label: str, unit: str, values: list[float], digits: int = 1) -> None:
    if not values:
        print(f"{label:<22} no samples")
        return
    fmt = lambda v: f"{v:,.{digits}f}"
    line = (
        f"min {fmt(min(values))} | med {fmt(statistics.median(values))} | "
        f"mean {fmt(statistics.fmean(values))} | max {fmt(max(values))}"
    )
    print(f"{label:<22} {line}  {unit}  (n={len(values)})")


def burst(base, api_key, model, prompt, max_tokens, parallel):
    """Fire `parallel` concurrent requests; returns (wall seconds, results, errors)."""
    if parallel == 1:
        t0 = time.perf_counter()
        try:
            return time.perf_counter() - t0, [stream_chat(base, api_key, model, prompt, max_tokens)], []
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as err:
            return time.perf_counter() - t0, [], [str(err)]
    wall = 0.0
    with cf.ThreadPoolExecutor(max_workers=parallel) as pool:
        pending = [pool.submit(stream_chat, base, api_key, model, prompt, max_tokens) for _ in range(parallel)]
        t0 = time.perf_counter()
        results, errors = [], []
        for fut in cf.as_completed(pending):
            try:
                results.append(fut.result())
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as err:
                errors.append(str(err))
        wall = time.perf_counter() - t0
    return wall, results, errors


def per_request(res, phase):
    if phase.startswith("decode"):
        window = res.elapsed if res.blockwise else res.elapsed - res.ttft
        return max(res.completion_tokens - 1, 1) / max(window, 1e-6)
    return res.prompt_tokens / max(res.ttft, 1e-6)


def run_phase(base, api_key, model, label, prompt_fn, max_tokens, runs, parallel):
    ttfts, per_req, aggregate = [], [], []
    for i in range(runs):
        prompt = prompt_fn(i) if callable(prompt_fn) else prompt_fn
        wall, results, errors = burst(base, api_key, model, prompt, max_tokens, parallel)
        for res in results:
            if res.error:
                errors.append(res.error)
        usable = [r for r in results if not r.error and r.completion_tokens and math.isfinite(r.ttft)]
        for err in errors:
            print(f"  run {i + 1}/{runs}: request failed: {err}")
        if not usable:
            print(f"  run {i + 1}/{runs}: no usable responses, skipped")
            continue
        results = usable
        if any(r.blockwise for r in results) and label.startswith("decode"):
            print("  note: endpoint streams whole blocks (diffusion) — decode rate spans the full request")
        in_tokens = sum(r.prompt_tokens for r in results)
        out_tokens = sum(r.completion_tokens for r in results)
        agg = (out_tokens if label.startswith("decode") else in_tokens) / max(wall, 1e-6)
        ttfts.extend(r.ttft for r in results)
        per_req.extend(per_request(r, label) for r in results)
        aggregate.append(agg)
        if parallel == 1 and not errors:
            res = results[0]
            print(
                f"  run {i + 1}/{runs}: {res.prompt_tokens} in / {res.completion_tokens} out tokens, "
                f"ttft {res.ttft * 1000:,.0f} ms, rate {per_request(res, label):,.0f} tok/s"
            )
        else:
            med = statistics.median(per_request(r, label) for r in results)
            print(
                f"  run {i + 1}/{runs}: {len(results)}/{parallel} ok, {in_tokens} in / {out_tokens} out tokens, "
                f"wall {wall:,.1f} s, aggregate {agg:,.0f} tok/s, per-req med {med:,.0f} tok/s"
            )
    report(f"{label} ttft", "ms", [t * 1000 for t in ttfts], digits=0)
    report(f"{label} per-req", "tok/s", per_req)
    if parallel > 1:
        report(f"{label} aggregate", "tok/s", aggregate)
    return ttfts, per_req, aggregate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("api", help="OpenAI-compatible base URL, e.g. http://localhost:9292/v1")
    ap.add_argument("--api-key", default="", help="bearer token (default: none)")
    ap.add_argument("--model", help="skip the interactive picker")
    ap.add_argument("--runs", type=int, default=3, help="iterations per phase (default 3)")
    ap.add_argument(
        "--parallel", type=int, default=1, help="concurrent requests per run, for throughput (default 1)"
    )
    ap.add_argument("--prefill-tokens", type=int, default=4096, help="approx prompt size for the prefill phase")
    ap.add_argument("--decode-tokens", type=int, default=256, help="max_tokens for the decode phase")
    args = ap.parse_args()

    if args.runs < 1 or args.parallel < 1:
        sys.exit("--runs and --parallel must both be >= 1")

    base = args.api if args.api.rstrip("/").endswith("/v1") else args.api.rstrip("/") + "/v1"
    try:
        model = pick_model(base, args.api_key, args.model)
    except urllib.error.HTTPError as err:
        sys.exit(f"GET {base}/models failed: {err}")
    except urllib.error.URLError as err:
        sys.exit(f"cannot reach {base}: {err.reason}")

    print(f"\nmodel: {model}\nendpoint: {base}\nruns: {args.runs} x {args.parallel} parallel\n")
    print("[1/3] warmup")
    stream_chat(base, args.api_key, model, "Reply with the single word: ready", 8)

    print("\n[2/3] prefill phase (~%d token prompt, 1 token out)" % args.prefill_tokens)
    run_phase(base, args.api_key, model, "prefill", lambda i: long_prompt(args.prefill_tokens, uuid4().hex[:12]), 1, args.runs, args.parallel)

    print(f"\n[3/3] decode phase (short prompt, up to {args.decode_tokens} tokens out)")
    run_phase(base, args.api_key, model, "decode", DECODE_PROMPT, args.decode_tokens, args.runs, args.parallel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
