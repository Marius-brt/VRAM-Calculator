#!/usr/bin/env bash
# Benchmarks a model on this machine for the VRAM Calculator: a vLLM or Unsloth server, or a GGUF file with
# llama-bench. `bash benchmark.sh --help` explains the modes and options.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash benchmark.sh [--unsloth | --gguf FILE] [options]

Measures how fast a model runs on this machine and writes report.md. Paste
the report into a benchmark issue so the VRAM Calculator's speed estimates
can be checked against your hardware:
https://github.com/Marius-brt/VRAM-Calculator/issues/new?template=benchmark.yml

Start the server with the model you want first, and run this while nothing
else uses the GPUs. It only sends requests; it loads nothing on the GPUs.

vLLM (the default)
  Benchmarks a running vLLM server with `vllm bench serve`.

    bash benchmark.sh
    bash benchmark.sh --users 12
    bash benchmark.sh --docker vllm-server-1

  --url URL       server address (default http://localhost:8000)

Unsloth
  Benchmarks a running `unsloth run` server with a built-in Python client,
  so vLLM doesn't need to be installed.

    bash benchmark.sh --unsloth --api-key KEY
    bash benchmark.sh --unsloth --api-key KEY --model unsloth/Qwen3-8B-GGUF

  --unsloth       benchmark Unsloth instead of vLLM
  --api-key KEY   the key `unsloth run` printed when it started (or set
                  BENCH_API_KEY); it never goes into the report
  --model ID      model to benchmark (default: the one Unsloth has loaded)
  --url URL       server address (default http://localhost:8888)

llama.cpp
  Runs llama-bench on a GGUF file: a 512-token prompt, then 128 tokens.

    bash benchmark.sh --gguf model.gguf

  --gguf FILE     the GGUF file to benchmark

Common options
  --users N       users at once in a server's second run (default 8);
                  match vLLM's --max-num-seqs or Unsloth's --parallel
  --docker NAME   run everything inside this Docker container (docker ps
                  shows the name); --url is then as seen from inside it
  -h, --help      show this help

What a server run sends
  Random prompts of 1,500 tokens, asking for 500 back: 20 requests from one
  user, then 10 per user from N users at once. Speculative decoding accepts
  fewer drafts on random text than on real chats.
EOF
}

URL=
USERS=8
DOCKER=
GGUF=
UNSLOTH=
MODEL=
# Options that take a value: stop with a clear message when it's missing.
need() { [ $# -ge 2 ] || { echo "$1 needs a value. See: bash benchmark.sh --help" >&2; exit 1; }; }
while [ $# -gt 0 ]; do
  case $1 in
    --url) need "$@"; URL=$2; shift 2 ;;
    --users) need "$@"; USERS=$2; shift 2 ;;
    --docker) need "$@"; DOCKER=$2; shift 2 ;;
    --unsloth) UNSLOTH=1; shift ;;
    --api-key) need "$@"; BENCH_API_KEY=$2; shift 2 ;;
    --model) need "$@"; MODEL=$2; shift 2 ;;
    --gguf) need "$@"; GGUF=$2; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown option: $1. See: bash benchmark.sh --help" >&2; exit 1 ;;
  esac
done
URL=${URL:-http://localhost:$([ -n "$UNSLOTH" ] && echo 8888 || echo 8000)}
# Passed to commands through the environment, so it stays out of process lists and the report.
export BENCH_API_KEY=${BENCH_API_KEY:-}

OUT=vram-benchmark-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
REPORT=$OUT/report.md

run() { if [ -n "$DOCKER" ]; then docker exec -i -e BENCH_API_KEY "$DOCKER" "$@"; else "$@"; fi; }
# Hide API keys and passwords if they appear in a launch command.
mask() { sed -E 's/(--api-key[ =]|--password[ =]|[Tt][Oo][Kk][Ee][Nn][=:])[^ ]+/\1***/g'; }
fence() { echo '```'; cat; echo '```'; }

gpus() {
  if run nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader 2>/dev/null; then
    [ "$(run nvidia-smi -L 2>/dev/null | wc -l)" -gt 1 ] && run nvidia-smi topo -m 2>/dev/null || true
  elif run rocm-smi --showproductname 2>/dev/null; then :
  elif command -v system_profiler >/dev/null; then system_profiler SPDisplaysDataType | grep -E "Chipset Model|Total Number of Cores"
  else echo "No GPU tool found (nvidia-smi, rocm-smi)"; fi
}

# Streams chat completions at a fixed concurrency and prints the same figures as `vllm bench serve`.
write_client() {
  cat > "$OUT/serve_bench.py" <<'PY'
import json, os, random, statistics, sys, threading, time, urllib.error, urllib.request

url, model, users, total = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
headers = {"Content-Type": "application/json"}
if os.environ.get("BENCH_API_KEY"):
    headers["Authorization"] = "Bearer " + os.environ["BENCH_API_KEY"]
WORDS = (
    "the of and to in is was for on that with as by at from his it an were are which this be or has had not first one "
    "their its new after but who they have her she two been other when there all during into school time may years more "
    "most only over city some world would where later up such used many can state about national out known university"
).split()


def request(seed, max_tokens, ignore_eos):
    # A different random prompt every time, so no request reuses another's cached prefix.
    rng = random.Random(seed)
    text = " ".join(rng.choice(WORDS) for _ in range(1400))
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Continue this text for as long as you can: " + text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), headers)
    start, first, chunks, usage = time.perf_counter(), None, 0, {}
    with urllib.request.urlopen(req, timeout=900) as res:
        for line in res:
            line = line.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            msg = json.loads(line[5:])
            usage = msg.get("usage") or usage
            for choice in msg.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    first = first or time.perf_counter()
                    chunks += 1
    if first is None:
        raise RuntimeError("the reply had no text")
    out = usage.get("completion_tokens") or chunks
    return usage.get("prompt_tokens") or 0, out, (first - start) * 1000, (time.perf_counter() - first) * 1000 / max(1, out - 1)


# Warm up, and find out whether the server keeps generating past the end-of-text token.
ignore_eos = True
try:
    request(random.randrange(1 << 30), 8, True)
except urllib.error.HTTPError as e:
    if e.code not in (400, 422):
        raise
    ignore_eos = False
    request(random.randrange(1 << 30), 8, False)

base, results, failures, lock = random.randrange(1 << 30), [], [], threading.Lock()
seeds = iter(range(total))


def worker():
    while True:
        with lock:
            seed = next(seeds, None)
        if seed is None:
            return
        try:
            result = request(base + seed, 500, ignore_eos)
            with lock:
                results.append(result)
        except Exception as e:
            with lock:
                failures.append(str(e))


began = time.perf_counter()
threads = [threading.Thread(target=worker) for _ in range(users)]
for t in threads:
    t.start()
for t in threads:
    t.join()
duration = time.perf_counter() - began
if not results:
    sys.exit("Every request failed: " + failures[0])

ttft, tpot = [r[2] for r in results], [r[3] for r in results]
rows = [
    ("Successful requests", len(results)),
    ("Failed requests", len(failures)),
    ("Maximum request concurrency", users),
    ("Benchmark duration (s)", f"{duration:.2f}"),
    ("Total input tokens", sum(r[0] for r in results)),
    ("Total generated tokens", sum(r[1] for r in results)),
    ("Output token throughput (tok/s)", f"{sum(r[1] for r in results) / duration:.2f}"),
    ("Mean TTFT (ms)", f"{statistics.mean(ttft):.2f}"),
    ("Median TTFT (ms)", f"{statistics.median(ttft):.2f}"),
    ("Mean TPOT (ms)", f"{statistics.mean(tpot):.2f}"),
    ("Median TPOT (ms)", f"{statistics.median(tpot):.2f}"),
]
print("{s:{c}^50}".format(s=" Serving Benchmark Result ", c="="))
for name, value in rows:
    print(f"{name + ':':<40}{value}")
print("=" * 50)
if not ignore_eos:
    print("The server doesn't take ignore_eos, so replies may end before 500 tokens.")
if failures:
    print("First failure:", failures[0])
PY
}

{
  echo "## Hardware"
  gpus | fence
} > "$REPORT"

if [ -n "$GGUF" ]; then
  echo "Running llama-bench on $GGUF (prompt of 512 tokens, then 128 generated)..." >&2
  {
    echo "## llama-bench"
    run llama-bench -m "$GGUF" -ngl 99 -fa 1 -p 512 -n 128 2>&1 | grep -E '^\||^build' | fence
  } >> "$REPORT"
else
  fetch() {
    run python3 -c 'import os, sys, urllib.request
req = urllib.request.Request(sys.argv[1])
if os.environ.get("BENCH_API_KEY"): req.add_header("Authorization", "Bearer " + os.environ["BENCH_API_KEY"])
print(urllib.request.urlopen(req, timeout=10).read().decode())' "$1"
  }
  name=$([ -n "$UNSLOTH" ] && echo "Unsloth (unsloth run)" || echo vLLM)
  if ! fetch "$URL/v1/models" > "$OUT/models.json" 2>/dev/null; then
    echo "No server answers at $URL/v1/models. Start $name first, or pass --url (and --docker if it runs in a container)." >&2
    [ -n "$UNSLOTH" ] && echo "Unsloth needs the API key it printed at start: pass --api-key." >&2
    exit 1
  fi
  if [ -n "$DOCKER" ]; then
    command=$(docker inspect -f '{{join .Config.Entrypoint " "}} {{join .Config.Cmd " "}}' "$DOCKER")
  elif [ -n "$UNSLOTH" ]; then
    command=$(ps -eo args | grep -E '^([^ ]*/)?unsloth (studio )?run' | head -1 || true)
  else
    command=$(ps -eo args | grep -E '^([^ ]*/)?(vllm serve|python[0-9.]* .*vllm\.entrypoints)' | head -1 || true)
  fi

  if [ -n "$UNSLOTH" ]; then
    write_client
    # Unsloth lists every downloaded model and flags the loaded ones; a pinned id (repo:quant) names the quant too.
    model=${MODEL:-$(run python3 -c 'import json, sys
rows = json.load(sys.stdin)["data"]
loaded = [m["id"] for m in rows if m.get("loaded")] if any("loaded" in m for m in rows) else [m["id"] for m in rows[:1]]
print(sorted(loaded, key=lambda i: ":" not in i)[0] if loaded else "")' < "$OUT/models.json")}
    if [ -z "$model" ]; then
      echo "Unsloth has no model loaded. Start one with unsloth run --model <repo>, or pass --model." >&2
      exit 1
    fi
    version=$(fetch "$URL/api/health" 2>/dev/null | run python3 -c 'import json, sys; print("Unsloth", json.load(sys.stdin)["version"])' 2>/dev/null ||
      run unsloth --version 2>/dev/null || echo "Unsloth, version unknown")
  else
    version="vLLM $(fetch "$URL/version" 2>/dev/null | sed -E 's/.*"version": *"([^"]+)".*/\1/' || echo "version unknown")"
  fi

  bench() {
    local log=$OUT/bench-$1-users.log
    echo "Running $1 concurrent user(s), $2 requests of 1,500 tokens in and 500 out..." >&2
    if [ -n "$UNSLOTH" ]; then
      run python3 - "$URL" "$model" "$1" "$2" < "$OUT/serve_bench.py" > "$log" 2>&1 || {
        tail -n 20 "$log" >&2
        echo "The benchmark failed; the full log is in $log" >&2
        exit 1
      }
      sed -n '/Serving Benchmark Result/,$p' "$log"
      return
    fi
    if ! run vllm bench serve --base-url "$URL" --dataset-name random --random-input-len 1500 --random-output-len 500 \
      --ignore-eos --num-prompts "$2" --max-concurrency "$1" > "$log" 2>&1; then
      tail -n 20 "$log" >&2
      echo "vllm bench serve failed; the full log is in $log" >&2
      exit 1
    fi
    sed -n '/Serving Benchmark Result/,/^==*$/p' "$log"
  }

  {
    echo "## Server"
    echo "$version"
    echo
    if [ -n "$UNSLOTH" ]; then
      # Just the one benchmarked: the full list names every model downloaded on the machine.
      echo "Model: $model"
    else
      echo "Models served:"
      fence < "$OUT/models.json"
    fi
    echo "Launch command:"
    echo "${command:-not found, please paste it in the issue}" | mask | fence
    echo "## 1 user"
    bench 1 20 | fence
    echo "## $USERS users"
    bench "$USERS" $((USERS * 10)) | fence
  } >> "$REPORT"
fi

cat "$REPORT"
echo
echo "Saved to $REPORT. Check it doesn't contain anything private, then paste it into a new issue:"
echo "https://github.com/Marius-brt/VRAM-Calculator/issues/new?template=benchmark.yml"
