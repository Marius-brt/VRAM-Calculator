#!/usr/bin/env bash
# Measures a model on your hardware and writes report.md to paste into a GitHub issue, so the VRAM Calculator's
# speed estimates can be checked and fitted against it:
# https://github.com/Marius-brt/VRAM-Calculator/issues/new?template=benchmark.yml
#
#   bash benchmark.sh                                    a vLLM server on http://localhost:8000
#   bash benchmark.sh --docker vllm-server-1 --users 12  vLLM inside a Docker container, 12 users at once
#   bash benchmark.sh --gguf model.gguf                  llama.cpp's llama-bench on a GGUF file
#
# Options:
#   --url URL        the server as seen from where vLLM runs (inside the container with --docker); default http://localhost:8000
#   --users N        concurrent users for the second run, e.g. your --max-num-seqs; default 8
#   --docker NAME    run everything inside this container (docker ps shows the name)
#   --gguf FILE      benchmark a GGUF file with llama-bench instead of a vLLM server
#
# vLLM runs use `vllm bench serve` against the running server: it only sends requests and loads nothing on the GPUs.
# Prompts are random, 1,500 tokens in and 500 out; speculative decoding accepts fewer drafts on random text than on
# real chats, and the report includes how many it accepted. Run it when nothing else uses the GPUs.
set -euo pipefail

URL=http://localhost:8000
USERS=8
DOCKER=
GGUF=
while [ $# -gt 0 ]; do
  case $1 in
    --url) URL=$2; shift 2 ;;
    --users) USERS=$2; shift 2 ;;
    --docker) DOCKER=$2; shift 2 ;;
    --gguf) GGUF=$2; shift 2 ;;
    -h | --help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
  esac
done

OUT=vram-benchmark-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
REPORT=$OUT/report.md

run() { if [ -n "$DOCKER" ]; then docker exec -i "$DOCKER" "$@"; else "$@"; fi; }
# Hide API keys if they appear in a launch command.
mask() { sed -E 's/(--api-key[ =]|[Tt][Oo][Kk][Ee][Nn][=:])[^ ]+/\1***/g'; }
fence() { echo '```'; cat; echo '```'; }

gpus() {
  if run nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader 2>/dev/null; then
    [ "$(run nvidia-smi -L 2>/dev/null | wc -l)" -gt 1 ] && run nvidia-smi topo -m 2>/dev/null || true
  elif run rocm-smi --showproductname 2>/dev/null; then :
  elif command -v system_profiler >/dev/null; then system_profiler SPDisplaysDataType | grep -E "Chipset Model|Total Number of Cores"
  else echo "No GPU tool found (nvidia-smi, rocm-smi)"; fi
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
  fetch() { run python3 -c 'import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1], timeout=10).read().decode())' "$1"; }
  if ! fetch "$URL/v1/models" > "$OUT/models.json" 2>/dev/null; then
    echo "No server answers at $URL/v1/models. Start vLLM first, or pass --url (and --docker if it runs in a container)." >&2
    exit 1
  fi
  if [ -n "$DOCKER" ]; then
    command=$(docker inspect -f '{{join .Config.Entrypoint " "}} {{join .Config.Cmd " "}}' "$DOCKER")
  else
    command=$(ps -eo args | grep -E '^([^ ]*/)?(vllm serve|python[0-9.]* .*vllm\.entrypoints)' | head -1 || true)
  fi

  bench() {
    local log=$OUT/vllm-bench-$1-users.log
    echo "Running $1 concurrent user(s), $2 requests of 1,500 tokens in and 500 out..." >&2
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
    echo "vLLM $(fetch "$URL/version" 2>/dev/null | sed -E 's/.*"version": *"([^"]+)".*/\1/' || echo "version unknown")"
    echo
    echo "Models served:"
    fence < "$OUT/models.json"
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
