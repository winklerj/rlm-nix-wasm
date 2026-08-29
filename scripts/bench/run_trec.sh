#!/bin/bash
# Usage: run_trec.sh <context-len> [output-suffix]
# Runs one trec_coarse eval at the given context length with task-level
# resume: completed tasks in the output JSONL are skipped on relaunch.
. "$(dirname "$0")/env.sh"
ctx=$1; suffix=${2:-}
out="results/oolong-trec-coarse-${ctx}-maxdepth1${suffix}.jsonl"
log="/tmp/oolong-trec-d1-${ctx}${suffix}.log"
# Probe a real completion, not just /models: during a service upgrade the
# model list answers 200 while completions still fail.
probe() {
  curl -s -m30 -o /dev/null -w "%{http_code}" "$OPENAI_BASE_URL/chat/completions" \
    -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d "{\"model\":\"${RLM_MODEL#openai/}\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1}"
}
until [ "$(probe)" = 200 ]; do
  echo "$(date -u) waiting for llama-server at $OPENAI_BASE_URL" | tee -a "$log"
  sleep 60
done
echo "=== Starting trec_coarse context ${ctx}${suffix} -> ${out} ===" | tee -a "$log"
nix-shell --run "uv run rlm eval run --dataset trec_coarse --context-len ${ctx} --model $RLM_MODEL --max-depth 1 --temperature 0.0 --wasm-python .wasm/python-3.12.0.wasm --output ${out} --resume --trace -v" >> "$log" 2>&1
echo "=== Finished trec_coarse context ${ctx}${suffix} ===" | tee -a "$log"
