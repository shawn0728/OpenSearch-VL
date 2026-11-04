#!/usr/bin/env bash
# Compare the base model and the SFT adapter on a prepared benchmark parquet.
#
#   bash opensearch_vl/run_sft_eval.sh <cases.parquet> <work_dir> <shards_per_model> [tags]
#
# tags defaults to "base sft". Trajectories and tool images go to <work_dir> on the
# instance disk (mutagen would otherwise mirror every intermediate PNG); results and
# trajectory JSONs are copied to $OUTPUT_ROOT. Finished cases are skipped on re-run.
set -uo pipefail
CASES=${1:?cases parquet}; WORK=${2:?work dir}; SHARDS=${3:-2}; TAGS=${4:-base sft}
HERE=$(cd "$(dirname "$0")" && pwd)
SFT_ENV=/root/miniconda3/envs/sft
OCR_ENV=/root/venvs/ocr
ADAPTER=${ADAPTER:-/root/autodl-fs/runs/imagentrl/exp-main/run-20261103-144117-305fea/adapter}
export PATH="$SFT_ENV/bin:/root/miniconda3/bin:$PATH"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QWEN35_4B_PATH=${QWEN35_4B_PATH:-/root/models/Qwen3.5-4B}
# Longest SFT trajectory has 8 assistant turns; p99 assistant message ≈ 1k tokens.
export AGENT_MAX_TURNS=${AGENT_MAX_TURNS:-15}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
SECRETS=/root/.imagentrl-secrets.env
[ -f "$SECRETS" ] || { echo "ERROR: $SECRETS missing (push it from .proj-tool/secrets.env)"; exit 1; }
set -a; . "$SECRETS"; set +a
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
mkdir -p "$WORK"

# ── OCR service for layout_parsing ───────────────────────────────────────
OCR_PID=""
cleanup() { [ -n "$OCR_PID" ] && kill "$OCR_PID" 2>/dev/null; }
trap cleanup EXIT
ocr_up() { curl -s -m 5 -o /dev/null http://127.0.0.1:8080/docs; }
if ! ocr_up; then
  bash "$HERE/start_ocr_service.sh" > "$OUTPUT_ROOT/ocr_service.log" 2>&1 &
  OCR_PID=$!
  for _ in $(seq 1 120); do ocr_up && break; kill -0 "$OCR_PID" 2>/dev/null || break; sleep 10; done
  ocr_up || { echo "ERROR: OCR service did not start"; tail -30 "$OUTPUT_ROOT/ocr_service.log"; exit 1; }
fi
echo "[$(date +%T)] OCR service ready"

# ── inference: all tags in parallel, shards share the GPU ────────────────
pids=()
for tag in $TAGS; do
  extra=(); [ "$tag" = sft ] && extra=(--adapter "$ADAPTER")
  mkdir -p "$WORK/$tag"
  for i in $(seq 0 $((SHARDS - 1))); do
    python "$HERE/run_infer.py" --model qwen3.5-4b --gpus 0 --dataset test \
      --data-path "$CASES" --output-dir "$WORK/$tag" --shard "$i/$SHARDS" --skip-existing \
      --max-tokens "$MAX_NEW_TOKENS" --log-level WARNING "${extra[@]}" \
      > "$OUTPUT_ROOT/infer_${tag}_$i.log" 2>&1 &
    pids+=($!)
  done
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
echo "[$(date +%T)] inference done (${failed} processes failed)"

# ── judge + summary ──────────────────────────────────────────────────────
runs=()
for tag in $TAGS; do
  python "$HERE/eval_with_gpt4o.py" --traj_dir "$WORK/$tag" --benchmark vdr --max_workers 8 \
    2>&1 | grep -vE "it/s|s/it" | tail -6
  runs+=(--run "$tag=$WORK/$tag")
done
baseline=(); case " $TAGS " in *" base "*) baseline=(--baseline base);; esac
python "$HERE/summarize_sft_eval.py" "${runs[@]}" "${baseline[@]}" --out "$OUTPUT_ROOT/summary"

for tag in $TAGS; do
  mkdir -p "$OUTPUT_ROOT/trajectories/$tag"
  cp "$WORK/$tag"/*_trajectory.json "$WORK/$tag"/eval_report_* "$OUTPUT_ROOT/trajectories/$tag/" 2>/dev/null || true
done
[ "$failed" -eq 0 ] || { echo "ERROR: $failed inference processes failed"; exit 1; }
echo SFT_EVAL_OK
