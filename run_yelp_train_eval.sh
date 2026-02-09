#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   conda activate gr26
#   bash run_yelp_train_eval.sh
#
# Optional overrides:
#   GIN=configs/yelp/hstu-sampled-softmax-n512-final.gin
#   MASTER_PORT=12345
#   LOG_DIR=logs/yelp
#   EVAL_ONLY=1                # skip training and evaluate latest checkpoint
#   RUN_PREPROCESS=1           # force preprocess
#   RUN_PREPROCESS=0           # skip preprocess even if files are missing

GIN="${GIN:-configs/yelp/hstu-sampled-softmax-n512-final.gin}"
MASTER_PORT="${MASTER_PORT:-12345}"
LOG_DIR="${LOG_DIR:-logs/yelp}"
EVAL_ONLY="${EVAL_ONLY:-0}"
RUN_PREPROCESS="${RUN_PREPROCESS:-auto}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "gr26" ]]; then
  echo "[error] Please run in conda env gr26. Current: ${CONDA_DEFAULT_ENV:-<none>}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
TS="$(date +%F_%H%M%S)"

PRE_LOG="${LOG_DIR}/preprocess_${TS}.log"
TRAIN_LOG="${LOG_DIR}/train_${TS}.log"
EVAL_LOG="${LOG_DIR}/eval_${TS}.log"
METRICS_JSON="${LOG_DIR}/metrics_${TS}.json"
METRICS_KEY="${LOG_DIR}/metrics_${TS}.txt"
RUN_SUMMARY="${LOG_DIR}/run_${TS}.summary"

echo "[info] ts=${TS}"
echo "[info] gin=${GIN}"
echo "[info] log_dir=${LOG_DIR}"

python - <<'PY'
import torch
import fbgemm_gpu  # noqa: F401
assert torch.cuda.is_available(), "CUDA not available in gr26"
assert hasattr(torch.ops.fbgemm, "asynchronous_complete_cumsum"), (
    "Missing fbgemm op asynchronous_complete_cumsum"
)
print("[info] runtime checks passed: cuda + fbgemm op")
PY

if [[ "${EVAL_ONLY}" != "1" ]]; then
  NEED_PREPROCESS=0
  if [[ "${RUN_PREPROCESS}" == "1" ]]; then
    NEED_PREPROCESS=1
  elif [[ "${RUN_PREPROCESS}" == "auto" ]]; then
    if [[ ! -f "tmp/yelp/sasrec_format_by_user_train.csv" || ! -f "tmp/yelp/sasrec_format_by_user_valid.csv" || ! -f "tmp/yelp/sasrec_format_by_user_test.csv" ]]; then
      NEED_PREPROCESS=1
    else
      if ! python - <<'PY'
from pathlib import Path
import pandas as pd

train_path = Path("tmp/processed/yelp/interactions_train.csv")
valid_path = Path("tmp/processed/yelp/interactions_valid.csv")
test_path = Path("tmp/processed/yelp/interactions_test.csv")
if not (train_path.exists() and valid_path.exists() and test_path.exists()):
    raise SystemExit(2)

train_df = pd.read_csv(train_path, usecols=["mapped_user_id", "timestamp"])
train_last_ts = train_df.groupby("mapped_user_id", sort=False)["timestamp"].max().rename(
    "train_last_timestamp"
)

def is_strict(path: Path) -> bool:
    df = pd.read_csv(path, usecols=["mapped_user_id", "timestamp"])
    if df.empty:
        return False
    joined = df.join(train_last_ts, on="mapped_user_id")
    if joined["train_last_timestamp"].isna().any():
        return False
    return bool((joined["timestamp"] > joined["train_last_timestamp"]).all())

strict_ok = is_strict(valid_path) and is_strict(test_path)
print(f"[info] strict-chrono cache check: {'ok' if strict_ok else 'stale'}")
raise SystemExit(0 if strict_ok else 3)
PY
      then
        NEED_PREPROCESS=1
      fi
    fi
  fi

  if [[ "${NEED_PREPROCESS}" == "1" ]]; then
    echo "[info] running preprocess..."
    python preprocess_yelp_data.py 2>&1 | tee "${PRE_LOG}"
  else
    echo "[info] skip preprocess"
  fi

  echo "[info] running training..."
  python main.py --gin_config_file "${GIN}" --master_port "${MASTER_PORT}" 2>&1 | tee "${TRAIN_LOG}"
fi

BEST_CKPT="$(ls -t ckpts/yelp-l50/*_best 2>/dev/null | head -1 || true)"
if [[ -z "${BEST_CKPT}" ]]; then
  BEST_CKPT="$(ls -t ckpts/yelp-l50/*_ep* | head -1 || true)"
fi
if [[ -z "${BEST_CKPT}" ]]; then
  echo "[error] No checkpoint found under ckpts/yelp-l50/" >&2
  exit 1
fi
echo "[info] evaluating checkpoint: ${BEST_CKPT}"

python evaluate_checkpoint.py \
  --gin_config_file "${GIN}" \
  --checkpoint_path "${BEST_CKPT}" \
  --device cuda \
  --eval_split test \
  --metrics_out "${METRICS_JSON}" \
  --key_metrics_out "${METRICS_KEY}" \
  2>&1 | tee "${EVAL_LOG}"

{
  echo "ts=${TS}"
  echo "gin=${GIN}"
  echo "checkpoint=${BEST_CKPT}"
  echo "preprocess_log=${PRE_LOG}"
  echo "train_log=${TRAIN_LOG}"
  echo "eval_log=${EVAL_LOG}"
  echo "metrics_json=${METRICS_JSON}"
  echo "metrics_key=${METRICS_KEY}"
} | tee "${RUN_SUMMARY}"

echo "[info] done. Summary: ${RUN_SUMMARY}"
