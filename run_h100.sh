#!/usr/bin/env bash
set -euo pipefail

# One-command helper for 1xH100 runs.
# - Creates/uses ./venv
# - Installs requirements
# - Runs timed training
# - Saves logs + artifacts into runs/<RUN_ID>/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

MAX_SECS="${MAX_WALLCLOCK_SECONDS:-4560}"  # 9.5 min * 8
RUN_ID="${RUN_ID:-h100x1_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="runs/${RUN_ID}"

DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"

TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"

# Linux/Triton acceleration knobs for train_gpt.py.
COMPILE_MODEL="${COMPILE_MODEL:-1}"
COMPILE_MUON="${COMPILE_MUON:-1}"
COMPILE_MODE="${COMPILE_MODE:-max-autotune-no-cudagraphs}"
SDP_BACKEND="${SDP_BACKEND:-auto}"

PREFLIGHT_CHECK_DATA="${PREFLIGHT_CHECK_DATA:-1}"

mkdir -p "${OUT_DIR}"
export OUT_DIR
export DATA_PATH
export RUN_ID

echo "[1/4] Creating venv if needed..."
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "[2/4] Installing dependencies..."
python -m pip install --upgrade pip setuptools wheel
# Install CUDA-enabled PyTorch explicitly for H100 (CUDA 12.4 wheels).
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
# Install the remaining requirements without re-resolving torch.
grep -viE '^\s*torch(==|\s|$)' requirements.txt > "${OUT_DIR}/requirements.no_torch.txt"
pip install -r "${OUT_DIR}/requirements.no_torch.txt"

echo "[2.5/4] Capturing reproducibility context..."
git rev-parse HEAD > "${OUT_DIR}/git_commit.txt" || true
git status --short > "${OUT_DIR}/git_status.txt" || true
git diff -- train_gpt.py run_h100.sh > "${OUT_DIR}/code_diff.patch" || true
python -m pip freeze > "${OUT_DIR}/pip_freeze.txt"
nvidia-smi > "${OUT_DIR}/nvidia_smi.txt" || true
cp -f requirements.txt "${OUT_DIR}/requirements.txt"

if [[ "${PREFLIGHT_CHECK_DATA}" == "1" ]]; then
  echo "[2.6/4] Validating dataset shards before training..."
  python - <<'PY'
import glob
import json
import os
import struct
from pathlib import Path

data_path = Path(os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024"))
train_files = sorted(glob.glob(str(data_path / "fineweb_train_*.bin")))
val_files = sorted(glob.glob(str(data_path / "fineweb_val_*.bin")))
all_files = train_files + val_files
if not all_files:
  raise SystemExit(f"No shard files found under {data_path}")

HEADER_INTS = 256
INT32 = 4
HEADER_BYTES = HEADER_INTS * INT32
TOKEN_BYTES = 2
MAGIC = 20240520
VERSION = 1

bad = []
report = []
for p in all_files:
  path = Path(p)
  size = path.stat().st_size
  ok = True
  err = ""
  expected = None
  num_tokens = None
  if size < HEADER_BYTES:
    ok = False
    err = f"too_small:{size}"
  else:
    with path.open("rb") as f:
      raw = f.read(HEADER_BYTES)
    hdr = struct.unpack("<" + "i" * HEADER_INTS, raw)
    if hdr[0] != MAGIC or hdr[1] != VERSION:
      ok = False
      err = f"bad_header:magic={hdr[0]} version={hdr[1]}"
    else:
      num_tokens = int(hdr[2])
      expected = HEADER_BYTES + num_tokens * TOKEN_BYTES
      if size != expected:
        ok = False
        err = f"size_mismatch:expected={expected} got={size}"
  entry = {
    "file": str(path),
    "ok": ok,
    "size": size,
    "num_tokens": num_tokens,
    "expected_size": expected,
    "error": err,
  }
  report.append(entry)
  if not ok:
    bad.append(entry)

out_dir = Path(os.environ["OUT_DIR"])
(out_dir / "dataset_shard_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
if bad:
  (out_dir / "dataset_shard_errors.json").write_text(json.dumps(bad, indent=2), encoding="utf-8")
  bad_preview = "\n".join(f" - {b['file']}: {b['error']}" for b in bad[:10])
  raise SystemExit(f"Dataset preflight failed with {len(bad)} bad shards:\n{bad_preview}")

print(f"dataset_preflight:ok shards={len(report)}")
PY
fi

echo "[3/4] Running training..."
echo "compile_model=${COMPILE_MODEL} compile_muon=${COMPILE_MUON} compile_mode=${COMPILE_MODE} sdp_backend=${SDP_BACKEND}"
TRAIN_RC=0
(
  export RUN_ID
  export MAX_WALLCLOCK_SECONDS="${MAX_SECS}"
  export DATA_PATH
  export TOKENIZER_PATH
  export VOCAB_SIZE
  export TRAIN_LOG_EVERY
  export VAL_LOSS_EVERY
  export COMPILE_MODEL
  export COMPILE_MUON
  export COMPILE_MODE
  export SDP_BACKEND
  export OUT_DIR

  torchrun --standalone --nproc_per_node=1 train_gpt.py
) 2>&1 | tee "${OUT_DIR}/console.log" || TRAIN_RC=$?
export TRAIN_RC

echo "[4/4] Saving artifacts..."
if [[ -f "logs/${RUN_ID}.txt" ]]; then
  cp -f "logs/${RUN_ID}.txt" "${OUT_DIR}/train.log"
fi

if [[ -f final_model.pt ]]; then
  cp -f final_model.pt "${OUT_DIR}/final_model.pt"
fi
if [[ -f final_model.int8.ptz ]]; then
  cp -f final_model.int8.ptz "${OUT_DIR}/final_model.int8.ptz"
fi
cp -f train_gpt.py "${OUT_DIR}/train_gpt.py"

python - <<'PY'
import hashlib
import json
import os
import pathlib
import time

run_id = os.environ["RUN_ID"]
out_dir = pathlib.Path("runs") / run_id
train_rc = int(os.environ.get("TRAIN_RC", "0"))

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

meta = {
    "run_id": run_id,
    "finished_unix": time.time(),
  "train_exit_code": train_rc,
}

m_pt = pathlib.Path("final_model.pt")
m_q = pathlib.Path("final_model.int8.ptz")
if m_pt.exists():
  meta["final_model_pt_bytes"] = m_pt.stat().st_size
  meta["sha256_final_model_pt"] = sha256(m_pt)
if m_q.exists():
  meta["final_model_int8_ptz_bytes"] = m_q.stat().st_size
  meta["sha256_final_model_int8_ptz"] = sha256(m_q)

(out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
PY

echo "Done. Results saved to ${OUT_DIR}"
if [[ "${TRAIN_RC}" != "0" ]]; then
  echo "Training failed with exit code ${TRAIN_RC}. Artifacts and logs were still preserved."
  exit "${TRAIN_RC}"
fi
