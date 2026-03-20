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
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
SDP_BACKEND="${SDP_BACKEND:-auto}"

mkdir -p "${OUT_DIR}"

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

echo "[3/4] Running training..."
echo "compile_model=${COMPILE_MODEL} compile_muon=${COMPILE_MUON} compile_mode=${COMPILE_MODE} sdp_backend=${SDP_BACKEND}"
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

  torchrun --standalone --nproc_per_node=1 train_gpt.py
) 2>&1 | tee "${OUT_DIR}/console.log"

echo "[4/4] Saving artifacts..."
if [[ -f "logs/${RUN_ID}.txt" ]]; then
  cp -f "logs/${RUN_ID}.txt" "${OUT_DIR}/train.log"
fi

cp -f final_model.pt "${OUT_DIR}/final_model.pt"
cp -f final_model.int8.ptz "${OUT_DIR}/final_model.int8.ptz"
cp -f train_gpt.py "${OUT_DIR}/train_gpt.py"

python - <<'PY'
import hashlib
import json
import os
import pathlib
import time

run_id = os.environ["RUN_ID"]
out_dir = pathlib.Path("runs") / run_id

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

m_pt = pathlib.Path("final_model.pt")
m_q = pathlib.Path("final_model.int8.ptz")
meta = {
    "run_id": run_id,
    "finished_unix": time.time(),
    "final_model_pt_bytes": m_pt.stat().st_size,
    "final_model_int8_ptz_bytes": m_q.stat().st_size,
    "sha256_final_model_pt": sha256(m_pt),
    "sha256_final_model_int8_ptz": sha256(m_q),
}
(out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
PY

echo "Done. Results saved to ${OUT_DIR}"
