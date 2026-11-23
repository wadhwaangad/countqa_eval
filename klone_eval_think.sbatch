#!/bin/bash
# CountQA eval of Qwen/Qwen3.5-4B (bf16) in thinking mode, served by vLLM on one GPU.
# Resumable: on preemption/requeue it continues from the predictions JSONL.
#SBATCH --job-name=countqa-think
#SBATCH --account=ckpt-krishna
#SBATCH --partition=ckpt-all
#SBATCH --gpus=1
#SBATCH --constraint=a100|a40|l40|l40s|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=logs/think-%j.out
#SBATCH --error=logs/think-%j.err

set -eo pipefail
# Several copies may be queued on different partitions: the first to start cancels the others;
# a copy that starts while another is running exits.
others_running=$(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t RUNNING -o %i | grep -vx "$SLURM_JOB_ID" || true)
if [ -n "$others_running" ]; then
  echo "another copy is running ($others_running); exiting"; exit 0
fi
for j in $(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING -o %i | grep -vx "$SLURM_JOB_ID" || true); do
  echo "cancelling pending sibling $j"; scancel "$j"
done

STORAGE_ROOT="/gscratch/krishna/$USER"
cd "$STORAGE_ROOT/countqa_eval"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
module load cuda/12.8.1
module load coenv/miniconda/3.13.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STORAGE_ROOT/conda-envs/title-filter"
export PYTHONNOUSERSITE=1
export HF_HOME="$STORAGE_ROOT/hf-cache"
export HF_HUB_OFFLINE=1
export no_proxy="127.0.0.1,localhost,${no_proxy:-}" NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export XDG_CACHE_HOME="$STORAGE_ROOT/xdg-cache"
export TMPDIR="$STORAGE_ROOT/tmp"
export TRITON_CACHE_DIR="$STORAGE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$STORAGE_ROOT/torchinductor"

# Override with env vars, e.g. for the SFT model:
#   sbatch --job-name=countqa-think-sft --export=ALL,MODEL=/gscratch/.../final,OUT=preds_sft.jsonl klone_eval_think.sbatch
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
OUT=${OUT:-preds_qwen3.5-4b_bf16_think.jsonl}
PORT=$((20000 + SLURM_JOB_ID % 10000))
nvidia-smi --query-gpu=name,memory.total --format=csv

vllm serve "$MODEL" --port "$PORT" --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":1}' \
  > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; tail -50 "logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
echo "vLLM ready"

python eval_countqa.py --data data --out "$OUT" \
  --url "http://127.0.0.1:$PORT/v1/chat/completions" --model "$MODEL" \
  --thinking --max-tokens 24576 --workers 32 ${EXTRA_ARGS:-}
