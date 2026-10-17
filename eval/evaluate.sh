#!/usr/bin/env bash
# Evaluate any ACT/LeRobot policy on an ALOHA sim task with lerobot-eval.
#
# One script for every model in the project (our own Transfer Cube ACT, later
# the Insertion models), with the same settings as the Phase 1 baseline
# (eval/phase1_baseline.sh), so all results are directly comparable:
# batch size 1, full fp32 precision, seed 1000 -> episode seeds 1000, 1001, ...
#
# Saves:
#   outputs/eval/<run_name>/eval_info.json   success rate + per-episode results
#   outputs/eval/<run_name>/videos/          videos of the first 10 episodes
#
# Usage:
#   conda activate lerobot && unset PYTHONPATH
#   bash eval/evaluate.sh <policy> <task> <run_name> [episodes]
#
#   <policy>    Hugging Face repo (e.g. Wolvered/act_aloha_transfer_cube) or a
#               local folder with config.json + model.safetensors. A Hub model is
#               downloaded automatically to ~/.cache/huggingface on first use.
#   <task>      AlohaTransferCube-v0 or AlohaInsertion-v0
#   <run_name>  name of the output folder, e.g. phase2_act_own_transfer_cube
#   [episodes]  default 50 (the full evaluation); use 2 for a quick test
#
# Example:
#   bash eval/evaluate.sh Wolvered/act_aloha_transfer_cube AlohaTransferCube-v0 phase2_act_own_transfer_cube 2
#
# Checked against LeRobot commit 8c920c42 (`lerobot-eval --help`).

set -euo pipefail   # stop on any error, on unset variables, and on failures inside pipes

# 1. Read the arguments. Show the usage text and stop if one is missing.
if [ "$#" -lt 3 ]; then
    echo "Usage: bash eval/evaluate.sh <policy> <task> <run_name> [episodes]" >&2
    exit 1
fi
POLICY="$1"
TASK="$2"
RUN_NAME="$3"
N_EPISODES="${4:-50}"

# 2. Only accept the two ALOHA tasks. A typo would otherwise make LeRobot fail
#    late, or (with no --env.task at all) silently use its default task, Insertion.
case "$TASK" in
    AlohaTransferCube-v0|AlohaInsertion-v0) ;;
    *) echo "Unknown task '$TASK'. Use AlohaTransferCube-v0 or AlohaInsertion-v0." >&2; exit 1 ;;
esac

# 3. One output folder per run. The episode count is part of the name, so a
#    2-episode test never mixes with the 50-episode run. Refuse to overwrite:
#    a finished 40-minute run should never be lost by accident.
#    outputs/ is in .gitignore; the small eval_info.json is copied to
#    analysis/results/ by hand once a run is final.
OUTPUT_DIR="outputs/eval/${RUN_NAME}_n${N_EPISODES}"
if [ -e "$OUTPUT_DIR" ]; then
    echo "$OUTPUT_DIR already exists. Delete it or rename it before re-running." >&2
    exit 1
fi

# 4. Draw camera images off-screen on the GPU (no window), as tested in Phase 0.
export MUJOCO_GL=egl

# 5. Run the evaluation, with exactly the Phase 1 baseline settings:
#    --eval.batch_size=1: one simulation at a time (4 GB GPU).
#    --policy.use_amp=false: full fp32 precision, no mixed-precision differences.
#    --seed=1000: fixed randomness, so every policy sees the same object positions.
echo "Evaluating $POLICY on $TASK for $N_EPISODES episodes -> $OUTPUT_DIR"
time lerobot-eval \
    --policy.path="$POLICY" \
    --env.type=aloha \
    --env.task="$TASK" \
    --eval.n_episodes="$N_EPISODES" \
    --eval.batch_size=1 \
    --policy.device=cuda \
    --policy.use_amp=false \
    --seed=1000 \
    --output_dir="$OUTPUT_DIR"

# 6. Print the headline numbers from the result file.
python - "$OUTPUT_DIR/eval_info.json" <<'EOF'
import json, sys
info = json.load(open(sys.argv[1]))
o = info["overall"]
lo, hi = o["pc_success_ci95"]
print(f"Success: {o['n_success']}/{o['n_episodes']} ({o['pc_success']:.1f}%), "
      f"95% CI {lo:.1f}-{hi:.1f}%  ->  {sys.argv[1]}")
EOF
