#!/usr/bin/env bash
# Phase 1: evaluate the pretrained ACT policy on ALOHA Transfer Cube.
#
# Downloads lerobot/act_aloha_sim_transfer_cube_human from the Hugging Face Hub,
# converts it to the current LeRobot format (step 0), lets it control the
# simulated robot for N episodes, and saves:
#   <output_dir>/eval_info.json   success rate + per-episode results
#   <output_dir>/videos/          videos of the first 10 episodes (fixed in LeRobot)
#
# Usage:
#   conda activate lerobot && unset PYTHONPATH
#   bash eval/phase1_baseline.sh        # full run, 50 episodes
#   bash eval/phase1_baseline.sh 2      # quick test run, 2 episodes
#
# Checked against LeRobot commit 8c920c42 (`lerobot-eval --help`).

set -euo pipefail   # stop on any error, on unset variables, and on failures inside pipes

# 0. Migrate the pretrained model to the current format (only once).
#    The Hub checkpoint was saved by an older LeRobot that kept the normalization
#    (mean/std) inside the model weights. Current LeRobot expects it in separate
#    policy_preprocessor/postprocessor files, so lerobot-eval fails with
#    "ProcessorMigrationError". LeRobot's own migration script moves the
#    statistics out of the weights; the network weights themselves are unchanged.
#    Saved locally only (no --push-to-hub).
HUB_MODEL="lerobot/act_aloha_sim_transfer_cube_human"
POLICY_DIR="outputs/models/act_aloha_sim_transfer_cube_human_migrated"
if [ ! -f "$POLICY_DIR/policy_preprocessor.json" ]; then
    echo "Migrating $HUB_MODEL -> $POLICY_DIR"
    python ~/lerobot/src/lerobot/processor/migrate_policy_normalization.py \
        --pretrained-path "$HUB_MODEL" \
        --output-dir "$POLICY_DIR"
fi

# 1. Number of episodes: first argument, default 50 (the full baseline).
N_EPISODES="${1:-50}"

# 2. One output folder per run size, so a test run never mixes with the full run.
#    outputs/ is in .gitignore (videos are large); we copy the small JSON to
#    analysis/results/ by hand once a run is final.
OUTPUT_DIR="outputs/eval/phase1_act_pretrained_transfer_cube_n${N_EPISODES}"
if [ -e "$OUTPUT_DIR" ]; then
    echo "$OUTPUT_DIR already exists. Delete it or rename it before re-running." >&2
    exit 1
fi

# 3. Draw images off-screen on the GPU (no window), as tested in Phase 0.
export MUJOCO_GL=egl

# 4. Run the evaluation.
#    --env.task is required: LeRobot's ALOHA default task is Insertion, which
#      would test this Transfer Cube policy on the wrong task.
#    --eval.batch_size=1: one simulation at a time (4 GB GPU).
#    --policy.use_amp=false: full fp32 precision, no mixed-precision surprises.
#    --seed=1000: fixed randomness, so every run sees the same cube positions.
echo "Running $N_EPISODES episodes -> $OUTPUT_DIR"
time lerobot-eval \
    --policy.path="$POLICY_DIR" \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes="$N_EPISODES" \
    --eval.batch_size=1 \
    --policy.device=cuda \
    --policy.use_amp=false \
    --seed=1000 \
    --output_dir="$OUTPUT_DIR"
