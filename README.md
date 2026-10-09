# Visual robustness of learned manipulation policies (ALOHA sim)

> **Status: work in progress.** The setup and the first baseline are done; training, perturbations and the main comparison are still being built. Results are added only when the runs behind them are finished.

## Question

How robust are learned robot manipulation policies to visual changes that happen on a real factory floor, such as different lighting, a shifted camera or clutter on the table?

| Success | Failure |
|---|---|
| ![ACT success, seed 1004](media/act_transfer_cube_success_seed1004.gif) | ![ACT failure, seed 1005](media/act_transfer_cube_failure_seed1005.gif) |
| Seed 1004: the cube is lifted and passed to the left arm. | Seed 1005: the cube is lifted but lost before the handover. |

Pretrained ACT on Transfer Cube, from the 50-episode baseline evaluation below. The view is the top camera, the only image the policy sees.

## Approach

Two policies are trained by imitation learning on the same simulated task and evaluated first in the standard scene, then under controlled visual perturbations.

| | |
|---|---|
| Policy A | ACT (Action Chunking Transformer) |
| Policy B | SmolVLA, fine-tuned from `lerobot/smolvla_base` |
| Framework | [Hugging Face LeRobot](https://github.com/huggingface/lerobot) |
| Simulator | ALOHA bimanual robot in MuJoCo ([gym-aloha](https://github.com/huggingface/gym-aloha)) |
| Tasks | Transfer Cube (pipeline check), Insertion (main experiment) |
| Perturbations (planned) | lighting, camera pose, distractor objects, table colour; 2-3 severity levels each |

Each condition is evaluated for 50 episodes with fixed seeds, and both policies see the same seeds.

## Progress

- [x] Setup verified: GPU, LeRobot, headless MuJoCo rendering
- [x] Baseline: pretrained ACT on Transfer Cube
- [ ] Train own ACT on Transfer Cube (Colab)
- [ ] Train ACT on Insertion
- [ ] Fine-tune SmolVLA on Insertion
- [ ] Visual perturbations
- [ ] Evaluation harness (per-episode CSV, resumable)
- [ ] Analysis and write-up

## Results so far

### Baseline: pretrained ACT, Transfer Cube, standard scene

| Policy | Task | Episodes | Seeds | Success | 95% CI |
|---|---|---|---|---|---|
| `lerobot/act_aloha_sim_transfer_cube_human` | AlohaTransferCube-v0 | 50 | 1000-1049 | 38/50 (76.0%) | 62.6-85.7% |

Highest reward stage reached per episode:

| Stage | Meaning | Episodes |
|---|---|---|
| 0 | cube never touched | 1 |
| 1 | right gripper touched the cube | 3 |
| 2 | cube lifted, no handover | 8 |
| 4 | success | 38 |

Most failures happen at the handover between the two arms, not at the first grasp.

Raw output: [`analysis/results/phase1_act_pretrained_transfer_cube.json`](analysis/results/phase1_act_pretrained_transfer_cube.json)

## Notes on the evaluation

- **Success criterion.** gym-aloha counts an episode as a success as soon as the left gripper touches the cube while the cube is off the table, and ends the episode at that moment. A fingertip touch counts the same as a full two-handed grasp, so the success rate is optimistic compared with a completed handover. All policies and conditions are scored with the same rule.
- **Checkpoint format.** The pretrained checkpoint on the Hub was saved by an older LeRobot version. It is converted locally with LeRobot's `migrate_policy_normalization.py`, which moves the normalization statistics out of the weights; the network weights are unchanged.
- **Uncertainty.** With 50 episodes per condition the 95% confidence interval spans up to about ±14 percentage points, so small differences between conditions are not meaningful.

## Repository layout

```
eval/
  phase1_baseline.sh      pretrained ACT baseline with lerobot-eval
scripts/
  check_setup.py          checks Python env, GPU, LeRobot and headless rendering
  view_scene.sh           opens a task scene in the MuJoCo 3D viewer
  watch_policy.py         watch a policy control the robot live in the viewer
  make_gif.py             turn an evaluation video into a small GIF for this README
analysis/results/         raw evaluation outputs (every reported number comes from here)
media/                    GIFs (and later plots) for this README
perturbations/            visual scene changes (planned)
notebooks/                Colab training notebook (planned)
```

## How to reproduce

Requirements: Linux or WSL2, an NVIDIA GPU, conda, and LeRobot installed from source with the `aloha` extra.

```bash
conda activate lerobot

# 1. check the setup
MUJOCO_GL=egl python scripts/check_setup.py

# 2. pretrained ACT baseline (50 episodes, about 40 min on a GTX 1650)
bash eval/phase1_baseline.sh

# optional: look at the scene, or watch the policy live (needs a display)
bash scripts/view_scene.sh
python scripts/watch_policy.py --episodes 3
```

## Environment

| | |
|---|---|
| LeRobot | 0.6.2, commit `8c920c4270460851cedd2737657584586d3dc66f` |
| PyTorch | 2.11.0 (CUDA 13.0) |
| MuJoCo | 3.8.1 |
| Python | 3.12 |
| Local GPU | NVIDIA GTX 1650, 4 GB (evaluation only) |
| Training | Google Colab, T4 (planned) |
