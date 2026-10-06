# Visual robustness of learned manipulation policies (ALOHA sim)

**Question:** How robust are learned manipulation policies to visual changes that happen on a real factory floor (lighting, camera position, clutter)?

We compare ACT and SmolVLA on the ALOHA Insertion task in MuJoCo simulation, first under standard conditions and then under visual perturbations.

Status: work in progress.

## Environment

- LeRobot commit: `8c920c4270460851cedd2737657584586d3dc66f` (lerobot 0.6.2)
- PyTorch 2.11.0 (CUDA 13.0), Python 3.12
- Headless rendering: `MUJOCO_GL=egl`

Check a machine with: `python scripts/check_setup.py`

Look at a simulation scene in MuJoCo's 3D viewer (needs a display, e.g. WSLg):
`bash scripts/view_scene.sh [transfer_cube|insertion]`

## Results

### Phase 1: pretrained ACT baseline (Transfer Cube, standard scene)

| Policy | Task | Episodes | Seeds | Success | 95% CI |
|---|---|---|---|---|---|
| `lerobot/act_aloha_sim_transfer_cube_human` (migrated) | AlohaTransferCube-v0 | 50 | 1000-1049 | 38/50 (76.0%) | 62.6-85.7% |

Raw output: `analysis/results/phase1_act_pretrained_transfer_cube.json`. Reproduce: `bash eval/phase1_baseline.sh`.

Note: gym-aloha counts an episode as a success when the left gripper touches the cube while it is off the table, and ends the episode at that moment. A fingertip touch counts the same as a full two-handed grasp.
