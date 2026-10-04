# Visual robustness of learned manipulation policies (ALOHA sim)

**Question:** How robust are learned manipulation policies to visual changes that happen on a real factory floor (lighting, camera position, clutter)?

We compare ACT and SmolVLA on the ALOHA Insertion task in MuJoCo simulation, first under standard conditions and then under visual perturbations.

Status: work in progress.

## Environment

- LeRobot commit: `8c920c4270460851cedd2737657584586d3dc66f` (lerobot 0.6.2)
- PyTorch 2.11.0 (CUDA 13.0), Python 3.12
- Headless rendering: `MUJOCO_GL=egl`

Check a machine with: `python scripts/check_setup.py`
