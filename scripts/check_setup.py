"""Check that this machine can run LeRobot policies in the ALOHA simulator.

Usage:
    conda activate lerobot && unset PYTHONPATH
    MUJOCO_GL=egl python scripts/check_setup.py [--save-frame frame.png]
"""

import argparse
import os
import subprocess
from pathlib import Path


def check_python_path():
    # ROS 2 Humble puts Python 3.10 packages on PYTHONPATH; they break imports in our 3.12 env.
    pp = os.environ.get("PYTHONPATH", "")
    assert "ros" not in pp.lower(), f"ROS paths in PYTHONPATH, run `unset PYTHONPATH`: {pp}"
    print("[ok] PYTHONPATH clean")


def check_gpu():
    import torch

    assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU"
    print(f"[ok] torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")


def check_lerobot():
    import lerobot

    repo = Path(lerobot.__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(f"[ok] lerobot {lerobot.__version__} at {repo} (commit {commit or 'unknown'})")


def check_aloha_render(save_frame=None):
    import gymnasium as gym
    import gym_aloha  # noqa: F401  (registers the gym_aloha/* environments)

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=0)
    img = obs["pixels"]["top"]
    assert img.mean() > 5, "Rendered frame is (almost) black, rendering backend is broken"
    obs, *_ = env.step(env.action_space.sample())
    print(
        f"[ok] ALOHA env: image {img.shape}, state {obs['agent_pos'].shape}, "
        f"action {env.action_space.shape}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}"
    )
    if save_frame:
        import imageio.v3 as iio

        iio.imwrite(save_frame, img)
        print(f"     saved frame to {save_frame}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-frame", help="optional path to save the rendered top-camera image")
    args = parser.parse_args()

    check_python_path()
    check_gpu()
    check_lerobot()
    check_aloha_render(args.save_frame)
    print("All checks passed.")
