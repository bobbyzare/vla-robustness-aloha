"""Check that this machine can run LeRobot policies in the ALOHA simulator.

Run this on a new machine, or whenever something looks broken. It tests the
setup as a chain of links. Each link depends on the one before it, so the
checks run in order and the first failure stops the script with a message
that says what to fix:

    1. Python env clean     -> no ROS 2 paths leaking into PYTHONPATH
    2. GPU visible          -> PyTorch can use CUDA (policy runs on the GPU)
    3. LeRobot imports      -> and print its git commit, for reproducibility
    4. ALOHA sim renders    -> env builds, steps, and returns a real (non-black) image

Usage:
    conda activate lerobot && unset PYTHONPATH
    MUJOCO_GL=egl python scripts/check_setup.py
    MUJOCO_GL=egl python scripts/check_setup.py --save-frame frame.png   # also save the camera image

MUJOCO_GL=egl makes MuJoCo draw images off-screen on the GPU (no window needed).
If EGL fails, try MUJOCO_GL=osmesa (CPU, slower, needs libosmesa6-dev).
"""

import argparse
import os
import subprocess
from pathlib import Path


def check_python_path():
    """Link 1: make sure ROS 2's Python 3.10 packages are not on PYTHONPATH.

    ROS 2 Humble adds its own folders to PYTHONPATH. Python searches those
    first, so it could load ROS's old (3.10) copy of a package instead of the
    one in our 3.12 conda env, which gives confusing import errors later.
    """
    pp = os.environ.get("PYTHONPATH", "")  # "" if the variable is not set
    assert "ros" not in pp.lower(), f"ROS paths in PYTHONPATH, run `unset PYTHONPATH`: {pp}"
    print("[ok] PYTHONPATH clean")


def check_gpu():
    """Link 2: make sure PyTorch can see the NVIDIA GPU through CUDA.

    Without it the policy network would run on the CPU, many times slower.
    """
    # Imported here (not at the top) so link 1 runs first: if the env is
    # broken, you see the clear PYTHONPATH message, not a torch import error.
    import torch

    assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU"
    print(f"[ok] torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")


def check_lerobot():
    """Link 3: make sure LeRobot imports, and print the exact version we use.

    LeRobot changes quickly, so we record its git commit hash in the README.
    Anyone reproducing our results can then check out the same code.
    """
    import lerobot

    # lerobot.__file__ = ~/lerobot/src/lerobot/__init__.py
    # parents[0] = .../src/lerobot, parents[1] = .../src, parents[2] = ~/lerobot (the git repo)
    repo = Path(lerobot.__file__).resolve().parents[2]

    # Same as running `git -C ~/lerobot rev-parse HEAD` in a shell.
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    # "unknown" if LeRobot was installed from pip rather than a git clone.
    print(f"[ok] lerobot {lerobot.__version__} at {repo} (commit {commit or 'unknown'})")


def check_aloha_render(save_frame=None):
    """Link 4: make sure the ALOHA simulator builds, steps and renders real images.

    Args:
        save_frame: optional file path; if given, the top-camera image is saved there.
    """
    import gymnasium as gym
    import gym_aloha  # noqa: F401  (looks unused, but importing it registers the gym_aloha/* env names)

    # Build the Transfer Cube world. "pixels_agent_pos" = camera images + 14 joint positions,
    # the same input format the ACT policy uses.
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")

    # Start an episode. seed=0 fixes the random cube position, so the run is repeatable.
    # reset() returns (observation, info); we don't need info.
    obs, _ = env.reset(seed=0)

    # obs = {"pixels": {"top": HxWx3 uint8 image}, "agent_pos": 14 numbers}
    img = obs["pixels"]["top"]

    # A broken renderer can return an image of the right shape that is all black.
    # Nothing would crash, but every policy would fail later for no visible reason.
    # Mean brightness is about 40 on a correct frame and about 0 on a black one.
    assert img.mean() > 5, "Rendered frame is (almost) black, rendering backend is broken"

    # Take one random action to prove the physics can advance in time.
    # step() returns (obs, reward, terminated, truncated, info); we keep only obs.
    obs, *_ = env.step(env.action_space.sample())

    print(
        f"[ok] ALOHA env: image {img.shape}, state {obs['agent_pos'].shape}, "
        f"action {env.action_space.shape}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}"
    )

    if save_frame:
        import imageio.v3 as iio

        iio.imwrite(save_frame, img)
        print(f"     saved frame to {save_frame}")

    env.close()  # free the simulator and its OpenGL context


if __name__ == "__main__":
    # Only runs when executed as a script, not when this file is imported.
    parser = argparse.ArgumentParser(description="Check the LeRobot + ALOHA sim setup.")
    parser.add_argument("--save-frame", help="optional path to save the rendered top-camera image")
    args = parser.parse_args()

    # Run the links in chain order; an assert failure stops here with its message.
    check_python_path()
    check_gpu()
    check_lerobot()
    check_aloha_render(args.save_frame)
    print("All checks passed.")
