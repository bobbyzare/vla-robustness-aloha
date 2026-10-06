"""Watch a trained policy control the ALOHA robot live in the MuJoCo 3D viewer.

For looking and understanding only: nothing is saved and no numbers are
reported. The official success rate comes from lerobot-eval (eval/phase1_baseline.sh).

The loop below does the same steps as lerobot-eval's rollout
(~/lerobot/src/lerobot/scripts/lerobot_eval.py, function `rollout`):

    observation -> preprocessor -> policy -> postprocessor -> env.step(action)

plus a viewer window that shows the simulation while our code drives it.

Usage:
    conda activate lerobot && unset PYTHONPATH
    python scripts/watch_policy.py                       # 3 episodes, seeds 1000, 1001, 1002
    python scripts/watch_policy.py --episodes 5 --seed 1010
    python scripts/watch_policy.py --speed 0.5           # half speed (slow motion)

The window stays open after the last episode until you close it.
Close it (or press Ctrl+C) at any time to stop early.
"""

import argparse
import os
import time

# Off-screen GPU rendering for the policy's camera image (as in Phase 0).
# Must be set before MuJoCo/dm_control are imported. The viewer window is
# separate and always uses a normal on-screen window.
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import gym_aloha  # noqa: F401  (registers the gym_aloha/* environment names)
import mujoco
import mujoco.viewer
import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

DEFAULT_POLICY = "outputs/models/act_aloha_sim_transfer_cube_human_migrated"
STEP_TIME = 0.02  # seconds of simulated time per env.step (gym-aloha runs at 50 Hz)
MAX_STEPS = 400  # same episode length as lerobot-eval uses for ALOHA
REWARD_NAMES = {0: "nothing", 1: "touched", 2: "lifted", 3: "handover attempt", 4: "SUCCESS"}


def load_policy(policy_dir, device):
    """Load the ACT network and its pre/post-processors (normalize / un-normalize)."""
    policy = ACTPolicy.from_pretrained(policy_dir)
    policy.to(device)
    policy.eval()  # inference mode: no dropout
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_dir,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def run_episode(env, viewer, policy, preprocessor, postprocessor, seed, speed):
    """Run one episode with the viewer open. Returns the highest reward reached (0-4)."""
    obs, _ = env.reset(seed=seed)  # same seed -> same cube position as in lerobot-eval
    policy.reset()  # empty ACT's queue of planned actions from the previous episode
    best_reward = 0

    for step in range(MAX_STEPS):
        if not viewer.is_running():  # user closed the window
            break
        t_start = time.time()

        # Observation -> policy input. preprocess_observation renames keys to
        # "observation.images.top" / "observation.state", converts the image to a
        # float tensor in [0, 1] with channels first, and adds a batch dimension.
        batch = preprocess_observation(obs)
        batch = preprocessor(batch)  # move to GPU, normalize with the training mean/std

        with torch.inference_mode():
            action = policy.select_action(batch)  # (1, 14), still normalized
        action = postprocessor(action)  # un-normalize -> real joint targets, back to CPU
        action = action.squeeze(0).numpy()  # (14,) numpy array for the env

        obs, reward, terminated, truncated, _ = env.step(action)
        viewer.sync()  # copy the new simulation state to the window

        if reward > best_reward:
            best_reward = reward
            print(f"  step {step:3d}: reward {reward} ({REWARD_NAMES[reward]})")
        if terminated:  # gym-aloha ends the episode at reward 4
            break

        # Wait so the episode plays at real speed (speed=1) instead of as fast as possible.
        time.sleep(max(0.0, STEP_TIME / speed - (time.time() - t_start)))

    return best_reward


def main():
    parser = argparse.ArgumentParser(description="Watch a policy control ALOHA in the MuJoCo viewer.")
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="folder with the (migrated) policy")
    parser.add_argument("--task", default="AlohaTransferCube-v0", help="gym-aloha task name")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000, help="seed of the first episode (+1 per episode)")
    parser.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0.5 = slow motion")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading policy from {args.policy} on {device}")
    policy, preprocessor, postprocessor = load_policy(args.policy, device)

    env = gym.make(f"gym_aloha/{args.task}", obs_type="pixels_agent_pos", max_episode_steps=MAX_STEPS)

    # gym-aloha wraps a dm_control environment; its physics holds the raw MuJoCo
    # model (the scene) and data (the current state). The viewer shows these, so
    # every env.step() becomes visible after viewer.sync().
    env.reset(seed=args.seed)
    physics = env.unwrapped._env.physics
    model, data = physics.model.ptr, physics.data.ptr

    # "Passive" viewer: the window only displays; our loop drives the simulation.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(args.episodes):
            seed = args.seed + i
            print(f"Episode {i} (seed {seed})")
            best = run_episode(env, viewer, policy, preprocessor, postprocessor, seed, args.speed)
            if not viewer.is_running():
                print("Viewer closed, stopping.")
                break
            print(f"  -> best reward {best} ({REWARD_NAMES[best]})")
            time.sleep(1.0)  # short pause between episodes

        # Keep the window open on the last frame until the user closes it.
        if viewer.is_running():
            print("Done. Close the viewer window to exit.")
        while viewer.is_running():
            time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    main()
