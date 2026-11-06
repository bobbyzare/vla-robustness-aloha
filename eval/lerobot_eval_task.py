"""Run lerobot-eval, but give the policy the same task instruction it was trained with.

Why this exists:
    During evaluation, lerobot-eval asks the simulator for a task description and passes
    it to the policy as the language instruction. gym-aloha has no `task_description`, so
    LeRobot falls back to the short task name, e.g. "insertion". The training datasets use
    a full sentence, e.g. "Insert the peg into the socket.". A language-conditioned policy
    such as SmolVLA would therefore see a different instruction at test time than in
    training. ACT ignores the instruction, so for ACT this changes nothing.

What it does:
    Adds a `task_description` to gym-aloha's environment class, with the exact sentence
    from the matching LeRobot dataset, and then starts lerobot-eval with all command-line
    arguments unchanged. LeRobot itself is not modified; the change only exists inside
    this Python process. With --eval.batch_size=1 the environment runs in this process,
    so lerobot-eval sees the added attribute.

Usage (same arguments as lerobot-eval; normally called through eval/evaluate.sh):
    python eval/lerobot_eval_task.py --policy.path=... --env.type=aloha --env.task=AlohaInsertion-v0 ...
"""

import sys

import gym_aloha.env

from lerobot.scripts.lerobot_eval import main

# Instruction per gym-aloha task name, copied from meta/tasks.parquet of the datasets
# lerobot/aloha_sim_insertion_human and lerobot/aloha_sim_transfer_cube_human.
TASK_INSTRUCTIONS = {
    "insertion": "Insert the peg into the socket.",
    "transfer_cube": "Pick up the cube with the right arm and transfer it to the left arm.",
}


def _task_description(self) -> str:
    # `self.task` is gym-aloha's short task name ("insertion" or "transfer_cube").
    return TASK_INSTRUCTIONS[self.task]


if __name__ == "__main__":
    gym_aloha.env.AlohaEnv.task_description = property(_task_description)
    sys.argv[0] = "lerobot-eval"
    main()
