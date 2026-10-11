"""Evaluate an RL-Games checkpoint: success rate over one episode per environment.

Runs every environment for one episode with deterministic actions, then prints the success,
failure and timeout rates overall and per asset. Some tasks anneal their success tolerance or
their reset distribution with the training step count; by default the evaluation uses the step
count the checkpoint was trained to (its epoch times the PPO horizon length).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--num_envs", type=int, default=400, help="Number of environments (one episode each).")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment.")
parser.add_argument(
    "--step_counter",
    type=int,
    default=None,
    help="Env step count to evaluate at. Defaults to the checkpoint's training step count.",
)
parser.add_argument("--max_steps", type=int, default=3000, help="Stop after this many env steps.")
parser.add_argument("--output", type=str, default=None, help="Optional path of a JSON file to write the results to.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os

import gymnasium as gym
import torch
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import isaaclabenvs.tasks  # noqa: F401

# TaskStatus values shared by all SkillWeaver tasks.
SUCCESS, FAILURE, TIMEOUT = 1, 2, 3


def asset_name(env, i: int) -> str:
    """Last two path components of the asset in env i, e.g. 'mug/asset_0016' or 'prismatic/90011_joint_1'."""
    return "/".join(env.dataset.path[int(env.data_object_id[i])].rstrip("/").split("/")[-2:])


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    """Evaluate with RL-Games agent."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.print_interval = 0
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    resume_path = retrieve_file_path(args_cli.checkpoint)
    step_counter = args_cli.step_counter
    if step_counter is None:
        epoch = int(torch.load(resume_path, map_location="cpu", weights_only=False).get("epoch", 0))
        step_counter = epoch * int(agent_cfg["params"]["config"]["horizon_length"])

    env = gym.make(args_cli.task, cfg=env_cfg)
    rl_device = agent_cfg["params"]["config"]["device"]
    env = RlGamesVecEnvWrapper(
        env,
        rl_device,
        agent_cfg["params"]["env"].get("clip_observations", math.inf),
        agent_cfg["params"]["env"].get("clip_actions", math.inf),
        agent_cfg["params"]["env"].get("obs_groups"),
        agent_cfg["params"]["env"].get("concate_obs_groups", True),
    )
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    uenv = env.unwrapped
    uenv.common_step_counter = step_counter
    n = uenv.num_envs
    outcome = torch.zeros(n, dtype=torch.int, device=uenv.device)  # 0 = still running
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    steps = 0
    while steps < args_cli.max_steps and bool((outcome == 0).any()) and simulation_app.is_running():
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
        steps += 1
        just_done = torch.nonzero(dones.bool() & (outcome == 0)).flatten()
        if len(just_done):
            outcome[just_done] = uenv.latest_status[just_done].int()

    finished = outcome > 0
    num = max(int(finished.sum()), 1)
    per_asset = {}
    for i in range(n):
        if bool(finished[i]):
            s, c = per_asset.get(asset_name(uenv, i), (0, 0))
            per_asset[asset_name(uenv, i)] = (s + int(outcome[i] == SUCCESS), c + 1)
    result = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "step_counter": step_counter,
        "episodes": int(finished.sum()),
        "success_rate": float((outcome == SUCCESS).sum()) / num,
        "failure_rate": float((outcome == FAILURE).sum()) / num,
        "timeout_rate": float((outcome == TIMEOUT).sum()) / num,
        "per_asset": {k: {"success": s, "episodes": c} for k, (s, c) in sorted(per_asset.items())},
    }

    print(f"\n{args_cli.task}  ({result['episodes']} episodes, step counter {step_counter})")
    print(f"  success {result['success_rate']:.3f}   failure {result['failure_rate']:.3f}   "
          f"timeout {result['timeout_rate']:.3f}")
    width = max(len(k) for k in per_asset) if per_asset else 0
    for k, (s, c) in sorted(per_asset.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {k:<{width}}  {s:>3}/{c:<3} {s / c:.2f}")
    if args_cli.output:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
        with open(args_cli.output, "w") as f:
            json.dump(result, f, indent=1)
        print(f"  wrote {args_cli.output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
