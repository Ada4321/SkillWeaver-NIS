import gymnasium as gym

from . import agents

##
# Register one Gym environment per SkillWeaver Neural Interaction Skill sub-policy (WidowX-250).
# Each pairs a task env class with the sub-policy cfg in subpolicy_cfgs.py.
##

gym.register(
    id="SkillWeaver-WidowX-Pick-v0",
    entry_point=f"{__name__}.pick_env:PickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subpolicy_cfgs:SkillWeaverWidowXPickCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
