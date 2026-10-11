import gymnasium as gym

from . import agents

##
# Register one Gym environment per SkillWeaver Neural Interaction Skill sub-policy.
# Each pairs a task env class with the sub-policy cfg in subpolicy_cfgs.py.
##

gym.register(
    id="SkillWeaver-Franka-Pick-General-v0",
    entry_point=f"{__name__}.pick_upright_env:PickUpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subpolicy_cfgs:SkillWeaverPickGeneralCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SkillWeaver-Franka-Pick-Upright-v0",
    entry_point=f"{__name__}.pick_upright_env:PickUpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subpolicy_cfgs:SkillWeaverPickUprightCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SkillWeaver-Franka-Pick-Concave-v0",
    entry_point=f"{__name__}.pick_upright_env:PickUpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subpolicy_cfgs:SkillWeaverPickConcaveCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SkillWeaver-Franka-Pick-ConcaveLying-v0",
    entry_point=f"{__name__}.pick_upright_env:PickUpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subpolicy_cfgs:SkillWeaverPickConcaveLyingCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SkillWeaver-Franka-Pick-Handle-v0",
    entry_point=f"{__name__}.pick_upright_env:PickUpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subpolicy_cfgs:SkillWeaverPickHandleCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SkillWeaver-Franka-Place-General-v0",
    entry_point=f"{__name__}.place_env:PlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subpolicy_cfgs:SkillWeaverPlaceGeneralCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SkillWeaver-Franka-Place-Upright-v0",
    entry_point=f"{__name__}.place_env:PlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subpolicy_cfgs:SkillWeaverPlaceUprightCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SkillWeaver-Franka-Open-Drawer-v0",
    entry_point=f"{__name__}.open_drawer_grasped_env:OpenDrawerGraspedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subpolicy_cfgs:SkillWeaverOpenDrawerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SkillWeaver-Franka-Open-Door-v0",
    entry_point=f"{__name__}.open_door_grasped_env:OpenDoorGraspedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subpolicy_cfgs:SkillWeaverOpenDoorCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SkillWeaver-Franka-Close-Drawer-v0",
    entry_point=f"{__name__}.close_drawer_push_env:CloseDrawerPushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subpolicy_cfgs:SkillWeaverCloseDrawerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SkillWeaver-Franka-Close-Door-v0",
    entry_point=f"{__name__}.close_door_push_env:CloseDoorPushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.subpolicy_cfgs:SkillWeaverCloseDoorCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
