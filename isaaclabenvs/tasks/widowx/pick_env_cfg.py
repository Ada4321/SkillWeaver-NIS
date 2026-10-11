from isaaclab.utils import configclass

from .base_env_cfg import BaseEnvCfg


@configclass
class PickEnvCfg(BaseEnvCfg):
    # env
    observation_space: int = 77
    state_space: int = 0

    # reward settings
    num_consecutive_success = 15
    lift_height: float = 0.10
    lift_up_thres_max: float = 0.35


    # dataset
    dataset: str = "rigid"
    dataset_split: str = "train"                                    
    # table height sampling (env-local z)
    # table_z_range: tuple[float, float] = (0.05, 0.25)
    # table_z_range: tuple[float, float] = (0.0, 0.01)
    # table_z_range: tuple[float, float] = (-0.06, 0.34)
    table_z_range: tuple[float, float] = (0.0, 0.15)

    # object xy sampling range (env-local)
    object_x_range: tuple[float, float] = (0.15, 0.55)
    object_y_range: tuple[float, float] = (-0.25, 0.25)

    # upright constraints (optional)
    upright_asset_file: str = "data/meta/asset_list_for_policy_upright_gt0.1.txt"
    upright_init_tilt_deg: float = 20.0
    upright_keep_tilt_deg: float = 30.0
    # Only consider points below this fraction of object height when upright is active (1.0 = no cut).
    upright_pcd_height_frac: float = 1.0
    # Contact reward only if side_score >= threshold when upright is active (0.0 disables).
    upright_contact_side_score_thres: float = 0.7

    # side-approach reward (optional, only for upright-active episodes)
    enable_side_approach_reward: bool = False
    side_approach_height_thres: float = 0.05  # height above table top
    w_rew_side_approach: float = 0.5

    # Reward Weights
    # gripper contact rewards
    w_rew_object_ee_dist: float = 50.0
    w_rew_contact: float = 1.0
    w_rew_dist: float = -0.5
    w_rew_open: float = -0.2
    # Dense reward: close gripper when fingers are near the object surface.
    # Gives the policy a direct gradient toward closure before contact is made.
    w_rew_gripper_close: float = 1.0
    gripper_close_dist_thresh: float = 0.04  # activate within this distance of object surface

    # gripper orientation rewards
    w_rew_wrist_rot_before_grasp: float = -0.5
    w_rew_wrist_rot_after_grasp: float = -2.0
    wrist_rot_penalty_start_deg: float = 0.0

    # smoothness and stability rewards
    w_rew_wrist_smooth: float = -0.5
    w_rew_height_stable: float = -20.0
    w_rew_vel_stable: float = -1.0
    w_rew_ee_stable: float = 0.0

    # lift
    w_rew_lift: float = 500.0
    w_rew_lifted_1: float = 100.0
    w_rew_lifted_2: float = 5.0
    w_rew_lifted_3: float = 400.0
    w_rew_lift_xy: float = 500.0
    w_rew_lift_xy_mult: float = 1.0

    # object xy offset
    w_rew_dist_xy: float = 0.1

    w_rew_success: float = 1500.0
    w_rew_success_pose_mult: float = 2.0
    w_rew_success_pose_alpha: float = 5.0
    w_rew_success_pose_max_scale: float = 100.0
    w_rew_lift_xy_alpha: float = 5.0
    w_rew_lift_xy_max_scale: float = 50.0
    pose_hold_min_denom: int = 3
    pose_hold_require_full_k: bool = False
    success_reward_only_on_success: bool = False
    lift_reward_cutoff_step: int = 4000
    w_rew_action: float = -0.001

    ## Margins
    wrist_rot_margin: float = 90.0
    threshold_decay: bool = False
    use_success_pose_constraints: bool = False
    success_xy_tol: float = 0.1
    success_xy_tol_min: float = 0.03
    lift_xy_success_ratio: float = 0.5
    warmup_steps: int = 20000
