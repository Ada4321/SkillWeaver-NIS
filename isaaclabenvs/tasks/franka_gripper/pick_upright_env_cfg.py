from isaaclab.utils import configclass

from .base_env_cfg import BaseEnvCfg


@configclass
class PickUpEnvCfg(BaseEnvCfg):
    # env
    observation_space: int = 80
    state_space: int = 0

    # reward settings
    num_consecutive_success = 15
    lift_height: float = 0.15
    lift_up_thres_max: float = 0.55
    # If True, compute lift using object's world-frame lowest point z instead of root-center z.
    lift_height_use_lowest_point: bool = False


    # dataset
    dataset: str = "rigid"
    dataset_split: str = "train"                                    
    # If True, use top_point_cloud for obs_dist/obs_closest_point (and thus rew_contact/rew_dist).
    # Lift and bottom_ref still use full point_cloud.
    use_top_point_cloud_for_obs_reward: bool = False
    # If True, use handle-side point cloud for obs_dist/obs_closest_point by reading:
    # - dataset config key `asset_json_path` to locate each asset json,
    # - json key `handle_side` (+x/-x/+y/-y/+z/-z),
    # - dataset config key `handle_point_cloud` to select the matching side pcd.
    # Mapping follows sample_collision_pcd.py: +x/-x/+y/-y/+z/-z -> front/back/right/left/top/bottom.
    # This mode is strict: no fallback path or side inference.
    use_handle_side_point_cloud_for_obs_reward: bool = False
    # table height sampling (env-local z)
    # table_z_range: tuple[float, float] = (0.05, 0.25)
    # table_z_range: tuple[float, float] = (0.0, 0.01)
    table_z_range: tuple[float, float] = (-0.06, 0.34)

    # object xy sampling range (env-local)
    object_x_range: tuple[float, float] = (0.2, 0.755)
    object_y_range: tuple[float, float] = (-0.355, 0.355)

    # upright constraints (optional)
    upright_asset_file: str = ""
    # Pose sampling mode for assets matched by upright_asset_file:
    # "upright" -> pose indices [0, 200), "lying" -> [200, end), "mixed" -> [0, end).
    upright_pose_mode: str = "mixed"
    upright_init_tilt_deg: float = 20.0
    upright_keep_tilt_deg: float = 30.0

    # object yaw (heading around world +z) max deviation from spawn pose
    obj_yaw_keep_deg: float = 30.0

    # object canonical z-axis tilt penalty (hinge)
    obj_tilt_free_deg: float = 10.0
    w_rew_obj_tilt_hinge: float = -1.0

    # Reward Weights
    # gripper contact rewards
    w_rew_object_ee_dist: float = 50.0
    w_rew_contact: float = 1.0
    w_rew_dist: float = -0.5
    w_rew_open: float = -0.2

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
    gate_success_with_upright_ok: bool = False
    gate_lift_reward_with_yaw_ok: bool = False
    gate_lifted_reward_with_yaw_ok: bool = False

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
    gate_success_with_xy_ratio: bool = True
    gate_lifted_pose_with_wrist_rot: bool = True
    gate_lift_rewards_with_xy: bool = True
    gate_success_with_yaw_ok: bool = False

    ## Margins
    wrist_rot_margin: float = 30.0
    threshold_decay: bool = False
    use_success_pose_constraints: bool = False
    success_xy_tol: float = 0.1
    success_xy_tol_min: float = 0.03
    lift_xy_success_ratio: float = 0.5
    lift_upright_success_ratio: float = 1.0
    lift_yaw_success_ratio: float = 1.0
    fail_on_xy_displacement: bool = False
    fail_xy_displacement_thresh: float = 0.15
    use_bottom_ref_for_xy_disp: bool = False
    xy_bottom_ref_z_band: float = 0.01
    enable_xy_radius_relax: bool = False
    xy_radius_relax_start: float = 0.55
    xy_radius_relax_slope: float = 0.2
    xy_radius_relax_max_bonus: float = 0.06
    xy_radius_relax_success_scale: float = 1.0
    xy_radius_relax_failure_scale: float = 1.0
    warmup_steps: int = 20000
