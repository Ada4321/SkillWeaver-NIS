"""SkillWeaver Neural Interaction Skill sub-policies (one cfg per paper sub-policy).

Generated from each paper run's params/env.yaml: every task field that differs from the
base cfg default is set here, and asset lists point at the released example assets.
"""

from isaaclab.utils import configclass

from .close_door_push_env_cfg import CloseDoorPushEnvCfg
from .close_drawer_push_env_cfg import CloseDrawerPushEnvCfg
from .open_door_grasped_env_cfg import OpenDoorGraspedEnvCfg
from .open_drawer_grasped_env_cfg import OpenDrawerGraspedEnvCfg
from .pick_upright_env_cfg import PickUpEnvCfg
from .place_env_cfg import PlaceEnvCfg


@configclass
class SkillWeaverPickGeneralCfg(PickUpEnvCfg):
    """Pick General. Reconstructed from run 2026-02-19_06-40-46_pick_sam3d_general_8gpu_tablez0.25."""

    def __post_init__(self):
        super().__post_init__()
        self.category_ratio = 1.0
        self.dataset = 'rigid'
        self.enable_xy_radius_relax = True
        self.fail_on_xy_displacement = True
        self.initial_pose_path = 'data/initial_pose/franka_gripper_table_closer_random_height_zstep0005_r02_zoff005_0205.npz'
        self.lift_upright_success_ratio = 0.34
        self.lift_xy_success_ratio = 0.34
        self.randomize_asset_use_usd_mass_prob = 0.15
        self.scenario_include = 'Kitchen_Table,libero'
        self.shuffle_asset_assignment = True
        self.shuffle_asset_assignment_seed = 0
        self.success_xy_tol = 0.1414
        self.success_xy_tol_min = 0.1414
        self.table_z_range = (-0.06, 0.25)
        self.upright_pose_mode = 'lying'
        self.w_rew_dist_xy = 0.0
        self.w_rew_obj_tilt_hinge = -0.0
        self.wrist_rot_penalty_start_deg = 30.0
        self.valid_asset_file = 'data/asset_lists/pick_general.txt'
        self.invalid_asset_file = ''
        self.upright_asset_file = 'data/asset_lists/pick_general_upright.txt'
        # Trained before arm stiffness was raised from 80 to 160 (2026-05-08).
        self.robot.actuators["panda_shoulder"].stiffness = 80.0
        self.robot.actuators["panda_forearm"].stiffness = 80.0


@configclass
class SkillWeaverPickUprightCfg(PickUpEnvCfg):
    """Pick Upright. Reconstructed from run 2026-02-19_19-30-54_pick_sam3d_upright_gt10_4gpu."""

    def __post_init__(self):
        super().__post_init__()
        self.category_ratio = 1.0
        self.dataset = 'rigid'
        self.enable_xy_radius_relax = True
        self.fail_on_xy_displacement = True
        self.gate_success_with_upright_ok = True
        self.initial_pose_path = 'data/initial_pose/franka_gripper_table_closer_random_height_zstep0005_r02_zoff005_0205.npz'
        self.lift_upright_success_ratio = 0.34
        self.lift_xy_success_ratio = 0.34
        self.obj_tilt_free_deg = 20.0
        self.object_x_range = (0.2, 0.65)
        self.object_y_range = (-0.35, 0.35)
        self.randomize_asset_use_usd_mass_prob = 0.15
        self.scenario_include = 'Kitchen_Table,libero'
        self.shuffle_asset_assignment = True
        self.shuffle_asset_assignment_seed = 0
        self.success_xy_tol = 0.1414
        self.success_xy_tol_min = 0.1414
        self.table_z_range = (-0.06, 0.25)
        self.upright_pose_mode = 'upright'
        self.use_bottom_ref_for_xy_disp = True
        self.w_rew_dist_xy = 0.0
        self.w_rew_obj_tilt_hinge = -0.0
        self.wrist_rot_margin = 88.0
        self.wrist_rot_penalty_start_deg = 88.0
        self.valid_asset_file = 'data/asset_lists/pick_upright.txt'
        self.invalid_asset_file = ''
        self.upright_asset_file = 'data/asset_lists/pick_upright_upright.txt'
        # Trained before arm stiffness was raised from 80 to 160 (2026-05-08).
        self.robot.actuators["panda_shoulder"].stiffness = 80.0
        self.robot.actuators["panda_forearm"].stiffness = 80.0


@configclass
class SkillWeaverPickConcaveCfg(PickUpEnvCfg):
    """Pick Concave. Reconstructed from run 2026-05-07_14-46-17_pick_sam3d_convex_8gpu_fail_30deg_small."""

    def __post_init__(self):
        super().__post_init__()
        self.category_include = 'bowl,cup,drinking_glass,measuring_cup,mug,small_teacup,red_coffee_mug,utensil_holder,white_mug,white_yellow_mug,black_bowl_1,ramekin'
        self.category_ratio = 1.0
        self.dataset = 'rigid'
        self.enable_xy_radius_relax = True
        self.fail_on_xy_displacement = True
        self.gate_success_with_upright_ok = True
        self.initial_pose_path = 'data/initial_pose/franka_gripper_table_closer_random_height_zstep0005_r02_zoff005_0205.npz'
        self.lift_height_use_lowest_point = True
        self.lift_upright_success_ratio = 0.34
        self.lift_xy_success_ratio = 0.34
        self.lift_yaw_success_ratio = 0.34
        self.obj_tilt_free_deg = 20.0
        self.randomize_asset_use_usd_mass_prob = 0.15
        self.scenario_include = 'Kitchen_Table,libero'
        self.shuffle_asset_assignment = True
        self.shuffle_asset_assignment_seed = 0
        self.success_xy_tol = 0.1414
        self.success_xy_tol_min = 0.1414
        self.upright_pose_mode = 'upright'
        self.use_bottom_ref_for_xy_disp = True
        self.w_rew_dist_xy = 0.0
        self.w_rew_obj_tilt_hinge = -0.0
        self.wrist_rot_penalty_start_deg = 30.0
        self.valid_asset_file = 'data/asset_lists/pick_concave.txt'
        self.invalid_asset_file = ''
        self.upright_asset_file = 'data/asset_lists/pick_concave_upright.txt'


@configclass
class SkillWeaverPickConcaveLyingCfg(PickUpEnvCfg):
    """Pick Concave Lying. Reconstructed from run 2026-05-07_20-23-42_pick_sam3d_convex_lying_top_8gpu_small."""

    def __post_init__(self):
        super().__post_init__()
        self.category_include = 'cup,drinking_glass,mug,red_coffee_mug,utensil_holder,white_mug,white_yellow_mug'
        self.category_ratio = 1.0
        self.dataset = 'rigid'
        self.enable_xy_radius_relax = True
        self.fail_on_xy_displacement = True
        self.initial_pose_path = 'data/initial_pose/franka_gripper_table_closer_random_height_zstep0005_r02_zoff005_0205.npz'
        self.lift_height_use_lowest_point = True
        self.lift_xy_success_ratio = 0.34
        self.randomize_asset_use_usd_mass_prob = 0.15
        self.scenario_include = 'Kitchen_Table,libero'
        self.shuffle_asset_assignment = True
        self.shuffle_asset_assignment_seed = 0
        self.success_xy_tol = 0.1414
        self.success_xy_tol_min = 0.1414
        self.upright_pose_mode = 'lying'
        self.use_top_point_cloud_for_obs_reward = True
        self.w_rew_dist_xy = 0.0
        self.w_rew_obj_tilt_hinge = -0.0
        self.wrist_rot_margin = 92.0
        self.wrist_rot_penalty_start_deg = 92.0
        self.valid_asset_file = 'data/asset_lists/pick_concave_lying.txt'
        self.invalid_asset_file = ''
        self.upright_asset_file = 'data/asset_lists/pick_concave_lying_upright.txt'


@configclass
class SkillWeaverPickHandleCfg(PickUpEnvCfg):
    """Pick Handle. Reconstructed from run 2026-05-09_00-50-35_pick_sam3d_handle_moka_r25."""

    def __post_init__(self):
        super().__post_init__()
        self.category_include = 'left_moka_pot,right_moka_pot'
        self.category_ratio = 1.0
        self.dataset = 'rigid'
        self.enable_xy_radius_relax = True
        self.fail_on_xy_displacement = True
        self.gate_success_with_upright_ok = True
        self.initial_pose_path = 'data/initial_pose/franka_gripper_table_closer_random_height_zstep0005_r02_zoff005_0205.npz'
        self.lift_height_use_lowest_point = True
        self.lift_upright_success_ratio = 0.34
        self.lift_xy_success_ratio = 0.34
        self.obj_tilt_free_deg = 30.0
        self.object_x_range = (0.2, 0.65)
        self.object_y_range = (-0.35, 0.35)
        self.randomize_asset_use_usd_mass_prob = 0.15
        self.scenario_include = 'Kitchen_Table,libero'
        self.shuffle_asset_assignment = True
        self.shuffle_asset_assignment_seed = 0
        self.success_xy_tol = 0.1414
        self.success_xy_tol_min = 0.1414
        self.table_z_range = (-0.06, 0.25)
        self.upright_pose_mode = 'upright'
        self.use_bottom_ref_for_xy_disp = True
        self.use_handle_side_point_cloud_for_obs_reward = True
        self.w_rew_dist_xy = 0.0
        self.w_rew_obj_tilt_hinge = -0.0
        self.wrist_rot_penalty_start_deg = 30.0
        self.valid_asset_file = 'data/asset_lists/pick_handle.txt'
        self.invalid_asset_file = ''
        self.upright_asset_file = 'data/asset_lists/pick_handle_upright.txt'


@configclass
class SkillWeaverPlaceGeneralCfg(PlaceEnvCfg):
    """Place General. Reconstructed from run 2026-04-30_14-51-44_place_sam3d_th200_lying_gengrasp_top5_trans_uptilt12p5_8gpu."""

    def __post_init__(self):
        super().__post_init__()
        self.category_ratio = 1.0
        self.goal_pose_orientation_topk = 5
        self.goal_pose_orientation_topk_enabled = True
        self.grasp_sampling_mode = 'general'
        self.initial_pose_path = 'data/initial_pose/franka_gripper_table_closer_random_height_zstep0005_r02_zoff005_0205.npz'
        self.post_reset_forced_steps = 2
        self.reach_once_pos_only = True
        self.require_orientation = False
        self.require_tilt_only = True
        self.scenario_include = 'Kitchen_Table,libero'
        self.shuffle_asset_assignment = True
        self.shuffle_asset_assignment_seed = 0
        self.threshold_decay = True
        self.tilt_only_for_upright_only = True
        self.tilt_rot_tol_deg = 12.5
        self.valid_asset_file = 'data/asset_lists/place_general.txt'
        self.invalid_asset_file = ''
        self.goal_pose_upright_asset_file = 'data/asset_lists/place_general_goal_pose_upright.txt'
        self.goal_pose_lying_asset_file = 'data/asset_lists/place_general_goal_pose_lying.txt'


@configclass
class SkillWeaverPlaceUprightCfg(PlaceEnvCfg):
    """Place Upright. Reconstructed from run 2026-04-04_22-11-03_place_sam3d_th200_upright_initarm_top5_trans_flip_tiltonly12p5_8gpu."""

    def __post_init__(self):
        super().__post_init__()
        self.category_ratio = 1.0
        self.flip_wrist_if_object_upside_down = True
        self.goal_pose_mode = 'upright'
        self.goal_pose_orientation_topk = 5
        self.goal_pose_orientation_topk_enabled = True
        self.goal_x_range = (0.2, 0.75)
        self.goal_y_range = (-0.355, 0.355)
        self.init_arm_upright = True
        self.init_arm_upright_z_jitter = 0.05
        self.init_arm_upright_z_offset = 0.2
        self.initial_pose_path = 'data/initial_pose/franka_gripper_table_closer_upright_zstep0005_r012_zoff000_th30_0326.npz'
        self.post_reset_forced_steps = 5
        self.reach_once_pos_only = True
        self.require_orientation = False
        self.require_tilt_only = True
        self.scenario_include = 'Kitchen_Table,libero'
        self.shuffle_asset_assignment = True
        self.shuffle_asset_assignment_seed = 0
        self.threshold_decay = True
        self.tilt_rot_tol_deg = 12.5
        self.valid_asset_file = 'data/asset_lists/place_upright.txt'
        self.invalid_asset_file = ''
        self.goal_pose_upright_asset_file = 'data/asset_lists/place_upright_goal_pose_upright.txt'


@configclass
class SkillWeaverOpenDrawerCfg(OpenDrawerGraspedEnvCfg):
    """Open Drawer. Reconstructed from run 2026-06-19_00-06-32_open_drawer_grasped."""

    def __post_init__(self):
        super().__post_init__()
        self.grasp_lost_fail_steps = 10
        self.init_arm_joint_pos_override = 'data/initial_pose/franka_gripper_arti_filtered.npz'
        self.post_reset_forced_steps = 10
        self.rew_open_require_pinch = True
        self.success_require_direction = False
        self.w_rew_opened_hold = 50.0
        self.asset_ids_file = 'data/asset_lists/open_drawer.txt'
        self.dataset = 'arti_prismatic'


@configclass
class SkillWeaverOpenDoorCfg(OpenDoorGraspedEnvCfg):
    """Open Door. Reconstructed from run 2026-06-28_00-15-47_open_door_grasped_microwave."""

    def __post_init__(self):
        super().__post_init__()
        self.init_arm_joint_pos_override = 'data/initial_pose/franka_gripper_arti_filtered.npz'
        self.object_init_x_range = (0.35, 0.55)
        self.object_init_y_range = (-0.2, 0.2)
        self.post_reset_forced_steps = 10
        self.start_curriculum = True
        self.success_require_direction = False
        self.w_rew_opened_hold = 50.0
        self.asset_ids_file = 'data/asset_lists/open_door.txt'
        self.dataset = 'arti_revolute'


@configclass
class SkillWeaverCloseDrawerCfg(CloseDrawerPushEnvCfg):
    """Close Drawer. Reconstructed from run 2026-06-21_18-44-34_close_drawer_push_require_touch_libero."""

    def __post_init__(self):
        super().__post_init__()
        self.close_gate_require_orient = True
        self.detach_fail_steps = 15
        self.handle_grasps_override = '*_cano_hd_closepush_init.npz'
        self.init_arm_joint_pos_override = 'data/initial_pose/franka_gripper_arti_filtered.npz'
        self.num_consecutive_success = 1
        self.regress_fail_norm = 0.3
        self.rew_close_require_touch = True
        self.w_rew_closed_hold = 50.0
        self.asset_ids_file = 'data/asset_lists/close_drawer.txt'
        self.dataset = 'arti_prismatic'


@configclass
class SkillWeaverCloseDoorCfg(CloseDoorPushEnvCfg):
    """Close Door. Reconstructed from run 2026-06-25_02-19-52_close_door_push_require_touch_microwave."""

    def __post_init__(self):
        super().__post_init__()
        self.close_gate_require_orient = True
        self.detach_fail_steps = 15
        self.handle_grasps_override = '*_cano_hd_closepush_init.npz'
        self.init_arm_joint_pos_override = 'data/initial_pose/franka_gripper_arti_filtered.npz'
        self.num_consecutive_success = 1
        self.regress_fail_norm = 0.3
        self.rew_close_require_touch = True
        self.w_rew_closed_hold = 50.0
        self.asset_ids_file = 'data/asset_lists/close_door.txt'
        self.dataset = 'arti_revolute'
