"""SkillWeaver Neural Interaction Skill sub-policies for the WidowX-250 (one cfg per sub-policy).

Every task field that differs from the base cfg default is set here, and asset lists point at the
released example assets.
"""

from isaaclab.utils import configclass

from .pick_env_cfg import PickEnvCfg


@configclass
class SkillWeaverWidowXPickCfg(PickEnvCfg):
    """WidowX Pick. Reconstructed from the training script of run widowx_pick_general_8gpu."""

    def __post_init__(self):
        super().__post_init__()
        self.category_ratio = 1.0
        self.dataset = 'rigid'
        self.gripper_close_dist_thresh = 0.02
        self.lift_xy_success_ratio = 0.34
        self.object_x_range = (0.19, 0.45)
        self.object_y_range = (-0.22, 0.22)
        self.randomize_asset_use_usd_mass_prob = 0.15
        self.scenario_include = 'Kitchen_Table,libero'
        self.shuffle_asset_assignment = True
        self.shuffle_asset_assignment_seed = 0
        self.success_xy_tol = 0.1414
        self.success_xy_tol_min = 0.1414
        self.w_rew_dist_xy = 0.0
        self.wrist_rot_penalty_start_deg = 120.0
        self.valid_asset_file = 'data/asset_lists/widowx_pick.txt'
        self.invalid_asset_file = ''
        self.upright_asset_file = 'data/asset_lists/widowx_pick_upright.txt'
