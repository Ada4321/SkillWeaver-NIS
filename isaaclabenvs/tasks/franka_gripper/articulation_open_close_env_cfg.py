import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclabenvs.utils import quat_compat as qc

from .base_env_cfg import BaseEnvCfg


@configclass
class ArticulatedOpenCloseEnvCfg(BaseEnvCfg):
    # env
    observation_space: int = 88
    state_space: int = 0
    ground_height: float = -0.1
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=False,
    )

    # articulated object settings
    objects: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[],  # NOTE: to be filled when setting up the environment.
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
            ),
            semantic_tags=[("class", "object"), ("color", "red")],
            random_choice=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0.0, 0.0), rot=qc.rot_xyzw((1.0, 0.0, 0.0, 0.0))),
        actuators={},
    )
    object_joint_index: int = 0
    joint_start_pos_range_norm: tuple[float, float] = (0.0, 0.2)
    target_joint_pos_norm: float = 1.0
    joint_eps: float = 1e-6
    # Object reset pose in each env-local frame.
    # Position: x, y sampled uniformly from cfg-provided ranges (env-local). No table boundary
    #   safety check; caller is responsible for keeping ranges within the table footprint.
    # Yaw: world quat = q_z(pi + theta); theta=0 makes asset forward (canonical +x) face world -x,
    #   theta = +/-pi/2 swings forward to world -y / +y respectively. theta is auto-set so the
    #   handle face always points toward y=0: y>=0 -> theta=+pi/2 (face -y), y<0 ->
    #   theta=-pi/2 (face +y).
    object_init_x_range: tuple[float, float] = (0.2, 0.7)
    object_init_y_range: tuple[float, float] = (-0.5, 0.5)
    # Asset canonical "forward" direction angle (CCW from local +x in the local xy plane).
    # 0 for assets where forward = local +x (e.g. partm). For LIBERO assets where forward = local
    # -y (handle face), set to -math.pi/2.
    asset_forward_local_angle_rad: float = 0.0
    # Init arm qpos sampling: filter by xy-plane cosine between stored EEF axis (fingertip->wrist)
    # and current task forward direction. Samples with cos > threshold are kept; if none pass,
    # falls back to argmax (sample with highest cosine).
    init_arm_eef_forward_cos_threshold: float = 0.15
    # Table top height augmentation (env-local z, sampled per reset). Object sits on table top.
    table_z_range: tuple[float, float] = (-0.06, 0.1)

    # dataset
    dataset: str = ""
    # Optional articulation asset-id filter.
    # Supports comma/space separated ids or "<asset>_joint_<idx>" tokens.
    # `asset_ids_file` accepts one token per line.
    asset_ids: str = ""
    asset_ids_file: str = ""
    # Optional override for the init-arm qpos seed npz (CLI: env.init_arm_joint_pos_override=/abs/path.npz).
    # Empty = use the dataset's config.toml value. Used e.g. to swap in a margin-filtered seed grid.
    init_arm_joint_pos_override: str = ""
    # Optional override for the handle_grasps filename GLOB (CLI: env.handle_grasps_override='*_cano_hd_hook_init.npz').
    # Empty = use the dataset's config.toml value. NOTE: a per-asset-dir glob pattern, NOT an abs path
    # (unlike init_arm_joint_pos_override). Used e.g. to swap grasps -> hook_init poses without a new dataset entry.
    handle_grasps_override: str = ""

    def __post_init__(self):
        super().__post_init__()
        # Articulation open tasks: lock the camera to ONE env (env 0) and frame that single table
        # close-up, front view along -x (from +x) toward the robot/drawer. origin_type="env" makes
        # eye/lookat offsets relative to env 0's origin, so the view stays on one workspace instead
        # of the multi-env grid center (which makes every table tiny).
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (1.3, 0.0, 0.45)
        self.viewer.lookat = (0.5, 0.0, 0.18)
