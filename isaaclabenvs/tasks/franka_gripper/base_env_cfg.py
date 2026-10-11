
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclabenvs.config import ASSET_ROOT
from isaaclabenvs.utils import quat_compat as qc

# Mean values from IsaacLabEnvs EventCfg (friction 0.6-1.2, mass scale 0.4-2.0).
EVENT_MEAN_STATIC_FRICTION = 1.0
EVENT_MEAN_DYNAMIC_FRICTION = 1.0
EVENT_MEAN_RESTITUTION = 0.0
EVENT_MEAN_MASS_SCALE = 1.0


@configclass
class EventCfg:
    # Robot Surface
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    # Object Surface
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    # Object Mass
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.4, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class BaseEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s: float = 10.0
    decimation: int = 25  # Number of simulation steps per control command
    action_space: int = 7
    initial_pose_path: str = "data/initial_pose/franka_gripper.npz"
    arm_cmd_limit: tuple[float, float, float, float, float, float] = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
    action_ema_alpha: float = 0.92

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.002,  # simulation frequency
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="min",
            static_friction=EVENT_MEAN_STATIC_FRICTION,
            dynamic_friction=EVENT_MEAN_DYNAMIC_FRICTION,
            restitution=EVENT_MEAN_RESTITUTION,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=False,
    )
    ground_height: float = -0.1

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str((ASSET_ROOT / "franka_gripper/usd/franka_gripper.usd").resolve()),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.3463,
                "panda_joint2": -0.0387,
                "panda_joint3": -0.3453,
                "panda_joint4": -2.3377,
                "panda_joint5": -0.0176,
                "panda_joint6": 2.3012,
                "panda_joint7": 0.7983,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=qc.rot_xyzw((1.0, 0.0, 0.0, 0.0)),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=160.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=160.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # other scales
    dof_velocity_scale: float = 0.1
    use_usd_mass: bool = True
    object_density: float = 500.0 * EVENT_MEAN_MASS_SCALE
    # Optional per-asset random mass mode:
    # for each asset, sample whether to keep USD mass; otherwise sample explicit mass in kg.
    randomize_asset_mass: bool = False
    randomize_asset_use_usd_mass_prob: float = 0.5
    randomize_asset_mass_min: float = 0.05
    randomize_asset_mass_max: float = 0.25
    randomize_asset_mass_seed: int | None = None

    # objects
    objects: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[],  # NOTE: to be filled when setting up the environment.
            mass_props=sim_utils.MassPropertiesCfg(density=object_density),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=qc.rot_xyzw((1.0, 0.0, 0.0, 0.0))),
    )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 2.0, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.06), rot=qc.rot_xyzw((1.0, 0.0, 0.0, 0.0))),
    )

    # contact forces
    contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_(left|right)finger(_tip|)",
        # filter_prim_paths_expr=["/World/envs/env_.*/Object_.*"],
        update_period=0.0,
        debug_vis=False,
    )

    # OSC controller
    controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    )

    # Domain Randomization
    events: EventCfg = EventCfg()

    # Print statistics / reward terms to the console every N env steps (0 disables).
    print_interval: int = 100

    # Dataset
    dataset: str = "rigid"
    # Optional per-run overrides for dataset-level whitelist/blacklist files.
    # If empty, values from config.toml are used.
    valid_asset_file: str = ""
    invalid_asset_file: str = ""
    indices: str = ""
    # Optional category whitelist for the rigid dataset.
    # Supports either comma/space separated categories (e.g. "apple,bowl")
    # or a txt file path with one category per line.
    category_include: str = ""
    # Optional scenario whitelist for the rigid dataset.
    # Supports either comma/space separated scenarios (e.g. "Kitchen_Table,libero")
    # or a txt file path with one scenario per line.
    scenario_include: str = ""
    category_ratio: float = 0.0
    # If None, falls back to env seed (and then 0 if env seed is also None).
    category_ratio_seed: int | None = None
    category_ratio_random: bool = False
    dataset_num_pose: int = 600
    # If enabled, assign assets to envs using a seed-shuffled order instead of fixed round-robin.
    shuffle_asset_assignment: bool = False
    # Optional seed override for shuffle_asset_assignment. Use -1 to fall back to env seed (then 0).
    shuffle_asset_assignment_seed: int = -1

    def __post_init__(self):
        # There's no need to call super().__post_init__; IsaacLab handles that.
        # Frame env 0's workspace (robot base at the env origin, table in front along +x).
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (1.35, 0.85, 0.95)
        self.viewer.lookat = (0.45, 0.0, 0.15)
        # Disable physics randomization to prefer USD mass/friction.
        self.events.robot_physics_material = None
        self.events.object_physics_material = None
        self.events.object_scale_mass = None
