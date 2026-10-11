from __future__ import annotations

import copy
import os
from typing import Callable

import isaaclab.sim as sim_utils
from isaaclabenvs.utils import quat_compat as qc
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclabenvs.config.config import PROJECT_ROOT
from isaaclabenvs.utils.dataset import AssetLoader, _resolve_asset_file
from isaaclabenvs.utils.parse import parse_indices

from .base_env_cfg import BaseEnvCfg


class BaseEnv(DirectRLEnv):
    cfg: BaseEnvCfg

    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        # set up distributed loading
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.dataset_indices = parse_indices(cfg.indices) if cfg.indices else None

        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # robot dof limits and targets
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # joint indices
        self.arm_joint_indices, _ = self.robot.find_joints("(waist|shoulder|elbow|forearm_roll|wrist_angle|wrist_rotate)")
        self.left_joint_idx = self.robot.find_joints("left_finger")[0][0]
        self.right_joint_idx = self.robot.find_joints("right_finger")[0][0]

        # link indices
        self.hand_link_idx = self.robot.find_bodies("gripper_link")[0][0]
        self.left_finger_idx = self.robot.find_bodies("left_finger_link")[0][0]
        self.right_finger_idx = self.robot.find_bodies("right_finger_link")[0][0]
        self.left_tip_idx = self.robot.find_bodies("left_finger_tip")[0][0]
        self.right_tip_idx = self.robot.find_bodies("right_finger_tip")[0][0]
        self.grasp_site_idx = self.robot.find_bodies("grip_site")[0][0]

        # NOTE: we need to re-order this
        self.keypoint_indices = [
            *[self.robot.find_bodies(name)[0][0] for name in self.contact.body_names],
            self.grasp_site_idx,
            self.hand_link_idx,
        ]

        # IK controller scale
        if len(self.cfg.arm_cmd_limit) != 6:
            raise ValueError(f"arm_cmd_limit must have 6 elements, got {len(self.cfg.arm_cmd_limit)}")
        self.cmd_limit = torch.tensor(self.cfg.arm_cmd_limit, dtype=torch.float, device=self.device)

        # unit
        self.unit_z = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device)
        self.unit_x = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device)

        # Action smoothing (EMA)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

    def _should_print(self) -> bool:
        """Console logging of statistics / reward terms every `print_interval` env steps (0 disables)."""
        interval = int(self.cfg.print_interval)
        return interval > 0 and self.common_step_counter % interval == 0

    def _compute_arm_max_abs_tracking_error(self) -> torch.Tensor:
        """Mean over envs of max-abs arm joint error |q_actual - q_target|."""
        arm_actual = self.robot.data.joint_pos[:, self.arm_joint_indices]
        arm_target = self.robot_dof_targets[:, self.arm_joint_indices]
        arm_abs_err = (arm_actual - arm_target).abs()
        if arm_abs_err.numel() == 0:
            return torch.zeros((), dtype=torch.float, device=self.device)
        return arm_abs_err.amax(dim=-1).mean()

    def _resolve_category_include(self, spec: str) -> list[str]:
        if not spec:
            return []
        text = str(spec).strip()
        if not text:
            return []

        list_path = text
        if not os.path.isabs(list_path):
            candidate = os.path.join(PROJECT_ROOT, list_path)
            if os.path.isfile(candidate):
                list_path = candidate
        if os.path.isfile(list_path):
            with open(list_path, "r") as f:
                return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]

        tokens = [token.strip() for token in text.replace(",", " ").split()]
        return [token for token in tokens if token]

    def _resolve_scenario_include(self, spec: str) -> list[str]:
        return self._resolve_category_include(spec)

    def _load_dataset(
        self,
        num_grasps: int = 0,
        num_poses: int = 0,
        pose_pre_sampler: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
        load_top_point_cloud: bool = False,
        load_handle_point_cloud: bool = False,
    ):
        loader = AssetLoader(
            self.cfg.dataset,
            valid_asset_file=self.cfg.valid_asset_file,
            invalid_asset_file=self.cfg.invalid_asset_file,
        )
        indices = self.dataset_indices

        if num_grasps > 0:
            # loader.filter_out("fg")
            loader.filter_out("widowx")

        if indices:
            loader.chunk(indices)

        if self.cfg.dataset == "rigid":
            user_scenarios = self._resolve_scenario_include(self.cfg.scenario_include)
            if user_scenarios:
                loader.filter_by_scenarios(user_scenarios)
                print(f"[dataset] apply user scenario filter: {sorted(set(user_scenarios))}")
            user_categories = self._resolve_category_include(self.cfg.category_include)
            if user_categories:
                loader.filter_by_categories(user_categories)
                print(f"[dataset] apply user category filter: {sorted(set(user_categories))}")
            if self.cfg.category_ratio > 0:
                if self.cfg.category_ratio_random:
                    seed = None
                else:
                    seed = self.cfg.category_ratio_seed
                    if seed is None:
                        seed = self.cfg.seed if self.cfg.seed is not None else 0
                loader.sample_by_category_ratio(self.cfg.category_ratio, seed=seed)

        if self.world_size > 1:
            average_size, remaining_size = divmod(len(loader), self.world_size)
            chunk_start = average_size * self.rank + min(self.rank, remaining_size)
            chunk_end = chunk_start + average_size + (1 if self.rank < remaining_size else 0)
            loader = loader.chunk(range(chunk_start, chunk_end))

        use_top_point_cloud_for_obs_reward = bool(getattr(self.cfg, "use_top_point_cloud_for_obs_reward", False))
        prefer_canonical_point_cloud = not use_top_point_cloud_for_obs_reward
        self.dataset = loader.load(
            num_grasps=num_grasps,
            num_poses=num_poses,
            robot="widowx",
            pose_pre_sampler=pose_pre_sampler,
            load_top_point_cloud=load_top_point_cloud,
            load_handle_point_cloud=load_handle_point_cloud,
            prefer_canonical_point_cloud=prefer_canonical_point_cloud,
        ).to(self.device)
        self.object_new_scale = torch.ones((len(self.dataset),), dtype=torch.float, device=self.device)

        # Data
        num_assets = len(self.dataset)
        if self.cfg.shuffle_asset_assignment:
            seed = int(self.cfg.shuffle_asset_assignment_seed)
            if seed < 0:
                seed = self.cfg.seed if self.cfg.seed is not None else 0
            seed = int(seed)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            spawn_asset_order = torch.randperm(num_assets, generator=gen, dtype=torch.int64)
            reps = (self.num_envs + num_assets - 1) // num_assets
            object_order = spawn_asset_order.repeat(reps)[: self.num_envs]
            self.data_object_id = object_order.to(device=self.device, dtype=torch.int32)
            print(f"[dataset] shuffled asset assignment enabled (seed={seed})")
        else:
            spawn_asset_order = torch.arange(num_assets, dtype=torch.int64)
            self.data_object_id = torch.tensor([i % num_assets for i in range(self.num_envs)], dtype=torch.int32, device=self.device)
        self.data_original_id = torch.tensor(
            [self.dataset.original_indices[i] for i in self.data_object_id.tolist()],
            dtype=torch.int32,
            device=self.device,
        )
        self.data_point_cloud = self.dataset.point_clouds[self.data_object_id]
        self.data_top_point_cloud = (
            self.dataset.top_point_clouds[self.data_object_id] if self.dataset.top_point_clouds is not None else None
        )
        self.data_handle_point_cloud = (
            self.dataset.handle_point_clouds[self.data_object_id]
            if self.dataset.handle_point_clouds is not None
            else None
        )
        self.data_object_scale = self.object_new_scale[self.data_object_id]

        use_usd_mask = None
        sampled_mass = None
        if self.cfg.randomize_asset_mass:
            use_usd_prob = float(self.cfg.randomize_asset_use_usd_mass_prob)
            if not (0.0 <= use_usd_prob <= 1.0):
                raise ValueError(
                    f"randomize_asset_use_usd_mass_prob must be in [0, 1], got {self.cfg.randomize_asset_use_usd_mass_prob}"
                )
            mass_min = float(self.cfg.randomize_asset_mass_min)
            mass_max = float(self.cfg.randomize_asset_mass_max)
            if mass_min <= 0.0 or mass_max <= 0.0 or mass_min > mass_max:
                raise ValueError(
                    "randomize_asset_mass_min/randomize_asset_mass_max must satisfy: 0 < min <= max, "
                    f"got min={mass_min}, max={mass_max}"
                )

            seed = self.cfg.randomize_asset_mass_seed
            if seed is None:
                seed = self.cfg.seed if self.cfg.seed is not None else 0
            seed = int(seed)

            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            use_usd_mask = torch.rand(num_assets, generator=gen) < use_usd_prob
            if mass_min == mass_max:
                sampled_mass = torch.full((num_assets,), mass_min, dtype=torch.float32)
            else:
                sampled_mass = torch.empty((num_assets,), dtype=torch.float32).uniform_(mass_min, mass_max, generator=gen)

            # Per-asset mass_props is set in each UsdFileCfg below.
            self.cfg.objects.spawn.mass_props = None
            print(
                f"[dataset] per-asset mass randomization enabled "
                f"(seed={seed}, use_usd_prob={use_usd_prob}, mass_range=[{mass_min}, {mass_max}], "
                f"use_usd={int(use_usd_mask.sum().item())}/{num_assets})"
            )
        elif self.cfg.use_usd_mass:
            self.cfg.objects.spawn.mass_props = None

        # Write the path into the config.
        assert isinstance(self.cfg.objects.spawn, sim_utils.MultiAssetSpawnerCfg)
        assets_cfg = []
        # Keep spawned asset order aligned with data_object_id assignment.
        for asset_idx in spawn_asset_order.tolist():
            path = self.dataset.path[asset_idx]
            spawn_kwargs = {
                "usd_path": _resolve_asset_file(
                    path,
                    self.dataset.config["usd_path"],
                    kind="usd_path",
                    fallback_glob=self.dataset.config.get("usd_path_glob"),
                ),
                "articulation_props": sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                "scale": (self.object_new_scale[asset_idx].item(),) * 3,
            }
            if use_usd_mask is not None and sampled_mass is not None and not bool(use_usd_mask[asset_idx].item()):
                spawn_kwargs["mass_props"] = sim_utils.MassPropertiesCfg(mass=float(sampled_mass[asset_idx].item()))
            assets_cfg.append(sim_utils.UsdFileCfg(**spawn_kwargs))
        if self.cfg.shuffle_asset_assignment:
            print(
                f"[dataset] aligned spawn order with shuffled assignment; "
                f"first_slots={spawn_asset_order[: min(8, num_assets)].tolist()}"
            )
        self.cfg.objects.spawn.assets_cfg = assets_cfg
        # IsaacLab 3.0's spawn_multi_asset no longer clones one asset per env from the regex prim
        # path: it spawns assets_cfg[i] at <prefix>/Object_i and leaves the per-env distribution to
        # InteractiveSceneCfg.random_heterogeneous_cloning, which would break the env <-> data_object_id
        # mapping every buffer here relies on. Reproduce the 2.x round-robin (env i gets
        # assets_cfg[i % num_assets]) with one cfg copy and one explicit spawn path per env.
        # Absent on 2.x -> skipped.
        if hasattr(self.cfg.objects.spawn, "spawn_paths"):
            self.cfg.objects.spawn.assets_cfg = [
                copy.deepcopy(assets_cfg[i % len(assets_cfg)]) for i in range(self.num_envs)
            ]
            self.cfg.objects.spawn.spawn_paths = [f"/World/envs/env_{i}/Object" for i in range(self.num_envs)]

        # Load initial grasp state: object pose in grasp_site frame (wxyz) + finger joints.
        if num_grasps > 0:
            self.hold_id = torch.randint(0, num_grasps, (self.num_envs,), device=self.device)
            self.hold_state = self.dataset.grasps[self.data_object_id, self.hold_id].clone()
            self.hold_rel_pos = self.hold_state[:, :3]
            self.hold_rel_quat = self.hold_state[:, 3:7]
            self.hold_rel_pose = self.hold_state[:, :7]
            self.hold_finger_joint = self.hold_state[:, 7:]


    def _setup_scene(self):

        self.robot = Articulation(self.cfg.robot)
        self.table = RigidObject(self.cfg.table) if self.cfg.table is not None else None
        self.contact = ContactSensor(self.cfg.contact)

        spawn_ground_plane(
            "/World/ground",
            GroundPlaneCfg(size=(1000.0, 1000.0)),
            translation=(0.0, 0.0, float(self.cfg.ground_height)),
        )


        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)

        # create objects now (must after envs are cloned)
        self.object = RigidObject(self.cfg.objects)

        # register to the scene
        self.scene.articulations["robot"] = self.robot
        if self.table is not None:
            self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["contact"] = self.contact
        self.controller = DifferentialIKController(self.cfg.controller, self.num_envs, self.device)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    ############################################################
    # pre-physics step calls
    ############################################################

    def _pre_physics_step(self, actions: torch.Tensor):
        # print(f"[_pre_physics_step] Pre-physics step ({self.common_step_counter})")
        raw_actions = actions.clone().clamp(-1.0, 1.0)
        alpha = self.cfg.action_ema_alpha
        if alpha is not None and alpha > 0.0:
            self.actions = alpha * self.prev_actions + (1.0 - alpha) * raw_actions
        else:
            self.actions = raw_actions
        self.prev_actions = self.actions.clone()
        arm_action, gripper_action = self.actions[:, :6], self.actions[:, 6:]
        # Scale
        arm_action *= self.cmd_limit
        gripper_action *= 0.01
        # We use DifferentialIKController to convert operational-space commands to joint-space commands.
        # NOTE: the IK controller speaks the sim's own quaternion convention on both stacks
        # (WXYZ on IsaacLab 2.x, XYZW on 3.0) and body_quat_w matches it, so these are passed
        # through unconverted -- only the buffer type is normalized (3.0 hands back ProxyArray).
        ee_pos = qc._as_tensor(self.robot.data.body_pos_w)[:, self.hand_link_idx]
        ee_quat_sim = qc._as_tensor(self.robot.data.body_quat_w)[:, self.hand_link_idx]
        self.controller.set_command(arm_action, ee_pos, ee_quat_sim)
        jacobian = qc._as_tensor(self.robot.root_physx_view.get_jacobians())[
            :, self.hand_link_idx - 1, :, self.arm_joint_indices
        ]
        arm_targets = self.controller.compute(
            ee_pos,
            ee_quat_sim,
            jacobian,
            qc._as_tensor(self.robot.data.joint_pos)[:, :-2],
        )
        gripper_targets = self.robot.data.joint_pos[:, -2:] + gripper_action
        # Set robot DoF targets.
        self.robot_dof_targets[:] = torch.clamp(
            torch.cat((arm_targets, gripper_targets), dim=-1),
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

    def _apply_action(self):
        self.robot.set_joint_position_target(self.robot_dof_targets)

    ############################################################
    # post-physics step calls
    ############################################################

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is not None:
            self.prev_actions[env_ids] = 0.0
        # reset robot


