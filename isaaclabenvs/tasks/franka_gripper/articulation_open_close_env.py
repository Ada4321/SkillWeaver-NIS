from __future__ import annotations

import copy
import math

import isaaclab.sim as sim_utils
from isaaclabenvs.utils import math_compat as torch_utils  # WXYZ; replaces isaacsim.core.utils.torch (gone in Isaac Sim 6.0)
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclabenvs.utils.articulation_dataset import (
    JOINT_MOTION_PRISMATIC,
    JOINT_MOTION_REVOLUTE,
    ArticulationAssetLoader,
    ArticulationMetaLoader,
)

from .articulation_open_close_env_cfg import ArticulatedOpenCloseEnvCfg
from isaaclabenvs.utils import quat_compat as qc

from .base_env import BaseEnv


class ArticulatedOpenCloseEnv(BaseEnv):
    cfg: ArticulatedOpenCloseEnvCfg

    def __init__(self, cfg: ArticulatedOpenCloseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.object.num_joints <= 0:
            raise ValueError("Articulated open/close tasks require object.num_joints >= 1.")
        if not (0 <= int(self.cfg.object_joint_index) < self.object.num_joints):
            raise ValueError(
                f"object_joint_index out of range: {self.cfg.object_joint_index}, num_joints={self.object.num_joints}"
            )

        self.object_joint_idx = int(self.cfg.object_joint_index)
        self.object_joint_lower = self.object.data.soft_joint_pos_limits[:, self.object_joint_idx, 0].to(self.device)
        self.object_joint_upper = self.object.data.soft_joint_pos_limits[:, self.object_joint_idx, 1].to(self.device)
        self.object_joint_denom = (self.object_joint_upper - self.object_joint_lower).abs().clamp_min(self.cfg.joint_eps)

        self.goal_joint_pos_norm = torch.full(
            (self.num_envs,),
            float(self.cfg.target_joint_pos_norm),
            dtype=torch.float,
            device=self.device,
        ).clamp(0.0, 1.0)
        self.object_joint_pos_norm = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.object_joint_vel_norm = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.object_joint_error_norm = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Table state: cached size and per-env pose (re-randomized in z on each reset).
        self.table_size: tuple[float, float, float] = getattr(self.cfg.table.spawn, "size", None) or (0.8, 2.0, 0.02)
        self.table_pose = torch.tensor(
            [[*self.cfg.table.init_state.pos, *self.cfg.table.init_state.rot]], device=self.device
        ).repeat(self.num_envs, 1)
        self.table_pose[:, :3] += self.scene.env_origins


    def _resolve_asset_ids(self, spec: str, file_spec: str) -> list[str]:
        tokens = self._resolve_category_include(spec)
        if file_spec:
            tokens.extend(self._resolve_category_include(file_spec))
        return list(dict.fromkeys(token for token in tokens if token))

    def _load_dataset(self):
        loader = ArticulationAssetLoader(
            self.cfg.dataset,
            valid_asset_file=self.cfg.valid_asset_file,
            invalid_asset_file=self.cfg.invalid_asset_file,
        )
        if self.dataset_indices:
            loader.chunk(self.dataset_indices)

        user_asset_ids = self._resolve_asset_ids(self.cfg.asset_ids, self.cfg.asset_ids_file)
        if user_asset_ids:
            loader.filter_by_asset_ids(user_asset_ids)
            print(f"[articulation_dataset] apply user asset-id filter: {user_asset_ids}")

        if self.world_size > 1:
            average_size, remaining_size = divmod(len(loader), self.world_size)
            chunk_start = average_size * self.rank + min(self.rank, remaining_size)
            chunk_end = chunk_start + average_size + (1 if self.rank < remaining_size else 0)
            loader.chunk(range(chunk_start, chunk_end))

        self.dataset = loader.load().to(self.device)
        self.object_new_scale = torch.ones((len(self.dataset),), dtype=torch.float, device=self.device)

        num_assets = len(self.dataset)
        if num_assets <= 0:
            raise ValueError("No articulated assets left after filtering.")

        if self.cfg.shuffle_asset_assignment:
            seed = int(self.cfg.shuffle_asset_assignment_seed)
            if seed < 0:
                seed = self.cfg.seed if self.cfg.seed is not None else 0
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            spawn_asset_order = torch.randperm(num_assets, generator=gen, dtype=torch.int64)
            reps = (self.num_envs + num_assets - 1) // num_assets
            object_order = spawn_asset_order.repeat(reps)[: self.num_envs]
            self.data_object_id = object_order.to(device=self.device, dtype=torch.int32)
            print(f"[articulation_dataset] shuffled asset assignment enabled (seed={int(seed)})")
        else:
            spawn_asset_order = torch.arange(num_assets, dtype=torch.int64)
            self.data_object_id = torch.tensor(
                [i % num_assets for i in range(self.num_envs)],
                dtype=torch.int32,
                device=self.device,
            )

        self.data_original_id = torch.tensor(
            [self.dataset.original_indices[i] for i in self.data_object_id.tolist()],
            dtype=torch.int32,
            device=self.device,
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
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            use_usd_mask = torch.rand(num_assets, generator=gen) < use_usd_prob
            if mass_min == mass_max:
                sampled_mass = torch.full((num_assets,), mass_min, dtype=torch.float32)
            else:
                sampled_mass = torch.empty((num_assets,), dtype=torch.float32).uniform_(mass_min, mass_max, generator=gen)

            self.cfg.objects.spawn.mass_props = None
            print(
                f"[articulation_dataset] per-asset mass randomization enabled "
                f"(seed={int(seed)}, use_usd_prob={use_usd_prob}, mass_range=[{mass_min}, {mass_max}], "
                f"use_usd={int(use_usd_mask.sum().item())}/{num_assets})"
            )
        elif self.cfg.use_usd_mass:
            self.cfg.objects.spawn.mass_props = None

        assert isinstance(self.cfg.objects.spawn, sim_utils.MultiAssetSpawnerCfg)
        assets_cfg = []
        for asset_idx in spawn_asset_order[: self.num_envs].tolist():
            spawn_kwargs = {
                "usd_path": self.dataset.usd_paths[asset_idx],
                "articulation_props": sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=True),
                "scale": (self.object_new_scale[asset_idx].item(),) * 3,
            }
            if use_usd_mask is not None and sampled_mass is not None and not bool(use_usd_mask[asset_idx].item()):
                spawn_kwargs["mass_props"] = sim_utils.MassPropertiesCfg(mass=float(sampled_mass[asset_idx].item()))
            assets_cfg.append(sim_utils.UsdFileCfg(**spawn_kwargs))
        self.cfg.objects.spawn.assets_cfg = assets_cfg
        # IsaacLab 3.0's spawn_multi_asset no longer clones one asset per env from a regex prim
        # path: it spawns asset i at <prefix>/Object_i and leaves the per-env distribution to
        # InteractiveSceneCfg.random_heterogeneous_cloning, which would break the deterministic
        # env <-> data_object_id mapping every buffer here relies on. assets_cfg is already built
        # in that order, so hand 3.0 one cfg copy and one explicit spawn path per env: env i gets
        # assets_cfg[i % len(assets_cfg)], the 2.x round-robin (also when num_envs > num_assets).
        # Absent on 2.x -> skipped.
        if hasattr(self.cfg.objects.spawn, "spawn_paths"):
            self.cfg.objects.spawn.assets_cfg = [
                copy.deepcopy(assets_cfg[i % len(assets_cfg)]) for i in range(self.num_envs)
            ]
            self.cfg.objects.spawn.spawn_paths = [
                f"/World/envs/env_{i}/Object" for i in range(self.num_envs)
            ]

        meta = ArticulationMetaLoader(
            self.dataset.path,
            urdf_spec=self.dataset.config["urdf_path"],
            handle_points_spec=self.dataset.config["handle_points"],
            handle_points_key=self.dataset.config["handle_points_key"],
            init_arm_joint_pos_spec=(self.cfg.init_arm_joint_pos_override or self.dataset.config["init_arm_joint_pos"]),
            init_arm_joint_pos_key=self.dataset.config["init_arm_joint_pos_key"],
            init_arm_eef_dir_key=self.dataset.config["init_arm_eef_dir_key"],
            link_forward_points_spec=self.dataset.config.get("link_forward_points", ""),
            link_forward_points_key=self.dataset.config.get("link_forward_points_key", ""),
            handle_grasps_spec=(self.cfg.handle_grasps_override or self.dataset.config.get("handle_grasps", "")),
            variant_stems=self.dataset.variant_stems,
        ).load(device=self.device)
        asset_ids = self.data_object_id.to(dtype=torch.long)
        self.data_handle_points = meta.handle_points[asset_ids]
        self.data_link_forward_points = (
            meta.link_forward_points[asset_ids] if meta.link_forward_points is not None else None
        )
        # Per-env hand-crafted handle grasps in q0/object frame (E, N_grasp, 8): [:3] grasp_site
        # pos, [3:7] grasp_site quat (wxyz), [7] finger joint. None when dataset has no grasps.
        self.data_handle_grasps = meta.handle_grasps[asset_ids] if meta.handle_grasps is not None else None
        # Default sources for closest-point computation; close subclasses override these to use a
        # merged (handle + link_forward) pcd.
        self.data_query_points = self.data_handle_points
        self.data_point_cloud = self.data_handle_points
        self.init_arm_joint_pos = meta.init_arm_joint_pos
        self.init_arm_eef_dir = meta.init_arm_eef_dir
        self.init_arm_grid = meta.init_arm_grid
        self.data_joint_motion_type = meta.joint_motion_type[asset_ids]
        self.data_joint_axis_obj = meta.joint_axis_obj[asset_ids]
        self.data_joint_origin_obj = meta.joint_origin_obj[asset_ids]

    def _setup_scene(self):
        self._load_dataset()


        self.robot = Articulation(self.cfg.robot)
        self.contact = ContactSensor(self.cfg.contact)
        self._handle_contact_sensors: list[ContactSensor] = []
        for attr_name, sensor_name in (
            ("contact_left_handle", "contact_left_handle"),
            ("contact_left_tip_handle", "contact_left_tip_handle"),
            ("contact_right_handle", "contact_right_handle"),
            ("contact_right_tip_handle", "contact_right_tip_handle"),
        ):
            if hasattr(self.cfg, attr_name):
                sensor = ContactSensor(getattr(self.cfg, attr_name))
                setattr(self, sensor_name, sensor)
                self._handle_contact_sensors.append(sensor)

        spawn_ground_plane(
            "/World/ground",
            GroundPlaneCfg(size=(1000.0, 1000.0)),
            translation=(0.0, 0.0, float(self.cfg.ground_height)),
        )

        self.table = RigidObject(self.cfg.table)


        self.scene.clone_environments(copy_from_source=True)
        self.object = Articulation(self.cfg.objects)

        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        self.scene.sensors["contact"] = self.contact
        for sensor_name in (
            "contact_left_handle",
            "contact_left_tip_handle",
            "contact_right_handle",
            "contact_right_tip_handle",
        ):
            if hasattr(self, sensor_name):
                self.scene.sensors[sensor_name] = getattr(self, sensor_name)
        self.controller = DifferentialIKController(self.cfg.controller, self.num_envs, self.device)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_buffer()
        # Placeholder: finalize success/failure logic later.
        self.success[:] = False
        self.failure[:] = False
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self.failure | self.success, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Placeholder: finalize reward shaping later.
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.extras.update(
            {
                "object_joint_pos_norm": self.object_joint_pos_norm.mean(),
                "object_joint_vel_norm": self.object_joint_vel_norm.abs().mean(),
                "object_joint_error_norm": self.object_joint_error_norm.abs().mean(),
                "arm_max_abs_tracking_error": self._compute_arm_max_abs_tracking_error(),
            }
        )
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Randomize table top height per env.
        table_z_low, table_z_high = sorted(float(x) for x in self.cfg.table_z_range)
        self.table_pose[env_ids, 2] = (
            torch.rand(len(env_ids), device=self.device) * (table_z_high - table_z_low) + table_z_low
        )
        # table_pose's quaternion comes straight from cfg.table.init_state.rot, i.e. already in
        # the sim's convention; only z is re-sampled here, so it is written back unconverted.
        self.table.write_root_link_pose_to_sim(qc.pose_to_xyzw(self.table_pose[env_ids]), env_ids=env_ids)

        n = len(env_ids)
        object_pose = torch.zeros((n, 7), dtype=torch.float, device=self.device)
        env_origins = self.scene.env_origins[env_ids]

        # Position: sample env-local x, y uniformly from cfg-provided ranges. No table boundary
        # safety check; caller is responsible for keeping ranges within the table footprint.
        x_lo, x_hi = sorted(float(v) for v in self.cfg.object_init_x_range)
        y_lo, y_hi = sorted(float(v) for v in self.cfg.object_init_y_range)
        obj_x_local = torch.rand((n,), device=self.device) * (x_hi - x_lo) + x_lo
        obj_y_local = torch.rand((n,), device=self.device) * (y_hi - y_lo) + y_lo

        # Yaw: hook computes per-env theta. Default (this class) faces toward y=0; subclasses
        # may override (e.g. CloseEnv adds away-flip + jitter for revolute joints).
        theta = self._sample_object_yaw_theta(obj_y_local, env_ids)

        # Baseline z-rotation aligns asset's canonical forward (= local +x rotated by
        # asset_forward_local_angle_rad in the local xy plane) onto world -x at theta=0.
        baseline = math.pi - float(self.cfg.asset_forward_local_angle_rad)
        phi = baseline + theta
        half = 0.5 * phi
        zeros = torch.zeros_like(half)

        object_pose[:, 0] = env_origins[:, 0] + obj_x_local
        object_pose[:, 1] = env_origins[:, 1] + obj_y_local
        # Assets are canonicalized with z_min=0; place root at table top (+ small buffer
        # to avoid initial penetration).
        table_top_z = self.table_pose[env_ids, 2] + 0.5 * float(self.table_size[2])
        object_pose[:, 2] = table_top_z + 0.002
        object_pose[:, 3:7] = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)
        # object_pose is built WXYZ above; qc converts it to whatever the sim expects.
        self.object.write_root_link_pose_to_sim(qc.pose_to_xyzw(object_pose), env_ids=env_ids)
        self.object.write_root_link_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)

        start_min, start_max = self.cfg.joint_start_pos_range_norm
        start_min = float(start_min)
        start_max = float(start_max)
        if start_min > start_max:
            start_min, start_max = start_max, start_min
        start_norm = torch.rand((len(env_ids),), device=self.device) * (start_max - start_min) + start_min
        object_joint_pos = self.object.data.default_joint_pos[env_ids].clone()
        object_joint_vel = torch.zeros_like(object_joint_pos)
        object_joint_pos[:, self.object_joint_idx] = (
            self.object_joint_lower[env_ids] + start_norm * self.object_joint_denom[env_ids]
        )
        self.object.write_joint_state_to_sim(object_joint_pos, object_joint_vel, env_ids=env_ids)

        self.success[env_ids] = False
        self.failure[env_ids] = False
        self.reset_buf[env_ids] = 0

        self._refresh_buffer()

    def _sample_object_yaw_theta(self, obj_y_local: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        # Default: handle face always points toward y=0 (theta=+pi/2 when y>=0, -pi/2 when y<0).
        # Subclasses may override to add per-task variation (e.g. away-flip + jitter).
        return torch.where(
            obj_y_local >= 0.0,
            torch.full_like(obj_y_local,  math.pi / 2),
            torch.full_like(obj_y_local, -math.pi / 2),
        )

    def _forward_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        # Asset canonical forward (= local +x rotated by asset_forward_local_angle_rad in xy)
        # transformed to world by the current object root quaternion.
        n = len(env_ids)
        angle = float(self.cfg.asset_forward_local_angle_rad)
        local_forward = torch.tensor(
            [math.cos(angle), math.sin(angle), 0.0], device=self.device
        ).expand(n, 3)
        return torch_utils.quat_apply(qc.root_quat_w(self.object)[env_ids], local_forward)

    def _compute_init_arm_query_xyz(
        self, pcd_q0: torch.Tensor, env_ids: torch.Tensor
    ) -> torch.Tensor:
        # Compute per-env init-arm query xyz in env-local frame:
        #   xy = the most-forward point of pcd_q0 (after q0 -> world -> env-local) projected on
        #        forward direction
        #   z  = mean z of the same pcd in env-local frame
        forward_w = self._forward_w(env_ids)
        forward_xy = forward_w[:, :2]
        forward_xy = forward_xy / forward_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        pcd_w = self._q0_points_to_world(pcd_q0[env_ids], env_ids=env_ids)
        pcd_env = pcd_w - self.scene.env_origins[env_ids].unsqueeze(1)

        proj = (pcd_env[..., :2] * forward_xy.unsqueeze(1)).sum(-1)
        fwd_idx = proj.argmax(dim=-1)
        gather_idx = fwd_idx.view(-1, 1, 1).expand(-1, 1, pcd_env.shape[-1])
        target_xy = pcd_env.gather(1, gather_idx).squeeze(1)[:, :2]

        target_z = pcd_env[..., 2].mean(dim=-1)
        return torch.cat([target_xy, target_z.unsqueeze(-1)], dim=-1)

    def _sample_init_arm_joint_pos(
        self, env_ids: torch.Tensor, target_xyz: torch.Tensor
    ) -> torch.Tensor:
        # Look up grid cell at (target_xyz) using sign-based ceil/floor on x/y (move toward
        # forward direction) and round on z (closest). Within the cell, filter the n_sample
        # poses by xy cosine between stored EEF axis and current forward; fallback to argmax
        # when no sample passes the threshold; finally uniform-sample one.
        grid = self.init_arm_grid
        forward_w = self._forward_w(env_ids)

        def discretize(t, vmin, vstep, vnum, sign_v):
            norm = (t - vmin) / vstep
            idx = torch.where(sign_v > 0, torch.ceil(norm), torch.floor(norm))
            return idx.clamp(0, vnum - 1).long()

        x_idx = discretize(target_xyz[:, 0], grid.x_min, grid.x_step, grid.x_num, forward_w[:, 0])
        y_idx = discretize(target_xyz[:, 1], grid.y_min, grid.y_step, grid.y_num, forward_w[:, 1])
        z_idx = ((target_xyz[:, 2] - grid.z_min) / grid.z_step).round().clamp(0, grid.z_num - 1).long()

        env_arange = torch.arange(len(env_ids), device=self.device)
        cell_qpos = self.init_arm_joint_pos[x_idx, y_idx, z_idx]
        cell_eef_dir = self.init_arm_eef_dir[x_idx, y_idx, z_idx]

        forward_xy = forward_w[:, :2]
        forward_xy = forward_xy / forward_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        eef_xy = cell_eef_dir[..., :2]
        eef_xy = eef_xy / eef_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos = (eef_xy * forward_xy.unsqueeze(1)).sum(-1)

        threshold = float(self.cfg.init_arm_eef_forward_cos_threshold)
        keep = cos > threshold
        fallback_needed = keep.sum(-1) == 0
        if fallback_needed.any():
            max_idx = cos.argmax(dim=-1)
            fallback_keep = torch.zeros_like(keep)
            fallback_keep.scatter_(1, max_idx.unsqueeze(-1), True)
            keep = torch.where(fallback_needed.unsqueeze(-1), fallback_keep, keep)

        sample_idx = torch.multinomial(keep.float(), num_samples=1).squeeze(-1)
        return cell_qpos[env_arange, sample_idx]

    def _q0_points_to_world(
        self, points_q0: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Batched version of _q0_point_to_world for (E, M, 3) input. Per-env dispatches by joint
        # motion type and applies revolute / prismatic transformation to all M points.
        # When env_ids is given, points_q0 is expected to be already sliced to [len(env_ids), M, 3]
        # and per-env state (joint_pos / origin / axis / motion_type / root pose) is sliced to match.
        if env_ids is None:
            joint_pos = self.object.data.joint_pos[:, self.object_joint_idx]
            joint_lower = self.object_joint_lower
            origin_obj = self.data_joint_origin_obj
            axis_obj = self.data_joint_axis_obj
            motion_type = self.data_joint_motion_type
            root_quat = qc.root_quat_w(self.object)
            root_pos = qc.root_pos_w(self.object)
        else:
            joint_pos = self.object.data.joint_pos[env_ids, self.object_joint_idx]
            joint_lower = self.object_joint_lower[env_ids]
            origin_obj = self.data_joint_origin_obj[env_ids]
            axis_obj = self.data_joint_axis_obj[env_ids]
            motion_type = self.data_joint_motion_type[env_ids]
            root_quat = qc.root_quat_w(self.object)[env_ids]
            root_pos = qc.root_pos_w(self.object)[env_ids]

        joint_delta = joint_pos - joint_lower

        origin = origin_obj.unsqueeze(1)  # (E, 1, 3)
        vec_from_hinge = points_q0 - origin
        vec_rot = self._rodrigues_rotate(vec_from_hinge, axis_obj, joint_delta)
        points_revolute_obj = vec_rot + origin

        points_prismatic_obj = points_q0 + (axis_obj.unsqueeze(1) * joint_delta.view(-1, 1, 1))

        is_revolute = (motion_type == JOINT_MOTION_REVOLUTE).view(-1, 1, 1)
        points_cur_obj = torch.where(is_revolute, points_revolute_obj, points_prismatic_obj)

        m = points_cur_obj.shape[1]
        root_quat_b = root_quat.unsqueeze(1).expand(-1, m, -1)
        return torch_utils.quat_apply(root_quat_b, points_cur_obj) + root_pos.unsqueeze(1)

    def _q0_point_to_world(self, points_q0: torch.Tensor) -> torch.Tensor:
        # Map (E, 3) points from canonical obj frame (joint=0) to current world frame.
        # Branches per-env by joint motion type: revolute -> rotate around joint axis at joint
        # origin; prismatic -> translate along joint axis.
        joint_delta = self.object.data.joint_pos[:, self.object_joint_idx] - self.object_joint_lower

        origin = self.data_joint_origin_obj
        vec_from_hinge = points_q0 - origin
        vec_rot = self._rodrigues_rotate(
            vec_from_hinge.unsqueeze(1),
            self.data_joint_axis_obj,
            joint_delta,
        ).squeeze(1)
        points_revolute_obj = vec_rot + origin

        points_prismatic_obj = points_q0 + self.data_joint_axis_obj * joint_delta.unsqueeze(-1)

        is_revolute = (self.data_joint_motion_type == JOINT_MOTION_REVOLUTE).unsqueeze(-1)
        points_cur_obj = torch.where(is_revolute, points_revolute_obj, points_prismatic_obj)

        return torch_utils.quat_apply(
            qc.root_quat_w(self.object), points_cur_obj
        ) + qc.root_pos_w(self.object)

    def _get_observations(self) -> dict:
        # NOTE: no _collect_states() hook here -- OpenBaseEnv / CloseEnv override
        # _get_observations without calling super(), so it would never fire for the real
        # tasks. The hook lives in those two classes, next to their _save_trajectory call.
        obs = torch.cat(
            (
                self.robot_joint_pos,
                self.robot.data.joint_vel * self.cfg.dof_velocity_scale,
                self.robot.data.joint_pos_target,
                self.wrist_state,
                self.robot.data.body_pos_w[:, self.left_finger_idx] - self.scene.env_origins,
                self.robot.data.body_pos_w[:, self.right_finger_idx] - self.scene.env_origins,
                self.robot.data.body_pos_w[:, self.grasp_site_idx] - self.scene.env_origins,
                qc.root_pos_w(self.object) - self.scene.env_origins,
                qc.root_quat_w(self.object),
                self.obs_dist,
                self.obs_closest_point.reshape(self.num_envs, -1),
                self.obs_contact,
                self.object_joint_pos_norm.unsqueeze(-1),
                self.object_joint_vel_norm.unsqueeze(-1),
                self.goal_joint_pos_norm.unsqueeze(-1),
                self.object_joint_error_norm.unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": obs}


    def _refresh_buffer(self):
        inv_object_quat, inv_object_pos = torch_utils.tf_inverse(
            qc.root_quat_w(self.object), qc.root_pos_w(self.object)
        )
        link_pos_obj_frame = torch_utils.tf_apply(
            inv_object_quat.unsqueeze(1).expand(-1, len(self.keypoint_indices), -1),
            inv_object_pos.unsqueeze(1),
            self.robot.data.body_pos_w[:, self.keypoint_indices, :],
        )
        joint_delta = self.object.data.joint_pos[:, 0] - self.object_joint_lower
        query_q0 = self._map_points_to_q0_frame(link_pos_obj_frame, joint_delta)
        distance = torch.cdist(query_q0, self.data_query_points)
        self.obs_dist, min_indices = distance.min(dim=-1)
        closest_direction_q0 = (
            torch.gather(self.data_query_points, dim=1, index=min_indices.unsqueeze(-1).expand(-1, -1, 3))
            - query_q0
        )
        closest_direction = self._rotate_vectors_from_q0_to_current(closest_direction_q0, joint_delta)
        self.obs_closest_point = torch_utils.quat_apply(
            qc.root_quat_w(self.object).unsqueeze(1).expand(-1, self.obs_dist.shape[1], -1),
            closest_direction,
        )

        self.obs_contact = self._compute_obs_contact()

        self.robot_joint_pos = torch_utils.scale_transform(
            self.robot.data.joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )
        self.wrist_state = qc.body_state_w(self.robot)[:, self.hand_link_idx].clone()
        self.wrist_state[:, :3] -= self.scene.env_origins

        object_joint_pos = self.object.data.joint_pos[:, self.object_joint_idx]
        object_joint_vel = self.object.data.joint_vel[:, self.object_joint_idx]
        self.object_joint_pos_norm = ((object_joint_pos - self.object_joint_lower) / self.object_joint_denom).clamp(0.0, 1.0)
        self.object_joint_vel_norm = object_joint_vel / self.object_joint_denom
        self.object_joint_error_norm = self.goal_joint_pos_norm - self.object_joint_pos_norm

    def _compute_obs_contact(self) -> torch.Tensor:
        if len(self._handle_contact_sensors) == 4:
            contact_cols = []
            for sensor in self._handle_contact_sensors:
                force = sensor.data.net_forces_w
                mag = force.norm(dim=-1).reshape(self.num_envs, -1)
                contact_cols.append(mag.max(dim=-1).values > 0.01)
            return torch.stack(contact_cols, dim=-1)

        force = self.contact.data.net_forces_w
        mag = force.norm(dim=-1).reshape(self.num_envs, -1)
        return mag > 0.01

    def _rodrigues_rotate(self, vec: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        axis_expand = axis.unsqueeze(1).expand(-1, vec.shape[1], -1)
        angle_expand = angle.unsqueeze(-1).unsqueeze(-1)
        cos_theta = torch.cos(angle_expand)
        sin_theta = torch.sin(angle_expand)
        cross = torch.cross(axis_expand, vec, dim=-1)
        dot = (vec * axis_expand).sum(dim=-1, keepdim=True)
        return vec * cos_theta + cross * sin_theta + axis_expand * dot * (1.0 - cos_theta)

    def _map_points_to_q0_frame(self, points_obj: torch.Tensor, joint_delta: torch.Tensor) -> torch.Tensor:
        mapped = points_obj.clone()

        prismatic_mask = self.data_joint_motion_type == JOINT_MOTION_PRISMATIC
        if prismatic_mask.any():
            mapped[prismatic_mask] = mapped[prismatic_mask] - (
                joint_delta[prismatic_mask].unsqueeze(-1).unsqueeze(-1) * self.data_joint_axis_obj[prismatic_mask].unsqueeze(1)
            )

        revolute_mask = self.data_joint_motion_type == JOINT_MOTION_REVOLUTE
        if revolute_mask.any():
            origin = self.data_joint_origin_obj[revolute_mask].unsqueeze(1)
            vec = mapped[revolute_mask] - origin
            vec_q0 = self._rodrigues_rotate(vec, self.data_joint_axis_obj[revolute_mask], -joint_delta[revolute_mask])
            mapped[revolute_mask] = vec_q0 + origin

        return mapped

    def _rotate_vectors_from_q0_to_current(self, vectors_q0: torch.Tensor, joint_delta: torch.Tensor) -> torch.Tensor:
        rotated = vectors_q0.clone()
        revolute_mask = self.data_joint_motion_type == JOINT_MOTION_REVOLUTE
        if revolute_mask.any():
            rotated[revolute_mask] = self._rodrigues_rotate(
                rotated[revolute_mask],
                self.data_joint_axis_obj[revolute_mask],
                joint_delta[revolute_mask],
            )
        return rotated
