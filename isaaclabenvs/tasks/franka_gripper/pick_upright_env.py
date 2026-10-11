from __future__ import annotations

import enum
import math
import os

from isaaclabenvs.utils import math_compat as torch_utils  # WXYZ; replaces isaacsim.core.utils.torch (gone in Isaac Sim 6.0)
from isaaclabenvs.utils import quat_compat as qc
import numpy as np
import torch
from isaaclab.utils import Timer

from isaaclabenvs.config import PROJECT_ROOT
from isaaclabenvs.utils.dataset import InitialPose

from .base_env import BaseEnv
from .pick_upright_env_cfg import PickUpEnvCfg


class TaskStatus(enum.IntEnum):
    SUCCESS = enum.auto()
    FAILURE = enum.auto()
    TIMEOUT = enum.auto()


# def quat_conjugate(self, q):
#     return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

# def quat_mul(self, q1, q2):
#     # Hamilton product
#     w1, x1, y1, z1 = q1.unbind(-1)
#     w2, x2, y2, z2 = q2.unbind(-1)
#     return torch.stack([
#         w1*w2 - x1*x2 - y1*y2 - z1*z2,
#         w1*x2 + x1*w2 + y1*z2 - z1*y2,
#         w1*y2 - x1*z2 + y1*w2 + z1*x2,
#         w1*z2 + x1*y2 - y1*x2 + z1*w2
#     ], dim=-1)


class PickUpEnv(BaseEnv):
    cfg: PickUpEnvCfg
    _POSE_MODE_SPLIT = 200

    def __init__(self, cfg: PickUpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.statistics = {
            "success_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "timeout_rate": torch.ones(1, dtype=torch.float, device=self.device),
            "failure_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "lifted_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "lifted_success_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "pose_success_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "arm_max_abs_tracking_error": torch.zeros(1, dtype=torch.float, device=self.device),
        }

        self.latest_status = torch.full((self.num_envs,), TaskStatus.TIMEOUT, dtype=torch.int, device=self.device)
        self.latest_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.min_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.max_lift = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.max_xy_drift_pre_lift = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cumulative_lifted = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_lift_xy = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_lifted_pose = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_lifted_pose_xy = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_lifted_pose_upright = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_lifted_pose_yaw = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.unit_x = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device)

        self.data_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.lift_up_thres = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.lift_ref_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._upright_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.prev_hand_quat = None
        # _load_dataset() is called during super().__init__ via _setup_scene().
        # Preserve values initialized there instead of clobbering them here.
        self._xy_bottom_ref_local = getattr(self, "_xy_bottom_ref_local", None)
        self.data_point_cloud_obs_reward = getattr(self, "data_point_cloud_obs_reward", None)

        # Table
        self.table_size: tuple[int, int, int] = getattr(self.cfg.table.spawn, "size", None) or (0.8, 2.0, 0.02)
        self.table_pose = torch.tensor(
            [[*self.cfg.table.init_state.pos, *self.cfg.table.init_state.rot]], device=self.device
        ).repeat(self.num_envs, 1)
        self.table_pose[:, :3] += self.scene.env_origins


        self.initial_pose = InitialPose(self.device, self.cfg.initial_pose_path)

    def _setup_scene(self):
        # load dataset
        self._load_dataset()
        # setup scene as usual
        with Timer("[INFO] Time to setup scene:"):
            super()._setup_scene()

    @staticmethod
    def _asset_parts_from_path(path: str) -> tuple[str, str, str]:
        parts = os.path.normpath(path.rstrip("/")).split(os.sep)
        if len(parts) >= 3:
            return parts[-3], parts[-2], parts[-1]
        if len(parts) == 2:
            return "unknown", parts[-2], parts[-1]
        if len(parts) == 1:
            return "unknown", "unknown", parts[-1]
        return "unknown", "unknown", "unknown"

    @staticmethod
    def _parse_upright_selector(text: str, line_no: int, list_path: str) -> tuple[str, str, str] | None:
        s = text.split("#", 1)[0].strip()
        if not s:
            return None
        parts = [part.strip() for part in s.split("/") if part.strip()]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid selector at {list_path}:{line_no}. "
                f"Expected 'scenario/category/asset_xxxx', got '{text.strip()}'."
            )
        scenario, category, asset = parts
        if not (asset.startswith("asset_") and asset[6:].isdigit()):
            raise ValueError(
                f"Invalid asset id at {list_path}:{line_no}. "
                f"Expected 'asset_xxxx', got '{asset}'."
            )
        return scenario, category, asset

    def _read_upright_selectors(self, path: str) -> list[tuple[str, str, str]]:
        if not path:
            return []
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        if not os.path.isfile(path):
            return []
        selectors: list[tuple[str, str, str]] = []
        with open(path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                selector = self._parse_upright_selector(line, line_no, path)
                if selector is not None:
                    selectors.append(selector)
        return selectors

    def _resolve_upright_mode(self) -> tuple[str, int]:
        mode_to_id = {"mixed": 0, "upright": 1, "lying": 2}
        cfg_mode = str(self.cfg.upright_pose_mode).strip().lower()
        if cfg_mode not in mode_to_id:
            raise ValueError(
                f"Invalid upright_pose_mode: {self.cfg.upright_pose_mode}. "
                "Expected one of: mixed, upright, lying."
            )
        return cfg_mode, mode_to_id[cfg_mode]

    def _load_dataset(self):
        if self.cfg.use_top_point_cloud_for_obs_reward and self.cfg.use_handle_side_point_cloud_for_obs_reward:
            raise ValueError(
                "use_top_point_cloud_for_obs_reward and use_handle_side_point_cloud_for_obs_reward "
                "cannot both be True."
            )

        selectors = self._read_upright_selectors(self.cfg.upright_asset_file)
        cfg_mode, selected_mode_id = self._resolve_upright_mode()
        selector_set = set(selectors)
        split_idx = self._POSE_MODE_SPLIT

        def _pose_pre_sampler(asset_path: str, pose_tensor: torch.Tensor) -> torch.Tensor:
            if selected_mode_id == 0 or not selector_set:
                return pose_tensor
            if self._asset_parts_from_path(asset_path) not in selector_set:
                return pose_tensor
            if selected_mode_id == 1:
                hi = min(split_idx, int(pose_tensor.shape[0]))
                return pose_tensor[:hi]
            lo = min(split_idx, int(pose_tensor.shape[0]))
            sliced = pose_tensor[lo:]
            if int(sliced.shape[0]) == 0:
                return pose_tensor
            return sliced

        super()._load_dataset(
            num_poses=self.cfg.dataset_num_pose,
            pose_pre_sampler=_pose_pre_sampler,
            load_top_point_cloud=self.cfg.use_top_point_cloud_for_obs_reward,
            load_handle_point_cloud=self.cfg.use_handle_side_point_cloud_for_obs_reward,
        )
        self._upright_asset_mask, self._upright_pose_mode = self._build_upright_asset_mask_and_mode(
            selectors=selectors,
            cfg_mode=cfg_mode,
            selected_mode_id=selected_mode_id,
        )
        if self._upright_asset_mask.any():
            matched_counts = self.dataset.pose_counts[self._upright_asset_mask]
            print(
                f"[upright] post-load pose_counts for matched assets: "
                f"min={int(matched_counts.min().item())} max={int(matched_counts.max().item())}"
            )
            if selected_mode_id == 1:
                max_count = int(matched_counts.max().item())
                if max_count > split_idx:
                    raise RuntimeError(
                        f"upright pre-sampler check failed: matched pose_count max={max_count} > split={split_idx}. "
                        "This indicates mode filtering was not applied before choose_from."
                    )
        self.data_pose_id = self._sample_pose_ids_for_assets(self.data_object_id)
        self.data_point_cloud_obs_reward = self.data_point_cloud
        if self.cfg.use_handle_side_point_cloud_for_obs_reward:
            if self.data_handle_point_cloud is None:
                raise RuntimeError(
                    "use_handle_side_point_cloud_for_obs_reward=True but dataset handle_point_clouds were not loaded."
                )
            self.data_point_cloud_obs_reward = self.data_handle_point_cloud
        elif self.cfg.use_top_point_cloud_for_obs_reward:
            if self.data_top_point_cloud is None:
                raise RuntimeError(
                    "use_top_point_cloud_for_obs_reward=True but dataset top_point_clouds were not loaded."
                )
            self.data_point_cloud_obs_reward = self.data_top_point_cloud
        if self.cfg.use_bottom_ref_for_xy_disp:
            self._xy_bottom_ref_local = self._build_xy_bottom_ref_local()

    def _sample_pose_ids_for_assets(self, asset_ids: torch.Tensor) -> torch.Tensor:
        num_pose = int(self.dataset.poses.shape[1])
        if num_pose <= 0:
            raise ValueError("dataset poses must be non-empty.")

        if hasattr(self.dataset, "pose_counts"):
            max_counts = self.dataset.pose_counts[asset_ids].to(device=self.device, dtype=torch.int64)
            max_counts = max_counts.clamp(min=1, max=num_pose)
        else:
            max_counts = torch.full((len(asset_ids),), num_pose, dtype=torch.int64, device=self.device)

        def _sample_range(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
            span = (high - low).clamp(min=1)
            rand = torch.rand((low.shape[0],), device=self.device)
            return (low + torch.floor(rand * span.to(torch.float32)).to(torch.int64)).to(torch.int32)

        zeros = torch.zeros_like(max_counts)
        return _sample_range(zeros, max_counts)

    def _build_upright_asset_mask_and_mode(
        self,
        selectors: list[tuple[str, str, str]] | None = None,
        cfg_mode: str | None = None,
        selected_mode_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if selectors is None:
            selectors = self._read_upright_selectors(self.cfg.upright_asset_file)
        if cfg_mode is None or selected_mode_id is None:
            cfg_mode, selected_mode_id = self._resolve_upright_mode()
        pose_mode = torch.zeros(len(self.dataset.path), dtype=torch.int64, device=self.device)
        if not selectors:
            return (
                torch.zeros(len(self.dataset.path), dtype=torch.bool, device=self.device),
                pose_mode,
            )

        selector_set = set(selectors)
        mask = torch.zeros(len(self.dataset.path), dtype=torch.bool, device=self.device)
        for i, p in enumerate(self.dataset.path):
            scenario, category, asset = self._asset_parts_from_path(p)
            if (scenario, category, asset) in selector_set:
                mask[i] = True
        pose_mode[mask] = selected_mode_id
        print(
            f"[upright] asset_file={self.cfg.upright_asset_file} "
            f"dataset={self.cfg.dataset} total={len(self.dataset.path)} matched={mask.sum().item()}"
        )
        if mask.any():
            first_idx = int(mask.nonzero(as_tuple=False)[0].item())
            print(f"[upright] example_path={self.dataset.path[first_idx]}")
        mode_ids = pose_mode[mask]
        if mode_ids.numel() > 0:
            print(
                f"[upright] pose_mode={cfg_mode} pose_mode_counts "
                f"upright={(mode_ids == 1).sum().item()} "
                f"lying={(mode_ids == 2).sum().item()} "
                f"mixed={(mode_ids == 0).sum().item()}"
            )
        # import ipdb; ipdb.set_trace()
        return mask, pose_mode

    def _compute_upright_ok(self) -> torch.Tensor:
        if self._upright_asset_mask is None or not self._upright_asset_mask.any():
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # object up direction in world
        up = torch_utils.quat_apply(qc.root_quat_w(self.object), self.unit_z.unsqueeze(0).expand(self.num_envs, -1))
        up_z = up[:, 2].clamp(-1.0, 1.0)
        cos_thres = math.cos(math.radians(self.cfg.upright_keep_tilt_deg))
        upright_ok = up_z >= cos_thres
        # import ipdb; ipdb.set_trace()
        return (~self._upright_active) | upright_ok

    def _compute_yaw_ok(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Heading-around-world-z: project object's canonical +x onto xy plane,
        # compare current vs spawn (data_pose). Robust to tilt; when projection
        # is degenerate (object pointing straight up), defer to the tilt gate.
        ux = self.unit_x.unsqueeze(0).expand(self.num_envs, -1)
        cur_x = torch_utils.quat_apply(qc.root_quat_w(self.object), ux)
        init_x = torch_utils.quat_apply(self.data_pose[:, 3:7], ux)
        cur_xy, init_xy = cur_x[:, :2], init_x[:, :2]
        cur_n = cur_xy.norm(dim=-1)
        init_n = init_xy.norm(dim=-1)
        degenerate = (cur_n < 1e-3) | (init_n < 1e-3)
        cos_yaw = (cur_xy * init_xy).sum(dim=-1) / (cur_n.clamp(min=1e-6) * init_n.clamp(min=1e-6))
        cos_yaw = cos_yaw.clamp(-1.0, 1.0)
        cos_thres = math.cos(math.radians(self.cfg.obj_yaw_keep_deg))
        yaw_ok = (cos_yaw >= cos_thres) | degenerate
        return yaw_ok, cos_yaw

    def _build_xy_bottom_ref_local(self) -> torch.Tensor:
        pcd = self.dataset.point_clouds
        z_min = pcd[..., 2].min(dim=1, keepdim=True).values
        z_band = max(0.0, float(self.cfg.xy_bottom_ref_z_band))
        bottom_mask = pcd[..., 2] <= (z_min + z_band)
        bottom_count = bottom_mask.sum(dim=1, keepdim=True).clamp(min=1).to(pcd.dtype)
        bottom_ref_local = (pcd * bottom_mask.unsqueeze(-1)).sum(dim=1) / bottom_count
        return bottom_ref_local

    def _compute_lowest_z(
        self,
        quat_w: torch.Tensor,
        pos_w: torch.Tensor,
        point_cloud_local: torch.Tensor,
    ) -> torch.Tensor:
        point_cloud_world = (
            torch_utils.quat_apply(
                quat_w.unsqueeze(1).expand(-1, point_cloud_local.shape[1], -1),
                point_cloud_local,
            )
            + pos_w.unsqueeze(1)
        )
        return point_cloud_world[..., 2].min(dim=1).values

    def _compute_lift_z_current(self) -> torch.Tensor:
        if not self.cfg.lift_height_use_lowest_point:
            return qc.root_pos_w(self.object)[:, 2]
        return self._compute_lowest_z(
            qc.root_quat_w(self.object),
            qc.root_pos_w(self.object),
            self.data_point_cloud,
        )

    def _compute_xy_disp(self) -> torch.Tensor:
        if self.cfg.use_bottom_ref_for_xy_disp and self._xy_bottom_ref_local is not None:
            bottom_ref_local = self._xy_bottom_ref_local[self.data_object_id]
            cur_bottom_w = (
                torch_utils.quat_apply(qc.root_quat_w(self.object), bottom_ref_local) + qc.root_pos_w(self.object)
            )
            tgt_bottom_w = torch_utils.quat_apply(self.data_pose[:, 3:7], bottom_ref_local) + self.data_pose[:, :3]
            return (cur_bottom_w[:, :2] - tgt_bottom_w[:, :2]).norm(dim=-1)
        return (qc.root_pos_w(self.object)[:, :2] - self.data_pose[:, :2]).norm(dim=-1)

    def _compute_xy_radius_relax_bonus(self) -> torch.Tensor:
        if not self.cfg.enable_xy_radius_relax:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        target_xy_local = self.data_pose[:, :2] - self.scene.env_origins[:, :2]
        target_radius = target_xy_local.norm(dim=-1)
        bonus = (target_radius - self.cfg.xy_radius_relax_start) * self.cfg.xy_radius_relax_slope
        return bonus.clamp(0.0, self.cfg.xy_radius_relax_max_bonus)

    ############################################################
    # post-physics step calls
    ############################################################

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_buffer()

        object_pos = qc.root_pos_w(self.object) - self.scene.env_origins
        xy_disp = self._compute_xy_disp()
        xy_relax_bonus = self._compute_xy_radius_relax_bonus()
        if self.cfg.fail_on_xy_displacement:
            fail_xy_thresh = (
                self.cfg.fail_xy_displacement_thresh + self.cfg.xy_radius_relax_failure_scale * xy_relax_bonus
            )
            xy_disp_fail = xy_disp > fail_xy_thresh
        else:
            xy_disp_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.failure = (
            (object_pos[:, 0].abs() > self.cfg.scene.env_spacing)
            | (object_pos[:, 1].abs() > self.cfg.scene.env_spacing)
            | (object_pos[:, 2] < self.table_pose[:, 2] + 0.5 * self.table_size[2])
            | xy_disp_fail
        )
        # Success: consecutive lifted steps >= k AND sufficient xy-in-range ratio within those steps
        lifted_streak = self.consecutive_lifted_pose >= self.cfg.num_consecutive_success
        xy_ratio = torch.where(
            self.consecutive_lifted_pose > 0,
            self.consecutive_lifted_pose_xy.float() / self.consecutive_lifted_pose.float(),
            torch.zeros_like(self.consecutive_lifted_pose, dtype=torch.float),
        )
        upright_ok = self._compute_upright_ok()
        upright_ratio = torch.where(
            self.consecutive_lifted_pose > 0,
            self.consecutive_lifted_pose_upright.float() / self.consecutive_lifted_pose.float(),
            torch.zeros_like(self.consecutive_lifted_pose, dtype=torch.float),
        )
        yaw_ratio = torch.where(
            self.consecutive_lifted_pose > 0,
            self.consecutive_lifted_pose_yaw.float() / self.consecutive_lifted_pose.float(),
            torch.zeros_like(self.consecutive_lifted_pose, dtype=torch.float),
        )
        success_upright_gate = (
            upright_ratio >= self.cfg.lift_upright_success_ratio
            if self.cfg.gate_success_with_upright_ok
            else torch.ones_like(upright_ok)
        )
        success_xy_gate = (
            xy_ratio >= self.cfg.lift_xy_success_ratio
            if self.cfg.gate_success_with_xy_ratio
            else torch.ones_like(upright_ok)
        )
        success_yaw_gate = (
            yaw_ratio >= self.cfg.lift_yaw_success_ratio
            if self.cfg.gate_success_with_yaw_ok
            else torch.ones_like(upright_ok)
        )
        self.success = lifted_streak & success_xy_gate & success_upright_gate & success_yaw_gate

        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self.failure | self.success, truncated

    def _get_rewards(self) -> torch.Tensor:

        # Approach: cumulative; in total 20
        object_ee_dist = (qc.root_pos_w(self.object) - qc.body_pos_w(self.robot)[:, self.grasp_site_idx]).norm(dim=-1)
        rew_object_ee_dist = self.cfg.w_rew_object_ee_dist * (self.min_dist - object_ee_dist).relu()
        self.min_dist = torch.minimum(self.min_dist, object_ee_dist)

        # Contact: bonus; max 0.1
        num_contact = ((self.obs_contact > 0) & (self.obs_dist[:, :4] < 0.02)).sum(dim=-1)
        upright_ok = self._compute_upright_ok()
        # rew_contact = self.cfg.w_rew_contact * num_contact.float().clip(0, 2) * upright_ok.float()
        rew_contact = self.cfg.w_rew_contact * num_contact.float().clip(0, 2)

        # Close: penalty; max -2
        dist = self.obs_dist[:, :4].mean(dim=-1)
        rew_dist = self.cfg.w_rew_dist * dist.clip(0, 0.25)

        # Gripper open: penalty; max -0.2
        jaw = qc._as_tensor(self.robot.data.joint_pos)[:, self.left_joint_idx]
        rew_open = self.cfg.w_rew_open * (jaw < 0.002).float()

        # XY alignment to target pose
        # XY alignment to target pose
        dist_xy = self._compute_xy_disp()
        xy_relax_bonus = self._compute_xy_radius_relax_bonus()
        rew_dist_xy = self.cfg.w_rew_dist_xy * torch.exp(-10 * dist_xy)

        # Wrist orientation: encourage gripper -Z aligned with world -Z (i.e. facing down)
        # Wrist orientation: encourage gripper -Z aligned with world -Z (i.e. facing down)
        wrist_direction = torch_utils.quat_apply(
            qc.body_quat_w(self.robot)[:, self.hand_link_idx],
            self.unit_z.unsqueeze(0).expand(self.num_envs, -1),
        )
        target_dir = -self.unit_z.unsqueeze(0).expand(self.num_envs, -1)
        cos_angle = (wrist_direction * target_dir).sum(dim=-1).clamp(-1.0, 1.0)
        wrist_angle = torch.acos(cos_angle)  # 0: perfectly down, pi: perfectly up

        before_grasp = num_contact == 0
        after_grasp = num_contact > 0

        rew_wrist_rot = torch.zeros_like(wrist_angle)
        # weaker orientation shaping before contact, stronger after contact
        rot_thres = math.radians(self.cfg.wrist_rot_penalty_start_deg)
        penalize_mask = wrist_angle > rot_thres
        rew_wrist_rot[before_grasp & penalize_mask] = (
            self.cfg.w_rew_wrist_rot_before_grasp * wrist_angle[before_grasp & penalize_mask]
        )
        rew_wrist_rot[after_grasp & penalize_mask] = (
            self.cfg.w_rew_wrist_rot_after_grasp * wrist_angle[after_grasp & penalize_mask]
        )

        # Wrist smoothness: penalize orientation change after grasp
        cur_quat = qc.body_quat_w(self.robot)[:, self.hand_link_idx]
        # lazy init to avoid large delta on the first step
        if (
            not hasattr(self, "prev_hand_quat")
            or self.prev_hand_quat is None
            or self.prev_hand_quat.shape != cur_quat.shape
        ):
            self.prev_hand_quat = cur_quat.clone()

        delta_q = torch_utils.quat_mul(torch_utils.quat_conjugate(self.prev_hand_quat), cur_quat)
        rot_axis_norm = delta_q[..., 1:].norm(dim=-1).clamp(1e-8)
        delta_angle = 2.0 * torch.atan2(rot_axis_norm, delta_q[..., 0].abs().clamp(1e-8))

        rew_wrist_smooth = self.cfg.w_rew_wrist_smooth * delta_angle * after_grasp.float()

        self.prev_hand_quat = cur_quat.clone()

        # Lift progress (relative to data_pose)
        lift_z = self._compute_lift_z_current()
        lift = (lift_z - self.lift_ref_z).relu()
        self.max_lift = torch.maximum(self.max_lift, lift)

        # Object tilt hinge penalty on canonical z-axis (world frame tilt from +Z)
        obj_up = torch_utils.quat_apply(
            qc.root_quat_w(self.object),
            self.unit_z.unsqueeze(0).expand(self.num_envs, -1),
        )
        obj_up_z = obj_up[:, 2].clamp(-1.0, 1.0)
        obj_tilt = torch.acos(obj_up_z)
        obj_tilt_free = math.radians(self.cfg.obj_tilt_free_deg)
        obj_tilt_hinge = (obj_tilt - obj_tilt_free).relu()
        rew_obj_tilt_hinge = (
            self.cfg.w_rew_obj_tilt_hinge
            * obj_tilt_hinge
            * self._upright_active.float()
            * after_grasp.float()
        )

        # Lifted detection: height + contact
        lifted_raw = (lift_z > self.lift_up_thres) & (num_contact > 0)

        # Pose-aware lifted: require good wrist orientation when lifted
        angle_thres = self.cfg.wrist_rot_margin / 180.0 * torch.pi
        good_orientation = wrist_angle < angle_thres
        lifted_pose = lifted_raw & good_orientation if self.cfg.gate_lifted_pose_with_wrist_rot else lifted_raw

        diff_xy = dist_xy

        # Success pose constraints: object xy/orientation close to initial pose
        if self.cfg.threshold_decay:
            decay = 0.9 ** (max(self.common_step_counter - self.cfg.warmup_steps, 0) / 10000)
            pos_tol = max(self.cfg.success_xy_tol_min, self.cfg.success_xy_tol * decay)
        else:
            pos_tol = self.cfg.success_xy_tol

        xy_tol_effective = pos_tol + self.cfg.xy_radius_relax_success_scale * xy_relax_bonus
        xy_ok_for_success = diff_xy < xy_tol_effective

        # Lift reward gated by contact + xy tolerance; height-graded below lift_height
        xy_ok = diff_xy < xy_tol_effective
        reward_xy_gate = xy_ok if self.cfg.gate_lift_rewards_with_xy else torch.ones_like(xy_ok, dtype=torch.bool)
        yaw_ok, cos_yaw = self._compute_yaw_ok()
        lift_yaw_factor = yaw_ok.float() if self.cfg.gate_lift_reward_with_yaw_ok else torch.ones_like(yaw_ok, dtype=torch.float)
        lifted_yaw_factor = yaw_ok.float() if self.cfg.gate_lifted_reward_with_yaw_ok else torch.ones_like(yaw_ok, dtype=torch.float)
        lift_xy = (lift_z - self.lift_ref_z).relu()
        rew_lift_xy = (
            self.cfg.w_rew_lift_xy
            * (lift_xy - self.max_lift_xy).relu()
            * (num_contact > 0).float()
            * reward_xy_gate.float()
            * lift_yaw_factor
            # * good_orientation.float()
        )
        self.max_lift_xy = torch.maximum(self.max_lift_xy, lift_xy)

        # Lifted reward: reward raw lifting, give extra bonus for pose-correct lifting
        lifted_raw_xy = lifted_raw & reward_xy_gate
        lifted_pose_xy = lifted_pose & reward_xy_gate
        # rew_lifted = (
        #     self.cfg.w_rew_lifted_1 * (lifted_raw_xy & ~self.lifted & upright_ok).float()
        #     + self.cfg.w_rew_lifted_2 * (lifted_pose_xy & upright_ok).float()
        #     + self.cfg.w_rew_lifted_3 * (lifted_pose_xy & ~self.lifted & upright_ok).float()
        # )
        rew_lifted = (
            self.cfg.w_rew_lifted_1 * (lifted_raw_xy & ~self.lifted).float()
            + self.cfg.w_rew_lifted_2 * lifted_pose_xy.float()
            + self.cfg.w_rew_lifted_3 * (lifted_pose_xy & ~self.lifted).float()
        ) * lifted_yaw_factor

        # Update lifted flag (tracks whether object has ever been lifted above the threshold)
        self.lifted[:] |= lifted_raw

        # Track consecutive lifted_pose regardless of xy/orientation constraints
        self.consecutive_lifted_pose[:] = torch.where(lifted_pose, self.consecutive_lifted_pose + 1, 0)
        self.consecutive_lifted_pose_xy[:] = torch.where(
            lifted_pose,
            self.consecutive_lifted_pose_xy + xy_ok_for_success.long(),
            0,
        )
        self.consecutive_lifted_pose_upright[:] = torch.where(
            lifted_pose,
            self.consecutive_lifted_pose_upright + upright_ok.long(),
            0,
        )
        self.consecutive_lifted_pose_yaw[:] = torch.where(
            lifted_pose,
            self.consecutive_lifted_pose_yaw + yaw_ok.long(),
            0,
        )

        # Consecutive lift steps (used for hold fraction)
        self.consecutive_success[:] = torch.where(lifted_pose, self.consecutive_success + 1, 0)

        # Height stability around lift_up_thres once pose is correct
        # Encourage the object to stay close to the desired height when lifted_pose is true
        z_error = lift_z - self.lift_up_thres
        rew_height_stable = self.cfg.w_rew_height_stable * (z_error).relu() * lifted_pose.float()

        # Velocity stability: discourage motion when lifted_pose is true
        obj_speed = qc._as_tensor(self.object.data.root_vel_w).norm(dim=-1)
        rew_vel_stable = self.cfg.w_rew_vel_stable * obj_speed * lifted_pose.float()

        # End-effector velocity stability: discourage wrist jitter when lifted
        ee_vel = qc._as_tensor(self.robot.data.body_vel_w)[:, self.hand_link_idx]
        ee_speed = ee_vel.norm(dim=-1)
        rew_ee_stable = self.cfg.w_rew_ee_stable * ee_speed * lifted_pose.float()

        # Success reward (pose constraints path only)
        lifted_success = self.consecutive_lifted_pose >= self.cfg.num_consecutive_success
        rew_success_lifted = torch.zeros_like(lifted_success, dtype=torch.float)
        rew_success_pose = self.cfg.w_rew_success * self.success.float()
        # Update success with lift-streak + xy-in-range ratio rule
        xy_ratio = torch.where(
            self.consecutive_lifted_pose > 0,
            self.consecutive_lifted_pose_xy.float() / self.consecutive_lifted_pose.float(),
            torch.zeros_like(self.consecutive_lifted_pose, dtype=torch.float),
        )
        upright_ratio = torch.where(
            self.consecutive_lifted_pose > 0,
            self.consecutive_lifted_pose_upright.float() / self.consecutive_lifted_pose.float(),
            torch.zeros_like(self.consecutive_lifted_pose, dtype=torch.float),
        )
        yaw_ratio = torch.where(
            self.consecutive_lifted_pose > 0,
            self.consecutive_lifted_pose_yaw.float() / self.consecutive_lifted_pose.float(),
            torch.zeros_like(self.consecutive_lifted_pose, dtype=torch.float),
        )
        success_upright_gate = (
            upright_ratio >= self.cfg.lift_upright_success_ratio
            if self.cfg.gate_success_with_upright_ok
            else torch.ones_like(upright_ok)
        )
        success_xy_gate = (
            xy_ratio >= self.cfg.lift_xy_success_ratio
            if self.cfg.gate_success_with_xy_ratio
            else torch.ones_like(upright_ok)
        )
        success_yaw_gate = (
            yaw_ratio >= self.cfg.lift_yaw_success_ratio
            if self.cfg.gate_success_with_yaw_ok
            else torch.ones_like(upright_ok)
        )
        self.success = (
            (self.consecutive_lifted_pose >= self.cfg.num_consecutive_success)
            & success_xy_gate
            & success_upright_gate
            & success_yaw_gate
        )
        rew_success = rew_success_lifted + rew_success_pose
        rew_success = rew_success * self.success.float()

        # Velocity penalty
        # rew_action = self.cfg.w_rew_action * 500 ** min(1, self.common_step_counter / 10000) * (obj_speed + ee_speed)
        weight_action = -0.1 * 100 ** min(1, self.common_step_counter / 20000)
        rew_action = weight_action * self.actions.norm(dim=-1)

        # Set latest status
        self.latest_status[self.success] = TaskStatus.SUCCESS
        self.latest_status[self.failure] = TaskStatus.FAILURE
        self.latest_status[self.reset_time_outs] = TaskStatus.TIMEOUT

        self.latest_lifted[self.reset_buf] = self.lifted[self.reset_buf]
        self.latest_lifted[:] |= self.lifted

        if self.reset_buf.any():
            self.statistics["success_rate"] = (self.latest_status == TaskStatus.SUCCESS).float().mean()
            self.statistics["timeout_rate"] = (self.latest_status == TaskStatus.TIMEOUT).float().mean()
            self.statistics["failure_rate"] = (self.latest_status == TaskStatus.FAILURE).float().mean()

        self.statistics["lifted_rate"] = self.latest_lifted.float().mean()
        self.statistics["lifted_success_rate"] = lifted_success.float().mean()
        self.statistics["pose_success_rate"] = self.success.float().mean()
        self.statistics["arm_max_abs_tracking_error"] = self._compute_arm_max_abs_tracking_error()

        if self._should_print():
            print({k: v.item() for k, v in self.statistics.items()})

        # import ipdb; ipdb.set_trace()

        # Record
        rewards = {
            "rew_object_ee_dist": rew_object_ee_dist.mean(),
            "rew_dist": rew_dist.mean(),
            "rew_open": rew_open.mean(),
            "rew_dist_xy": rew_dist_xy.mean(),
            "rew_wrist_rot": rew_wrist_rot.mean(),
            "rew_wrist_smooth": rew_wrist_smooth.mean(),
            "rew_contact": rew_contact.mean(),
            "rew_lifted": rew_lifted.mean(),
            "rew_lift_xy": rew_lift_xy.mean(),
            "xy_radius_relax_bonus_mean": xy_relax_bonus.mean(),
            "xy_tol_effective_mean": xy_tol_effective.mean(),
            "upright_ok_rate": upright_ok.float().mean(),
            "upright_ratio_mean": upright_ratio.mean(),
            "yaw_ok_rate": yaw_ok.float().mean(),
            "yaw_ratio_mean": yaw_ratio.mean(),
            "obj_yaw_dev_deg_mean": (torch.acos(cos_yaw) * 180.0 / torch.pi).mean(),
            "obj_tilt_deg_mean": (obj_tilt * 180.0 / torch.pi).mean(),
            "rew_obj_tilt_hinge": rew_obj_tilt_hinge.mean(),
            "rew_height_stable": rew_height_stable.mean(),
            "rew_vel_stable": rew_vel_stable.mean(),
            "rew_ee_stable": rew_ee_stable.mean(),
            "rew_action": rew_action.mean(),
            "rew_success_lifted": rew_success_lifted.mean(),
            "rew_success_pose": rew_success_pose.mean(),
            "rew_success": rew_success.mean(),
        }
        self.extras.update(rewards)
        self.extras.update(self.statistics)

        if self._should_print():
            print("; ".join(f"{k}: {v:.4f}" for k, v in rewards.items()))


        reward = (
            rew_object_ee_dist
            + rew_dist
            + rew_open
            + rew_dist_xy
            + rew_wrist_rot
            + rew_wrist_smooth
            + rew_contact
            + rew_lifted
            + rew_lift_xy
            + rew_obj_tilt_hinge
            + rew_height_stable
            + rew_vel_stable
            + rew_ee_stable
            + rew_action
        ) + rew_success
        return reward


    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        #import ipdb; ipdb.set_trace()


        # randomize table height
        z_min, z_max = self.cfg.table_z_range
        self.table_pose[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (z_max - z_min) + z_min
        self.table.write_root_link_pose_to_sim(qc.pose_to_xyzw(self.table_pose[env_ids]), env_ids=env_ids)

        # reset object
        self.data_pose_id[env_ids] = self._sample_pose_ids_for_assets(self.data_object_id[env_ids])
        sampled_pose = self.dataset.poses[self.data_object_id[env_ids], self.data_pose_id[env_ids]]
        if sampled_pose.shape[-1] == 4:
            # Dataset provides only orientation (quat wxyz).
            self.data_pose[env_ids, :3] = 0.0
            self.data_pose[env_ids, 3:7] = sampled_pose
        else:
            self.data_pose[env_ids] = sampled_pose
        x_min, x_max = self.cfg.object_x_range
        y_min, y_max = self.cfg.object_y_range
        self.data_pose[env_ids, 0] = torch.rand((len(env_ids),), device=self.device) * (x_max - x_min) + x_min
        self.data_pose[env_ids, 1] = torch.rand((len(env_ids),), device=self.device) * (y_max - y_min) + y_min
        # Preserve canonical-frame z before adding table height (used for initial pose sampling).
        data_pose_z_canonical = self.data_pose[env_ids, 2].clone()
        self.data_pose[env_ids, 2] += self.table_pose[env_ids, 2] + 0.5 * self.table_size[2] + 0.002
        self.data_pose[env_ids, :3] += self.scene.env_origins[env_ids]
        if self._upright_asset_mask is None:
            self._upright_active[env_ids] = False
        else:
            up = torch_utils.quat_apply(
                self.data_pose[env_ids, 3:7],
                self.unit_z.unsqueeze(0).expand(len(env_ids), -1),
            )
            up_z = up[:, 2].clamp(-1.0, 1.0)
            cos_thres = math.cos(math.radians(self.cfg.upright_init_tilt_deg))
            upright_episode = up_z >= cos_thres
            asset_mask = self._upright_asset_mask[self.data_object_id[env_ids]]
            self._upright_active[env_ids] = asset_mask & upright_episode
        # Lift threshold: ignore table height compensation; clamp to absolute max z
        if self.cfg.lift_height_use_lowest_point:
            self.lift_ref_z[env_ids] = self._compute_lowest_z(
                self.data_pose[env_ids, 3:7],
                self.data_pose[env_ids, :3],
                self.data_point_cloud[env_ids],
            )
        else:
            self.lift_ref_z[env_ids] = self.data_pose[env_ids, 2]
        lift_thres = self.lift_ref_z[env_ids] + self.cfg.lift_height
        self.lift_up_thres[env_ids] = torch.minimum(
            lift_thres, torch.tensor(self.cfg.lift_up_thres_max, device=self.device)
        )

        self.object.write_root_link_pose_to_sim(qc.pose_to_xyzw(self.data_pose[env_ids]), env_ids=env_ids)
        self.object.write_root_link_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)

        # reset hand: joint
        joint_pos = qc._as_tensor(self.robot.data.default_joint_pos)[env_ids]
        # Use canonical z (assumed center height) to shift table_z upward and avoid collisions.
        table_z_eff = self.table_pose[env_ids, 2] + 2.0 * data_pose_z_canonical
        joint_pos[:, :7] = self.initial_pose.sample(
            self.data_pose[env_ids, :2] - self.scene.env_origins[env_ids, :2],
            table_z=table_z_eff,
        )
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        if self.prev_hand_quat is None:
            # first time allocation
            self.prev_hand_quat = qc.body_quat_w(self.robot)[:, self.hand_link_idx].clone()
        else:
            # only update the envs that are being reset
            self.prev_hand_quat[env_ids] = qc.body_quat_w(self.robot)[env_ids, self.hand_link_idx]

        self.max_xy_drift_pre_lift[env_ids] = 0.0

        self.reset_buf[env_ids] = 0
        self.min_dist[env_ids] = 0.2
        self.max_lift[env_ids] = 0.0
        self.lifted[env_ids] = False
        self.cumulative_lifted[env_ids] = 0
        self.max_lift_xy[env_ids] = 0.0
        self.consecutive_lifted_pose[env_ids] = 0
        self.consecutive_lifted_pose_xy[env_ids] = 0
        self.consecutive_lifted_pose_upright[env_ids] = 0
        self.consecutive_lifted_pose_yaw[env_ids] = 0
        self.consecutive_success[env_ids] = 0

        self._refresh_buffer()

    def _get_observations(self) -> dict:

        obs = torch.cat(
            (
                self.robot_joint_pos,
                qc._as_tensor(self.robot.data.joint_vel) * self.cfg.dof_velocity_scale,
                qc._as_tensor(self.robot.data.joint_pos_target),
                self.wrist_state,
                qc.body_pos_w(self.robot)[:, self.left_finger_idx] - self.scene.env_origins,
                qc.body_pos_w(self.robot)[:, self.right_finger_idx] - self.scene.env_origins,
                qc.body_pos_w(self.robot)[:, self.grasp_site_idx] - self.scene.env_origins,
                qc.root_pos_w(self.object) - self.scene.env_origins,
                self.obs_dist,
                self.obs_closest_point.reshape(self.num_envs, -1),
                self.obs_contact,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _refresh_buffer(self):
        # Distance Feature
        pcd_obs_reward = self.data_point_cloud_obs_reward
        if pcd_obs_reward is None:
            pcd_obs_reward = self.data_point_cloud
        if pcd_obs_reward is None:
            raise RuntimeError("Point cloud for observation reward is not initialized.")

        inv_object_quat, inv_object_pos = torch_utils.tf_inverse(
            qc.root_quat_w(self.object), qc.root_pos_w(self.object)
        )
        link_pos_obj_frame = torch_utils.tf_apply(
            inv_object_quat.unsqueeze(1).expand(-1, len(self.keypoint_indices), -1),
            inv_object_pos.unsqueeze(1),
            qc.body_pos_w(self.robot)[:, self.keypoint_indices, :],
        )
        distance = torch.cdist(
            link_pos_obj_frame,
            pcd_obs_reward,
        )  # (num_envs, num_query, point_cloud_size)
        self.obs_dist, min_indices = distance.min(dim=-1)
        closest_direction = (
            torch.gather(
                pcd_obs_reward,
                dim=1,
                index=min_indices.unsqueeze(-1).expand(-1, -1, 3),
            )
            - link_pos_obj_frame
        )  # (num_envs, num_query, 3)
        self.obs_closest_point: torch.Tensor = torch_utils.quat_apply(
            qc.root_quat_w(self.object).unsqueeze(1).expand(-1, self.obs_dist.shape[1], -1),
            closest_direction,
        )

        # Contact Forces
        force = qc._as_tensor(self.contact.data.net_forces_w)
        self.obs_contact = force.squeeze(1).norm(dim=-1) > 0.01

        # Others
        self.robot_joint_pos = torch_utils.scale_transform(
            qc._as_tensor(self.robot.data.joint_pos), self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )
        self.wrist_state = qc.body_state_w(self.robot)[:, self.hand_link_idx]
        self.wrist_state[:, :3] -= self.scene.env_origins


    def _resample_pcd(self, pcd: torch.Tensor, n_points: int) -> torch.Tensor:
        """pcd: (N_raw, 3) -> (n_points, 3)."""
        N = pcd.shape[0]
        if N == n_points:
            return pcd
        if N > n_points:
            # random downsample
            idx = torch.randperm(N, device=pcd.device)[:n_points]
            return pcd[idx]
        # N < n_points: upsample by tiling + random
        reps = (n_points + N - 1) // N
        pcd_big = pcd.repeat(reps, 1)[:n_points]
        return pcd_big


def sample_points_from_mesh(verts: np.ndarray, faces: np.ndarray, n_samples: int) -> np.ndarray:
    """
    verts: (V, 3)
    faces: (F, 3) triangle vertex indices
    returns: (n_samples, 3) points in mesh local frame
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Triangle areas
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)
    probs = areas / areas.sum()

    # Sample triangles
    face_idx = np.random.choice(len(faces), size=n_samples, p=probs)
    v0 = v0[face_idx]
    v1 = v1[face_idx]
    v2 = v2[face_idx]

    # Sample barycentric coords
    u = np.random.rand(n_samples, 1)
    v = np.random.rand(n_samples, 1)
    mask = u + v > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]

    points = v0 + u * (v1 - v0) + v * (v2 - v0)
    print(points)
    return points
