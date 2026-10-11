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
from .pick_env_cfg import PickEnvCfg


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


class PickEnv(BaseEnv):
    cfg: PickEnvCfg

    def __init__(self, cfg: PickEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.consecutive_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.data_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.lift_up_thres = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._upright_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.prev_hand_quat = None

        # Table
        self.table_size: tuple[int, int, int] = self.cfg.table.spawn.size
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

    def _load_dataset(self):
        super()._load_dataset(num_poses=self.cfg.dataset_num_pose)
        # Pose ID: randomly sampled (only from each asset's non-padded pose range)
        self._lying_pose_mask: torch.Tensor | None = None
        self.data_pose_id = self._sample_pose_ids_for_assets(self.data_object_id)
        self._upright_asset_mask = self._build_upright_asset_mask()

    def _sample_pose_ids_for_assets(self, asset_ids: torch.Tensor) -> torch.Tensor:
        num_pose = int(self.dataset.poses.shape[1])
        if num_pose <= 0:
            raise ValueError("dataset poses must be non-empty.")
        if hasattr(self.dataset, "pose_counts"):
            max_counts = self.dataset.pose_counts[asset_ids].to(device=self.device, dtype=torch.int64)
            max_counts = max_counts.clamp(min=1, max=num_pose)
            rand = torch.rand((len(asset_ids),), device=self.device)
            pose_ids = torch.floor(rand * max_counts.to(torch.float32)).to(torch.int32)
        else:
            pose_ids = torch.randint(0, num_pose, (len(asset_ids),), dtype=torch.int32, device=self.device)

        # For upright assets, resample from lying poses only.
        if self._lying_pose_mask is not None and self._upright_asset_mask is not None:
            upright_envs = self._upright_asset_mask[asset_ids]
            if upright_envs.any():
                upright_indices = upright_envs.nonzero(as_tuple=False).squeeze(1)
                upright_asset_ids = asset_ids[upright_indices]
                lying_mask = self._lying_pose_mask[upright_asset_ids]  # [N, num_pose]
                # For each upright asset, uniformly sample among its lying poses.
                # Fall back to full range if no lying poses exist.
                counts = lying_mask.sum(dim=1).clamp(min=1)  # [N]
                rand = torch.rand((len(upright_indices),), device=self.device)
                # Map rand to a local index within lying poses, then convert to actual pose id.
                local_idx = torch.floor(rand * counts.float()).long().clamp(max=num_pose - 1)
                # cumsum to find the actual pose index for the local_idx-th True in each row
                cum = lying_mask.long().cumsum(dim=1)  # [N, num_pose]
                target = local_idx.unsqueeze(1) + 1  # [N, 1], 1-indexed
                # For rows with no lying poses, use full range (already sampled)
                has_lying = lying_mask.any(dim=1)
                actual_idx = (cum >= target).float().argmax(dim=1).to(torch.int32)
                pose_ids[upright_indices] = torch.where(has_lying, actual_idx, pose_ids[upright_indices])

        return pose_ids

    def _build_upright_asset_mask(self) -> torch.Tensor:
        def _asset_parts_from_path(p: str) -> tuple[str, str, str]:
            parts = os.path.normpath(p.rstrip("/")).split(os.sep)
            if len(parts) >= 3:
                return parts[-3], parts[-2], parts[-1]
            if len(parts) == 2:
                return "unknown", parts[-2], parts[-1]
            if len(parts) == 1:
                return "unknown", "unknown", parts[-1]
            return "unknown", "unknown", "unknown"

        def _parse_selector(text: str, line_no: int, list_path: str) -> tuple[str, str, str] | None:
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

        def _read_list(path: str) -> set[tuple[str, str, str]]:
            if not path:
                return set()
            if not os.path.isabs(path):
                path = os.path.join(PROJECT_ROOT, path)
            if not os.path.isfile(path):
                return set()
            selectors: set[tuple[str, str, str]] = set()
            with open(path, "r") as f:
                for line_no, line in enumerate(f, start=1):
                    selector = _parse_selector(line, line_no, path)
                    if selector is not None:
                        selectors.add(selector)
            return selectors

        selectors = _read_list(self.cfg.upright_asset_file)

        if not selectors:
            return torch.zeros(len(self.dataset.path), dtype=torch.bool, device=self.device)

        mask = torch.zeros(len(self.dataset.path), dtype=torch.bool, device=self.device)
        for i, p in enumerate(self.dataset.path):
            scenario, category, asset = _asset_parts_from_path(p)
            if (scenario, category, asset) in selectors:
                mask[i] = True
        print(
            f"[upright] asset_file={self.cfg.upright_asset_file} "
            f"dataset={self.cfg.dataset} total={len(self.dataset.path)} matched={mask.sum().item()}"
        )
        if mask.any():
            first_idx = int(mask.nonzero(as_tuple=False)[0].item())
            print(f"[upright] example_path={self.dataset.path[first_idx]}")

        # Pre-compute lying pose mask for upright assets so _sample_pose_ids_for_assets
        # can restrict them to non-upright spawn orientations.
        cos_thres = math.cos(math.radians(self.cfg.upright_init_tilt_deg))
        # Compute up_z for every (asset, pose) pair.
        poses_dev = torch.as_tensor(self.dataset.poses, dtype=torch.float32, device=self.device)  # [A, P, 4or7]
        quats = poses_dev[..., -4:]  # last 4 = wxyz quat regardless of shape
        unit_z = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)
        unit_z_exp = unit_z.unsqueeze(0).unsqueeze(0).expand(poses_dev.shape[0], poses_dev.shape[1], -1)
        poses_flat = quats.reshape(-1, 4)
        unit_z_flat = unit_z_exp.reshape(-1, 3)
        up_flat = torch_utils.quat_apply(poses_flat, unit_z_flat)
        up_z = up_flat[:, 2].reshape(poses_dev.shape[0], poses_dev.shape[1])  # [A, P]
        # lying = NOT upright (up_z below threshold)
        self._lying_pose_mask = up_z < cos_thres  # [A, P]
        lying_counts = self._lying_pose_mask[mask].sum(dim=1)
        print(
            f"[upright] lying_pose_mask built: upright_assets={mask.sum().item()} "
            f"avg_lying_poses_per_upright_asset={lying_counts.float().mean().item():.1f}"
        )

        return mask

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

    ############################################################
    # post-physics step calls
    ############################################################

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_buffer()

        object_pos = qc.root_pos_w(self.object) - self.scene.env_origins
        self.failure = (
            (object_pos[:, 0].abs() > self.cfg.scene.env_spacing)
            | (object_pos[:, 1].abs() > self.cfg.scene.env_spacing)
            | (object_pos[:, 2] < self.table_pose[:, 2] + 0.5 * self.table_size[2])
        )
        # Success: consecutive lifted steps >= k AND sufficient xy-in-range ratio within those steps
        lifted_streak = self.consecutive_lifted_pose >= self.cfg.num_consecutive_success
        xy_ratio = torch.where(
            self.consecutive_lifted_pose > 0,
            self.consecutive_lifted_pose_xy.float() / self.consecutive_lifted_pose.float(),
            torch.zeros_like(self.consecutive_lifted_pose, dtype=torch.float),
        )
        upright_ok = self._compute_upright_ok()
        self.success = lifted_streak & (xy_ratio >= self.cfg.lift_xy_success_ratio) & upright_ok

        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self.failure | self.success, truncated

    def _get_rewards(self) -> torch.Tensor:

        # Approach: cumulative; in total 20
        object_ee_dist = (qc.root_pos_w(self.object) - qc.body_pos_w(self.robot)[:, self.grasp_site_idx]).norm(dim=-1)
        rew_object_ee_dist = self.cfg.w_rew_object_ee_dist * (self.min_dist - object_ee_dist).relu()
        self.min_dist = torch.minimum(self.min_dist, object_ee_dist)

        # Contact: bonus; max 0.1
        num_contact = ((self.obs_contact > 0) & (self.obs_dist[:, :4] < 0.04)).sum(dim=-1)
        upright_ok = self._compute_upright_ok()
        # rew_contact = self.cfg.w_rew_contact * num_contact.float().clip(0, 2) * upright_ok.float()
        rew_contact = self.cfg.w_rew_contact * num_contact.float().clip(0, 2)
        

        # Close: penalty only before contact; stops penalizing once fingers are on the object
        dist = self.obs_dist[:, :4].mean(dim=-1)
        rew_dist = self.cfg.w_rew_dist * dist.clip(0, 0.25) * (num_contact == 0).float()

        # Gripper open: penalty only when NOT in contact (don't penalize a successful grip)
        jaw = qc._as_tensor(self.robot.data.joint_pos)[:, self.left_joint_idx]
        # WidowX left_finger lower limit is 0.015 m; 0.018 catches "nearly fully closed".
        rew_open = self.cfg.w_rew_open * (jaw < 0.018).float() * (num_contact == 0).float()


        # XY alignment to target pose
        dist_xy = torch.norm(qc.root_pos_w(self.object)[:, :2] - self.data_pose[:, :2], dim=-1)
        rew_dist_xy = self.cfg.w_rew_dist_xy * torch.exp(-10 * dist_xy)

        # Wrist orientation: encourage gripper body +X (approach axis) pointing toward world -Z.
        # WidowX EEF approaches along body +X; penalize if it points too far from downward.
        wrist_direction = torch_utils.quat_apply(
            qc.body_quat_w(self.robot)[:, self.hand_link_idx],
            self.unit_x.unsqueeze(0).expand(self.num_envs, -1),
        )
        target_dir = -self.unit_z.unsqueeze(0).expand(self.num_envs, -1)
        cos_angle = (wrist_direction * target_dir).sum(dim=-1).clamp(-1.0, 1.0)
        wrist_angle = torch.acos(cos_angle)  # 0: body +X pointing down, pi: pointing up
        cos_vert = (wrist_direction * self.unit_z.unsqueeze(0).expand(self.num_envs, -1)).sum(dim=-1).abs().clamp(0.0, 1.0)
        side_score = 1.0 - cos_vert  # 1.0 when horizontal, 0.0 when vertical
        contact_gate = torch.where(
            self._upright_active,
            (side_score >= self.cfg.upright_contact_side_score_thres).float(),
            torch.ones_like(side_score),
        )
        rew_contact = rew_contact * contact_gate

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
        lift = (qc.root_pos_w(self.object)[:, 2] - self.data_pose[:, 2]).relu()
        self.max_lift = torch.maximum(self.max_lift, lift)

        # Lifted detection: height + contact
        obj_z = qc.root_pos_w(self.object)[:, 2]
        lifted_raw = (obj_z > self.lift_up_thres) & (num_contact > 0)

        # Side-approach reward for upright-active objects above a height threshold
        rew_side_approach = torch.zeros_like(obj_z)
        upright_active_rate = torch.zeros_like(obj_z)
        height_ok_rate = torch.zeros_like(obj_z)
        if self.cfg.enable_side_approach_reward:
            table_top_z = self.table_pose[:, 2] + 0.5 * self.table_size[2]
            height_ok = (obj_z - table_top_z) > self.cfg.side_approach_height_thres
            upright_active_rate = self._upright_active.float()
            height_ok_rate = height_ok.float()
            rew_side_approach = (
                self.cfg.w_rew_side_approach
                * side_score
                * height_ok.float()
                * self._upright_active.float()
            )

        # Pose-aware lifted: require good wrist orientation when lifted
        angle_thres = self.cfg.wrist_rot_margin / 180.0 * torch.pi
        good_orientation = wrist_angle < angle_thres
        lifted_pose = lifted_raw & good_orientation

        diff_xy = (qc.root_pos_w(self.object)[:, :2] - self.data_pose[:, :2]).norm(dim=-1)

        # Success pose constraints: object xy/orientation close to initial pose
        if self.cfg.threshold_decay:
            decay = 0.9 ** (max(self.common_step_counter - self.cfg.warmup_steps, 0) / 10000)
            pos_tol = max(self.cfg.success_xy_tol_min, self.cfg.success_xy_tol * decay)
        else:
            pos_tol = self.cfg.success_xy_tol

        xy_tol_effective = pos_tol
        xy_ok_for_success = diff_xy < xy_tol_effective

        # Lift reward gated by contact + xy tolerance; height-graded below lift_height
        xy_ok = diff_xy < xy_tol_effective
        lift_xy = (obj_z - self.data_pose[:, 2]).relu()
        rew_lift_xy = (
            self.cfg.w_rew_lift_xy
            * (lift_xy - self.max_lift_xy).relu()
            * (num_contact > 0).float()
            * xy_ok.float()
            # * upright_ok.float()
            # * good_orientation.float()
        )
        self.max_lift_xy = torch.maximum(self.max_lift_xy, lift_xy)

        # Lifted reward: reward raw lifting, give extra bonus for pose-correct lifting
        lifted_raw_xy = lifted_raw & xy_ok
        lifted_pose_xy = lifted_pose & xy_ok
        # rew_lifted = (
        #     self.cfg.w_rew_lifted_1 * (lifted_raw_xy & ~self.lifted & upright_ok).float()
        #     + self.cfg.w_rew_lifted_2 * (lifted_pose_xy & upright_ok).float()
        #     + self.cfg.w_rew_lifted_3 * (lifted_pose_xy & ~self.lifted & upright_ok).float()
        # )
        rew_lifted = (
            self.cfg.w_rew_lifted_1 * (lifted_raw_xy & ~self.lifted).float()
            + self.cfg.w_rew_lifted_2 * (lifted_pose_xy & upright_ok).float()
            + self.cfg.w_rew_lifted_3 * (lifted_pose_xy & ~self.lifted & upright_ok).float()
        )

        # Update lifted flag (tracks whether object has ever been lifted above the threshold)
        self.lifted[:] |= lifted_raw

        # Track consecutive lifted_pose regardless of xy/orientation constraints
        self.consecutive_lifted_pose[:] = torch.where(lifted_pose, self.consecutive_lifted_pose + 1, 0)
        self.consecutive_lifted_pose_xy[:] = torch.where(
            lifted_pose,
            self.consecutive_lifted_pose_xy + xy_ok_for_success.long(),
            0,
        )

        # Consecutive lift steps (used for hold fraction)
        self.consecutive_success[:] = torch.where(lifted_pose, self.consecutive_success + 1, 0)

        # Height stability around lift_up_thres once pose is correct
        # Encourage the object to stay close to the desired height when lifted_pose is true
        z_error = obj_z - self.lift_up_thres
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
        self.success = (self.consecutive_lifted_pose >= self.cfg.num_consecutive_success) & (
            xy_ratio >= self.cfg.lift_xy_success_ratio
        ) & upright_ok
        rew_success = rew_success_lifted + rew_success_pose
        rew_success = rew_success * self.success.float()

        # Velocity penalty
        # rew_action = self.cfg.w_rew_action * 500 ** min(1, self.common_step_counter / 10000) * (obj_speed + ee_speed)
        rew_action = self.cfg.w_rew_action * self.actions.norm(dim=-1)

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
            "rew_side_approach": rew_side_approach.mean(),
            "upright_active_rate": upright_active_rate.mean(),
            "height_ok_rate": height_ok_rate.mean(),
            "side_score_mean": side_score.mean(),
            "upright_ok_rate": upright_ok.float().mean(),
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
            + rew_side_approach
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
        lift_thres = self.data_pose[env_ids, 2] + self.cfg.lift_height
        self.lift_up_thres[env_ids] = torch.minimum(
            lift_thres, torch.tensor(self.cfg.lift_up_thres_max, device=self.device)
        )

        self.object.write_root_link_pose_to_sim(qc.pose_to_xyzw(self.data_pose[env_ids]), env_ids=env_ids)
        self.object.write_root_link_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)

        # reset hand: joint
        joint_pos = qc._as_tensor(self.robot.data.default_joint_pos)[env_ids]
        # Use canonical z (assumed center height) to shift table_z upward and avoid collisions.
        table_z_eff = self.table_pose[env_ids, 2] + 2.0 * data_pose_z_canonical
        joint_pos[:, :6] = self.initial_pose.sample(
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
        inv_object_quat, inv_object_pos = torch_utils.tf_inverse(
            qc.root_quat_w(self.object), qc.root_pos_w(self.object)
        )
        link_pos_obj_frame = torch_utils.tf_apply(
            inv_object_quat.unsqueeze(1).expand(-1, len(self.keypoint_indices), -1),
            inv_object_pos.unsqueeze(1),
            qc.body_pos_w(self.robot)[:, self.keypoint_indices, :],
        )
        distance = torch.cdist(link_pos_obj_frame, self.data_point_cloud)  # (num_envs, num_query, point_cloud_size)
        if self.cfg.upright_pcd_height_frac < 1.0:
            frac = max(0.0, float(self.cfg.upright_pcd_height_frac))
            pcd_z = self.data_point_cloud[..., 2]  # (num_envs, N)
            z_min = pcd_z.min(dim=1).values
            z_max = pcd_z.max(dim=1).values
            z_cut = z_min + frac * (z_max - z_min)
            allowed = pcd_z <= z_cut.unsqueeze(1)
            allowed = torch.where(self._upright_active.unsqueeze(1), allowed, torch.ones_like(allowed))
            allowed_any = allowed.any(dim=1)
            allowed = torch.where(allowed_any.unsqueeze(1), allowed, torch.ones_like(allowed))
            distance = distance + (~allowed).unsqueeze(1).to(distance.dtype) * 1e6
        self.obs_dist, min_indices = distance.min(dim=-1)
        closest_direction = (
            torch.gather(self.data_point_cloud, dim=1, index=min_indices.unsqueeze(-1).expand(-1, -1, 3))
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
