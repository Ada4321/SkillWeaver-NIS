from __future__ import annotations

import enum
import math

import torch

from .articulation_open_close_env import ArticulatedOpenCloseEnv
from .open_env_cfg import OpenEnvCfg


class TaskStatus(enum.IntEnum):
    SUCCESS = enum.auto()
    FAILURE = enum.auto()
    TIMEOUT = enum.auto()


class OpenBaseEnv(ArticulatedOpenCloseEnv):
    cfg: OpenEnvCfg

    def __init__(self, cfg: OpenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Open-specific query points: four finger links + grasp site + hand.
        self.keypoint_indices = [
            self.left_finger_idx,
            self.right_finger_idx,
            self.left_tip_idx,
            self.right_tip_idx,
            self.grasp_site_idx,
            self.hand_link_idx,
        ]

        self.statistics = {
            "success_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "failure_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "timeout_rate": torch.ones(1, dtype=torch.float, device=self.device),
            "joint_target_reached_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "opened_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "arm_max_abs_tracking_error": torch.zeros(1, dtype=torch.float, device=self.device),
        }
        self.latest_status = torch.full((self.num_envs,), TaskStatus.TIMEOUT, dtype=torch.int, device=self.device)
        self.latest_opened = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.consecutive_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_joint_error_abs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.max_open_pos_norm = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.opened = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Per-env handle centroid in canonical obj frame (joint=0); transformed to current world
        # frame each step via _q0_point_to_world for use as the obs target reference.
        self.data_handle_center_q0 = self.data_handle_points.mean(dim=1)

    def _compute_obs_contact(self) -> torch.Tensor:
        if len(self._handle_contact_sensors) != 4:
            raise RuntimeError(
                "Open tasks require 4 handle contact sensors "
                "(left, left_tip, right, right_tip)."
            )
        # Keep single-sensor behavior in logic: contact columns follow the same order
        # as distance query keypoints [:4] = [left, right, left_tip, right_tip].
        ordered_sensors = (
            self.contact_left_handle,
            self.contact_right_handle,
            self.contact_left_tip_handle,
            self.contact_right_tip_handle,
        )
        contact_cols = []
        for sensor in ordered_sensors:
            force = sensor.data.net_forces_w
            mag = force.norm(dim=-1).reshape(self.num_envs, -1)
            contact_cols.append(mag.max(dim=-1).values > 0.01)
        return torch.stack(contact_cols, dim=-1)

    def _get_observations(self) -> dict:

        obs = torch.cat(
            (
                self.robot_joint_pos,
                self.robot.data.joint_vel * self.cfg.dof_velocity_scale,
                self.robot.data.joint_pos_target,
                self.wrist_state,
                self.robot.data.body_pos_w[:, self.left_finger_idx] - self.scene.env_origins,
                self.robot.data.body_pos_w[:, self.right_finger_idx] - self.scene.env_origins,
                self.robot.data.body_pos_w[:, self.grasp_site_idx] - self.scene.env_origins,
                self._q0_point_to_world(self.data_handle_center_q0) - self.scene.env_origins,
                self.obs_dist,
                self.obs_closest_point.reshape(self.num_envs, -1),
                self.obs_contact,
                self.object_joint_pos_norm.unsqueeze(-1),
                self.goal_joint_pos_norm.unsqueeze(-1),
                self.object_joint_error_norm.unsqueeze(-1),
                self.object_joint_vel_norm.unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _compute_direction_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return wrist/finger direction scores in [0, 1] for gating.

        Base open env has no task-specific direction target, so default to pass-through.
        """
        ones = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        return ones, ones

    def _direction_thresholds_from_angles(
        self, wrist_max_angle_deg: float, finger_perp_max_angle_deg: float
    ) -> tuple[float, float]:
        # wrist_score = (dot + 1) / 2, dot = cos(theta)
        wrist_deg = max(0.0, min(180.0, float(wrist_max_angle_deg)))
        wrist_score_min = 0.5 * (1.0 + math.cos(math.radians(wrist_deg)))
        # finger_score = 1 - |cos(phi)|, phi target is 90deg, |phi-90| <= delta
        # => |cos(phi)| <= sin(delta), so finger_score >= 1 - sin(delta)
        finger_deg = max(0.0, min(90.0, float(finger_perp_max_angle_deg)))
        finger_score_min = 1.0 - math.sin(math.radians(finger_deg))
        return wrist_score_min, finger_score_min

    def _compute_direction_gate(
        self, require_direction: bool, wrist_max_angle_deg: float, finger_perp_max_angle_deg: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wrist_score, finger_score = self._compute_direction_scores()
        if bool(require_direction):
            wrist_min, finger_min = self._direction_thresholds_from_angles(
                wrist_max_angle_deg,
                finger_perp_max_angle_deg,
            )
            gate = (wrist_score >= wrist_min) & (finger_score >= finger_min)
        else:
            gate = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return gate, wrist_score, finger_score

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_buffer()
        self.failure[:] = False

        # Optional regress-to-fail: terminate as FAILURE if the drawer falls back more than
        # regress_fail_norm (normalized) from its peak opening. max_open_pos_norm only grows (and is
        # reset to the start opening), so this can only fire after the drawer has opened that much.
        # 0 = disabled. (max_open_pos_norm is from the previous step's _get_rewards -- 1-step stale,
        # negligible. Grasped subclass ORs its grasp_lost fail on top via super().)
        if float(self.cfg.regress_fail_norm) > 0.0:
            regressed = (self.max_open_pos_norm - self.object_joint_pos_norm) > float(self.cfg.regress_fail_norm)
            self.failure |= regressed

        joint_target_reached = self.object_joint_pos_norm >= self.goal_joint_pos_norm
        success_dir_gate, _, _ = self._compute_direction_gate(
            self.cfg.success_require_direction,
            self.cfg.success_direction_wrist_max_angle_deg,
            self.cfg.success_direction_finger_perp_max_angle_deg,
        )
        target_reached = joint_target_reached & success_dir_gate

        self.consecutive_success[:] = torch.where(
            target_reached,
            self.consecutive_success + 1,
            torch.zeros_like(self.consecutive_success),
        )
        self.success = self.consecutive_success >= int(self.cfg.num_consecutive_success)


        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self.failure | self.success, truncated

    def _compute_num_contact_and_pinch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Validated-contact count and two-sided pinch flag.

        Columns of obs_dist/obs_contact: [0]=left_finger, [1]=right_finger, [2]=left_tip,
        [3]=right_tip. When cfg.contact_use_tips_only is True, restrict both to the two
        fingertip columns (better suited to thin handles grasped at the tips); otherwise use all
        four finger/tip keypoints (original behavior).
        """
        if bool(self.cfg.contact_use_tips_only):
            left_idx, right_idx = 2, 3  # left_tip, right_tip
            cols = [left_idx, right_idx]
            num_contact = ((self.obs_contact[:, cols] > 0) & (self.obs_dist[:, cols] < 0.02)).sum(dim=-1)
            left_contact = self.obs_contact[:, left_idx]
            right_contact = self.obs_contact[:, right_idx]
            left_near = self.obs_dist[:, left_idx] < 0.02
            right_near = self.obs_dist[:, right_idx] < 0.02
        else:
            num_contact = ((self.obs_contact > 0) & (self.obs_dist[:, :4] < 0.02)).sum(dim=-1)
            left_contact = self.obs_contact[:, 0] | self.obs_contact[:, 2]
            right_contact = self.obs_contact[:, 1] | self.obs_contact[:, 3]
            left_near = torch.minimum(self.obs_dist[:, 0], self.obs_dist[:, 2]) < 0.02
            right_near = torch.minimum(self.obs_dist[:, 1], self.obs_dist[:, 3]) < 0.02
        pinch = left_contact & right_contact & left_near & right_near
        return num_contact, pinch

    def _get_rewards(self) -> torch.Tensor:
        # Same as pick_upright.
        num_contact, pinch = self._compute_num_contact_and_pinch()
        rew_contact = self.cfg.w_rew_contact * num_contact.float().clip(0, 2)
        dist = self.obs_dist[:, :4].mean(dim=-1)
        rew_dist = self.cfg.w_rew_dist * dist.clip(0, 0.25)
        # rew_jaw_open disabled (judged unhelpful): penalized a fully-closed empty jaw.
        # jaw = self.robot.data.joint_pos[:, self.left_joint_idx]
        # rew_jaw_open = self.cfg.w_rew_jaw_open * (jaw < float(self.cfg.jaw_close_thres)).float()

        rew_pinch = self.cfg.w_rew_pinch * pinch.float()
        rew_contact_pinch_bonus = self.cfg.w_rew_contact_pinch_bonus * num_contact.float().clip(0, 2) * pinch.float()

        rew_dir_gate, rew_wrist_score, rew_finger_score = self._compute_direction_gate(
            self.cfg.rew_open_require_direction,
            self.cfg.rew_open_direction_wrist_max_angle_deg,
            self.cfg.rew_open_direction_finger_perp_max_angle_deg,
        )
        success_dir_gate, success_wrist_score, success_finger_score = self._compute_direction_gate(
            self.cfg.success_require_direction,
            self.cfg.success_direction_wrist_max_angle_deg,
            self.cfg.success_direction_finger_perp_max_angle_deg,
        )

        open_pos = self.object_joint_pos_norm
        open_delta = (open_pos - self.max_open_pos_norm).relu()
        if bool(self.cfg.rew_open_require_pinch):
            open_gate = pinch & rew_dir_gate
            # open_gate = rew_dir_gate
            rew_open = self.cfg.w_rew_open * open_delta * open_gate.float()
            self.max_open_pos_norm = torch.where(
                open_gate,
                torch.maximum(self.max_open_pos_norm, open_pos),
                self.max_open_pos_norm,
            )
        else:
            open_gate = rew_dir_gate
            pinch_scale = 1.0 + float(self.cfg.rew_open_pinch_bonus) * pinch.float()
            rew_open = self.cfg.w_rew_open * open_delta * pinch_scale * open_gate.float()
            self.max_open_pos_norm = torch.maximum(self.max_open_pos_norm, open_pos)
        opened_now = open_pos >= self.goal_joint_pos_norm
        if bool(self.cfg.rew_open_require_pinch):
            # Only credit "opened" (once + hold) while genuinely pinching, so ramming the drawer
            # open and letting go does not earn the bonus.
            opened_now = opened_now & pinch
        rew_opened = (
            self.cfg.w_rew_opened_once * (opened_now & (~self.opened)).float()
            + self.cfg.w_rew_opened_hold * opened_now.float()
        )
        self.opened[:] |= opened_now
        # Same success bonus design as pose_env: reward earlier success more.
        # rew_success = 25 * self.success.float() * (20 + (self.max_episode_length - self.episode_length_buf).float())
        rew_success = self.cfg.w_rew_success * self.success.float()

        # Align statistics bookkeeping style with pose_env.
        self.latest_status[self.success] = int(TaskStatus.SUCCESS)
        self.latest_status[self.failure] = int(TaskStatus.FAILURE)
        self.latest_status[self.reset_time_outs] = int(TaskStatus.TIMEOUT)

        self.latest_opened[self.reset_buf] = self.opened[self.reset_buf]
        self.latest_opened[:] |= self.opened

        joint_error_abs = self.object_joint_error_norm.abs()
        target_reached = self.object_joint_pos_norm >= self.goal_joint_pos_norm

        if self.reset_buf.any():
            self.statistics["success_rate"] = (self.latest_status == int(TaskStatus.SUCCESS)).float().mean()
            self.statistics["timeout_rate"] = (self.latest_status == int(TaskStatus.TIMEOUT)).float().mean()
            self.statistics["failure_rate"] = (self.latest_status == int(TaskStatus.FAILURE)).float().mean()

        self.statistics["opened_rate"] = self.latest_opened.float().mean()
        self.statistics["joint_target_reached_rate"] = target_reached.float().mean()
        self.statistics["arm_max_abs_tracking_error"] = self._compute_arm_max_abs_tracking_error()
        if self._should_print():
            print({k: v.item() for k, v in self.statistics.items()})

        reward = rew_contact + rew_contact_pinch_bonus + rew_pinch + rew_dist + rew_open + rew_opened + rew_success  # + rew_jaw_open (disabled)

        rewards = {
            "rew_dist": rew_dist.mean(),
            "rew_contact": rew_contact.mean(),
            "rew_contact_pinch_bonus": rew_contact_pinch_bonus.mean(),
            "rew_pinch": rew_pinch.mean(),
            # "rew_jaw_open": rew_jaw_open.mean(),  # disabled
            "rew_open": rew_open.mean(),
            "pinch_rate": pinch.float().mean(),
            "rew_open_gate_rate": open_gate.float().mean(),
            "rew_open_dir_gate_rate": rew_dir_gate.float().mean(),
            "success_dir_gate_rate": success_dir_gate.float().mean(),
            "rew_open_wrist_score": rew_wrist_score.mean(),
            "rew_open_finger_score": rew_finger_score.mean(),
            "success_wrist_score": success_wrist_score.mean(),
            "success_finger_score": success_finger_score.mean(),
            "rew_opened": rew_opened.mean(),
            "rew_success": rew_success.mean(),
            "joint_error_abs": joint_error_abs.mean(),
            "joint_pos_norm": self.object_joint_pos_norm.mean(),
            "joint_goal_norm": self.goal_joint_pos_norm.mean(),
        }
        self.extras.update(rewards)
        self.extras.update(self.statistics)
        if self._should_print():
            print("; ".join(f"{k}: {v:.4f}" for k, v in rewards.items()))
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return

        target_xyz = self._compute_init_arm_query_xyz(self.data_query_points, env_ids)
        sampled_arm_joint_pos = self._sample_init_arm_joint_pos(env_ids, target_xyz)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        if sampled_arm_joint_pos.shape[-1] == len(self.arm_joint_indices):
            joint_pos[:, self.arm_joint_indices] = sampled_arm_joint_pos
        elif sampled_arm_joint_pos.shape[-1] == joint_pos.shape[-1]:
            joint_pos[:] = sampled_arm_joint_pos
        else:
            raise RuntimeError(
                "init_arm_arti.npz data dof mismatch. "
                f"Expected arm dof {len(self.arm_joint_indices)} or robot dof {joint_pos.shape[-1]}, "
                f"got {sampled_arm_joint_pos.shape[-1]}."
            )
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        start_min, start_max = self.cfg.joint_start_pos_range_norm
        start_min = float(start_min)
        start_max = float(start_max)
        if start_min > start_max:
            start_min, start_max = start_max, start_min
        start_norm = torch.rand((len(env_ids),), device=self.device) * (start_max - start_min) + start_min

        object_joint_pos = self.object.data.default_joint_pos[env_ids].clone()
        object_joint_vel = torch.zeros_like(object_joint_pos)
        sampled_joint_pos = self.object_joint_lower[env_ids] + start_norm * self.object_joint_denom[env_ids]
        object_joint_pos[:, self.object_joint_idx] = sampled_joint_pos
        self.object.write_joint_state_to_sim(object_joint_pos, object_joint_vel, env_ids=env_ids)

        self._refresh_buffer()
        self.max_open_pos_norm[env_ids] = self.object_joint_pos_norm[env_ids]
        self.opened[env_ids] = False
        self.prev_joint_error_abs[env_ids] = self.object_joint_error_norm[env_ids].abs()
        self.consecutive_success[env_ids] = 0
        # NOTE: do NOT reset latest_status to TIMEOUT here. The terminal step already wrote the real
        # outcome (SUCCESS/FAILURE/TIMEOUT) in _get_rewards; overwriting it on reset made latest_status
        # hold TIMEOUT for every ongoing env, so success_rate/failure_rate only counted the single step
        # an env terminated (per-step, ~L x too small) instead of the last-episode outcome (per-episode,
        # comparable to opened_rate). latest_status keeps the last terminal result across the episode.
