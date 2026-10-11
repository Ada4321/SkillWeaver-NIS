from __future__ import annotations

import enum

import torch

from .articulation_open_close_env import ArticulatedOpenCloseEnv
from .close_env_cfg import CloseEnvCfg


class TaskStatus(enum.IntEnum):
    SUCCESS = enum.auto()
    FAILURE = enum.auto()
    TIMEOUT = enum.auto()


class CloseEnv(ArticulatedOpenCloseEnv):
    cfg: CloseEnvCfg

    def __init__(self, cfg: CloseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Close-specific query points: four finger links + grasp site + hand.
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
            "closed_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "arm_max_abs_tracking_error": torch.zeros(1, dtype=torch.float, device=self.device),
        }
        self.latest_status = torch.full((self.num_envs,), TaskStatus.TIMEOUT, dtype=torch.int, device=self.device)
        self.latest_closed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.consecutive_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.min_close_pos_norm = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.closed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _load_dataset(self):
        super()._load_dataset()
        if self.data_link_forward_points is None:
            raise RuntimeError(
                "CloseEnv requires link_forward_points; configure dataset toml "
                "(link_forward_points + link_forward_points_key)."
            )
        merged = torch.cat([self.data_handle_points, self.data_link_forward_points], dim=1)
        self.data_query_points = merged
        self.data_point_cloud = merged
        # Per-env movable_link centroid in canonical obj frame (joint=0); transformed to current
        # world frame each step via _q0_point_to_world for use as the obs target reference.
        self.data_movable_center_q0 = self.data_link_forward_points.mean(dim=1)

    def _compute_obs_contact(self) -> torch.Tensor:
        if len(self._handle_contact_sensors) != 4:
            raise RuntimeError(
                "Close tasks require 4 handle contact sensors "
                "(left, left_tip, right, right_tip)."
            )
        # Contact columns follow the same order as keypoints[:4] = [left, right, left_tip, right_tip].
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
                self._q0_point_to_world(self.data_movable_center_q0) - self.scene.env_origins,
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

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_buffer()
        self.failure[:] = False

        # Optional regress-to-fail: terminate as FAILURE if the drawer is pulled back OPEN more than
        # regress_fail_norm from its closest-to-closed peak. min_close_pos_norm only shrinks (and is
        # reset to the start opening), so this can only fire after the drawer has closed that much.
        # 0 = disabled. (min_close_pos_norm is from the previous step's _get_rewards -- 1-step stale.)
        if float(self.cfg.regress_fail_norm) > 0.0:
            regressed = (self.object_joint_pos_norm - self.min_close_pos_norm) > float(self.cfg.regress_fail_norm)
            self.failure |= regressed

        joint_target_reached = self.object_joint_pos_norm <= self.goal_joint_pos_norm
        self.consecutive_success[:] = torch.where(
            joint_target_reached,
            self.consecutive_success + 1,
            torch.zeros_like(self.consecutive_success),
        )
        self.success = self.consecutive_success >= int(self.cfg.num_consecutive_success)


        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self.failure | self.success, truncated

    def _close_reward_gate(self) -> torch.Tensor:
        """Per-env multiplier (in [0,1]) applied to the closing rewards (rew_close/rew_closed).

        Default 1.0 -> ungated, so existing close tasks are unaffected. Subclasses override to require
        the gripper to be engaged with the drawer (e.g. push-to-close gates on gripper-near-drawer so
        the policy is not paid for closing the drawer while flying off it).
        """
        return torch.ones(self.num_envs, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        num_contact = ((self.obs_contact > 0) & (self.obs_dist[:, :4] < 0.02)).sum(dim=-1)
        rew_contact = self.cfg.w_rew_contact * num_contact.float().clip(0, 2)
        dist = self.obs_dist[:, :4].mean(dim=-1)
        rew_dist = self.cfg.w_rew_dist * dist.clip(0, 0.25)

        close_pos = self.object_joint_pos_norm
        close_delta = (self.min_close_pos_norm - close_pos).relu()
        # Optional engagement gate (default 1.0). Gates the reward VALUE only -- progress tracking
        # (min_close_pos_norm) below is unchanged so the best-closed peak is always remembered.
        close_gate = self._close_reward_gate()
        rew_close = self.cfg.w_rew_close * close_delta * close_gate
        self.min_close_pos_norm = torch.minimum(self.min_close_pos_norm, close_pos)

        closed_now = close_pos <= self.goal_joint_pos_norm
        rew_closed = (
            self.cfg.w_rew_closed_once * (closed_now & (~self.closed)).float()
            + self.cfg.w_rew_closed_hold * closed_now.float()
        ) * close_gate
        self.closed[:] |= closed_now
        rew_success = self.cfg.w_rew_success * self.success.float()

        self.latest_status[self.success] = int(TaskStatus.SUCCESS)
        self.latest_status[self.failure] = int(TaskStatus.FAILURE)
        self.latest_status[self.reset_time_outs] = int(TaskStatus.TIMEOUT)

        self.latest_closed[self.reset_buf] = self.closed[self.reset_buf]
        self.latest_closed[:] |= self.closed

        joint_error_abs = self.object_joint_error_norm.abs()
        target_reached = self.object_joint_pos_norm <= self.goal_joint_pos_norm

        if self.reset_buf.any():
            self.statistics["success_rate"] = (self.latest_status == int(TaskStatus.SUCCESS)).float().mean()
            self.statistics["timeout_rate"] = (self.latest_status == int(TaskStatus.TIMEOUT)).float().mean()
            self.statistics["failure_rate"] = (self.latest_status == int(TaskStatus.FAILURE)).float().mean()

        self.statistics["closed_rate"] = self.latest_closed.float().mean()
        self.statistics["joint_target_reached_rate"] = target_reached.float().mean()
        self.statistics["arm_max_abs_tracking_error"] = self._compute_arm_max_abs_tracking_error()
        if self._should_print():
            print({k: v.item() for k, v in self.statistics.items()})

        reward = rew_contact + rew_dist + rew_close + rew_closed + rew_success

        rewards = {
            "rew_dist": rew_dist.mean(),
            "rew_contact": rew_contact.mean(),
            "rew_close": rew_close.mean(),
            "rew_closed": rew_closed.mean(),
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

        self._refresh_buffer()
        self.min_close_pos_norm[env_ids] = self.object_joint_pos_norm[env_ids]
        self.closed[env_ids] = False
        self.consecutive_success[env_ids] = 0
        # NOTE: do NOT reset latest_status to TIMEOUT here -- the terminal step already recorded the
        # real outcome (SUCCESS/FAILURE/TIMEOUT); overwriting it on reset made success_rate a per-step
        # rate (~episode-length x too small) instead of the last-episode outcome. Keep last terminal.
