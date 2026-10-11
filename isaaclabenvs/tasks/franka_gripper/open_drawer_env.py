from isaaclabenvs.utils import math_compat as torch_utils  # WXYZ; replaces isaacsim.core.utils.torch (gone in Isaac Sim 6.0)
from isaaclabenvs.utils import quat_compat as qc
import torch

from isaaclabenvs.utils.articulation_dataset import JOINT_MOTION_PRISMATIC

from .open_base_env import OpenBaseEnv
from .open_drawer_env_cfg import OpenDrawerEnvCfg


class OpenDrawerEnv(OpenBaseEnv):
    cfg: OpenDrawerEnvCfg

    def _compute_handle_axis_q0(self) -> torch.Tensor:
        # Handle long axis in q0/object frame via PCA principal component.
        points = self.data_handle_points
        centered = points - points.mean(dim=1, keepdim=True)
        cov = torch.matmul(centered.transpose(1, 2), centered) / float(points.shape[1])
        eigvals, eigvecs = torch.linalg.eigh(cov)
        axis_q0 = eigvecs[..., -1]
        return axis_q0 / axis_q0.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _compute_drawer_goal_dir(self) -> torch.Tensor:
        prismatic_mask = self.data_joint_motion_type == JOINT_MOTION_PRISMATIC
        if not bool(prismatic_mask.all()):
            invalid = int((~prismatic_mask).sum().item())
            raise RuntimeError(f"OpenDrawerEnv expects prismatic-only assets, but found {invalid} non-prismatic samples.")
        axis_w = torch_utils.quat_apply(qc.root_quat_w(self.object), self.data_joint_axis_obj)
        axis_w = axis_w / axis_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        drawer_goal_dir = axis_w
        return drawer_goal_dir

    def _reward_force_alignment(self, target_dir: torch.Tensor) -> torch.Tensor:
        # Contact sensor reports force on fingers; use reaction force on object for alignment.
        force_w = qc._as_tensor(self.contact.data.net_forces_w).sum(dim=1)
        force_obj_w = -force_w
        dot = (force_obj_w * target_dir).sum(dim=-1)
        force_norm = force_obj_w.norm(dim=-1).clamp_min(1e-8)
        return dot / force_norm

    def _reward_wrist_direction_alignment(self, target_dir: torch.Tensor) -> torch.Tensor:
        wrist_dir = torch_utils.quat_apply(
            qc.body_quat_w(self.robot)[:, self.hand_link_idx],
            self.unit_z.unsqueeze(0).expand(self.num_envs, -1),
        )
        wrist_dir = wrist_dir / wrist_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        target_dir = target_dir / target_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (wrist_dir * target_dir).sum(dim=-1)

    def _reward_finger_line_perp_to_handle(self, handle_axis_w: torch.Tensor) -> torch.Tensor:
        left = qc.body_pos_w(self.robot)[:, self.left_finger_idx]
        right = qc.body_pos_w(self.robot)[:, self.right_finger_idx]
        finger_line = right - left
        finger_line = finger_line / finger_line.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        handle_axis = handle_axis_w / handle_axis_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos = (finger_line * handle_axis).sum(dim=-1).abs().clamp(0.0, 1.0)
        return 1.0 - cos

    def _compute_direction_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        drawer_goal_dir = self._compute_drawer_goal_dir()
        handle_axis_w = torch_utils.quat_apply(qc.root_quat_w(self.object), self._compute_handle_axis_q0())
        wrist_raw = self._reward_wrist_direction_alignment(-drawer_goal_dir).clamp(-1.0, 1.0)
        wrist_score = ((wrist_raw + 1.0) * 0.5).clamp(0.0, 1.0)
        finger_score = self._reward_finger_line_perp_to_handle(handle_axis_w).clamp(0.0, 1.0)
        return wrist_score, finger_score

    def _get_observations(self) -> dict:
        base_obs = super()._get_observations()["policy"]
        drawer_goal_dir = self._compute_drawer_goal_dir()
        obs = torch.cat((base_obs, drawer_goal_dir), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        drawer_goal_dir = self._compute_drawer_goal_dir()
        handle_axis_w = torch_utils.quat_apply(qc.root_quat_w(self.object), self._compute_handle_axis_q0())
        min_handle_dist = self.obs_dist[:, :4].min(dim=-1).values
        pose_gate = torch.exp(-float(self.cfg.rew_pose_align_dist_alpha) * min_handle_dist)
        rew_force_align = self.cfg.w_rew_force_align * self._reward_force_alignment(drawer_goal_dir)
        rew_wrist_dir_align = (
            self.cfg.w_rew_wrist_dir_align * self._reward_wrist_direction_alignment(-drawer_goal_dir) * pose_gate
        )
        rew_finger_perp_align = (
            self.cfg.w_rew_finger_perp_align * self._reward_finger_line_perp_to_handle(handle_axis_w) * pose_gate
        )
        self.extras["rew_force_align"] = rew_force_align.mean()
        self.extras["rew_wrist_dir_align"] = rew_wrist_dir_align.mean()
        self.extras["rew_finger_perp_align"] = rew_finger_perp_align.mean()
        self.extras["pose_align_gate"] = pose_gate.mean()
        print(
            f"rew_force_align: {self.extras['rew_force_align'].item():.4f}; "
            f"rew_wrist_dir_align: {self.extras['rew_wrist_dir_align'].item():.4f}; "
            f"rew_finger_perp_align: {self.extras['rew_finger_perp_align'].item():.4f}; "
            f"pose_align_gate: {self.extras['pose_align_gate'].item():.4f}"
        )
        return reward + rew_force_align + rew_wrist_dir_align + rew_finger_perp_align
