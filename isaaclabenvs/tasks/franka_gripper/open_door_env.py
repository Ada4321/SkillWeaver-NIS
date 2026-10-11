from isaaclabenvs.utils import math_compat as torch_utils  # WXYZ; replaces isaacsim.core.utils.torch (gone in Isaac Sim 6.0)
from isaaclabenvs.utils import quat_compat as qc
import torch

from isaaclabenvs.utils.articulation_dataset import JOINT_MOTION_REVOLUTE

from .open_base_env import OpenBaseEnv
from .open_door_env_cfg import OpenDoorEnvCfg


class OpenDoorEnv(OpenBaseEnv):
    cfg: OpenDoorEnvCfg

    def _compute_handle_axis_q0(self) -> torch.Tensor:
        # Handle long axis in q0/object frame via PCA principal component.
        points = self.data_handle_points
        centered = points - points.mean(dim=1, keepdim=True)
        cov = torch.matmul(centered.transpose(1, 2), centered) / float(points.shape[1])
        eigvals, eigvecs = torch.linalg.eigh(cov)
        axis_q0 = eigvecs[..., -1]
        return axis_q0 / axis_q0.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _compute_direction_terms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        root_quat_w = qc.root_quat_w(self.object)
        root_pos_w = qc.root_pos_w(self.object)

        revolute_mask = self.data_joint_motion_type == JOINT_MOTION_REVOLUTE
        if not bool(revolute_mask.all()):
            invalid = int((~revolute_mask).sum().item())
            raise RuntimeError(f"OpenDoorEnv expects revolute-only assets, but found {invalid} non-revolute samples.")

        axis_w = torch_utils.quat_apply(root_quat_w, self.data_joint_axis_obj)
        axis_w = axis_w / axis_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        handle_center_q0 = self.data_handle_points.mean(dim=1)
        joint_delta = self.object.data.joint_pos[:, self.object_joint_idx] - self.object_joint_lower
        handle_center_cur_obj = handle_center_q0.clone()
        origin = self.data_joint_origin_obj
        vec_from_hinge = handle_center_cur_obj - origin
        vec_rot = self._rodrigues_rotate(
            vec_from_hinge.unsqueeze(1),
            self.data_joint_axis_obj,
            joint_delta,
        ).squeeze(1)
        handle_center_cur_obj = vec_rot + origin

        handle_center_cur_w = torch_utils.quat_apply(root_quat_w, handle_center_cur_obj) + root_pos_w
        hinge_w = torch_utils.quat_apply(root_quat_w, self.data_joint_origin_obj) + root_pos_w

        radial_w = handle_center_cur_w - hinge_w
        radial_w_proj = radial_w - (radial_w * axis_w).sum(dim=-1, keepdim=True) * axis_w
        radial_dir = radial_w_proj / radial_w_proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        tangent_w = torch.cross(axis_w, radial_dir, dim=-1)
        tangent_dir = tangent_w / tangent_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return axis_w, radial_dir, tangent_dir, radial_w_proj

    def _reward_force_alignment(self, axis_w: torch.Tensor, lever_w: torch.Tensor) -> torch.Tensor:
        # Use torque around hinge axis induced by object reaction force at the handle.
        force_finger_w = self.contact.data.net_forces_w.sum(dim=1)
        force_obj_w = -force_finger_w
        torque_w = torch.cross(lever_w, force_obj_w, dim=-1)
        torque_axis = (torque_w * axis_w).sum(dim=-1)
        lever_norm = lever_w.norm(dim=-1).clamp_min(1e-8)
        force_norm = force_obj_w.norm(dim=-1).clamp_min(1e-8)
        return torque_axis / (lever_norm * force_norm)

    def _reward_wrist_direction_alignment(self, target_dir: torch.Tensor) -> torch.Tensor:
        wrist_dir = torch_utils.quat_apply(
            qc.body_quat_w(self.robot)[:, self.hand_link_idx],
            self.unit_z.unsqueeze(0).expand(self.num_envs, -1),
        )
        wrist_dir = wrist_dir / wrist_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        target_dir = target_dir / target_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (wrist_dir * target_dir).sum(dim=-1)

    def _reward_finger_line_perp_to_handle(self, handle_axis_w: torch.Tensor) -> torch.Tensor:
        left = self.robot.data.body_pos_w[:, self.left_finger_idx]
        right = self.robot.data.body_pos_w[:, self.right_finger_idx]
        finger_line = right - left
        finger_line = finger_line / finger_line.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        handle_axis = handle_axis_w / handle_axis_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos = (finger_line * handle_axis).sum(dim=-1).abs().clamp(0.0, 1.0)
        return 1.0 - cos

    def _compute_direction_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        axis_w, _, tangent_dir, _ = self._compute_direction_terms()
        joint_delta = self.object.data.joint_pos[:, self.object_joint_idx] - self.object_joint_lower
        handle_axis_q0 = self._compute_handle_axis_q0()
        handle_axis_cur_obj = self._rodrigues_rotate(
            handle_axis_q0.unsqueeze(1),
            self.data_joint_axis_obj,
            joint_delta,
        ).squeeze(1)
        handle_axis_w = torch_utils.quat_apply(qc.root_quat_w(self.object), handle_axis_cur_obj)
        wrist_raw = self._reward_wrist_direction_alignment(-tangent_dir).clamp(-1.0, 1.0)
        wrist_score = ((wrist_raw + 1.0) * 0.5).clamp(0.0, 1.0)
        finger_score = self._reward_finger_line_perp_to_handle(handle_axis_w).clamp(0.0, 1.0)
        return wrist_score, finger_score

    def _get_observations(self) -> dict:
        base_obs = super()._get_observations()["policy"]
        axis_w, radial_dir, tangent_dir, _ = self._compute_direction_terms()

        obs = torch.cat((base_obs, axis_w, radial_dir, tangent_dir), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        axis_w, _, tangent_dir, lever_w = self._compute_direction_terms()
        joint_delta = self.object.data.joint_pos[:, self.object_joint_idx] - self.object_joint_lower
        handle_axis_q0 = self._compute_handle_axis_q0()
        handle_axis_cur_obj = self._rodrigues_rotate(
            handle_axis_q0.unsqueeze(1),
            self.data_joint_axis_obj,
            joint_delta,
        ).squeeze(1)
        handle_axis_w = torch_utils.quat_apply(qc.root_quat_w(self.object), handle_axis_cur_obj)
        min_handle_dist = self.obs_dist[:, :4].min(dim=-1).values
        pose_gate = torch.exp(-float(self.cfg.rew_pose_align_dist_alpha) * min_handle_dist)
        rew_force_align = self.cfg.w_rew_force_align * self._reward_force_alignment(axis_w, lever_w)
        rew_wrist_dir_align = (
            self.cfg.w_rew_wrist_dir_align * self._reward_wrist_direction_alignment(-tangent_dir) * pose_gate
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
