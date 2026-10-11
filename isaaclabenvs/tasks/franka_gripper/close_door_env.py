from isaaclabenvs.utils import math_compat as torch_utils  # WXYZ; replaces isaacsim.core.utils.torch (gone in Isaac Sim 6.0)
from isaaclabenvs.utils import quat_compat as qc
import torch

from isaaclabenvs.utils.articulation_dataset import JOINT_MOTION_REVOLUTE

from .close_door_env_cfg import CloseDoorEnvCfg
from .close_env import CloseEnv


class CloseDoorEnv(CloseEnv):
    cfg: CloseDoorEnvCfg

    def _sample_object_yaw_theta(self, obj_y_local: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        theta = super()._sample_object_yaw_theta(obj_y_local, env_ids)
        n = obj_y_local.shape[0]
        is_revolute = self.data_joint_motion_type[env_ids] == JOINT_MOTION_REVOLUTE

        away_prob = float(self.cfg.yaw_away_prob)
        if away_prob > 0.0:
            flip = (torch.rand((n,), device=self.device) < away_prob) & is_revolute
            theta = torch.where(flip, -theta, theta)

        jitter_rad = float(self.cfg.yaw_jitter_rad)
        if jitter_rad > 0.0:
            jitter = (torch.rand((n,), device=self.device) * 2.0 - 1.0) * jitter_rad
            jitter = torch.where(is_revolute, jitter, torch.zeros_like(jitter))
            theta = theta + jitter
        return theta

    def _compute_direction_terms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns (axis_w, radial_dir, open_tangent_dir, lever_w) in world frame.
        # axis_w / radial_dir / lever_w are direction-neutral geometry; open_tangent points along
        # handle motion when the joint OPENS (caller flips for close-direction obs).
        revolute_mask = self.data_joint_motion_type == JOINT_MOTION_REVOLUTE
        if not bool(revolute_mask.all()):
            invalid = int((~revolute_mask).sum().item())
            raise RuntimeError(f"CloseDoorEnv expects revolute-only assets, but found {invalid} non-revolute samples.")

        root_quat_w = qc.root_quat_w(self.object)
        root_pos_w = qc.root_pos_w(self.object)
        axis_w = torch_utils.quat_apply(root_quat_w, self.data_joint_axis_obj)
        axis_w = axis_w / axis_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        movable_center_cur_w = self._q0_point_to_world(self.data_movable_center_q0)
        hinge_w = torch_utils.quat_apply(root_quat_w, self.data_joint_origin_obj) + root_pos_w

        radial_w = movable_center_cur_w - hinge_w
        radial_w_proj = radial_w - (radial_w * axis_w).sum(dim=-1, keepdim=True) * axis_w
        radial_dir = radial_w_proj / radial_w_proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        tangent_w = torch.cross(axis_w, radial_dir, dim=-1)
        open_tangent_dir = tangent_w / tangent_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return axis_w, radial_dir, open_tangent_dir, radial_w_proj

    def _reward_force_alignment(self, axis_w: torch.Tensor, lever_w: torch.Tensor) -> torch.Tensor:
        # Reaction torque around hinge axis induced by finger contact force; negated so positive
        # reward corresponds to torque in the CLOSE direction.
        force_finger_w = qc._as_tensor(self.contact.data.net_forces_w).sum(dim=1)
        force_obj_w = -force_finger_w
        torque_w = torch.cross(lever_w, force_obj_w, dim=-1)
        torque_axis = (torque_w * axis_w).sum(dim=-1)
        lever_norm = lever_w.norm(dim=-1).clamp_min(1e-8)
        force_norm = force_obj_w.norm(dim=-1).clamp_min(1e-8)
        return -torque_axis / (lever_norm * force_norm)

    def _get_observations(self) -> dict:
        base_obs = super()._get_observations()["policy"]
        axis_w, radial_dir, open_tangent_dir, _ = self._compute_direction_terms()
        close_tangent_dir = -open_tangent_dir
        obs = torch.cat((base_obs, axis_w, radial_dir, close_tangent_dir), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        axis_w, _, _, lever_w = self._compute_direction_terms()
        rew_force_align = self.cfg.w_rew_force_align * self._reward_force_alignment(axis_w, lever_w)
        self.extras["rew_force_align"] = rew_force_align.mean()
        print(f"rew_force_align: {self.extras['rew_force_align'].item():.4f}")
        return reward + rew_force_align
