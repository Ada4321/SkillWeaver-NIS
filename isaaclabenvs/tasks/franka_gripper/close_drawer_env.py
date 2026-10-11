from isaaclabenvs.utils import math_compat as torch_utils  # WXYZ; replaces isaacsim.core.utils.torch (gone in Isaac Sim 6.0)
from isaaclabenvs.utils import quat_compat as qc
import torch

from isaaclabenvs.utils.articulation_dataset import JOINT_MOTION_PRISMATIC

from .close_drawer_env_cfg import CloseDrawerEnvCfg
from .close_env import CloseEnv


class CloseDrawerEnv(CloseEnv):
    cfg: CloseDrawerEnvCfg

    def _compute_drawer_pull_dir(self) -> torch.Tensor:
        # Pull-out direction in world frame (= open direction for a drawer).
        prismatic_mask = self.data_joint_motion_type == JOINT_MOTION_PRISMATIC
        if not bool(prismatic_mask.all()):
            invalid = int((~prismatic_mask).sum().item())
            raise RuntimeError(f"CloseDrawerEnv expects prismatic-only assets, but found {invalid} non-prismatic samples.")
        axis_w = torch_utils.quat_apply(qc.root_quat_w(self.object), self.data_joint_axis_obj)
        return axis_w / axis_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _reward_force_alignment(self, pull_dir: torch.Tensor) -> torch.Tensor:
        # Finger reaction force projected onto pull direction; negated so positive reward
        # corresponds to force in the CLOSE (push-in) direction.
        force_w = self.contact.data.net_forces_w.sum(dim=1)
        force_obj_w = -force_w
        dot = (force_obj_w * pull_dir).sum(dim=-1)
        force_norm = force_obj_w.norm(dim=-1).clamp_min(1e-8)
        return -dot / force_norm

    def _get_observations(self) -> dict:
        base_obs = super()._get_observations()["policy"]
        push_dir = -self._compute_drawer_pull_dir()
        obs = torch.cat((base_obs, push_dir), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        pull_dir = self._compute_drawer_pull_dir()
        rew_force_align = self.cfg.w_rew_force_align * self._reward_force_alignment(pull_dir)
        self.extras["rew_force_align"] = rew_force_align.mean()
        print(f"rew_force_align: {self.extras['rew_force_align'].item():.4f}")
        return reward + rew_force_align
