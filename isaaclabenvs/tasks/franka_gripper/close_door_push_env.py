from __future__ import annotations

from isaaclabenvs.utils import math_compat as torch_utils  # WXYZ; replaces isaacsim.core.utils.torch (gone in Isaac Sim 6.0)
from isaaclabenvs.utils import quat_compat as qc
import numpy as np
import torch

from .close_door_grasped_env import CloseDoorGraspedEnv
from .close_door_push_env_cfg import CloseDoorPushEnvCfg


class CloseDoorPushEnv(CloseDoorGraspedEnv):
    """Close a revolute DOOR by PRESSING the gripper against the outer face and swinging it shut.

    Reset / reset-time IK are inherited unchanged from CloseDoorGraspedEnv: the arm is solved onto a
    sampled hand-crafted close-push pose (palm flush on the door's outer face beside the handle, jaws
    OPEN). With cfg.post_reset_forced_steps=0 the jaws are NEVER forced closed, so the policy presses
    with the open gripper and pushes -- it does not grasp the handle.

    Reward extends CloseDoorEnv's (closing progress / closed / success / hinge force) with a single
    location-agnostic term: +w_rew_touch when ANY sampled gripper-surface point (palm + both fingers,
    both +/-x faces) comes within touch_eps of the door point cloud (handle U outer-face). The
    finger-only contact/distance shaping is disabled via the cfg weights (=0).

    Revolute analogue of CloseDrawerPushEnv; the push-reward body is geometry-agnostic (the q0 mapping
    in _map_points_to_q0_frame already handles revolute), so it is identical bar naming.

    Two optional anti-fly-away constraints (default OFF, see cfg):
      - rew_close_require_touch : gate the closing reward on the gripper being within close_gate_dist
        of the door (so closing it while flying off earns nothing).
      - detach_fail_steps       : once the gripper has touched the door, FAIL the episode if it stays
        farther than detach_dist for that many consecutive steps.
    """

    cfg: CloseDoorPushEnvCfg

    def __init__(self, cfg: CloseDoorPushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        pts = np.load(cfg.gripper_surface_points_path)
        # Link-local surface points: hand pts in panda_hand frame; finger pts in the finger link frame
        # (reused for both fingers -- the right finger's Rz(pi) is baked into its body_quat_w in sim).
        self._grip_pts_hand = torch.tensor(pts["hand"], dtype=torch.float, device=self.device)
        self._grip_pts_finger = torch.tensor(pts["finger"], dtype=torch.float, device=self.device)
        # Subsample the door pcd for the distance/touch test (contact detection does not need all
        # 2048 pts); keeps the cdist (n, M, P) cheap. data_query_points = [handle, outer-face] (q0).
        self._touch_query = self.data_query_points[:, ::8].contiguous()

        # Per-step gripper<->door signal, computed in _get_dones and reused in _get_rewards.
        self._grip_door_dist = torch.zeros(self.num_envs, device=self.device)
        self._grip_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Detach-fail bookkeeping: has this env ever touched the door since reset, and for how many
        # consecutive steps it has been detached afterwards.
        self._touched_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._detach_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Hand orientation captured at reset (the correct pressing pose); the pose-hold reward keeps the
        # policy near it so it presses with one steady pose. Init to identity (overwritten on reset).
        self._reset_hand_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._reset_hand_quat[:, 0] = 1.0
        # cos(orientation-error angle) vs the reset pose; cached in _refresh_grip_door so the close
        # gate and the rew_orient term share it (the gate runs inside super()._get_rewards, earlier).
        self._orient_cos = torch.ones(self.num_envs, device=self.device)

    # --------------------------------------------------------------- gripper <-> door geometry
    def _gripper_surface_points_q0(self) -> torch.Tensor:
        """All sampled gripper-surface points, mapped into the object q0 frame -> (n, M, 3).

        Each link's local points are placed in world via its live body pose, then world -> object ->
        q0 (undo the revolute opening) exactly like the keypoint distances in _refresh_buffer, so the
        comparison against the static q0 door pcd is apples-to-apples.
        """
        n = self.num_envs
        pieces = []
        for body_idx, local in (
            (self.hand_link_idx, self._grip_pts_hand),
            (self.left_finger_idx, self._grip_pts_finger),
            (self.right_finger_idx, self._grip_pts_finger),
        ):
            m = local.shape[0]
            quat = qc.body_quat_w(self.robot)[:, body_idx].unsqueeze(1).expand(-1, m, -1)
            pos = qc.body_pos_w(self.robot)[:, body_idx].unsqueeze(1)
            pieces.append(torch_utils.tf_apply(quat, pos, local.unsqueeze(0).expand(n, -1, -1)))
        pts_w = torch.cat(pieces, dim=1)  # (n, M, 3) world

        inv_quat, inv_pos = torch_utils.tf_inverse(
            qc.root_quat_w(self.object), qc.root_pos_w(self.object)
        )
        m_tot = pts_w.shape[1]
        pts_obj = torch_utils.tf_apply(
            inv_quat.unsqueeze(1).expand(-1, m_tot, -1), inv_pos.unsqueeze(1), pts_w
        )
        joint_delta = qc._as_tensor(self.object.data.joint_pos)[:, 0] - self.object_joint_lower
        return self._map_points_to_q0_frame(pts_obj, joint_delta)

    def _refresh_grip_door(self) -> None:
        """Compute and cache the gripper<->door min distance and touched flag for this step."""
        pts_q0 = self._gripper_surface_points_q0()
        nearest = torch.cdist(pts_q0, self._touch_query).min(dim=-1).values  # (n, M) per gripper point
        self._grip_door_dist = nearest.min(dim=-1).values  # (n,) closest gripper point -> door
        self._grip_touched = (nearest < float(self.cfg.touch_eps)).any(dim=-1)  # (n,)
        # Hand-orientation alignment vs the reset pressing pose: cos of the error angle, in [-1, 1].
        q = qc.body_quat_w(self.robot)[:, self.hand_link_idx]
        dot = (q * self._reset_hand_quat).sum(dim=-1)
        self._orient_cos = 2.0 * dot * dot - 1.0

    # ----------------------------------------------------------------------------- gate / dones
    def _close_reward_gate(self) -> torch.Tensor:
        gate = torch.ones(self.num_envs, device=self.device)
        if bool(self.cfg.rew_close_require_touch):
            gate = gate * (self._grip_door_dist < float(self.cfg.close_gate_dist)).float()
        # Optional (CLI opt-in): also require the gripper to hold its reset pressing orientation, so the
        # 500-scale close/closed rewards are unreachable while flailing -- a much harder constraint than
        # the dense rew_orient nudge alone.
        if bool(self.cfg.close_gate_require_orient):
            gate = gate * (self._orient_cos >= float(self.cfg.close_gate_orient_cos)).float()
        return gate

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        # Refresh the gripper<->door signal here (before _get_rewards) so the gate and the touch
        # bonus both reuse it without a second cdist.
        self._refresh_grip_door()

        steps = int(self.cfg.detach_fail_steps)
        if steps <= 0:
            return terminated, truncated
        # Only judge detachment after the env has actually engaged the door once.
        active = self.episode_length_buf >= int(self.cfg.post_reset_forced_steps)
        self._touched_once |= self._grip_touched & active
        detached = self._grip_door_dist > float(self.cfg.detach_dist)
        lost = self._touched_once & active & detached
        self._detach_steps = torch.where(
            lost, self._detach_steps + 1, torch.zeros_like(self._detach_steps)
        )
        detach_fail = self._detach_steps >= steps
        self.failure = self.failure | detach_fail
        return terminated | detach_fail, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return
        self._touched_once[env_ids] = False
        self._detach_steps[env_ids] = 0
        # Capture the reset (pressing) hand orientation as the pose-hold target. super() has already
        # placed the arm on the close-push grasp and refreshed kinematics, so body_quat_w is current.
        self._reset_hand_quat[env_ids] = qc.body_quat_w(self.robot)[env_ids, self.hand_link_idx]

    # ---------------------------------------------------------------------------------- rewards
    def _get_rewards(self) -> torch.Tensor:
        # super() (CloseDoorEnv -> CloseEnv) applies _close_reward_gate() to rew_close/rew_closed.
        reward = super()._get_rewards()

        rew_touch = self.cfg.w_rew_touch * self._grip_touched.float()
        # Dense approach/press shaping: pull the gripper surface continuously onto the face (0 once
        # flush, saturated past approach_clip). _grip_door_dist was cached in _get_dones above.
        approach = self._grip_door_dist.clamp(0.0, float(self.cfg.approach_clip))
        rew_approach = -self.cfg.w_rew_approach * approach

        # Pose-hold: keep the hand orientation at the reset pressing pose (cached in _refresh_grip_door).
        rew_orient = self.cfg.w_rew_orient * self._orient_cos
        # Smoothness: penalize the rotation part of the operational-space action delta (dims 3:6).
        rew_action_rot = self.cfg.w_rew_action_rot * self.actions[:, 3:6].norm(dim=-1)

        self.extras["rew_touch"] = rew_touch.mean()
        self.extras["rew_approach"] = rew_approach.mean()
        self.extras["rew_orient"] = rew_orient.mean()
        self.extras["rew_action_rot"] = rew_action_rot.mean()
        self.extras["touched_rate"] = self._grip_touched.float().mean()
        self.extras["grip_door_dist"] = self._grip_door_dist.mean()
        print(
            f"rew_touch: {rew_touch.mean():.4f}; rew_approach: {rew_approach.mean():.4f}; "
            f"rew_orient: {rew_orient.mean():.4f}; rew_action_rot: {rew_action_rot.mean():.4f}; "
            f"touched_rate: {self._grip_touched.float().mean():.4f}; "
            f"grip_door_dist: {self._grip_door_dist.mean():.4f}"
        )
        return reward + rew_touch + rew_approach + rew_orient + rew_action_rot
