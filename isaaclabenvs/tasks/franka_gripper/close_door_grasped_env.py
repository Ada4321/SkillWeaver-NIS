from __future__ import annotations

from isaaclabenvs.utils import math_compat as torch_utils  # WXYZ; replaces isaacsim.core.utils.torch (gone in Isaac Sim 6.0)
from isaaclabenvs.utils import quat_compat as qc
import torch
from isaaclab.controllers import DifferentialIKController
from isaaclab.utils import math as math_utils

from .close_door_env import CloseDoorEnv
from .close_door_grasped_env_cfg import CloseDoorGraspedEnvCfg


class CloseDoorGraspedEnv(CloseDoorEnv):
    """Close a revolute DOOR starting already grasped on the handle (Phase 2).

    Reward / observation / done are inherited unchanged from CloseDoorEnv (the door starts (nearly)
    OPEN and the goal is closed). Only the reset and the post-reset forced close differ: at reset the
    arm is solved (reset-time DLS IK) onto a sampled hand-crafted handle grasp with the jaws OPEN,
    then the first `post_reset_forced_steps` steps hold the arm and command the jaws closed so the
    fingers settle onto the handle; afterwards the policy swings the door closed.

    Mirror of CloseDrawerGraspedEnv; the only structural difference is the q0 -> current grasp mapping,
    which rotates about the hinge (revolute) instead of translating along the joint axis (prismatic).
    """

    cfg: CloseDoorGraspedEnvCfg

    def __init__(self, cfg: CloseDoorGraspedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if getattr(self, "data_handle_grasps", None) is None:
            raise RuntimeError(
                "CloseDoorGraspedEnv requires per-asset handle grasps. Set `handle_grasps` for the "
                f"dataset '{self.cfg.dataset}' in config.toml (only the grasp-carrying trees have them)."
            )
        # grasp_site (panda_grip_site) sits at +z * offset from panda_hand in the hand frame.
        self._grip_site_offset = torch.tensor(
            [0.0, 0.0, float(self.cfg.grasp_init_grip_site_offset_z)], device=self.device
        )
        # Arm joint solution from the reset IK; the forced-close steps rigidly hold the arm here so
        # the closing reaction cannot drift the gripper off the handle.
        self._grasp_arm_q = torch.zeros((self.num_envs, len(self.arm_joint_indices)), device=self.device)
        # Grasp-maintenance bookkeeping (active only when cfg.grasp_lost_fail_steps > 0): did this env
        # ever pinch since reset, and for how many consecutive steps the grasp has been lost afterwards.
        self._grasped_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lost_grasp_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Base reset: place object (door starts ~open), sample a reachable arm config, write
        # close-task bookkeeping. We then OVERRIDE the arm with the grasped IK pose.
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return

        self._place_arm_on_grasp(env_ids)

        # Re-sync close-task bookkeeping for the new (grasped) arm state. The door joint itself was
        # not moved here, so min_close / closed stay consistent with the sampled start.
        self._refresh_buffer()
        self.min_close_pos_norm[env_ids] = self.object_joint_pos_norm[env_ids]
        self.closed[env_ids] = False
        self.consecutive_success[env_ids] = 0
        self._grasped_once[env_ids] = False
        self._lost_grasp_steps[env_ids] = 0

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        fail_steps = int(self.cfg.grasp_lost_fail_steps)
        if fail_steps <= 0:
            return terminated, truncated
        # Only judge grasp loss after the forced-close window has established the grasp.
        active = self.episode_length_buf >= int(self.cfg.post_reset_forced_steps)
        _, pinch = self._compute_num_contact_and_pinch()
        self._grasped_once |= pinch & active
        lost = self._grasped_once & active & (~pinch)
        self._lost_grasp_steps = torch.where(
            lost, self._lost_grasp_steps + 1, torch.zeros_like(self._lost_grasp_steps)
        )
        grasp_fail = self._lost_grasp_steps >= fail_steps
        self.failure = self.failure | grasp_fail
        return terminated | grasp_fail, truncated

    def _compute_num_contact_and_pinch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Validated-contact count and two-sided pinch flag (copied from OpenBaseEnv; CloseEnv lacks it).

        Columns of obs_dist/obs_contact: [0]=left_finger, [1]=right_finger, [2]=left_tip, [3]=right_tip.
        cfg.contact_use_tips_only True restricts to the two fingertip columns; else uses all four.
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

    def _place_arm_on_grasp(self, env_ids: torch.Tensor):
        n = len(env_ids)
        device = self.device

        # 1) sample one grasp per env (grasp_site pose + finger in q0/object frame).
        num_grasp = int(self.data_handle_grasps.shape[1])
        gid = torch.randint(0, num_grasp, (n,), device=device)
        grasp = self.data_handle_grasps[env_ids, gid]  # (n, 8/9)
        p_g_q0 = grasp[:, :3]
        q_g_q0 = grasp[:, 3:7]

        # 2) q0 -> current object frame. REVOLUTE door: rotate the grasp_site about the hinge by the
        #    current opening angle (position AND orientation). The door starts ~open, so the grasp_site
        #    lands on the handle at its open position. Then object -> world.
        joint_delta = (
            self.object.data.joint_pos[env_ids, self.object_joint_idx] - self.object_joint_lower[env_ids]
        )
        axis = self.data_joint_axis_obj[env_ids]
        origin = self.data_joint_origin_obj[env_ids]
        vec_rot = self._rodrigues_rotate(
            (p_g_q0 - origin).unsqueeze(1), axis, joint_delta
        ).squeeze(1)
        p_cur_obj = vec_rot + origin
        q_delta = qc.quat_to_wxyz(math_utils.quat_from_angle_axis(joint_delta, axis))  # (n, 4) wxyz
        q_cur_q0 = torch_utils.quat_mul(q_delta, q_g_q0)
        root_quat = qc.root_quat_w(self.object)[env_ids]
        root_pos = qc.root_pos_w(self.object)[env_ids]
        grasp_pos_w = torch_utils.quat_apply(root_quat, p_cur_obj) + root_pos
        grasp_quat_w = torch_utils.quat_mul(root_quat, q_cur_q0)

        # 3) grasp_site target -> panda_hand target (grip_site is +z*offset from the hand origin).
        offset = self._grip_site_offset.unsqueeze(0).expand(n, -1)
        hand_pos_w = grasp_pos_w - torch_utils.quat_apply(grasp_quat_w, offset)
        hand_quat_w = grasp_quat_w
        target_pose = torch.cat([hand_pos_w, hand_quat_w], dim=-1)  # (n, 7)

        # 4) reset-time DLS IK seeded from the grid arm config super()._reset_idx just placed; jaws open
        #    to the grasp's density-aware width (col 8, clears neighbour handles), clamped to the finger
        #    limit. sim.forward() after each write refreshes kinematics without advancing dynamics.
        if grasp.shape[-1] >= 9:
            finger_open = grasp[:, 8:9].clamp(max=float(self.robot_dof_upper_limits[-1])).expand(n, 2)
        else:
            finger_open = self.robot_dof_upper_limits[-2:]
        q_full = self.robot.data.joint_pos[env_ids].clone()
        q_full[:, -2:] = finger_open
        joint_vel = torch.zeros_like(q_full)
        arm_lower = self.robot_dof_lower_limits[self.arm_joint_indices]
        arm_upper = self.robot_dof_upper_limits[self.arm_joint_indices]

        controller = DifferentialIKController(self.cfg.reset_ik_controller, n, device)
        controller.set_command(qc.pose_to_xyzw(target_pose))

        for _ in range(int(self.cfg.grasp_init_ik_iters)):
            self.robot.write_joint_state_to_sim(q_full, joint_vel, env_ids=env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.robot.update(self.physics_dt)

            cur_pos = qc.body_pos_w(self.robot)[env_ids, self.hand_link_idx]
            # sim convention, matching controller.set_command above -- do NOT convert to WXYZ
            cur_quat = qc._as_tensor(self.robot.data.body_quat_w)[env_ids, self.hand_link_idx]
            jacobian = qc._as_tensor(self.robot.root_physx_view.get_jacobians())[
                :, self.hand_link_idx - 1, :, self.arm_joint_indices
            ][env_ids]
            arm_joint_pos = self.robot.data.joint_pos[env_ids][:, : len(self.arm_joint_indices)]
            arm_targets = controller.compute(cur_pos, cur_quat, jacobian, arm_joint_pos)
            q_full[:, self.arm_joint_indices] = torch.clamp(arm_targets, arm_lower, arm_upper)
            if (cur_pos - hand_pos_w).norm(dim=-1).max() < float(self.cfg.grasp_init_pos_tol):
                break

        # Final state: solved arm, jaws still open (the forced steps will close them).
        q_full[:, -2:] = finger_open
        self.robot.write_joint_state_to_sim(q_full, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = q_full
        # Remember the IK arm solution so the forced-close steps can pin the arm to it.
        self._grasp_arm_q[env_ids] = q_full[:, self.arm_joint_indices]

        if bool(self.cfg.grasp_init_debug):
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.robot.update(self.physics_dt)
            res = (self.robot.data.body_pos_w[env_ids, self.hand_link_idx] - hand_pos_w).norm(dim=-1)
            print(
                f"[grasp_init] reset {n} envs | IK pos residual mean={res.mean().item()*1000:.1f}mm "
                f"max={res.max().item()*1000:.1f}mm"
            )

    # ----------------------------------------------------- post-reset forced close
    def _pre_physics_step(self, actions: torch.Tensor):
        raw_actions = actions.clone().clamp(-1.0, 1.0)
        forced_steps = max(0, int(self.cfg.post_reset_forced_steps))
        forced_mask = self.episode_length_buf < forced_steps if forced_steps > 0 else None
        if forced_mask is not None and forced_mask.any():
            raw_actions[forced_mask, :6] = 0.0  # hold the IK'd arm pose
            raw_actions[forced_mask, 6:] = -1.0  # close the jaws onto the handle
        super()._pre_physics_step(raw_actions)
        if forced_mask is not None and forced_mask.any():
            self.actions[forced_mask] = raw_actions[forced_mask]
            self.prev_actions[forced_mask] = raw_actions[forced_mask]
            # Rigidly pin the arm to the reset-IK solution while the jaws close (super() set the arm
            # target from "hold current pose", which follows the closing reaction and drifts off).
            n_arm = len(self.arm_joint_indices)
            self.robot_dof_targets[forced_mask, :n_arm] = self._grasp_arm_q[forced_mask]
