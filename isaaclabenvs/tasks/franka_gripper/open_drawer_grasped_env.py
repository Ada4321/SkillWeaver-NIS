from __future__ import annotations

from isaaclabenvs.utils import math_compat as torch_utils  # WXYZ; replaces isaacsim.core.utils.torch (gone in Isaac Sim 6.0)
from isaaclabenvs.utils import quat_compat as qc
import torch
from isaaclab.controllers import DifferentialIKController

from .open_base_env import TaskStatus
from .open_drawer_env import OpenDrawerEnv
from .open_drawer_grasped_env_cfg import OpenDrawerGraspedEnvCfg


class OpenDrawerGraspedEnv(OpenDrawerEnv):
    """Open the drawer starting already grasped on the handle.

    Reward / observation / done are inherited unchanged from OpenDrawerEnv. Only the reset and
    the post-reset forced close differ.
    """

    cfg: OpenDrawerGraspedEnvCfg

    def __init__(self, cfg: OpenDrawerGraspedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if getattr(self, "data_handle_grasps", None) is None:
            raise RuntimeError(
                "OpenDrawerGraspedEnv requires per-asset handle grasps. Set `handle_grasps` for the "
                f"dataset '{self.cfg.dataset}' in config.toml (only the thickened-handle tree has them)."
            )
        # grasp_site (panda_grip_site) sits at +z * offset from panda_hand in the hand frame.
        self._grip_site_offset = torch.tensor(
            [0.0, 0.0, float(self.cfg.grasp_init_grip_site_offset_z)], device=self.device
        )
        # Grasp-maintenance bookkeeping (active only when cfg.grasp_lost_fail_steps > 0): did this
        # env ever pinch since reset, and for how many consecutive steps the grasp has been lost.
        self._grasped_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lost_grasp_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Arm joint solution from the reset IK; the forced-close steps rigidly hold the arm here so
        # the closing reaction cannot drift the gripper off the handle.
        self._grasp_arm_q = torch.zeros((self.num_envs, len(self.arm_joint_indices)), device=self.device)
        # Grasp-maintenance bookkeeping: did this env ever pinch since reset, and for how many
        # consecutive steps has the grasp been lost afterwards.
        self._grasped_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lost_grasp_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Base reset: place object (upright, randomized), sample drawer start, sample a reachable
        # arm config and write bookkeeping. We then OVERRIDE the arm with the grasped IK pose.
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return

        self._place_arm_on_grasp(env_ids)

        # Re-sync open-task bookkeeping for the new (grasped) arm state. The drawer joint itself
        # was not moved here, so max_open / opened stay consistent with the sampled start.
        self._refresh_buffer()
        self.max_open_pos_norm[env_ids] = self.object_joint_pos_norm[env_ids]
        self.opened[env_ids] = False
        self.prev_joint_error_abs[env_ids] = self.object_joint_error_norm[env_ids].abs()
        self.consecutive_success[env_ids] = 0
        # NOTE: do NOT reset latest_status here (see open_base_env._reset_idx) -- the terminal step
        # already recorded the real outcome; resetting it corrupted success_rate into a per-step rate.
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

    def _place_arm_on_grasp(self, env_ids: torch.Tensor):
        n = len(env_ids)
        device = self.device

        # 1) sample one grasp per env (grasp_site pose + finger in q0/object frame).
        num_grasp = int(self.data_handle_grasps.shape[1])
        gid = torch.randint(0, num_grasp, (n,), device=device)
        grasp = self.data_handle_grasps[env_ids, gid]  # (n, 8)
        p_g_q0 = grasp[:, :3]
        q_g_q0 = grasp[:, 3:7]

        # 2) q0 -> current object frame (prismatic drawer: translate grasp_site along the joint
        #    axis by the current opening; orientation unchanged). Then object -> world.
        joint_delta = (
            self.object.data.joint_pos[env_ids, self.object_joint_idx] - self.object_joint_lower[env_ids]
        )
        axis = self.data_joint_axis_obj[env_ids]
        p_cur_obj = p_g_q0 + axis * joint_delta.unsqueeze(-1)
        root_quat = qc.root_quat_w(self.object)[env_ids]
        root_pos = qc.root_pos_w(self.object)[env_ids]
        grasp_pos_w = torch_utils.quat_apply(root_quat, p_cur_obj) + root_pos
        grasp_quat_w = torch_utils.quat_mul(root_quat, q_g_q0)

        # 3) grasp_site target -> panda_hand target (grip_site is +z*offset from the hand origin).
        offset = self._grip_site_offset.unsqueeze(0).expand(n, -1)
        hand_pos_w = grasp_pos_w - torch_utils.quat_apply(grasp_quat_w, offset)
        hand_quat_w = grasp_quat_w
        target_pose = torch.cat([hand_pos_w, hand_quat_w], dim=-1)  # (n, 7)

        # 4) reset-time DLS IK from the default (ready) seed; jaws open to the grasp's density-aware
        #    width (col 8; capped so the open fingers clear neighbouring drawer handles), clamped to
        #    the robot finger limit. sim.forward() after each write refreshes kinematics WITHOUT
        #    advancing dynamics, so other envs are untouched.
        if grasp.shape[-1] >= 9:
            open_w = grasp[:, 8:9].clamp(max=float(self.robot_dof_upper_limits[-1]))
            finger_open = open_w.expand(n, 2)  # (n, 2) per-env
        else:
            finger_open = self.robot_dof_upper_limits[-2:]
        # Seed the IK from the reachable-grid arm config that super()._reset_idx already placed
        # (its grasp_site is already at the handle front), NOT the far default ready pose. The
        # default seed left a long tail of non-converging envs (IK residual up to ~40cm) that flung
        # the gripper off the handle; the grid seed starts the DLS iteration right next to the goal.
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
            # Early stop once every env is within tolerance (keeps the per-reset step count low).
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
            # Rigidly pin the arm to the reset-IK solution while the jaws close. super() set the arm
            # target from the controller's "hold current pose", which follows the closing reaction
            # and drifts the gripper off the handle; overwrite it with the fixed IK joint solution.
            n_arm = len(self.arm_joint_indices)
            self.robot_dof_targets[forced_mask, :n_arm] = self._grasp_arm_q[forced_mask]
