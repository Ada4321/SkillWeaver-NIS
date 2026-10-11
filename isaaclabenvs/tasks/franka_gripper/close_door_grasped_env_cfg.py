from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from .close_door_env_cfg import CloseDoorEnvCfg


@configclass
class CloseDoorGraspedEnvCfg(CloseDoorEnvCfg):
    """Close-door task initialized from a grasped state (revolute analogue of CloseDrawerGraspedEnvCfg).

    The door starts (nearly) fully OPEN, the arm is solved (reset-time DLS IK) onto a sampled hand-
    crafted handle grasp with the jaws OPEN (q0->current via rotation about the hinge), the first
    `post_reset_forced_steps` steps force the jaws closed onto the handle, then the policy swings the
    door closed. Reward / observation / done are inherited from CloseDoorEnv (goal target_joint_pos_norm
    = 0.0 = closed, plus the hinge-torque force-alignment reward).
    """

    # Doors with hand-crafted handle grasps.
    dataset: str = "arti_revolute"

    # Grasp maintenance: once the env has pinched the handle, terminate as FAILURE if the grasp is
    # lost for this many consecutive steps. 0 = disabled (default off); opt in via env.grasp_lost_fail_steps=N.
    grasp_lost_fail_steps: int = 0

    # The door starts (nearly) fully OPEN; the policy only has to swing it closed from the grasp.
    joint_start_pos_range_norm: tuple[float, float] = (0.9, 1.0)

    # Post-reset forced close: first N steps force arm delta = 0 and gripper = close so the open jaws
    # settle onto the handle before the policy takes over.
    post_reset_forced_steps: int = 8

    # Reset-time IK that places the (open) gripper on the sampled grasp (one physics step per iter to
    # refresh the Jacobian; early-stops once the worst-env hand position residual < grasp_init_pos_tol).
    grasp_init_ik_iters: int = 50
    grasp_init_pos_tol: float = 0.005
    # grasp_site (panda_grip_site) offset from panda_hand origin, in the hand frame [m].
    grasp_init_grip_site_offset_z: float = 0.1025
    # Print mean/max reset IK position residual for the just-reset envs (debug/validation).
    grasp_init_debug: bool = True

    # Used by _compute_num_contact_and_pinch (CloseEnvCfg lacks it). False = 4-keypoint pinch.
    contact_use_tips_only: bool = False

    reset_ik_controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
