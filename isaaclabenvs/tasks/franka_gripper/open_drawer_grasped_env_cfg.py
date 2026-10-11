from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from .open_drawer_env_cfg import OpenDrawerEnvCfg


@configclass
class OpenDrawerGraspedEnvCfg(OpenDrawerEnvCfg):
    """Open-drawer task initialized from a grasped state.

    Same reward / observation / done as OpenDrawerEnv (per OmniReset: the grasped reset only
    changes the init-state distribution, not the reward). At reset the arm is solved (reset-time
    DLS IK) onto a sampled hand-crafted handle grasp with the jaws OPEN; the first
    `post_reset_forced_steps` rollout steps then hold the arm and command the gripper closed so
    the fingers settle onto the handle (mirrors place_env's post-reset forced close).
    """

    # Grasps only exist for the thickened-handle tree.
    dataset: str = "arti_prismatic"

    # Grasp maintenance: once the env has pinched the handle, terminate as FAILURE if the grasp is
    # lost for this many consecutive steps (makes letting go costly instead of free).
    # 0 = disabled (default off); opt in from the training script via env.grasp_lost_fail_steps=N.
    grasp_lost_fail_steps: int = 0

    # The drawer starts (nearly) closed; the policy only has to pull it open from the grasp.
    joint_start_pos_range_norm: tuple[float, float] = (0.0, 0.05)

    # Post-reset forced close: first N steps force arm delta = 0 and gripper = close so the
    # open jaws settle onto the handle before the policy takes over.
    post_reset_forced_steps: int = 8

    # Reset-time IK that places the (open) gripper on the sampled grasp. Each iteration takes one
    # physics step (to refresh the Jacobian), so the loop early-stops once the worst-env hand
    # position residual is below grasp_init_pos_tol to keep the per-reset step count small.
    grasp_init_ik_iters: int = 50
    grasp_init_pos_tol: float = 0.005
    # grasp_site (panda_grip_site) offset from panda_hand origin, in the hand frame [m].
    grasp_init_grip_site_offset_z: float = 0.1025
    # Print mean/max reset IK position residual for the just-reset envs (debug/validation).
    grasp_init_debug: bool = True

    reset_ik_controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
