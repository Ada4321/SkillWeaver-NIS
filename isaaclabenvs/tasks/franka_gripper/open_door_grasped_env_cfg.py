from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from .open_door_env_cfg import OpenDoorEnvCfg


@configclass
class OpenDoorGraspedEnvCfg(OpenDoorEnvCfg):
    """Open-door (revolute) task initialized from a grasped state -- revolute analogue of
    OpenDrawerGraspedEnvCfg.

    Same reward / observation / done as OpenDoorEnv (the grasped reset only changes the init-state
    distribution, not the reward). At reset the arm is solved (reset-time DLS IK) onto a sampled
    hand-crafted handle grasp with the jaws OPEN (q0->current via rotation about the hinge); the first
    `post_reset_forced_steps` rollout steps then hold the arm and command the gripper closed so the
    fingers settle onto the handle, after which the policy pulls the door open.
    """

    # Doors with hand-crafted handle grasps.
    dataset: str = "arti_revolute"

    # Grasp maintenance: once the env has pinched the handle, terminate as FAILURE if the grasp is
    # lost for this many consecutive steps (makes letting go costly instead of free).
    # 0 = disabled (default off); opt in from the training script via env.grasp_lost_fail_steps=N.
    grasp_lost_fail_steps: int = 0

    # The door starts (nearly) closed; the policy only has to pull it open from the grasp.
    joint_start_pos_range_norm: tuple[float, float] = (0.0, 0.05)

    # Start-distribution curriculum (default OFF; opt in via env.start_curriculum=True). When on, the
    # UPPER bound of the reset start-open range is annealed from start_curriculum_hi_start down to the
    # real upper (joint_start_pos_range_norm[1]) over start_curriculum_steps env steps; the lower bound
    # stays joint_start_pos_range_norm[0]. Early training thus includes near-open starts (short pull to
    # goal -> dense open/success signal) and gradually concentrates on the real closed-start task,
    # breaking the "grasp-and-hold" local optimum. Keyed on common_step_counter (one per env.step()).
    start_curriculum: bool = False
    start_curriculum_hi_start: float = 0.9
    start_curriculum_steps: int = 20000

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
