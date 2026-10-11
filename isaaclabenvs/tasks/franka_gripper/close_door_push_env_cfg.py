from isaaclab.utils import configclass

from isaaclabenvs.config import ASSET_ROOT

from .close_door_grasped_env_cfg import CloseDoorGraspedEnvCfg


@configclass
class CloseDoorPushEnvCfg(CloseDoorGraspedEnvCfg):
    """Push-to-close for revolute DOORS (microwaves): the gripper starts pressed flush on the door's
    OUTER FACE beside the handle (hand-crafted close-push init poses, jaws OPEN), is NOT forced to
    pinch, and the policy presses + swings the door shut. Reward only cares that SOME gripper surface
    touches the door and that it closes -- no handle pinch, no contact-location.

    Revolute analogue of CloseDrawerPushEnvCfg (same push-reward machinery); differences:
      - post_reset_forced_steps=0   : no forced jaw-close (the open gripper presses, never grasps).
      - w_rew_contact/w_rew_dist=0  : drop the finger-only contact/distance shaping.
      - w_rew_touch                 : reward when ANY sampled gripper-surface point touches the door.
    """

    # Doors with hand-crafted close-push init poses.
    dataset: str = "arti_revolute"

    # No forced jaw-close: the open gripper presses the face and pushes -- it does not grasp the handle.
    post_reset_forced_steps: int = 0
    grasp_lost_fail_steps: int = 0

    # Disable the finger-only contact/distance shaping -- we reward ANY gripper surface touching the
    # door (palm included), measured densely below instead of via the 4 finger contact sensors.
    w_rew_contact: float = 0.0
    w_rew_dist: float = 0.0

    # Dense "touched the door" bonus: +w_rew_touch when the nearest sampled gripper-surface point is
    # within touch_eps of the door point cloud (handle U outer-face). Location-agnostic.
    w_rew_touch: float = 1.0
    touch_eps: float = 0.01  # [m]

    # The inherited finger-only force-alignment reward (CloseDoorEnv hinge torque) is blind to the PALM,
    # which is the body that actually presses in push mode -- with no finger contact it normalizes
    # numerical noise into a +/-1 signal. Disable it here; the dense approach shaping carries the gradient.
    w_rew_force_align: float = 0.0

    # Dense approach/press shaping: -w_rew_approach * clamp(grip_door_dist, 0, approach_clip). Pulls the
    # gripper surface continuously onto the face (0 once flush, saturates past approach_clip so a far
    # start is not yanked in too hard). Bridges "near -> pressed" that touch/close alone don't shape.
    w_rew_approach: float = 5.0
    approach_clip: float = 0.10  # [m]

    # Pose-hold shaping so the gripper presses with ONE steady pose instead of wobbling: reward keeping
    # the hand orientation at the (correct) pose it was reset into. cos_err in [-1,1] = cos of the
    # orientation-error angle vs the reset quaternion (double-cover robust).
    w_rew_orient: float = 3.0
    # Smoothness: penalize the ROTATION part of the operational-space action delta (translation is left
    # free so push-in is not discouraged). Negative weight.
    w_rew_action_rot: float = -0.5

    # Precomputed gripper surface points (panda_hand frame + finger link frame), both +/-x faces.
    gripper_surface_points_path: str = str(ASSET_ROOT / "franka_gripper/gripper_surface_points.npz")

    # ---- anti-fly-away constraints (default OFF; opt in via CLI / train script) -----------------
    # Engagement gate: pay rew_close/rew_closed only while the gripper is within close_gate_dist of
    # the door, so the policy is not rewarded for closing it while flying off.
    rew_close_require_touch: bool = False
    close_gate_dist: float = 0.05  # [m]
    # Orientation gate: also require the hand to hold its reset pressing orientation before paying the
    # close/closed rewards (cos of the error angle >= close_gate_orient_cos). Default OFF; enable via
    # CLI (env.close_gate_require_orient=True). Much stronger than the dense rew_orient nudge.
    close_gate_require_orient: bool = False
    close_gate_orient_cos: float = 0.8  # cos(err angle) >= 0.8 ~= within ~37 deg
    # Detach fail: once the gripper has touched the door, terminate as FAILURE if it stays farther than
    # detach_dist from the door for detach_fail_steps consecutive steps. 0 = disabled.
    detach_fail_steps: int = 0
    detach_dist: float = 0.10  # [m]
