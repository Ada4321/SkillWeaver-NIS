"""WXYZ <-> XYZW adapters for the IsaacLab asset API, active only on IsaacLab 3.0.

IsaacLab 3.0 changed every quaternion to XYZW ("Changed the quaternion ordering to match
warp, PhysX, and Newton native XYZW quaternion ordering", CHANGELOG). This repo is WXYZ
throughout and stays that way: the observation vectors the RL checkpoints were trained on
contain `root_quat_w` / `wrist_state` quaternions in WXYZ, the recorded trajectory npz keys
are WXYZ, and `isaacsim.core.utils.torch` (the retired math module this repo was written
against) was WXYZ. Converting the repo instead would silently change the observation and
invalidate every trained checkpoint, so the adaptation happens here, at the boundary.

VERSION ADAPTIVE. Under IsaacLab 2.x (Isaac Sim 4.5, the stack `train*.sh` still uses)
every function here is a pass-through, so the same source tree keeps training there. Under
3.0 the conversions kick in. See `_IS_XYZW` below for how the two are told apart.

BOTH DIRECTIONS MATTER. Converting only on write is worse than converting nowhere: reset
code reads a pose out of the sim and writes it back, so a write-only conversion corrupts
exactly the poses that were already correct.

Nothing here raises on a wrong-convention input -- a quaternion is a valid quaternion
either way. The failure mode is silent: a WXYZ identity `[1,0,0,0]` read as XYZW is a
180-degree rotation about X.

Usage -- read through the accessors, write through the converters:

    from isaaclabenvs.utils import quat_compat as qc

    quat = qc.root_quat_w(obj)                     # WXYZ, ready for repo math
    obj.write_root_link_pose_to_sim(qc.pose_to_xyzw(pose_wxyz), env_ids=env_ids)
"""

from __future__ import annotations

import importlib.util

import numpy as np
import torch

__all__ = [
    "_as_tensor",
    "quat_to_xyzw", "quat_to_wxyz", "pose_to_xyzw", "pose_to_wxyz",
    "state_to_wxyz", "root_pose_w", "root_quat_w", "root_state_w",
    "body_quat_w", "body_state_w", "cam_quat_w_ros", "rot_xyzw",
    "IS_XYZW",
]


def _detect_xyzw() -> bool:
    """True when the installed IsaacLab uses XYZW quaternions (3.0+).

    Probe: `isaaclab.utils.warp.proxy_array` exists only in 3.0 (it is the ProxyArray
    buffer type introduced alongside the XYZW switch). `find_spec` RAISES rather than
    returning None when a PARENT package is missing, so it must be guarded. Fall back to
    the retired `isaacsim.core.utils.torch`: it ships with Isaac Sim 4.5 (IsaacLab 2.x,
    WXYZ) and is gone in 6.0 (IsaacLab 3.0, XYZW).
    """
    try:
        if importlib.util.find_spec("isaaclab.utils.warp.proxy_array") is not None:
            return True
    except (ImportError, ModuleNotFoundError, ValueError):
        pass
    try:
        return importlib.util.find_spec("isaacsim.core.utils.torch") is None
    except (ImportError, ModuleNotFoundError, ValueError):
        return True


IS_XYZW: bool = _detect_xyzw()


def _as_tensor(arr):
    """Normalize an IsaacLab 3.0 data buffer to something sliceable.

    3.0's `data.*` properties return `isaaclab.utils.warp.ProxyArray`, a warp-backed
    wrapper -- NOT a `torch.Tensor`. It forwards indexing, so `isinstance` checks pass
    it through to the numpy branch, where `np.asarray` on a CUDA-backed buffer raises
    "can't convert cuda:0 device type tensor to numpy". `.torch` is the documented
    zero-copy view, so take it when present. Also covers the raw warp arrays the PhysX
    view hands back (`get_jacobians()`), which reject list/tensor fancy indexing.
    """
    if isinstance(arr, torch.Tensor):
        return arr
    t = getattr(arr, "torch", None)
    if isinstance(t, torch.Tensor):
        return t
    try:  # raw warp array (physx view buffers); wp.to_torch is zero-copy
        import warp as wp

        if isinstance(arr, wp.array):
            return wp.to_torch(arr)
    except Exception:
        pass
    return arr


def _swap(arr, i, j):
    """Reorder the last axis, moving the scalar component between front and back."""
    arr = _as_tensor(arr)
    if isinstance(arr, torch.Tensor):
        return torch.cat([arr[..., i], arr[..., j]], dim=-1)
    a = np.asarray(arr)
    return np.concatenate([a[..., i], a[..., j]], axis=-1)


# --- bare quaternions, shape (..., 4) -------------------------------------------------
def quat_to_xyzw(q):
    """WXYZ -> XYZW (identity on IsaacLab 2.x)."""
    if not IS_XYZW:
        return _as_tensor(q)
    return _swap(q, slice(1, 4), slice(0, 1))


def quat_to_wxyz(q):
    """XYZW -> WXYZ (identity on IsaacLab 2.x)."""
    if not IS_XYZW:
        return _as_tensor(q)
    return _swap(q, slice(3, 4), slice(0, 3))


# --- 7-vector poses [pos(3), quat(4)] -------------------------------------------------
def pose_to_xyzw(pose):
    """Pose with a WXYZ quaternion -> pose the sim accepts (identity on 2.x)."""
    pose = _as_tensor(pose)
    if not IS_XYZW:
        return pose
    if isinstance(pose, torch.Tensor):
        return torch.cat([pose[..., :3], pose[..., 4:7], pose[..., 3:4]], dim=-1)
    a = np.asarray(pose)
    return np.concatenate([a[..., :3], a[..., 4:7], a[..., 3:4]], axis=-1)


def pose_to_wxyz(pose):
    """Pose read out of the sim -> pose with a WXYZ quaternion (identity on 2.x)."""
    pose = _as_tensor(pose)
    if not IS_XYZW:
        return pose
    if isinstance(pose, torch.Tensor):
        return torch.cat([pose[..., :3], pose[..., 6:7], pose[..., 3:6]], dim=-1)
    a = np.asarray(pose)
    return np.concatenate([a[..., :3], a[..., 6:7], a[..., 3:6]], axis=-1)


# --- 13-vector states [pos(3), quat(4), lin_vel(3), ang_vel(3)] -----------------------
def state_to_wxyz(state):
    """State whose quaternion slice [3:7] is XYZW -> the same state in WXYZ."""
    state = _as_tensor(state)
    if not IS_XYZW:
        return state
    if isinstance(state, torch.Tensor):
        return torch.cat(
            [state[..., :3], state[..., 6:7], state[..., 3:6], state[..., 7:]], dim=-1
        )
    a = np.asarray(state)
    return np.concatenate([a[..., :3], a[..., 6:7], a[..., 3:6], a[..., 7:]], axis=-1)


# --- accessors: read an IsaacLab asset and hand back WXYZ ------------------------------
def root_pose_w(asset):
    """Root pose (..., 7) in WXYZ."""
    return pose_to_wxyz(asset.data.root_pose_w)


def root_quat_w(asset):
    """Root orientation (..., 4) in WXYZ."""
    return quat_to_wxyz(asset.data.root_quat_w)


def root_pos_w(asset):
    """Root position (..., 3). No convention change; normalizes the buffer type."""
    return _as_tensor(asset.data.root_pos_w)


def root_state_w(asset):
    """Root state (..., 13) with its quaternion slice in WXYZ."""
    return state_to_wxyz(asset.data.root_state_w)


def body_quat_w(asset):
    """Per-body orientations (..., num_bodies, 4) in WXYZ."""
    return quat_to_wxyz(asset.data.body_quat_w)


def body_pos_w(asset):
    """Per-body positions (..., num_bodies, 3). Buffer-type normalization only."""
    return _as_tensor(asset.data.body_pos_w)


def body_state_w(asset):
    """Per-body states (..., num_bodies, 13) with quaternion slices in WXYZ."""
    return state_to_wxyz(asset.data.body_state_w)


def cam_quat_w_ros(camera):
    """Camera orientation (..., 4) in WXYZ, ROS frame (+Z forward, -Y up).

    `CameraData.quat_w_ros` went from `(w, x, y, z)` in IsaacLab 2.x to `(x, y, z, w)`
    in 3.0. These values are baked into the `cam_extr` of every recorded trajectory, so
    getting this wrong silently corrupts the dataset rather than the render.
    """
    return quat_to_wxyz(camera.data.quat_w_ros)


def rot_xyzw(rot):
    """WXYZ 4-tuple -> the tuple `InitialStateCfg.rot` / `OffsetCfg.rot` expects.

    Documented `(x, y, z, w)` in IsaacLab 3.0 (default `(0, 0, 0, 1)`); `(w, x, y, z)`
    in 2.x (default `(1, 0, 0, 0)`). Identity on 2.x.
    """
    w, x, y, z = (float(v) for v in rot)
    return (x, y, z, w) if IS_XYZW else (w, x, y, z)
