"""Drop-in replacement for `isaacsim.core.utils.torch`, which Isaac Sim 6.0 no longer ships.

QUATERNION CONVENTION -- the whole point of this module.

This repo is WXYZ throughout: the observation vectors the RL checkpoints were trained on,
the recorded trajectory npz, and every call site that used to import
`isaacsim.core.utils.torch` (which was itself WXYZ).

IsaacLab 3.0 switched to XYZW: `quat_apply` in `isaaclab.utils.math` documents
`quat: The quaternion in (x, y, z, w)`, and the asset API follows suit. Re-exporting those
under these names would silently change what every call site means -- no error, just wrong
orientations. So when the retired module is gone the helpers below are implemented
WXYZ-natively instead. Conversion to XYZW happens only at the IsaacLab API boundary --
see `isaaclabenvs/utils/quat_compat.py`.

VERSION ADAPTIVE. On Isaac Sim 4.5 the original module is still there and is re-exported
verbatim, so behaviour on that stack is bit-for-bit unchanged.

`scale_transform` is not quaternion-related (it normalizes to [-1, 1]); it is wrapped
rather than re-exported only because it is `torch.jit.script`-compiled and so refuses a
`ProxyArray` outright ("Unable to cast ProxyArray ... to Tensor").

Every entry point runs its inputs through `quat_compat._as_tensor`. IsaacLab 3.0 hands
back `ProxyArray`, not `torch.Tensor`, and a warp `vec3`-typed array reports `.shape ==
(N,)` rather than `(N, 3)` -- so a `.view(v.shape)` at the end silently becomes
`.view([N])` and raises. Normalizing once here covers every call site.

Import as:  from isaaclabenvs.utils import math_compat as torch_utils
"""

from __future__ import annotations

import torch

from isaaclabenvs.utils.quat_compat import _as_tensor

__all__ = [
    "quat_apply",
    "quat_conjugate",
    "quat_mul",
    "quat_unit",
    "scale_transform",
    "tf_apply",
    "tf_inverse",
]

try:  # Isaac Sim 4.5 / IsaacLab 2.x -- keep the original implementations verbatim.
    import isaacsim.core.utils.torch as _legacy
except (ImportError, ModuleNotFoundError):
    _legacy = None


if _legacy is not None:
    quat_apply = _legacy.quat_apply
    quat_conjugate = _legacy.quat_conjugate
    quat_mul = _legacy.quat_mul
    quat_unit = _legacy.quat_unit
    scale_transform = _legacy.scale_transform
    tf_apply = _legacy.tf_apply
    tf_inverse = _legacy.tf_inverse

else:
    from isaaclab.utils.math import scale_transform as _scale_transform

    def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        """Conjugate of a WXYZ quaternion. Shape (..., 4)."""
        q = _as_tensor(q)
        return torch.cat((q[..., 0:1], -q[..., 1:4]), dim=-1)

    def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Hamilton product of two WXYZ quaternions.

        Reshapes to (-1, 4) like the original so mismatched-but-broadcastable batch
        shapes work -- call sites pass e.g. (1, 4) against (N, 4).
        """
        a, b = _as_tensor(a), _as_tensor(b)
        shape = b.shape
        a = a.reshape(-1, 4)
        b = b.reshape(-1, 4)
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return torch.stack(
            (
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ),
            dim=-1,
        ).view(shape)

    def quat_unit(q: torch.Tensor) -> torch.Tensor:
        """Normalize a quaternion. Convention-agnostic; shape (..., 4)."""
        q = _as_tensor(q)
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vectors `v` by WXYZ quaternions `q`.

        Mirrors the reshape-then-broadcast form of the original: call sites pass a
        single quaternion (1, 4) against a whole point cloud (1, N, 3), which a
        same-shape-only implementation rejects with "linalg.cross: inputs must have
        the same number of dimensions".
        """
        q, v = _as_tensor(q), _as_tensor(v)
        shape = v.shape
        q = q.reshape(-1, 4)
        v = v.reshape(-1, 3)
        xyz = q[:, 1:]
        t = torch.cross(xyz, v, dim=-1) * 2
        return (v + q[:, 0:1] * t + torch.cross(xyz, t, dim=-1)).view(shape)

    def tf_inverse(q: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of the rigid transform (q, t). Quaternions are WXYZ."""
        q_inv = quat_conjugate(_as_tensor(q))
        return q_inv, -quat_apply(q_inv, _as_tensor(t))

    def tf_apply(q: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply the rigid transform (q, t) to points v. Quaternions are WXYZ."""
        return quat_apply(q, v) + _as_tensor(t)

    def scale_transform(x, lower, upper):
        """Normalize `x` from [lower, upper] to [-1, 1]. ProxyArray-tolerant wrapper."""
        return _scale_transform(_as_tensor(x), _as_tensor(lower), _as_tensor(upper))
