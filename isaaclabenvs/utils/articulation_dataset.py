from __future__ import annotations

import dataclasses
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from glob import glob, has_magic

import numpy as np
import torch
from typing_extensions import Self

from isaaclabenvs.config import DATASET_CONFIG
from isaaclabenvs.config.config import PROJECT_ROOT

JOINT_MOTION_FIXED = 0
JOINT_MOTION_REVOLUTE = 1
JOINT_MOTION_PRISMATIC = 2
REVOLUTE_TYPES = {"revolute", "continuous"}
PRISMATIC_TYPES = {"prismatic"}


@dataclasses.dataclass
class ArticulationAssetDataset:
    path: list[str]
    original_indices: list[int]
    config: dict[str, str]
    usd_paths: list[str]
    variant_stems: list[str] = dataclasses.field(default_factory=list)

    def __len__(self) -> int:
        return len(self.path)

    def to(self, device: str | torch.device | int) -> Self:
        # Keep API parity with other dataset classes.
        _ = device
        return self


class ArticulationAssetLoader:
    name: str
    config: dict[str, str]
    path: list[str]
    original_indices: list[int]

    def __init__(self, name: str, valid_asset_file: str = "", invalid_asset_file: str = ""):
        self.name = name.lower()
        self.config = dict(DATASET_CONFIG[self.name])
        required_fields = (
            "path",
            "usd_path",
            "urdf_path",
            "handle_points",
            "handle_points_key",
            "init_arm_joint_pos",
            "init_arm_joint_pos_key",
        )
        missing = [key for key in required_fields if not str(self.config.get(key, "")).strip()]
        if missing:
            raise KeyError(
                f"Dataset '{self.name}' missing required articulated config fields: {missing}. "
                "Please define them in config.toml."
            )
        if valid_asset_file:
            self.config["valid_asset_file"] = valid_asset_file
        if invalid_asset_file:
            self.config["invalid_asset_file"] = invalid_asset_file

        paths = [path for path in glob(self.config["path"]) if os.path.isdir(path)]
        self.path = sorted(paths)
        self.original_indices = list(range(len(paths)))
        if not self.path:
            raise ValueError(f"Failed to match any asset path for '{self.config['path']}'")

        def _asset_parts_from_path(p: str) -> tuple[str, str, str]:
            parts = os.path.normpath(p.rstrip("/")).split(os.sep)
            if len(parts) >= 3:
                return parts[-3], parts[-2], parts[-1]
            if len(parts) == 2:
                return "unknown", parts[-2], parts[-1]
            if len(parts) == 1:
                return "unknown", "unknown", parts[-1]
            return "unknown", "unknown", "unknown"

        def _selector_from_line(line: str) -> tuple[str, ...] | None:
            s = line.strip()
            if not s or s.startswith("#"):
                return None

            if "+" in s:
                parts = [part.strip() for part in s.split("+") if part.strip()]
            elif "/" in s:
                parts = [part.strip() for part in s.split("/") if part.strip()]
            else:
                parts = [s]

            if not parts:
                return None
            if len(parts) > 3:
                parts = parts[-3:]
            return tuple(parts)

        def _match_selector(selector: tuple[str, ...], scenario: str, category: str, asset: str) -> bool:
            if len(selector) == 3:
                return selector == (scenario, category, asset)
            if len(selector) == 2:
                left, right = selector
                return (
                    (scenario == left and category == right)
                    or (scenario == left and asset == right)
                    or (category == left and asset == right)
                )
            if len(selector) == 1:
                token = selector[0]
                return token in {scenario, category, asset}
            return False

        def _resolve_optional_path(path: str) -> str:
            if not path:
                return ""
            if os.path.isabs(path):
                return path
            return os.path.join(PROJECT_ROOT, path)

        valid_asset_file = _resolve_optional_path(self.config.get("valid_asset_file", ""))
        invalid_asset_file = _resolve_optional_path(self.config.get("invalid_asset_file", ""))

        valid_selectors: set[tuple[str, ...]] = set()
        if valid_asset_file:
            with open(valid_asset_file, "r") as f:
                valid_selectors = {selector for selector in (_selector_from_line(line) for line in f) if selector}

        invalid_selectors: set[tuple[str, ...]] = set()
        if invalid_asset_file:
            with open(invalid_asset_file, "r") as f:
                invalid_selectors = {selector for selector in (_selector_from_line(line) for line in f) if selector}

        if valid_selectors:
            new_paths = []
            new_indices = []
            for idx, p in enumerate(self.path):
                scenario, category, asset = _asset_parts_from_path(p)
                if any(_match_selector(selector, scenario, category, asset) for selector in valid_selectors):
                    new_paths.append(p)
                    new_indices.append(self.original_indices[idx])
            self.path = new_paths
            self.original_indices = new_indices

        if invalid_selectors:
            new_paths = []
            new_indices = []
            for idx, p in enumerate(self.path):
                scenario, category, asset = _asset_parts_from_path(p)
                if not any(_match_selector(selector, scenario, category, asset) for selector in invalid_selectors):
                    new_paths.append(p)
                    new_indices.append(self.original_indices[idx])
            self.path = new_paths
            self.original_indices = new_indices

        print(f"Loaded {len(self.path)} assets from '{self.name}' articulated dataset.")

    def __len__(self) -> int:
        return len(self.path)

    def chunk(self, indices: list[int] | range) -> Self:
        if not indices:
            return self
        index_set = set(indices)
        new_paths = []
        new_indices = []
        for p, orig_idx in zip(self.path, self.original_indices, strict=True):
            if orig_idx in index_set:
                new_paths.append(p)
                new_indices.append(orig_idx)
        self.path = new_paths
        self.original_indices = new_indices
        return self

    def filter_by_asset_ids(self, asset_ids: list[str]) -> Self:
        if not asset_ids:
            return self
        tokens = [str(token).strip() for token in asset_ids if str(token).strip()]
        if not tokens:
            return self

        exact_tokens = set(tokens)
        base_ids = {token for token in tokens if "_joint_" not in token}

        new_paths = []
        new_indices = []
        for p, orig_idx in zip(self.path, self.original_indices, strict=True):
            asset_name = os.path.basename(os.path.normpath(p))
            asset_base = asset_name.split("_joint_", 1)[0]
            if asset_name in exact_tokens or asset_base in base_ids:
                new_paths.append(p)
                new_indices.append(orig_idx)

        self.path = new_paths
        self.original_indices = new_indices
        if not self.path:
            raise ValueError(f"No assets left after filtering asset_ids: {tokens}")
        return self

    def _resolve_asset_file(
        self,
        asset_dir: str,
        spec: str,
        kind: str,
        *,
        stem: str | None = None,
    ) -> str:
        def _candidates(pattern: str) -> list[str]:
            return sorted([p for p in glob(pattern) if os.path.isfile(p)])

        pattern = spec if os.path.isabs(spec) else os.path.join(asset_dir, spec)
        if has_magic(pattern):
            matches = _candidates(pattern)
            if not matches:
                raise FileNotFoundError(
                    f"Failed to locate {kind} for asset '{asset_dir}'. "
                    f"Glob pattern '{pattern}' matched no files."
                )
            if stem:
                stem_matches = [m for m in matches if os.path.basename(m).startswith(stem)]
                if stem_matches:
                    if len(stem_matches) > 1:
                        print(
                            f"[articulation_dataset] {kind}: multiple stem matches for "
                            f"'{pattern}' (stem='{stem}'), using '{stem_matches[0]}'"
                        )
                    return stem_matches[0]
            if len(matches) > 1:
                print(f"[articulation_dataset] {kind}: multiple matches for '{pattern}', using '{matches[0]}'")
            return matches[0]
        if os.path.isfile(pattern):
            return pattern

        raise FileNotFoundError(
            f"Failed to locate {kind} for asset '{asset_dir}'. Expected file: '{pattern}'."
        )

    @staticmethod
    def _stem_from_usd(usd_path: str) -> str:
        base = os.path.basename(usd_path)
        for suffix in ("_physics.usd", ".usd"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return os.path.splitext(base)[0]

    def _expand_variant_usd_paths(self, asset_dir: str) -> list[str]:
        spec = self.config["usd_path"]
        pattern = spec if os.path.isabs(spec) else os.path.join(asset_dir, spec)
        if has_magic(pattern):
            return sorted(p for p in glob(pattern) if os.path.isfile(p))
        if os.path.isfile(pattern):
            return [pattern]
        return []

    def load(self) -> ArticulationAssetDataset:
        if len(self.path) == 0:
            raise ValueError("No articulated assets to load.")

        path: list[str] = []
        indices: list[int] = []
        usd_paths: list[str] = []
        variant_stems: list[str] = []
        next_index = 0
        for asset_dir in self.path:
            variant_usds = self._expand_variant_usd_paths(asset_dir)
            if not variant_usds:
                raise FileNotFoundError(
                    f"Failed to locate usd_path for asset '{asset_dir}'. "
                    f"Pattern '{self.config['usd_path']}' matched no files."
                )
            for usd in variant_usds:
                path.append(asset_dir)
                indices.append(next_index)
                usd_paths.append(usd)
                variant_stems.append(self._stem_from_usd(usd))
                next_index += 1

        return ArticulationAssetDataset(
            path=path,
            original_indices=indices,
            config=self.config,
            usd_paths=usd_paths,
            variant_stems=variant_stems,
        )


@dataclasses.dataclass
class _JointRef:
    name: str
    joint_type: str
    parent: str
    child: str
    axis_xyz: np.ndarray
    origin_T: np.ndarray


def _parse_xyz(raw: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not raw:
        return default
    values = [x for x in str(raw).split() if x]
    if len(values) != 3:
        return default
    return (float(values[0]), float(values[1]), float(values[2]))


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return v / n


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_elem_to_T(origin_elem: ET.Element | None) -> np.ndarray:
    xyz = _parse_xyz(origin_elem.attrib.get("xyz") if origin_elem is not None else None, (0.0, 0.0, 0.0))
    rpy = _parse_xyz(origin_elem.attrib.get("rpy") if origin_elem is not None else None, (0.0, 0.0, 0.0))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy_to_matrix(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def _parse_urdf_joint_refs(urdf_path: str) -> tuple[list[str], list[_JointRef]]:
    root = ET.parse(urdf_path).getroot()
    links: list[str] = []
    joints: list[_JointRef] = []

    for link in root.findall("link"):
        name = str(link.attrib.get("name", "")).strip()
        if name:
            links.append(name)

    for joint in root.findall("joint"):
        name = str(joint.attrib.get("name", "")).strip()
        joint_type = str(joint.attrib.get("type", "")).strip()
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        parent = str(parent_elem.attrib.get("link", "")).strip() if parent_elem is not None else ""
        child = str(child_elem.attrib.get("link", "")).strip() if child_elem is not None else ""
        if not name or not parent or not child:
            continue
        axis_elem = joint.find("axis")
        axis_xyz = np.asarray(
            _parse_xyz(axis_elem.attrib.get("xyz") if axis_elem is not None else None, (1.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        joints.append(
            _JointRef(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                axis_xyz=_normalize(axis_xyz),
                origin_T=_origin_elem_to_T(joint.find("origin")),
            )
        )
    return links, joints


def _compute_link_transforms_at_zero(links: list[str], joints: list[_JointRef]) -> dict[str, np.ndarray]:
    children = {joint.child for joint in joints}
    root_candidates = [ln for ln in links if ln not in children]
    roots = sorted(root_candidates) if root_candidates else (sorted(links)[:1] if links else [])
    if not roots:
        raise ValueError("No root link found in URDF.")

    children_joints: dict[str, list[_JointRef]] = defaultdict(list)
    for joint in joints:
        children_joints[joint.parent].append(joint)

    T_link_obj: dict[str, np.ndarray] = {}
    queue: list[str] = []
    for root in roots:
        T_link_obj[root] = np.eye(4, dtype=np.float64)
        queue.append(root)

    while queue:
        parent = queue.pop()
        T_parent = T_link_obj[parent]
        for joint in children_joints.get(parent, []):
            if joint.child in T_link_obj:
                continue
            # At q=0, motion transform is identity.
            T_child = T_parent @ joint.origin_T
            T_link_obj[joint.child] = T_child
            queue.append(joint.child)

    for link in links:
        if link not in T_link_obj:
            T_link_obj[link] = np.eye(4, dtype=np.float64)
    return T_link_obj


@dataclasses.dataclass
class ArticulationInitArmGrid:
    x_min: float
    x_step: float
    x_num: int
    y_min: float
    y_step: float
    y_num: int
    z_min: float
    z_step: float
    z_num: int
    n_sample: int


@dataclasses.dataclass
class ArticulationMetaDataset:
    path: list[str]
    handle_points: torch.Tensor  # (N, N_handle, 3)
    init_arm_joint_pos: torch.Tensor  # (x_num, y_num, z_num, n_sample, N_arm_dof) — global, shared across assets
    init_arm_eef_dir: torch.Tensor  # (x_num, y_num, z_num, n_sample, 3) — global EEF axis (fingertip->wrist) in world frame
    init_arm_grid: ArticulationInitArmGrid
    joint_motion_type: torch.Tensor  # (N,), {0: fixed, 1: revolute, 2: prismatic}
    joint_axis_obj: torch.Tensor  # (N, 3)
    joint_origin_obj: torch.Tensor  # (N, 3)
    link_forward_points: torch.Tensor | None = None  # (N, N_forward, 3) or None when not configured
    # (N, N_grasp, 8) per-asset hand-crafted handle grasps in q0/object frame, or None when
    # not configured. Layout: [:3] grasp_site pos, [3:7] grasp_site quat (wxyz), [7] finger joint.
    handle_grasps: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.path)

    def to(self, device: str | torch.device | int) -> Self:
        self.handle_points = self.handle_points.to(device)
        self.init_arm_joint_pos = self.init_arm_joint_pos.to(device)
        self.init_arm_eef_dir = self.init_arm_eef_dir.to(device)
        self.joint_motion_type = self.joint_motion_type.to(device)
        self.joint_axis_obj = self.joint_axis_obj.to(device)
        self.joint_origin_obj = self.joint_origin_obj.to(device)
        if self.link_forward_points is not None:
            self.link_forward_points = self.link_forward_points.to(device)
        if self.handle_grasps is not None:
            self.handle_grasps = self.handle_grasps.to(device)
        return self


class ArticulationMetaLoader:
    path: list[str]
    variant_stems: list[str]
    urdf_spec: str
    handle_points_spec: str
    handle_points_key: str
    init_arm_joint_pos_spec: str
    init_arm_joint_pos_key: str
    init_arm_eef_dir_key: str
    link_forward_points_spec: str
    link_forward_points_key: str
    handle_grasps_spec: str

    def __init__(
        self,
        path: list[str],
        *,
        urdf_spec: str,
        handle_points_spec: str,
        handle_points_key: str,
        init_arm_joint_pos_spec: str,
        init_arm_joint_pos_key: str,
        init_arm_eef_dir_key: str,
        link_forward_points_spec: str = "",
        link_forward_points_key: str = "",
        handle_grasps_spec: str = "",
        variant_stems: list[str] | None = None,
    ):
        self.path = path
        if variant_stems is None:
            self.variant_stems = ["" for _ in path]
        else:
            if len(variant_stems) != len(path):
                raise ValueError(
                    f"variant_stems length {len(variant_stems)} does not match path length {len(path)}."
                )
            self.variant_stems = list(variant_stems)
        self.urdf_spec = str(urdf_spec).strip()
        self.handle_points_spec = str(handle_points_spec).strip()
        self.handle_points_key = str(handle_points_key).strip()
        self.init_arm_joint_pos_spec = str(init_arm_joint_pos_spec).strip()
        self.init_arm_joint_pos_key = str(init_arm_joint_pos_key).strip()
        self.init_arm_eef_dir_key = str(init_arm_eef_dir_key).strip()
        self.link_forward_points_spec = str(link_forward_points_spec).strip()
        self.link_forward_points_key = str(link_forward_points_key).strip()
        self.handle_grasps_spec = str(handle_grasps_spec).strip()

        missing = []
        if not self.urdf_spec:
            missing.append("urdf_spec")
        if not self.handle_points_spec:
            missing.append("handle_points_spec")
        if not self.handle_points_key:
            missing.append("handle_points_key")
        if not self.init_arm_joint_pos_spec:
            missing.append("init_arm_joint_pos_spec")
        if not self.init_arm_joint_pos_key:
            missing.append("init_arm_joint_pos_key")
        if not self.init_arm_eef_dir_key:
            missing.append("init_arm_eef_dir_key")
        if missing:
            raise ValueError(f"ArticulationMetaLoader missing required specs: {missing}")
        if bool(self.link_forward_points_spec) != bool(self.link_forward_points_key):
            raise ValueError(
                "link_forward_points_spec and link_forward_points_key must both be set or both empty."
            )
        # init_arm is global (shared across all assets), so it is a single file, never a per-asset
        # glob; a relative path resolves against the repo root like the other dataset paths.
        if not os.path.isabs(self.init_arm_joint_pos_spec):
            self.init_arm_joint_pos_spec = os.path.join(PROJECT_ROOT, self.init_arm_joint_pos_spec)

    def _resolve_asset_file(
        self,
        asset_dir: str,
        spec: str,
        *,
        kind: str,
        stem: str | None = None,
    ) -> str:
        pattern = spec if os.path.isabs(spec) else os.path.join(asset_dir, spec)
        if has_magic(pattern):
            matches = sorted([p for p in glob(pattern) if os.path.isfile(p)])
            if not matches:
                raise FileNotFoundError(
                    f"Failed to locate {kind} for asset '{asset_dir}'. "
                    f"Glob pattern '{pattern}' matched no files."
                )
            if stem:
                stem_matches = [m for m in matches if os.path.basename(m).startswith(stem)]
                if stem_matches:
                    if len(stem_matches) > 1:
                        print(
                            f"[articulation_dataset] {kind}: multiple stem matches for "
                            f"'{pattern}' (stem='{stem}'), using '{stem_matches[0]}'"
                        )
                    return stem_matches[0]
            if len(matches) > 1:
                print(f"[articulation_dataset] {kind}: multiple matches for '{pattern}', using '{matches[0]}'")
            return matches[0]
        if os.path.isfile(pattern):
            return pattern
        raise FileNotFoundError(f"Failed to locate {kind} for asset '{asset_dir}'. Expected file: '{pattern}'.")

    def _load_npz_array_exact(self, path: str, *, key: str, kind: str) -> np.ndarray:
        data = np.load(path)
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {kind} npz: {path}. Keys: {list(data.files)}")
        return np.asarray(data[key])

    def _extract_joint_motion_meta(self, urdf_path: str) -> tuple[int, np.ndarray, np.ndarray]:
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"urdf not found: {urdf_path}")

        links, joints = _parse_urdf_joint_refs(urdf_path)
        movable = [
            j for j in joints
            if j.joint_type.lower() in REVOLUTE_TYPES or j.joint_type.lower() in PRISMATIC_TYPES
        ]
        if len(movable) != 1:
            names = [j.name for j in movable]
            raise ValueError(
                f"Expected exactly 1 non-fixed joint in URDF '{urdf_path}', got {len(movable)}: {names}"
            )
        target_joint = movable[0]

        T_link_obj = _compute_link_transforms_at_zero(links, joints)
        if target_joint.parent not in T_link_obj:
            raise ValueError(
                f"target joint parent '{target_joint.parent}' transform unavailable in URDF: {urdf_path}"
            )

        parent_T = T_link_obj[target_joint.parent]
        axis_obj = _normalize(parent_T[:3, :3] @ target_joint.axis_xyz)
        joint_origin_obj = (parent_T @ target_joint.origin_T)[:3, 3]

        joint_type = target_joint.joint_type.lower()
        if joint_type in REVOLUTE_TYPES:
            motion_type = JOINT_MOTION_REVOLUTE
        else:
            motion_type = JOINT_MOTION_PRISMATIC

        return motion_type, axis_obj.astype(np.float32), joint_origin_obj.astype(np.float32)

    def load(self, device: str | torch.device | int) -> ArticulationMetaDataset:
        num_assets = len(self.path)
        handle_points: list[torch.Tensor] = []
        joint_motion_type = torch.full((num_assets,), JOINT_MOTION_FIXED, dtype=torch.int64)
        joint_axis_obj = torch.zeros((num_assets, 3), dtype=torch.float)
        joint_axis_obj[:, 0] = 1.0
        joint_origin_obj = torch.zeros((num_assets, 3), dtype=torch.float)
        load_link_forward = bool(self.link_forward_points_spec)
        link_forward_points: list[torch.Tensor] = []
        load_handle_grasps = bool(self.handle_grasps_spec)
        handle_grasps: list[torch.Tensor] = []

        init_arm_path = self.init_arm_joint_pos_spec
        if not os.path.isfile(init_arm_path):
            raise FileNotFoundError(f"init_arm_joint_pos file not found: {init_arm_path}")
        init_arm_npz = np.load(init_arm_path)
        arm_data = np.asarray(init_arm_npz[self.init_arm_joint_pos_key], dtype=np.float32)
        if arm_data.ndim != 5:
            raise ValueError(
                f"Invalid init arm data shape at {init_arm_path}: {arm_data.shape}. "
                "Expected (x_num, y_num, z_num, n_sample, arm_dof)."
            )
        eef_dir_data = np.asarray(init_arm_npz[self.init_arm_eef_dir_key], dtype=np.float32)
        if eef_dir_data.shape != arm_data.shape[:4] + (3,):
            raise ValueError(
                f"Invalid init_arm_eef_dir shape at {init_arm_path}: {eef_dir_data.shape}. "
                f"Expected {arm_data.shape[:4] + (3,)} to match grid + n_sample of init_arm data."
            )
        init_arm_joint_pos_tensor = torch.from_numpy(arm_data).to(torch.float)
        init_arm_eef_dir_tensor = torch.from_numpy(eef_dir_data).to(torch.float)
        init_arm_grid = ArticulationInitArmGrid(
            x_min=float(init_arm_npz["x_min"]),
            x_step=float(init_arm_npz["x_step"]),
            x_num=int(arm_data.shape[0]),
            y_min=float(init_arm_npz["y_min"]),
            y_step=float(init_arm_npz["y_step"]),
            y_num=int(arm_data.shape[1]),
            z_min=float(init_arm_npz["z_min"]),
            z_step=float(init_arm_npz["z_step"]),
            z_num=int(arm_data.shape[2]),
            n_sample=int(arm_data.shape[3]),
        )

        for i, asset_path in enumerate(self.path):
            stem = self.variant_stems[i] if i < len(self.variant_stems) else ""
            stem_arg = stem or None
            urdf_path = self._resolve_asset_file(asset_path, self.urdf_spec, kind="urdf_path", stem=stem_arg)
            motion_type, axis_obj, origin_obj = self._extract_joint_motion_meta(urdf_path)
            joint_motion_type[i] = int(motion_type)
            joint_axis_obj[i] = torch.from_numpy(axis_obj)
            joint_origin_obj[i] = torch.from_numpy(origin_obj)

            handle_points_path = self._resolve_asset_file(
                asset_path, self.handle_points_spec, kind="handle_points", stem=stem_arg
            )
            points = self._load_npz_array_exact(
                handle_points_path,
                key=self.handle_points_key,
                kind="handle_points",
            )
            points = np.asarray(points, dtype=np.float32)
            if points.ndim != 2 or points.shape[-1] != 3:
                raise ValueError(
                    f"Invalid handle points shape at {handle_points_path}: {points.shape}. "
                    "Expected (N, 3)."
                )
            handle_points.append(torch.from_numpy(points))

            if load_link_forward:
                fwd_path = self._resolve_asset_file(
                    asset_path, self.link_forward_points_spec, kind="link_forward_points", stem=stem_arg
                )
                fwd = self._load_npz_array_exact(
                    fwd_path,
                    key=self.link_forward_points_key,
                    kind="link_forward_points",
                )
                fwd = np.asarray(fwd, dtype=np.float32)
                if fwd.ndim != 2 or fwd.shape[-1] != 3:
                    raise ValueError(
                        f"Invalid link_forward_points shape at {fwd_path}: {fwd.shape}. "
                        "Expected (N, 3)."
                    )
                link_forward_points.append(torch.from_numpy(fwd))

            if load_handle_grasps:
                grasps_path = self._resolve_asset_file(
                    asset_path, self.handle_grasps_spec, kind="handle_grasps", stem=stem_arg
                )
                grasp_npz = np.load(grasps_path)
                for gk in ("pos", "quat", "finger"):
                    if gk not in grasp_npz:
                        raise KeyError(
                            f"Missing key '{gk}' in handle_grasps npz: {grasps_path}. Keys: {list(grasp_npz.files)}"
                        )
                g_pos = np.asarray(grasp_npz["pos"], dtype=np.float32)
                g_quat = np.asarray(grasp_npz["quat"], dtype=np.float32)
                g_finger = np.asarray(grasp_npz["finger"], dtype=np.float32).reshape(-1, 1)
                if g_pos.ndim != 2 or g_pos.shape[-1] != 3:
                    raise ValueError(
                        f"Invalid handle_grasps 'pos' shape at {grasps_path}: {g_pos.shape}. Expected (N_grasp, 3)."
                    )
                if g_quat.shape != (g_pos.shape[0], 4):
                    raise ValueError(
                        f"Invalid handle_grasps 'quat' shape at {grasps_path}: {g_quat.shape}. "
                        f"Expected ({g_pos.shape[0]}, 4) (wxyz)."
                    )
                if g_finger.shape != (g_pos.shape[0], 1):
                    raise ValueError(
                        f"Invalid handle_grasps 'finger' shape at {grasps_path}: {g_finger.shape}. "
                        f"Expected ({g_pos.shape[0]},)."
                    )
                # Optional density-aware jaw OPEN width (per-finger approach opening); default to fully
                # open (0.04) when absent so legacy grasp npz behave exactly as before.
                if "open" in grasp_npz:
                    g_open = np.asarray(grasp_npz["open"], dtype=np.float32).reshape(-1, 1)
                else:
                    g_open = np.full((g_pos.shape[0], 1), 0.04, dtype=np.float32)
                grasp = np.concatenate([g_pos, g_quat, g_finger, g_open], axis=-1)  # (N_grasp, 9)
                handle_grasps.append(torch.from_numpy(grasp))

        if len(handle_points) != num_assets:
            raise RuntimeError(
                f"Internal error when loading handle points: loaded={len(handle_points)} expected={num_assets}."
            )
        try:
            handle_points_tensor = torch.stack(handle_points, dim=0).to(torch.float)
        except RuntimeError as e:
            shapes = [tuple(t.shape) for t in handle_points[:8]]
            raise RuntimeError(
                "Failed to stack handle points across assets. "
                f"Ensure all assets have same point count. Example shapes: {shapes}"
            ) from e

        link_forward_points_tensor: torch.Tensor | None = None
        if load_link_forward:
            if len(link_forward_points) != num_assets:
                raise RuntimeError(
                    f"Internal error when loading link_forward points: loaded={len(link_forward_points)} "
                    f"expected={num_assets}."
                )
            try:
                link_forward_points_tensor = torch.stack(link_forward_points, dim=0).to(torch.float)
            except RuntimeError as e:
                shapes = [tuple(t.shape) for t in link_forward_points[:8]]
                raise RuntimeError(
                    "Failed to stack link_forward points across assets. "
                    f"Ensure all assets have same point count. Example shapes: {shapes}"
                ) from e

        handle_grasps_tensor: torch.Tensor | None = None
        if load_handle_grasps:
            if len(handle_grasps) != num_assets:
                raise RuntimeError(
                    f"Internal error when loading handle_grasps: loaded={len(handle_grasps)} "
                    f"expected={num_assets}."
                )
            try:
                handle_grasps_tensor = torch.stack(handle_grasps, dim=0).to(torch.float)
            except RuntimeError as e:
                shapes = [tuple(t.shape) for t in handle_grasps[:8]]
                raise RuntimeError(
                    "Failed to stack handle_grasps across assets. "
                    f"Ensure all assets have same grasp count. Example shapes: {shapes}"
                ) from e

        return ArticulationMetaDataset(
            path=list(self.path),
            handle_points=handle_points_tensor,
            init_arm_joint_pos=init_arm_joint_pos_tensor,
            init_arm_eef_dir=init_arm_eef_dir_tensor,
            init_arm_grid=init_arm_grid,
            joint_motion_type=joint_motion_type,
            joint_axis_obj=joint_axis_obj,
            joint_origin_obj=joint_origin_obj,
            link_forward_points=link_forward_points_tensor,
            handle_grasps=handle_grasps_tensor,
        ).to(device)
