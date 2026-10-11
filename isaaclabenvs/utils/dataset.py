import dataclasses
import json
import os
import re
import time
from glob import glob, has_magic
from typing import Callable

import numpy as np
import torch
from typing_extensions import Self

from isaaclabenvs.config import DATASET_CONFIG
from isaaclabenvs.config.config import PROJECT_ROOT


_HANDLE_SIDE_TO_SURFACE = {
    "+x": "front",
    "-x": "back",
    "+y": "right",
    "-y": "left",
    "+z": "top",
    "-z": "bottom",
}

_HANDLE_SIDE_ALIASES = {
    "+x": "+x",
    "x+": "+x",
    "-x": "-x",
    "x-": "-x",
    "+y": "+y",
    "y+": "+y",
    "-y": "-y",
    "y-": "-y",
    "+z": "+z",
    "z+": "+z",
    "-z": "-z",
    "z-": "-z",
}


@dataclasses.dataclass
class AssetDataset:
    path: list[str]
    original_indices: list[int]
    config: dict[str, str]
    poses: torch.Tensor  # (N, N_pose, 7), in (x, y, z, qw, qx, qy, qz) format
    pose_counts: torch.Tensor  # (N,), per-asset number of non-padded poses available for sampling
    point_clouds: torch.Tensor  # (N, N_point, 3)
    top_point_clouds: torch.Tensor | None  # (N, N_top_point, 3)
    handle_point_clouds: torch.Tensor | None  # (N, N_handle_point, 3)
    grasps: torch.Tensor | None  # (N, N_grasp, 3 [grasp_site_rel_pos] + 4 [rel_quat_wxyz] + N_finger)

    def __len__(self) -> int:
        return len(self.path)

    def to(self, device: str | torch.device | int) -> Self:
        for field_meta in dataclasses.fields(self):
            field_name = field_meta.name
            current_value = getattr(self, field_name)
            if isinstance(current_value, torch.Tensor):
                setattr(self, field_name, current_value.to(device))
        return self


class AssetLoader:
    name: str
    config: dict[str, str]
    path: list[str]
    original_indices: list[int]
    categories: list[str]
    scenarios: list[str]

    def __init__(self, name: str, valid_asset_file: str = "", invalid_asset_file: str = ""):
        self.name = name.lower()
        # Make a local copy so per-run CLI overrides don't mutate global config.
        self.config = dict(DATASET_CONFIG[self.name])
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

        def _selector_text(selector: tuple[str, ...]) -> str:
            return "+".join(selector)

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

        valid_selectors: list[tuple[str, ...]] = []
        if valid_asset_file:
            with open(valid_asset_file, "r") as f:
                valid_selectors = [selector for selector in (_selector_from_line(line) for line in f) if selector]

        invalid_selectors: list[tuple[str, ...]] = []
        if invalid_asset_file:
            with open(invalid_asset_file, "r") as f:
                invalid_selectors = [selector for selector in (_selector_from_line(line) for line in f) if selector]

        self.valid_asset_names = sorted(_selector_text(selector) for selector in valid_selectors)
        self.invalid_asset_names = sorted(_selector_text(selector) for selector in invalid_selectors)

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

        self.categories = _build_categories(self.path)
        self.scenarios = _build_scenarios(self.path)

        print(f"Loaded {len(self.path)} assets from '{self.name}' dataset.")

    def __len__(self) -> int:
        return len(self.path)

    def chunk(self, indices: list[int]) -> Self:
        if not indices:
            # import ipdb; ipdb.set_trace()
            return self
        
        index_set = set(indices)
        new_paths = []
        new_indices = []
        new_categories = []
        new_scenarios = []

        for p, orig_idx, cat, scenario in zip(
            self.path, self.original_indices, self.categories, self.scenarios, strict=True
        ):
            if orig_idx in index_set:
                new_paths.append(p)
                new_indices.append(orig_idx)
                new_categories.append(cat)
                new_scenarios.append(scenario)

        self.path = new_paths
        self.original_indices = new_indices
        self.categories = new_categories
        self.scenarios = new_scenarios
        return self

    def filter_out(self, robot: str) -> Self:
        new_path = []
        new_indices = []
        new_categories = []
        new_scenarios = []
        for i, p in enumerate(self.path):
            try:
                _ = np.load(os.path.join(p, f"grasp_pose.{robot}.npz"))["state"]
                new_path.append(p)
                new_indices.append(self.original_indices[i])
                new_categories.append(self.categories[i])
                new_scenarios.append(self.scenarios[i])
            except Exception:
                pass

        self.path = new_path
        self.original_indices = new_indices
        self.categories = new_categories
        self.scenarios = new_scenarios

        return self

    def filter_by_scenarios(self, scenarios: list[str]) -> Self:
        if not scenarios:
            return self
        allowed = {s for s in scenarios if s}
        if not allowed:
            return self

        new_path = []
        new_indices = []
        new_categories = []
        new_scenarios = []
        for p, orig_idx, cat, scenario in zip(
            self.path, self.original_indices, self.categories, self.scenarios, strict=True
        ):
            if scenario in allowed:
                new_path.append(p)
                new_indices.append(orig_idx)
                new_categories.append(cat)
                new_scenarios.append(scenario)

        self.path = new_path
        self.original_indices = new_indices
        self.categories = new_categories
        self.scenarios = new_scenarios
        if not self.path:
            raise ValueError(f"No assets left after filtering scenarios: {sorted(allowed)}")

        return self

    def filter_by_categories(self, categories: list[str]) -> Self:
        if not categories:
            return self
        allowed = {c for c in categories if c}
        if not allowed:
            return self

        new_path = []
        new_indices = []
        new_categories = []
        new_scenarios = []
        for p, orig_idx, cat, scenario in zip(
            self.path, self.original_indices, self.categories, self.scenarios, strict=True
        ):
            if cat in allowed:
                new_path.append(p)
                new_indices.append(orig_idx)
                new_categories.append(cat)
                new_scenarios.append(scenario)

        self.path = new_path
        self.original_indices = new_indices
        self.categories = new_categories
        self.scenarios = new_scenarios
        if not self.path:
            raise ValueError(f"No assets left after filtering categories: {sorted(allowed)}")

        return self

    def load(
        self,
        num_grasps: int = 0,
        num_poses: int = 0,
        pose_first_m: int = 0,
        robot: str | None = None,
        pose_pre_sampler: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
        load_top_point_cloud: bool = False,
        load_handle_point_cloud: bool = False,
        prefer_canonical_point_cloud: bool = True,
    ) -> AssetDataset:
        print(f"Loading {len(self.path)} assets from {self.name} dataset ...")

        path = []
        indices = []
        poses = []
        pose_counts = []
        point_clouds = []
        top_point_clouds = []
        handle_point_clouds = []
        grasps = []

        if robot is None and num_grasps > 0:
            raise ValueError("Please specify the robot name when loading grasps.")

        if load_top_point_cloud:
            top_pcd_spec = self.config.get("top_point_cloud", "")
            top_pcd_fallback_glob = self.config.get("top_point_cloud_glob")
            if not top_pcd_spec and not top_pcd_fallback_glob:
                raise KeyError(
                    f"Dataset '{self.name}' is missing top point-cloud config. "
                    "Expected key 'top_point_cloud' or 'top_point_cloud_glob'."
                )
            top_pcd_key = self.config.get("top_point_cloud_key") or self.config.get("point_cloud_key")

        if load_handle_point_cloud:
            handle_pcd_spec = self.config.get("handle_point_cloud", "")
            asset_json_spec = self.config.get("asset_json_path", "")
            if not handle_pcd_spec:
                raise KeyError(
                    f"Dataset '{self.name}' is missing handle point-cloud config. "
                    "Expected key 'handle_point_cloud'."
                )
            if not asset_json_spec:
                raise KeyError(
                    f"Dataset '{self.name}' is missing asset json config. "
                    "Expected key 'asset_json_path'."
                )
            handle_pcd_key = self.config.get("handle_point_cloud_key") or self.config.get("point_cloud_key")
            expected_handle_points: int | None = None

        #import ipdb; ipdb.set_trace()

        for i, p in zip(self.original_indices, self.path, strict=True):
            if num_grasps > 0:
                possible_grasps = torch.tensor(np.load(os.path.join(p, f"grasp_pose.{robot}.npz"))["state"])
                grasps.append(choose_from(possible_grasps, num_grasps))
            # import ipdb; ipdb.set_trace()
            pose_path = _resolve_asset_file(
                p,
                self.config["table_pose"],
                kind="table_pose",
                fallback_glob=self.config.get("table_pose_glob"),
            )
            pcd_path = _resolve_asset_file(
                p,
                self.config["point_cloud"],
                kind="point_cloud",
                fallback_glob=self.config.get("point_cloud_glob"),
                prefer_canonical_point_cloud=prefer_canonical_point_cloud,
            )
            pose_tensor = torch.tensor(_load_array(pose_path, key=self.config.get("table_pose_key"), kind="table_pose"))
            if pose_pre_sampler is not None:
                pose_tensor = pose_pre_sampler(p, pose_tensor)
            original_pose_count = int(pose_tensor.shape[0])
            if original_pose_count <= 0:
                raise ValueError(f"Empty table_pose for asset: {p}")
            if pose_first_m > 0:
                pose_tensor = pose_tensor[:pose_first_m]
                valid_pose_count = min(original_pose_count, int(pose_first_m))
            elif num_poses > 0:
                valid_pose_count = min(original_pose_count, int(num_poses))
                pose_tensor = choose_from(pose_tensor, num_poses)
            else:
                valid_pose_count = original_pose_count
            poses.append(pose_tensor)
            pose_counts.append(valid_pose_count)
            point_clouds.append(
                torch.tensor(_load_array(pcd_path, key=self.config.get("point_cloud_key"), kind="point_cloud"))
            )
            if load_top_point_cloud:
                top_pcd_path = _resolve_asset_file(
                    p,
                    top_pcd_spec,
                    kind="top_point_cloud",
                    fallback_glob=top_pcd_fallback_glob,
                )
                top_point_clouds.append(
                    torch.tensor(_load_array(top_pcd_path, key=top_pcd_key, kind="top_point_cloud"))
                )

            if load_handle_point_cloud:
                json_matches = _resolve_required_matches(p, asset_json_spec, kind="asset_json_path")
                if len(json_matches) != 1:
                    raise RuntimeError(
                        f"Expected exactly one asset json for '{p}', got {len(json_matches)}: {json_matches}"
                    )
                with open(json_matches[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                handle_side = _normalize_handle_side(metadata.get("handle_side"))
                side_name = _HANDLE_SIDE_TO_SURFACE[handle_side]

                handle_matches = _resolve_required_matches(p, handle_pcd_spec, kind="handle_point_cloud")
                side_matches = [
                    cand for cand in handle_matches if f"_{side_name}_" in os.path.basename(cand).lower()
                ]
                if len(side_matches) != 1:
                    raise RuntimeError(
                        f"Expected exactly one handle-side point cloud for asset '{p}', "
                        f"handle_side='{handle_side}' -> side='{side_name}', got {len(side_matches)} "
                        f"matches from {handle_matches}."
                    )

                handle_points = torch.tensor(
                    _load_array(side_matches[0], key=handle_pcd_key, kind="handle_point_cloud")
                )
                if handle_points.ndim != 2 or int(handle_points.shape[1]) != 3:
                    raise ValueError(
                        f"Invalid handle point cloud shape for '{side_matches[0]}': {tuple(handle_points.shape)}. "
                        "Expected (N, 3)."
                    )
                if expected_handle_points is None:
                    expected_handle_points = int(handle_points.shape[0])
                elif int(handle_points.shape[0]) != expected_handle_points:
                    raise ValueError(
                        f"Inconsistent handle-side point count for '{p}': got {int(handle_points.shape[0])}, "
                        f"expected {expected_handle_points}."
                    )
                handle_point_clouds.append(handle_points)

            path.append(p)
            indices.append(i)

        return AssetDataset(
            path=path,
            original_indices=indices,
            config=self.config,
            poses=torch.stack(poses, dim=0).to(torch.float),
            pose_counts=torch.tensor(pose_counts, dtype=torch.int64),
            point_clouds=torch.stack(point_clouds, dim=0).to(torch.float),
            top_point_clouds=torch.stack(top_point_clouds, dim=0).to(torch.float) if load_top_point_cloud else None,
            handle_point_clouds=torch.stack(handle_point_clouds, dim=0).to(torch.float) if load_handle_point_cloud else None,
            grasps=torch.stack(grasps, dim=0).to(torch.float) if num_grasps > 0 else None,
        )

    def sample_by_category_ratio(self, ratio: float, seed: int | None = 0) -> Self:
        if ratio <= 0:
            return self
        if ratio > 1:
            raise ValueError("category_ratio must be in (0, 1].")

        cat_to_indices: dict[str, list[int]] = {}
        for idx, cat in enumerate(self.categories):
            cat_to_indices.setdefault(cat, []).append(idx)

        if seed is None:
            rng = np.random.default_rng(int(time.time_ns()))
        else:
            rng = np.random.default_rng(int(seed))
        selected = []
        for idxs in cat_to_indices.values():
            count = int(np.floor(len(idxs) * ratio))
            if count <= 0:
                continue
            chosen = rng.choice(idxs, size=count, replace=False)
            selected.extend(chosen.tolist())

        selected_set = set(selected)
        self.path = [p for i, p in enumerate(self.path) if i in selected_set]
        self.original_indices = [idx for i, idx in enumerate(self.original_indices) if i in selected_set]
        self.categories = [c for i, c in enumerate(self.categories) if i in selected_set]
        self.scenarios = [s for i, s in enumerate(self.scenarios) if i in selected_set]
        return self


class InitialPose:
    data: torch.Tensor  # [x_num, y_num, num_pose, num_dof]
    x_min: float
    x_step: float
    y_min: float
    y_step: float
    z_min: float | None
    z_step: float | None

    def __init__(self, device: str | torch.device | int, path: str | None = None):
        self.device = device
        if path is None:
            path = os.path.join(PROJECT_ROOT, "data/initial_pose/franka_gripper.npz")
        elif not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        raw = np.load(path)
        self.data = torch.tensor(raw["data"], dtype=torch.float, device=self.device)
        self.x_min, self.x_step = raw["x_min"].item(), raw["x_step"].item()
        self.y_min, self.y_step = raw["y_min"].item(), raw["y_step"].item()
        self.has_z = self.data.ndim == 5
        if self.has_z:
            if "z_min" not in raw or "z_step" not in raw:
                raise ValueError("Initial pose data includes z dimension but z_min/z_step are missing.")
            self.z_min, self.z_step = raw["z_min"].item(), raw["z_step"].item()
            self.x_num, self.y_num, self.z_num, self.n_sample, _ = self.data.shape
        else:
            self.z_min, self.z_step = None, None
            self.z_num = 1
            self.x_num, self.y_num, self.n_sample, _ = self.data.shape

    def sample(self, object_xy: torch.Tensor, table_z: torch.Tensor | None = None) -> torch.Tensor:
        object_i = ((object_xy[:, 0] - self.x_min) / self.x_step).round().clamp(0, self.x_num - 1).long()
        object_j = ((object_xy[:, 1] - self.y_min) / self.y_step).round().clamp(0, self.y_num - 1).long()
        selected = torch.randint(0, self.n_sample, (object_xy.shape[0],), device=object_xy.device)
        if not self.has_z:
            return self.data[object_i, object_j, selected]
        if table_z is None:
            raise ValueError("table_z is required when initial pose data includes a z dimension.")
        if not isinstance(table_z, torch.Tensor):
            table_z = torch.tensor(table_z, dtype=object_xy.dtype, device=object_xy.device)
        elif table_z.device != object_xy.device:
            table_z = table_z.to(object_xy.device)
        object_k = ((table_z - self.z_min) / self.z_step).ceil().clamp(0, self.z_num - 1).long()
        return self.data[object_i, object_j, object_k, selected]


############################################################
# Helper Functions
############################################################


def choose_from(data: torch.Tensor, num: int) -> torch.Tensor:
    assert data.numel() > 0
    if data.shape[0] <= num:
        indices = torch.randint(0, data.shape[0], (num - data.shape[0],), device=data.device)
        return torch.cat([data, data[indices]], dim=0)
    else:
        indices = torch.randperm(data.shape[0], device=data.device)[:num]
        return data[indices]


def _load_array(path: str, *, key: str | None, kind: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if key and key in data:
            return data[key]
        if "arr_0" in data:
            return data["arr_0"]
        if key:
            raise KeyError(f"Missing key '{key}' in {kind} npz: {path}. Keys: {list(data.keys())}")
        if len(data.files) == 1:
            return data[data.files[0]]
        raise KeyError(f"Ambiguous {kind} npz: {path}. Keys: {list(data.keys())}")
    return data


def _build_categories(paths: list[str]) -> list[str]:
    categories: list[str] = []
    for p in paths:
        categories.append(_category_from_path(p, 1))
    return categories


def _build_scenarios(paths: list[str]) -> list[str]:
    scenarios: list[str] = []
    for p in paths:
        scenarios.append(_category_from_path(p, 2))
    return scenarios


def _category_from_path(path: str, level: int) -> str:
    if level < 1:
        return "unknown"
    parts = os.path.normpath(path.rstrip("/")).split(os.sep)
    if len(parts) < level + 1:
        return "unknown"
    return parts[-(level + 1)]



def _normalize_handle_side(value: object) -> str:
    if value is None:
        raise ValueError("Missing 'handle_side' in asset json.")
    text = str(value).strip().lower().replace(" ", "")
    normalized = _HANDLE_SIDE_ALIASES.get(text)
    if normalized is None:
        raise ValueError(
            f"Invalid handle_side '{value}'. Expected one of {sorted(_HANDLE_SIDE_TO_SURFACE)}."
        )
    return normalized


def _resolve_required_matches(asset_dir: str, spec: str, *, kind: str) -> list[str]:
    if not spec:
        raise KeyError(f"Missing dataset config key for {kind}.")

    pattern = spec if os.path.isabs(spec) else os.path.join(asset_dir, spec)
    if has_magic(pattern):
        matches = sorted([path for path in glob(pattern) if os.path.isfile(path)])
        if not matches:
            raise FileNotFoundError(
                f"Failed to locate {kind} for asset '{asset_dir}' with pattern '{pattern}'."
            )
        return matches

    if not os.path.isfile(pattern):
        raise FileNotFoundError(
            f"Failed to locate {kind} for asset '{asset_dir}' with path '{pattern}'."
        )
    return [pattern]


def _resolve_asset_file(
    asset_dir: str,
    spec: str,
    *,
    kind: str,
    fallback_glob: str | None = None,
    prefer_canonical_point_cloud: bool = False,
) -> str:
    # Supports either an explicit relative path (legacy) or a glob pattern.
    def _candidates(pattern: str) -> list[str]:
        return sorted([p for p in glob(pattern) if os.path.isfile(p)])

    def _pick_match(pattern: str, matches: list[str]) -> str:
        chosen = matches[0]
        if kind == "point_cloud" and prefer_canonical_point_cloud and len(matches) > 1:
            strict = [
                m
                for m in matches
                if re.fullmatch(r"asset_\d+_pcd\.npz", os.path.basename(m).lower()) is not None
            ]
            if strict:
                chosen = strict[0]
            else:
                token_blocklist = ("front", "back", "left", "right", "top", "bottom", "rerun")
                generic = []
                for m in matches:
                    name = os.path.basename(m).lower()
                    if not name.endswith("_pcd.npz"):
                        continue
                    if re.search(r"_r\d+_pcd\.npz$", name):
                        continue
                    if any(token in name for token in token_blocklist):
                        continue
                    generic.append(m)
                if generic:
                    chosen = generic[0]
        # debug print            
        # if len(matches) > 1:
        #     print(f"[dataset] {kind}: multiple matches for '{pattern}', using '{chosen}'")
        return chosen

    if spec:
        pattern = spec if os.path.isabs(spec) else os.path.join(asset_dir, spec)
        if has_magic(pattern):
            matches = _candidates(pattern)
            if matches:
                return _pick_match(pattern, matches)
        else:
            if os.path.isfile(pattern):
                return pattern

    if fallback_glob:
        pattern = fallback_glob if os.path.isabs(fallback_glob) else os.path.join(asset_dir, fallback_glob)
        matches = _candidates(pattern)
        if matches:
            return _pick_match(pattern, matches)

    raise FileNotFoundError(
        f"Failed to locate {kind} for asset '{asset_dir}'. "
        f"Tried spec='{spec}' and fallback_glob='{fallback_glob}'."
    )
