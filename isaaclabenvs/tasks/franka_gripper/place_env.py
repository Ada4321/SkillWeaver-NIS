from __future__ import annotations

import enum
import os

from isaaclabenvs.utils import math_compat as torch_utils  # WXYZ; replaces isaacsim.core.utils.torch (gone in Isaac Sim 6.0)
from isaaclabenvs.utils import quat_compat as qc
import torch

from isaaclabenvs.config import PROJECT_ROOT
from isaaclabenvs.utils.dataset import InitialPose

from .base_env import BaseEnv
from .place_env_cfg import PlaceEnvCfg


class TaskStatus(enum.IntEnum):
    SUCCESS = enum.auto()
    FAILURE = enum.auto()
    TIMEOUT = enum.auto()


class PlaceEnv(BaseEnv):
    cfg: PlaceEnvCfg
    _GOAL_POSE_MODE_SPLIT = 200

    def __init__(self, cfg: PlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.statistics = {
            "success_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "timeout_rate": torch.ones(1, dtype=torch.float, device=self.device),
            "failure_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "reached_rate": torch.zeros(1, dtype=torch.float, device=self.device),
            "arm_max_abs_tracking_error": torch.zeros(1, dtype=torch.float, device=self.device),
        }

        self.latest_status = torch.full((self.num_envs,), TaskStatus.TIMEOUT, dtype=torch.int, device=self.device)
        self.latest_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.min_diff_pos = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.min_diff_rot = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.min_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reached_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.consecutive_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.goal_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.diff_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)

        # Table
        self.table_size: tuple[int, int, int] = getattr(self.cfg.table.spawn, "size", None) or (0.8, 2.0, 0.02)
        self.table_pose = torch.tensor(
            [[*self.cfg.table.init_state.pos, *self.cfg.table.init_state.rot]], device=self.device
        ).repeat(self.num_envs, 1)
        self.table_pose[:, :3] += self.scene.env_origins


        self.initial_pose = InitialPose(self.device, self.cfg.initial_pose_path)

        self._goal_pose_debug_draw = None
        self._goal_pose_debug_draw_failed = False
        if self.cfg.visualize_goal_pose_marker:
            self._init_goal_pose_debug_draw()

    def _setup_scene(self):
        # load dataset
        self._load_dataset()
        # setup scene as usual
        super()._setup_scene()


    def _init_goal_pose_debug_draw(self):
        if self._goal_pose_debug_draw is not None or self._goal_pose_debug_draw_failed:
            return
        try:
            from omni.isaac.debug_draw import _debug_draw as debug_draw_module
        except Exception:
            try:
                import omni.isaac.debug_draw._debug_draw as debug_draw_module
            except Exception as exc:
                self._goal_pose_debug_draw_failed = True
                print(f"[place_goal_pose_marker] debug draw unavailable: {exc}")
                return
        try:
            self._goal_pose_debug_draw = debug_draw_module.acquire_debug_draw_interface()
        except Exception as exc:
            self._goal_pose_debug_draw_failed = True
            print(f"[place_goal_pose_marker] failed to acquire debug draw interface: {exc}")

    def _update_goal_pose_debug_draw(self):
        if not self.cfg.visualize_goal_pose_marker:
            return
        self._init_goal_pose_debug_draw()
        if self._goal_pose_debug_draw is None:
            return

        max_envs = int(self.cfg.goal_pose_marker_max_envs)
        if max_envs <= 0:
            self._goal_pose_debug_draw.clear_lines()
            return

        num_vis = min(max_envs, self.num_envs)
        env_ids = torch.arange(num_vis, device=self.device, dtype=torch.long)
        pos_world = self.scene.env_origins[env_ids] + self.goal_pose[env_ids, :3]
        quat_world = self.goal_pose[env_ids, 3:7]

        axis_len = float(self.cfg.goal_pose_marker_axis_length)
        local_axes = torch.tensor(
            [[axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, axis_len]],
            dtype=torch.float32,
            device=self.device,
        )
        axis_world = torch_utils.quat_apply(
            quat_world.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 4),
            local_axes.unsqueeze(0).expand(num_vis, -1, -1).reshape(-1, 3),
        ).reshape(num_vis, 3, 3)
        end_world = pos_world.unsqueeze(1) + axis_world

        starts: list[tuple[float, float, float]] = []
        ends: list[tuple[float, float, float]] = []
        colors: list[tuple[float, float, float, float]] = []
        widths: list[float] = []
        axis_colors = (
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
        )
        line_width = float(self.cfg.goal_pose_marker_line_width)
        for i in range(num_vis):
            s = tuple(float(v) for v in pos_world[i].tolist())
            for j in range(3):
                e = tuple(float(v) for v in end_world[i, j].tolist())
                starts.append(s)
                ends.append(e)
                colors.append(axis_colors[j])
                widths.append(line_width)

        try:
            self._goal_pose_debug_draw.clear_lines()
            self._goal_pose_debug_draw.draw_lines(starts, ends, colors, widths)
        except Exception as exc:
            if not self._goal_pose_debug_draw_failed:
                print(f"[place_goal_pose_marker] draw failed: {exc}")
            self._goal_pose_debug_draw_failed = True
            self._goal_pose_debug_draw = None

    @staticmethod
    def _asset_parts_from_path(path: str) -> tuple[str, str, str]:
        parts = os.path.normpath(path.rstrip("/")).split(os.sep)
        if len(parts) >= 3:
            return parts[-3], parts[-2], parts[-1]
        if len(parts) == 2:
            return "unknown", parts[-2], parts[-1]
        if len(parts) == 1:
            return "unknown", "unknown", parts[-1]
        return "unknown", "unknown", "unknown"

    @staticmethod
    def _parse_goal_pose_selector(text: str, line_no: int, list_path: str) -> tuple[str, str, str] | None:
        s = text.split("#", 1)[0].strip()
        if not s:
            return None
        parts = [part.strip() for part in s.split("/") if part.strip()]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid selector at {list_path}:{line_no}. "
                f"Expected 'scenario/category/asset_xxxx', got '{text.strip()}'."
            )
        scenario, category, asset = parts
        if not (asset.startswith("asset_") and asset[6:].isdigit()):
            raise ValueError(
                f"Invalid asset id at {list_path}:{line_no}. "
                f"Expected 'asset_xxxx', got '{asset}'."
            )
        return scenario, category, asset

    def _read_goal_pose_selectors(self, path: str) -> list[tuple[str, str, str]]:
        if not path:
            return []
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        if not os.path.isfile(path):
            return []
        selectors: list[tuple[str, str, str]] = []
        with open(path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                selector = self._parse_goal_pose_selector(line, line_no, path)
                if selector is not None:
                    selectors.append(selector)
        return selectors

    def _resolve_goal_pose_mode(self) -> tuple[str, int]:
        mode_to_id = {"mixed": 0, "upright": 1, "lying": 2}
        cfg_mode = str(self.cfg.goal_pose_mode).strip().lower()
        if cfg_mode not in mode_to_id:
            raise ValueError(
                f"Invalid goal_pose_mode: {self.cfg.goal_pose_mode}. "
                "Expected one of: mixed, upright, lying."
            )
        return cfg_mode, mode_to_id[cfg_mode]

    def _load_dataset(self):
        legacy_selectors = self._read_goal_pose_selectors(self.cfg.goal_pose_asset_file)
        upright_selectors = self._read_goal_pose_selectors(self.cfg.goal_pose_upright_asset_file)
        lying_selectors = self._read_goal_pose_selectors(self.cfg.goal_pose_lying_asset_file)
        cfg_mode, selected_mode_id = self._resolve_goal_pose_mode()
        legacy_selector_set = set(legacy_selectors)
        upright_selector_set = set(upright_selectors)
        lying_selector_set = set(lying_selectors)
        use_split_files = bool(upright_selector_set or lying_selector_set)

        if use_split_files and legacy_selector_set:
            raise ValueError(
                "goal_pose_asset_file cannot be used together with "
                "goal_pose_upright_asset_file / goal_pose_lying_asset_file."
            )

        overlap = upright_selector_set & lying_selector_set
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(
                "Found assets listed in both goal_pose_upright_asset_file and goal_pose_lying_asset_file. "
                f"Example conflict: {example[0]}/{example[1]}/{example[2]}"
            )

        split_idx = self._GOAL_POSE_MODE_SPLIT

        def _mode_id_for_asset(asset_path: str) -> int:
            selector = self._asset_parts_from_path(asset_path)
            if use_split_files:
                if selector in upright_selector_set:
                    return 1
                if selector in lying_selector_set:
                    return 2
                return selected_mode_id
            if not legacy_selector_set:
                return 0
            return selected_mode_id if selector in legacy_selector_set else 0

        def _pose_pre_sampler(asset_path: str, pose_tensor: torch.Tensor) -> torch.Tensor:
            mode_id = _mode_id_for_asset(asset_path)
            if mode_id == 0:
                return pose_tensor
            if mode_id == 1:
                hi = min(split_idx, int(pose_tensor.shape[0]))
                return pose_tensor[:hi]
            lo = min(split_idx, int(pose_tensor.shape[0]))
            sliced = pose_tensor[lo:]
            if int(sliced.shape[0]) == 0:
                return pose_tensor
            return sliced

        super()._load_dataset(
            num_grasps=self.cfg.dataset_num_grasp,
            num_poses=self.cfg.dataset_num_pose,
            pose_pre_sampler=_pose_pre_sampler,
        )
        asset_mode_ids = torch.zeros(len(self.dataset.path), dtype=torch.int64, device=self.device)
        for i, p in enumerate(self.dataset.path):
            asset_mode_ids[i] = _mode_id_for_asset(p)

        # Per-env asset mode (0=mixed, 1=upright, 2=lying); used by orientation reward to
        # selectively apply tilt-only relaxation when tilt_only_for_upright_only=True.
        self.data_asset_mode_ids = asset_mode_ids
        self.data_object_mode = asset_mode_ids[self.data_object_id.to(torch.long)]

        if use_split_files:
            print(
                f"[place_goal_pose] split_files upright={self.cfg.goal_pose_upright_asset_file} "
                f"lying={self.cfg.goal_pose_lying_asset_file} default_mode={cfg_mode} "
                f"dataset={self.cfg.dataset} total={len(self.dataset.path)} "
                f"upright={(asset_mode_ids == 1).sum().item()} "
                f"lying={(asset_mode_ids == 2).sum().item()} "
                f"mixed={(asset_mode_ids == 0).sum().item()}"
            )
        elif legacy_selector_set:
            matched = torch.zeros(len(self.dataset.path), dtype=torch.bool, device=self.device)
            for i, p in enumerate(self.dataset.path):
                if self._asset_parts_from_path(p) in legacy_selector_set:
                    matched[i] = True
            print(
                f"[place_goal_pose] asset_file={self.cfg.goal_pose_asset_file} "
                f"mode={cfg_mode} dataset={self.cfg.dataset} "
                f"total={len(self.dataset.path)} matched={matched.sum().item()}"
            )
            if matched.any():
                matched_counts = self.dataset.pose_counts[matched]
                print(
                    f"[place_goal_pose] post-load pose_counts for matched assets: "
                    f"min={int(matched_counts.min().item())} max={int(matched_counts.max().item())}"
                )
                if selected_mode_id == 1:
                    max_count = int(matched_counts.max().item())
                    if max_count > split_idx:
                        raise RuntimeError(
                            f"place goal_pose pre-sampler check failed: matched pose_count max={max_count} > split={split_idx}. "
                            "This indicates mode filtering was not applied before choose_from."
                        )

        if (asset_mode_ids == 1).any():
            upright_counts = self.dataset.pose_counts[asset_mode_ids == 1]
            max_count = int(upright_counts.max().item())
            print(
                f"[place_goal_pose] upright pose_counts min={int(upright_counts.min().item())} max={max_count}"
            )
            if max_count > split_idx:
                raise RuntimeError(
                    f"place goal_pose pre-sampler check failed: upright pose_count max={max_count} > split={split_idx}. "
                    "This indicates mode filtering was not applied before choose_from."
                )
        if (asset_mode_ids == 2).any():
            lying_counts = self.dataset.pose_counts[asset_mode_ids == 2]
            print(
                f"[place_goal_pose] lying pose_counts min={int(lying_counts.min().item())} max={int(lying_counts.max().item())}"
            )
        if (asset_mode_ids == 0).any():
            mixed_counts = self.dataset.pose_counts[asset_mode_ids == 0]
            print(
                f"[place_goal_pose] mixed pose_counts min={int(mixed_counts.min().item())} max={int(mixed_counts.max().item())}"
            )
        self.goal_pose_id = self._sample_pose_ids_for_assets(self.data_object_id)

    def _sample_pose_ids_for_assets(self, asset_ids: torch.Tensor) -> torch.Tensor:
        num_pose = int(self.dataset.poses.shape[1])
        if num_pose <= 0:
            raise ValueError("dataset poses must be non-empty.")
        if hasattr(self.dataset, "pose_counts"):
            max_counts = self.dataset.pose_counts[asset_ids].to(device=self.device, dtype=torch.int64)
            max_counts = max_counts.clamp(min=1, max=num_pose)
            rand = torch.rand((len(asset_ids),), device=self.device)
            return torch.floor(rand * max_counts.to(torch.float32)).to(torch.int32)
        return torch.randint(0, num_pose, (len(asset_ids),), dtype=torch.int32, device=self.device)

    def _sample_pose_ids_by_orientation_topk(self, asset_ids: torch.Tensor, ref_quat_wxyz: torch.Tensor) -> torch.Tensor:
        num_env = len(asset_ids)
        if num_env == 0:
            return torch.empty((0,), dtype=torch.int32, device=self.device)

        num_pose = int(self.dataset.poses.shape[1])
        if num_pose <= 0:
            raise ValueError("dataset poses must be non-empty.")

        topk = int(self.cfg.goal_pose_orientation_topk)
        if topk <= 0:
            raise ValueError(
                f"goal_pose_orientation_topk must be > 0, got {self.cfg.goal_pose_orientation_topk}."
            )

        if hasattr(self.dataset, "pose_counts"):
            max_counts = self.dataset.pose_counts[asset_ids].to(device=self.device, dtype=torch.int64)
            max_counts = max_counts.clamp(min=1, max=num_pose)
        else:
            max_counts = torch.full((num_env,), num_pose, dtype=torch.int64, device=self.device)

        pose_quat = self.dataset.poses[asset_ids, :, 3:7].to(torch.float32)
        pose_quat = pose_quat / pose_quat.norm(dim=-1, keepdim=True).clamp(min=1.0e-12)

        ref_quat = ref_quat_wxyz.to(device=self.device, dtype=torch.float32)
        ref_quat = ref_quat / ref_quat.norm(dim=-1, keepdim=True).clamp(min=1.0e-12)

        # Tilt-only similarity: compare world-frame up vectors, ignore yaw around the up axis.
        unit_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
        pose_up = torch_utils.quat_apply(
            pose_quat.reshape(-1, 4),
            unit_z.unsqueeze(0).expand(num_env * num_pose, -1),
        ).reshape(num_env, num_pose, 3)
        ref_up = torch_utils.quat_apply(ref_quat, unit_z.unsqueeze(0).expand(num_env, -1))
        similarity = (pose_up * ref_up.unsqueeze(1)).sum(dim=-1)
        valid_mask = torch.arange(num_pose, device=self.device, dtype=torch.int64).unsqueeze(0) < max_counts.unsqueeze(1)
        similarity = similarity.masked_fill(~valid_mask, -2.0)

        k_per_env = torch.minimum(max_counts, torch.full_like(max_counts, topk))
        max_k = int(k_per_env.max().item())
        topk_ids = torch.topk(similarity, k=max_k, dim=1, largest=True, sorted=True).indices

        rand_in_topk = torch.floor(torch.rand((num_env,), device=self.device) * k_per_env.to(torch.float32)).to(torch.int64)
        row_ids = torch.arange(num_env, device=self.device)
        return topk_ids[row_ids, rand_in_topk].to(torch.int32)

    def _resolve_grasp_mode(self) -> str:
        mode = str(self.cfg.grasp_sampling_mode).strip().lower()
        valid_modes = {"mixed", "general", "upright", "convex"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid grasp_sampling_mode: {self.cfg.grasp_sampling_mode}. "
                "Expected one of: mixed, general, upright, convex."
            )
        return mode

    def _sample_hold_ids(self, num_env: int) -> torch.Tensor:
        mode = self._resolve_grasp_mode()
        num_grasp = int(self.dataset.grasps.shape[1])
        if num_grasp <= 0:
            raise ValueError("dataset grasps must be non-empty.")

        if mode == "mixed":
            low, high = 0, num_grasp
        else:
            split1 = int(max(0, min(self.cfg.grasp_mode_split_1, num_grasp)))
            split2 = int(max(split1, min(self.cfg.grasp_mode_split_2, num_grasp)))
            if mode == "general":
                low, high = 0, split1
            elif mode == "upright":
                low, high = split1, split2
            else:  # convex
                low, high = split2, num_grasp

            if high <= low:
                low, high = 0, num_grasp

        span = high - low
        rand = torch.floor(torch.rand((num_env,), device=self.device) * float(span)).to(torch.int64)
        return (rand + low).to(torch.int64)

    def _compute_goal_top_z(self, env_ids: torch.Tensor) -> torch.Tensor:
        point_cloud_local = self.data_point_cloud[env_ids]
        goal_quat = self.goal_pose[env_ids, 3:7]
        point_cloud_world = (
            torch_utils.quat_apply(
                goal_quat.unsqueeze(1).expand(-1, point_cloud_local.shape[1], -1),
                point_cloud_local,
            )
            + self.scene.env_origins[env_ids].unsqueeze(1)
            + self.goal_pose[env_ids, :3].unsqueeze(1)
        )
        return point_cloud_world[..., 2].max(dim=1).values

    def _compute_object_goal_rot_error(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            goal_quat = self.goal_pose[:, 3:7]
            object_quat = qc.root_quat_w(self.object)
            modes = self.data_object_mode
        else:
            goal_quat = self.goal_pose[env_ids, 3:7]
            object_quat = qc.root_quat_w(self.object)[env_ids]
            modes = self.data_object_mode[env_ids]

        # Full orientation error (always computed; cheap quat op).
        diff_quat = torch_utils.quat_mul(goal_quat, torch_utils.quat_conjugate(object_quat))
        full_err = 2 * diff_quat[:, 0].abs().clip(0, 1).arccos()

        if not self.cfg.require_tilt_only:
            return full_err

        unit_z = self.unit_z.unsqueeze(0).expand(goal_quat.shape[0], -1)
        goal_up = torch_utils.quat_apply(goal_quat, unit_z)
        object_up = torch_utils.quat_apply(object_quat, unit_z)
        tilt_cos = (goal_up * object_up).sum(dim=-1).clamp(-1.0, 1.0)
        tilt_err = tilt_cos.arccos()

        if not self.cfg.tilt_only_for_upright_only:
            return tilt_err

        # Upright envs use tilt; non-upright fall back to full orientation error (the caller
        # decides whether to actually use it based on require_orientation).
        return torch.where(modes == 1, tilt_err, full_err)

    ############################################################
    # pre-physics step calls
    ############################################################

    def _pre_physics_step(self, actions: torch.Tensor):
        raw_actions = actions.clone().clamp(-1.0, 1.0)
        forced_steps = max(0, int(self.cfg.post_reset_forced_steps))
        forced_mask = self.episode_length_buf < forced_steps if forced_steps > 0 else None
        if forced_mask is not None and forced_mask.any():
            raw_actions[forced_mask, :6] = 0.0
            raw_actions[forced_mask, 6:] = -1.0
        super()._pre_physics_step(raw_actions)
        if forced_mask is not None and forced_mask.any():
            self.actions[forced_mask] = raw_actions[forced_mask]
            self.prev_actions[forced_mask] = raw_actions[forced_mask]

    ############################################################
    # post-physics step calls
    ############################################################

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_buffer()

        object_pos = qc.root_pos_w(self.object) - self.scene.env_origins
        table_bottom_z_local = self.table_pose[:, 2] - self.scene.env_origins[:, 2] - 0.5 * float(self.table_size[2])
        dropped = object_pos[:, 2] < (table_bottom_z_local - float(self.cfg.drop_fail_margin))
        self.failure = (
            (object_pos[:, 0].abs() > self.cfg.scene.env_spacing)
            | (object_pos[:, 1].abs() > self.cfg.scene.env_spacing)
            | dropped
            | (self.obs_dist.min(dim=-1).values > 0.2)
        )
        self.success = self.consecutive_success >= self.cfg.num_consecutive_success

        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self.failure | self.success, truncated

    def _get_rewards(self) -> torch.Tensor:
        # POS_TOL = 0.2
        if self.cfg.threshold_decay:
            decay = 0.9 ** (max(self.common_step_counter - 20000, 0) / 10000)
            POS_TOL = max(0.02, 0.05 * decay)
            ROT_TOL = max(0.25, 0.5 * decay)
        else:
            POS_TOL = 0.03
            ROT_TOL = 0.3

        num_contact = (self.obs_contact > 0).sum(dim=-1)

        dist = self.obs_dist.mean(dim=-1)
        rew_dist = self.cfg.w_rew_dist * dist.clip(0, 0.25)

        # Goal position
        # X, Y
        diff_object_pos = self.diff_pose[:, :2].norm(dim=-1)
        near_release = (diff_object_pos < float(self.cfg.release_near_goal_tol_xy)) & (
            self.diff_pose[:, 2].abs() < float(self.cfg.release_near_goal_tol_z)
        )
        if self.cfg.enable_near_goal_release_contact_penalty:
            rew_contact = torch.where(
                near_release,
                -float(self.cfg.w_rew_contact_near_goal) * num_contact.float(),
                float(self.cfg.w_rew_contact) * num_contact.float(),
            )
        else:
            rew_contact = float(self.cfg.w_rew_contact) * num_contact.float()
        rew_object_pos = self.cfg.w_rew_object_pos * (self.min_diff_pos - diff_object_pos).relu()
        self.min_diff_pos = torch.min(self.min_diff_pos, diff_object_pos)
        # Z
        rew_object_height = self.cfg.w_rew_object_height * (self.min_height - qc.root_pos_w(self.object)[:, 2]).relu()
        self.min_height = torch.min(self.min_height, qc.root_pos_w(self.object)[:, 2])
        # Goal orientation: per-env decide tilt / full / none based on cfg + asset mode.
        require_orientation = bool(self.cfg.require_orientation)
        require_tilt_only = bool(self.cfg.require_tilt_only)
        tilt_only_for_upright_only = bool(self.cfg.tilt_only_for_upright_only)

        is_upright = self.data_object_mode == 1
        # Per-env: upright uses tilt iff require_tilt_only; non-upright uses tilt iff
        # require_tilt_only and not tilt_only_for_upright_only.
        env_uses_tilt = torch.where(
            is_upright,
            torch.full_like(is_upright, require_tilt_only),
            torch.full_like(is_upright, require_tilt_only and not tilt_only_for_upright_only),
        )
        # Envs that don't use tilt fall back to full orientation iff require_orientation.
        env_uses_full = (~env_uses_tilt) & require_orientation
        env_uses_orient = env_uses_tilt | env_uses_full

        if env_uses_orient.any():
            diff_object_rot_raw = self._compute_object_goal_rot_error()
            # Mask out non-orient envs so they don't pollute reward / min_diff_rot.
            diff_object_rot = torch.where(
                env_uses_orient, diff_object_rot_raw, torch.zeros_like(diff_object_rot_raw)
            )
            rew_object_rot = self.cfg.w_rew_object_rot * (self.min_diff_rot - diff_object_rot).relu()
            rew_object_rot = torch.where(env_uses_orient, rew_object_rot, torch.zeros_like(rew_object_rot))
            new_min = torch.minimum(self.min_diff_rot, diff_object_rot)
            self.min_diff_rot = torch.where(env_uses_orient, new_min, self.min_diff_rot)

            tilt_rot_tol = max(0.0, float(self.cfg.tilt_rot_tol_deg)) * torch.pi / 180.0
            tol = torch.where(
                env_uses_tilt,
                torch.full_like(diff_object_rot, tilt_rot_tol),
                torch.full_like(diff_object_rot, ROT_TOL),
            )
            rot_ok = (diff_object_rot < tol) | (~env_uses_orient)
        else:
            rew_object_rot = torch.zeros_like(diff_object_pos)
            rot_ok = torch.ones_like(diff_object_pos, dtype=torch.bool)
        # Reach
        reached_pos = (
            (diff_object_pos < POS_TOL)
            & (self.diff_pose[:, 2] > -0.05)
            & (qc._as_tensor(self.object.data.root_lin_vel_w).norm(dim=-1) < 0.1)
        )
        reached = reached_pos & rot_ok
        reached_once = reached_pos if self.cfg.reach_once_pos_only else reached
        rew_reached = (
            self.cfg.w_rew_reached_once * (reached_once & (~self.reached_once)).float()
            + self.cfg.w_rew_reached * reached.float()
            + self.cfg.w_rew_reached_release * reached.float() * (num_contact == 0)
        )
        self.reached_once[:] |= reached_once
        self.reached[:] |= reached
        # Success
        self.consecutive_success[:] = torch.where(reached, self.consecutive_success + 1, 0)
        rew_success = (
            self.cfg.w_rew_success
            * self.success.float()
            * (20 + (self.max_episode_length - self.episode_length_buf).float())
        )

        # Others
        # Velocity penalty
        rew_action = self.cfg.w_rew_action * (
            qc._as_tensor(self.robot.data.joint_vel).pow(2).sum(dim=-1) + qc._as_tensor(self.object.data.root_vel_w).pow(2).sum(dim=-1)
        ).clamp(0.0, 10000.0)
        # Failure penalty
        rew_failure = self.cfg.w_rew_failure * self.failure.float()

        # Set latest status
        self.latest_status[self.success] = TaskStatus.SUCCESS
        self.latest_status[self.failure] = TaskStatus.FAILURE
        self.latest_status[self.reset_time_outs] = TaskStatus.TIMEOUT

        self.latest_reached[self.reset_buf] = self.reached[self.reset_buf]
        self.latest_reached[:] |= self.reached

        if self.reset_buf.any():
            self.statistics["success_rate"] = (self.latest_status == TaskStatus.SUCCESS).float().mean()
            self.statistics["timeout_rate"] = (self.latest_status == TaskStatus.TIMEOUT).float().mean()
            self.statistics["failure_rate"] = (self.latest_status == TaskStatus.FAILURE).float().mean()

        self.statistics["reached_rate"] = self.latest_reached.float().mean()
        self.statistics["arm_max_abs_tracking_error"] = self._compute_arm_max_abs_tracking_error()

        if self._should_print():
            print({k: v.item() for k, v in self.statistics.items()})

        # Record
        self.extras.update(
            {
                "rew_contact": rew_contact.mean(),
                "rew_dist": rew_dist.mean(),
                "rew_object_pos": rew_object_pos.mean(),
                "rew_object_rot": rew_object_rot.mean(),
                "rew_object_height": rew_object_height.mean(),
                "rew_reached": rew_reached.mean(),
                "rew_success": rew_success.mean(),
                "rew_action": rew_action.mean(),
                "rew_failure": rew_failure.mean(),
            }
        )
        self.extras.update(self.statistics)

        reward = (
            rew_contact
            + rew_dist
            + rew_object_pos
            + rew_object_rot
            + rew_object_height
            + rew_reached
            + rew_action
            + rew_success
            + rew_failure
        )
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # randomize table height
        z_min, z_max = self.cfg.table_z_range
        self.table_pose[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (z_max - z_min) + z_min
        self.table.write_root_link_pose_to_sim(qc.pose_to_xyzw(self.table_pose[env_ids]), env_ids=env_ids)

        # sample goal pose (env-local)
        if self.cfg.goal_pose_orientation_topk_enabled:
            self.goal_pose_id[env_ids] = self._sample_pose_ids_by_orientation_topk(
                self.data_object_id[env_ids],
                qc.root_quat_w(self.object)[env_ids],
            )
        else:
            self.goal_pose_id[env_ids] = self._sample_pose_ids_for_assets(self.data_object_id[env_ids])
        self.goal_pose[env_ids] = self.dataset.poses[self.data_object_id[env_ids], self.goal_pose_id[env_ids]]
        x_min, x_max = self.cfg.goal_x_range
        y_min, y_max = self.cfg.goal_y_range
        self.goal_pose[env_ids, 0] = torch.rand((len(env_ids),), device=self.device) * (x_max - x_min) + x_min
        self.goal_pose[env_ids, 1] = torch.rand((len(env_ids),), device=self.device) * (y_max - y_min) + y_min
        self.goal_pose[env_ids, 2] += self.table_pose[env_ids, 2] + 0.5 * self.table_size[2] + 0.002
        if self.cfg.visualize_goal_pose_marker:
            self._update_goal_pose_debug_draw()
        goal_top_z = self._compute_goal_top_z(env_ids)
        table_z_eff = goal_top_z
        if self.cfg.init_arm_upright:
            table_z_eff = table_z_eff + float(self.cfg.init_arm_upright_z_offset)
            jitter = float(self.cfg.init_arm_upright_z_jitter)
            if jitter < 0.0:
                raise ValueError(f"init_arm_upright_z_jitter should be >= 0, got {jitter}.")
            if jitter > 0.0:
                noise = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * jitter
                table_z_eff = table_z_eff + noise

        # sample grasp state (ee-relative object pose + finger joints)
        self.hold_id[env_ids] = self._sample_hold_ids(len(env_ids))
        self.hold_state[env_ids] = self.dataset.grasps[self.data_object_id[env_ids], self.hold_id[env_ids]]

        # reset hand: sample arm around goal, then set finger joints for grasp
        joint_pos = qc._as_tensor(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_pos[:, :7] = self.initial_pose.sample(
            self.goal_pose[env_ids, :2],
            table_z=table_z_eff,
        )
        finger_dofs = self.hold_finger_joint.shape[1]
        joint_pos[:, -finger_dofs:] = self.hold_finger_joint[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Optionally rotate wrist by +pi when initialized object points downward.
        ee_pos = qc.body_pos_w(self.robot)[env_ids, self.grasp_site_idx]
        ee_quat = qc.body_quat_w(self.robot)[env_ids, self.grasp_site_idx]
        if self.cfg.flip_wrist_if_object_upside_down:
            init_obj_quat = torch_utils.quat_mul(ee_quat, self.hold_rel_quat[env_ids])
            init_obj_up = torch_utils.quat_apply(init_obj_quat, self.unit_z.unsqueeze(0).expand(len(env_ids), -1))
            upside_down_mask = init_obj_up[:, 2] < 0.0
            if upside_down_mask.any():
                wrist_joint_idx = int(self.arm_joint_indices[-1])
                wrist_lower = float(self.robot_dof_lower_limits[wrist_joint_idx].item())
                wrist_upper = float(self.robot_dof_upper_limits[wrist_joint_idx].item())
                wrist_q = joint_pos[upside_down_mask, wrist_joint_idx]
                plus_target = wrist_q + torch.pi
                minus_target = wrist_q - torch.pi
                plus_valid = (plus_target >= wrist_lower) & (plus_target <= wrist_upper)
                minus_valid = (minus_target >= wrist_lower) & (minus_target <= wrist_upper)

                # Prefer exact +/-pi flip when feasible; otherwise pick the clamped target with larger rotation.
                plus_clamped = plus_target.clamp(min=wrist_lower, max=wrist_upper)
                minus_clamped = minus_target.clamp(min=wrist_lower, max=wrist_upper)
                choose_plus = torch.where(
                    plus_valid,
                    torch.ones_like(plus_valid),
                    torch.where(minus_valid, torch.zeros_like(plus_valid), (plus_clamped - wrist_q).abs() >= (minus_clamped - wrist_q).abs()),
                )
                wrist_q_new = torch.where(choose_plus, plus_clamped, minus_clamped)
                joint_pos[upside_down_mask, wrist_joint_idx] = wrist_q_new
                self.robot.write_joint_state_to_sim(
                    joint_pos[upside_down_mask],
                    joint_vel[upside_down_mask],
                    env_ids=env_ids[upside_down_mask],
                )
                ee_pos = qc.body_pos_w(self.robot)[env_ids, self.grasp_site_idx]
                ee_quat = qc.body_quat_w(self.robot)[env_ids, self.grasp_site_idx]

        # reset object from ee-relative grasp pose (grasp_site frame)
        obj_pos = torch_utils.tf_apply(ee_quat, ee_pos, self.hold_rel_pos[env_ids])
        obj_quat = torch_utils.quat_mul(ee_quat, self.hold_rel_quat[env_ids])
        obj_pose = torch.cat((obj_pos, obj_quat), dim=-1)
        self.object.write_root_link_pose_to_sim(qc.pose_to_xyzw(obj_pose), env_ids=env_ids)
        self.object.write_root_link_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)

        self.reset_buf[env_ids] = 0
        self.reached[env_ids] = False
        self.reached_once[env_ids] = False
        self.consecutive_success[env_ids] = 0

        self._refresh_buffer()

        self.min_diff_pos[env_ids] = self.diff_pose[env_ids, :3].norm(dim=-1).clip(max=0.4)
        self.min_diff_rot[env_ids] = self._compute_object_goal_rot_error(env_ids).clip(max=0.5)
        self.min_height[env_ids] = qc.root_pos_w(self.object)[env_ids, 2]

    def _get_observations(self) -> dict:

        obs = torch.cat(
            (
                self.robot_joint_pos,
                qc._as_tensor(self.robot.data.joint_vel) * self.cfg.dof_velocity_scale,
                qc._as_tensor(self.robot.data.joint_pos_target),
                self.wrist_state,
                qc.body_pos_w(self.robot)[:, self.left_finger_idx] - self.scene.env_origins,
                qc.body_pos_w(self.robot)[:, self.right_finger_idx] - self.scene.env_origins,
                qc.body_pos_w(self.robot)[:, self.grasp_site_idx] - self.scene.env_origins,
                qc.root_pos_w(self.object) - self.scene.env_origins,
                self.obs_dist,
                self.obs_closest_point.reshape(self.num_envs, -1),
                self.obs_contact,
                self.goal_pose,
                self.diff_pose,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _refresh_buffer(self):
        # Distance Feature
        inv_object_quat, inv_object_pos = torch_utils.tf_inverse(
            qc.root_quat_w(self.object), qc.root_pos_w(self.object)
        )
        link_pos_obj_frame = torch_utils.tf_apply(
            inv_object_quat.unsqueeze(1).expand(-1, len(self.keypoint_indices), -1),
            inv_object_pos.unsqueeze(1),
            qc.body_pos_w(self.robot)[:, self.keypoint_indices, :],
        )
        distance = torch.cdist(link_pos_obj_frame, self.data_point_cloud)  # (num_envs, num_query, point_cloud_size)
        self.obs_dist, min_indices = distance.min(dim=-1)
        closest_direction = (
            torch.gather(self.data_point_cloud, dim=1, index=min_indices.unsqueeze(-1).expand(-1, -1, 3))
            - link_pos_obj_frame
        )  # (num_envs, num_query, 3)
        self.obs_closest_point: torch.Tensor = torch_utils.quat_apply(
            qc.root_quat_w(self.object).unsqueeze(1).expand(-1, self.obs_dist.shape[1], -1),
            closest_direction,
        )

        # Contact Forces
        force = qc._as_tensor(self.contact.data.net_forces_w)
        self.obs_contact = force.squeeze(1).norm(dim=-1) > 0.01

        # Others
        self.robot_joint_pos = torch_utils.scale_transform(
            qc._as_tensor(self.robot.data.joint_pos), self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )
        self.diff_pose[:, :3] = self.scene.env_origins + self.goal_pose[:, :3] - qc.root_pos_w(self.object)
        self.diff_pose[:, 3:] = torch_utils.quat_mul(
            self.goal_pose[:, 3:], torch_utils.quat_conjugate(qc.root_quat_w(self.object))
        )
        self.wrist_state = qc.body_state_w(self.robot)[:, self.hand_link_idx]
        self.wrist_state[:, :3] -= self.scene.env_origins

        if self.cfg.visualize_goal_pose_marker:
            self._update_goal_pose_debug_draw()


