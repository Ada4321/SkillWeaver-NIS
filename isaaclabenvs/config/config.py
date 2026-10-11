from pathlib import Path

import toml

# Basic paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
ASSET_ROOT = PROJECT_ROOT / "assets"

# User-defined paths: config.toml if present, otherwise the template, which points at the
# data bundle extracted to <repo>/data. Relative dataset paths resolve against the repo root.
CONFIG_ROOT = Path(__file__).parent
_config_file = CONFIG_ROOT / "config.toml"
if not _config_file.exists():
    _config_file = CONFIG_ROOT / "config.template.toml"
DATASET_CONFIG: dict[str, dict[str, str]] = toml.load(str(_config_file))["dataset"]
for _section in DATASET_CONFIG.values():
    if _section.get("path") and not Path(_section["path"]).is_absolute():
        _section["path"] = str(PROJECT_ROOT / _section["path"])
