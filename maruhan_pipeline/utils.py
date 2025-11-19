"""Shared helpers for the Maruhan pipeline wrappers."""
from __future__ import annotations

from typing import Any

from mmcv import Config


def override_data_root(cfg: Config, data_root: str):
    """Update dataset paths inside the config when the root changes."""

    def replace_path(value: Any):
        if isinstance(value, str):
            original_root = cfg.get('data_root', '')
            if original_root and original_root in value:
                return value.replace(original_root, data_root)
        return value

    cfg.data_root = data_root
    for split in ('train', 'val', 'test'):
        split_cfg = cfg.data.get(split)
        if split_cfg is None:
            continue
        if 'ann_file' in split_cfg:
            split_cfg.ann_file = replace_path(split_cfg.ann_file)
        if 'img_prefix' in split_cfg:
            split_cfg.img_prefix = replace_path(split_cfg.img_prefix)


def parse_device_id(device: str) -> int | None:
    """Return the CUDA device index if available."""
    if device.startswith('cuda'):
        return int(device.split(':')[1]) if ':' in device else 0
    return None
