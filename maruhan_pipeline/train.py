#!/usr/bin/env python3
"""Single-GPU training shortcut for the Maruhan car keypoint task."""
import argparse
import os
import sys
import time

import mmcv
import torch
from mmcv import Config
from mmcv.runner import set_random_seed

from mmpose.apis import init_random_seed, train_model
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.utils import setup_multi_processes

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from maruhan_pipeline.utils import override_data_root, parse_device_id

import mmcv_custom  # noqa: F401  # ensures custom modules are registered


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train ViTPose on the Maruhan car keypoint dataset')
    parser.add_argument(
        '--config',
        default='configs/maruhan/vitpose_maruhan_256x192.py',
        help='Path to the Maruhan config.')
    parser.add_argument(
        '--work-dir',
        default='runs/exp2',
        help='Directory for logs/checkpoints (defaults to config work_dir).')
    parser.add_argument(
        '--device',
        default='cuda:2',
        help='PyTorch device string, e.g., cuda:0 or cpu.')
    parser.add_argument(
        '--seed', type=int, default=3407, help='Random seed (default: 3407).')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Disable validation during training.')
    parser.add_argument(
        '--resume-from', default=None, help='Checkpoint path to resume from.')
    parser.add_argument(
        '--pretrained',
        default='checkpoints/ViTPose/vitpose-b.pth',
        help='Override model.pretrained with a ViTPose checkpoint.')
    parser.add_argument(
        '--data-root',
        default=None,
        help='Override the dataset root used in the config.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.data_root:
        override_data_root(cfg, args.data_root)

    cfg.work_dir = args.work_dir or cfg.get('work_dir',
                                            'work_dirs/vitpose_maruhan')
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.pretrained:
        cfg.model['pretrained'] = args.pretrained

    if args.resume_from:
        cfg.resume_from = args.resume_from

    device_id = parse_device_id(args.device)
    if device_id is not None:
        torch.cuda.set_device(device_id)
        cfg.gpu_ids = [device_id]
    else:
        cfg.gpu_ids = []

    seed = args.seed if args.seed is not None else init_random_seed(
        device='cuda' if device_id is not None else 'cpu')
    set_random_seed(seed, deterministic=False)
    cfg.seed = seed

    model = build_posenet(cfg.model)
    datasets = [build_dataset(cfg.data.train)]

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    meta = dict(seed=seed)

    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=not args.no_validate,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
