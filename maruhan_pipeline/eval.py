#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the Maruhan car keypoint dataset."""
import argparse
import os
import sys

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmpose.apis import single_gpu_test
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import setup_multi_processes

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from maruhan_pipeline.utils import override_data_root, parse_device_id

import mmcv_custom  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate ViTPose on the Maruhan dataset.')
    parser.add_argument(
        '--config',
        default='configs/maruhan/vitpose_maruhan_256x192.py',
        help='Config used for training.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path.')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device for inference (cuda:0 or cpu).')
    parser.add_argument(
        '--metric',
        nargs='+',
        default=None,
        help='Metrics to compute (defaults to config evaluation.metric).')
    parser.add_argument(
        '--data-root',
        default=None,
        help='Override the dataset root used inside the config.')
    parser.add_argument(
        '--work-dir',
        default=None,
        help='Directory to place evaluation logs/results.')
    parser.add_argument(
        '--out',
        default=None,
        help='Optional path to dump raw pose estimations as a pickle.')
    return parser.parse_args()


def build_test_loader(cfg: Config):
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    test_loader_cfg = dict(
        samples_per_gpu=cfg.data.get('test_dataloader', {}).get(
            'samples_per_gpu', cfg.data.get('samples_per_gpu', 1)),
        workers_per_gpu=cfg.data.get('test_dataloader', {}).get(
            'workers_per_gpu', cfg.data.get('workers_per_gpu', 1)),
        shuffle=False,
        drop_last=False,
        dist=False,
        persistent_workers=cfg.data.get('persistent_workers', False),
        seed=cfg.get('seed', None))
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    return dataset, data_loader


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.data_root:
        override_data_root(cfg, args.data_root)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.work_dir = args.work_dir or cfg.get('work_dir',
                                            'work_dirs/vitpose_maruhan')
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    setup_multi_processes(cfg)

    dataset, data_loader = build_test_loader(cfg)

    model = build_posenet(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    device_id = parse_device_id(args.device)
    if device_id is not None:
        torch.cuda.set_device(device_id)
        model = MMDataParallel(model.cuda(device_id), device_ids=[device_id])
    else:
        model = model.cpu()

    outputs = single_gpu_test(model, data_loader)

    if args.out:
        mmcv.dump(outputs, args.out)

    eval_cfg = cfg.get('evaluation', {}).copy()
    if args.metric is not None:
        eval_cfg['metric'] = args.metric

    results = dataset.evaluate(outputs, cfg.work_dir, **eval_cfg)
    for key, value in sorted(results.items()):
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
