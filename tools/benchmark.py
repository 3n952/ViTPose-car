"""Benchmark ViTPose forward-pass latency on a CUDA device."""
import argparse
import os
import sys
import time
import statistics as stat
from pathlib import Path
from loguru import logger

import torch
import numpy as np

from xtcocotools.coco import COCO
from mmpose.apis import inference_top_down_pose_model, init_pose_model
from mmpose.datasets import DatasetInfo

VITPOSE_ROOT = Path(__file__).parent.parent
if str(VITPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(VITPOSE_ROOT))

import mmcv_custom  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure ViTPose inference latency on Maruhan dataset')
    parser.add_argument(
        '--config',
        default='configs/maruhan/vitpose_maruhan_256x192.py',
        help='config file path used for training')
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='checkpoint file path')
    # parser.add_argument(
    #     '--json-file',
    #     default='../../../benchmark_dataset/Parking-.v1i.coco/train/_annotations.coco.json',
    #     help='COCO format annotation file')
    # parser.add_argument(
    #     '--img-root',
    #     default='../../../benchmark_dataset/Parking-.v1i.coco/train',
    #     help='root directory containing images')
    parser.add_argument(
        '--device',
        default='cuda:2',
        help='device for inference (cuda:0 or cpu)')
    parser.add_argument(
        '--warmup',
        type=int,
        default=1,
        help='number of warmup iterations to exclude from statistics')
    parser.add_argument(
        '--num-iterations',
        type=int,
        default=10,
        help='number of benchmark iterations')
    # parser.add_argument(
    #     '--work-dir',
    #     default='runs/benchmark_latency',
    #     help='directory to save benchmark results')
    return parser.parse_args()


class VitposeLatencyBenchmark:
    """ViTPose inference latency benchmark"""
    def __init__(self, config_path: str, checkpoint: str,
                 device: str, num_warmup: int, num_iterations: int):

        self.config_path = config_path
        self.checkpoint = checkpoint
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.device = device
        
        # init
        self.pose_model = init_pose_model(self.config_path, self.checkpoint, device=self.device.lower())
        self.dataset = self.pose_model.cfg.data['test']['type']
        dataset_info_cfg = self.pose_model.cfg.data['test'].get('dataset_info', None)
        self.dataset_info = DatasetInfo(dataset_info_cfg) if dataset_info_cfg else None
        self.coco = COCO(self.pose_model.cfg.data['benchmark']['ann_file'])  #json_file
        self.img_ids = list(self.coco.imgs.keys())
        
        logger.info(f'model loaded: {self.checkpoint}')
        logger.info(f'device: {self.device}')
        logger.info(f'total image number: {len(self.img_ids)}')

    def _prepare_image_data(self, image_id):
        """prepare image and bounding box information"""
        image_meta = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.pose_model.cfg.data['benchmark']['img_prefix'], image_meta['file_name'])
        ann_ids = self.coco.getAnnIds(image_id)
        ann_info = self.coco.loadAnns(ann_ids)
        dets = [{'bbox': ann['bbox']} for ann in ann_info]
        return image_path, dets, image_meta

    def _warmup(self):
        if not (torch.cuda.is_available() and 'cuda' in self.device):
            raise RuntimeError('CUDA is not available.')

        # Warmup
        logger.info(f'Warmup started ({self.num_warmup} iterations)...')
        for i in range(self.num_warmup):
            for image_id in self.img_ids:
                image_path, dets, _ = self._prepare_image_data(image_id)

                torch.cuda.synchronize()
                _ = inference_top_down_pose_model(
                    self.pose_model,
                    image_path,
                    dets,
                    bbox_thr=None, 
                    format='xywh',
                    dataset=self.dataset,
                    dataset_info=self.dataset_info,
                    return_heatmap=False)
            
            # torch.cuda.empty_cache()

        torch.cuda.synchronize()
        logger.info(f'  Warmup {self.num_warmup} completed')
        

    def run(self):
        """ViTPose latency measurement"""
        
        self._warmup()

        logger.info(f'Latency measurement started with {self.num_iterations} iterations...')
        img_latencies = []
        car_latencies = []

        for i in range(self.num_iterations):
            for _, image_id in enumerate(self.img_ids):
                image_path, dets, _ = self._prepare_image_data(image_id)
                num_cars = len(dets)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = inference_top_down_pose_model(
                    self.pose_model,
                    image_path,
                    dets,
                    bbox_thr=None,
                    format='xywh',
                    dataset=self.dataset,
                    dataset_info=self.dataset_info,
                    return_heatmap=False)
                
                torch.cuda.synchronize()
                
                end = time.perf_counter()
                latency_ms = (end - start) * 1000
                img_latencies.append(latency_ms)  # ms per image
                car_latencies.append(latency_ms / num_cars)  # mean ms per car
            
            car_latency_mean = stat.mean(car_latencies[-10:])
            car_latencies = []
            logger.info(f'  [{i + 1}/{self.num_iterations}] processed - '
                    f'image latency: {latency_ms:.3f} ms, car latency: {car_latency_mean:.3f} ms')

            # torch.cuda.empty_cache()

        # 통계 계산
        latencies = np.array(img_latencies)
        results = {
            "num_images": len(self.images),
            "num_iterations": self.num_iterations,
            "total_inferences": len(latencies),
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
        
        return results


def main():
    args = parse_args()
    benchmark = VitposeLatencyBenchmark(
        args.config,
        args.checkpoint,
        args.device,
        args.warmup,
        args.num_iterations,
    )
    
    benchmark_results = benchmark.run()
    
    # calculate statistics
    logger.info("\n" + "=" * 80)
    logger.info("Benchmark Results")
    logger.info("=" * 80)
    logger.info(f"Total inferences: {benchmark_results['total_inferences']}")
    logger.info(f"  ({benchmark_results['num_images']} images × {benchmark_results['num_iterations']} iterations)")
    logger.info("")
    logger.info("Latency (ms):")
    logger.info(f"  Mean:   {benchmark_results['mean_ms']:.2f} ms")
    logger.info(f"  Median: {benchmark_results['median_ms']:.2f} ms")
    logger.info(f"  Std:    {benchmark_results['std_ms']:.2f} ms")
    logger.info(f"  Min:    {benchmark_results['min_ms']:.2f} ms")
    logger.info(f"  Max:    {benchmark_results['max_ms']:.2f} ms")
    logger.info("")
    logger.info("Percentiles:")
    logger.info(f"  P50:    {benchmark_results['p50_ms']:.2f} ms")
    logger.info(f"  P90:    {benchmark_results['p90_ms']:.2f} ms")
    logger.info(f"  P95:    {benchmark_results['p95_ms']:.2f} ms")
    logger.info(f"  P99:    {benchmark_results['p99_ms']:.2f} ms")
    logger.info("=" * 80 + "\n")
    
    logger.success(f"P95 Latency: {benchmark_results['p95_ms']:.2f} ms")


if __name__ == '__main__':
    main()
