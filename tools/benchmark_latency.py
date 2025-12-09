"""Benchmark ViTPose forward-pass latency on a CUDA device."""
import argparse
import os
import sys
import time
from loguru import logger

import torch
from xtcocotools.coco import COCO

from mmpose.apis import inference_top_down_pose_model, init_pose_model
from mmpose.datasets import DatasetInfo

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mmcv_custom  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Maruhan 데이터셋에서 ViTPose inference latency 측정')
    parser.add_argument(
        '--config',
        default='configs/maruhan/vitpose_maruhan_256x192.py',
        help='학습에 사용된 config 파일 경로')
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='체크포인트 파일 경로')
    parser.add_argument(
        '--json-file',
        default='../Maruhan-car-kp/train/_annotations.coco.json',
        help='COCO 형식의 annotation 파일')
    parser.add_argument(
        '--img-root',
        default='../Maruhan-car-kp/train',
        help='이미지가 포함된 루트 디렉토리')
    parser.add_argument(
        '--device',
        default='cuda:2',
        help='추론에 사용할 디바이스 (cuda:0 또는 cpu)')
    parser.add_argument(
        '--warmup',
        type=int,
        default=5,
        help='통계에서 제외할 warmup iteration 수')
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='측정할 이미지 수')
    parser.add_argument(
        '--work-dir',
        default='runs/benchmark_latency',
        help='벤치마크 결과를 저장할 디렉토리')
    return parser.parse_args()


class VitposeLatencyMeasurer:
    def __init__(self, config_path: str, checkpoint: str, json_file: str, 
                 img_root: str, num_images: int, warmup_num: int, device: str):
        self.config_path = config_path
        self.checkpoint = checkpoint
        self.json_file = json_file
        self.img_root = img_root
        self.num_images = num_images
        self.warmup_num = warmup_num
        self.device = device
        
        # 모델 초기화 (visualize.py와 동일한 방식)
        self.pose_model = init_pose_model(config_path, checkpoint, device=device.lower())
        
        # 데이터셋 정보 가져오기
        dataset = self.pose_model.cfg.data['test']['type']
        dataset_info_cfg = self.pose_model.cfg.data['test'].get('dataset_info', None)
        self.dataset_info = DatasetInfo(dataset_info_cfg) if dataset_info_cfg else None
        self.dataset = dataset
        
        # COCO 데이터 로드
        self.coco = COCO(json_file)
        # self.img_ids = list(self.coco.imgs.keys())[:num_images]
        self.img_ids = list(self.coco.imgs.keys())
        
        logger.info(f'모델 로드 완료: {checkpoint}')
        logger.info(f'디바이스: {device}')
        logger.info(f'측정할 이미지 수: {len(self.img_ids)}')

    def _prepare_image_data(self, image_id):
        """이미지와 bounding box 정보를 준비합니다."""
        image_meta = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.img_root, image_meta['file_name'])
        ann_ids = self.coco.getAnnIds(image_id)
        ann_info = self.coco.loadAnns(ann_ids)
        dets = [{'bbox': ann['bbox']} for ann in ann_info]
        return image_path, dets, image_meta

    def run(self):
        """Latency 측정을 수행합니다."""
        if not torch.cuda.is_available() and 'cuda' in self.device:
            raise RuntimeError('CUDA is not available.')

        # Warmup
        logger.info(f'\nWarmup 시작 ({self.warmup_num} iterations)...')
        for i in range(min(self.warmup_num, len(self.img_ids))):
            image_id = self.img_ids[i]
            image_path, dets, _ = self._prepare_image_data(image_id)
            
            if 'cuda' in self.device:
                torch.cuda.synchronize()
            
            inference_top_down_pose_model(
                self.pose_model,
                image_path,
                dets,
                bbox_thr=None,
                format='xywh',
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                return_heatmap=False)
            
            if 'cuda' in self.device:
                torch.cuda.synchronize()
            
            logger.info(f'  Warmup {i + 1}/{self.warmup_num} 완료')
        
        # 실제 측정
        logger.info(f'\nLatency 측정 시작 ({len(self.img_ids)} 이미지)...')
        latencies = []
        batch_latencies = []
        
        for idx, image_id in enumerate(self.img_ids):
            image_path, dets, image_meta = self._prepare_image_data(image_id)
            num_cars = len(dets)
            
            if 'cuda' in self.device:
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            inference_top_down_pose_model(
                self.pose_model,
                image_path,
                dets,
                bbox_thr=None,
                format='xywh',
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                return_heatmap=False)
            
            if 'cuda' in self.device:
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            batch_latencies.append(elapsed)
            
            # 이미지당 latency (배치 크기 = 차량 수)
            if num_cars > 0:
                per_car_latency = elapsed / num_cars
                latencies.append(per_car_latency)
            
            if (idx + 1) % 10 == 0:
                logger.info(f'  [{idx + 1}/{len(self.img_ids)}] 처리 완료 - '
                          f'배치 시간: {elapsed * 1000:.3f} ms, 차량 수: {num_cars}')

        return batch_latencies, latencies


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    
    benchmark = VitposeLatencyMeasurer(
        args.config,
        args.checkpoint,
        args.json_file,
        args.img_root,
        args.num_images,
        args.warmup,
        args.device
    )
    
    batch_latencies, per_car_latencies = benchmark.run()
    
    # 통계 계산
    batch_tensor = torch.tensor(batch_latencies, dtype=torch.float64, device='cpu')
    batch_avg = batch_tensor.mean().item()
    batch_std = batch_tensor.std().item()
    batch_min = batch_tensor.min().item()
    batch_max = batch_tensor.max().item()
    
    car_tensor = torch.tensor(per_car_latencies, dtype=torch.float64, device='cpu')
    car_avg = car_tensor.mean().item()
    car_std = car_tensor.std().item()
    car_min = car_tensor.min().item()
    car_max = car_tensor.max().item()
    
    # 결과 출력
    print('\n' + '=' * 70)
    print('Latency 벤치마크 결과')
    print('=' * 70)
    print(f'Config       : {args.config}')
    print(f'Checkpoint   : {args.checkpoint}')
    print(f'Device       : {args.device}')
    print(f'측정 이미지 수: {len(batch_latencies)}')
    print(f'Warmup iters : {args.warmup}')
    print('-' * 70)
    print('배치당 처리 시간 (이미지당):')
    print(f'  평균: {batch_avg * 1000:.3f} ms')
    print(f'  표준편차: {batch_std * 1000:.3f} ms')
    print(f'  최소: {batch_min * 1000:.3f} ms')
    print(f'  최대: {batch_max * 1000:.3f} ms')
    print('-' * 70)
    print('차량당 처리 시간:')
    print(f'  평균: {car_avg * 1000:.3f} ms')
    print(f'  표준편차: {car_std * 1000:.3f} ms')
    print(f'  최소: {car_min * 1000:.3f} ms')
    print(f'  최대: {car_max * 1000:.3f} ms')
    print('-' * 70)
    if car_avg > 0:
        throughput = 1.0 / car_avg
        print(f'처리량 (throughput): {throughput:.2f} 차량/초')
    print('=' * 70)


if __name__ == '__main__':
    main()
