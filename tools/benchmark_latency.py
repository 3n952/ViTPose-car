"""Benchmark ViTPose forward-pass latency on a CUDA device."""
import argparse
import os
import sys
import time

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

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
        '--device',
        default='cuda:2',
        help='추론에 사용할 디바이스 (cuda:0 또는 cpu)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Forward pass당 샘플 수')
    parser.add_argument(
        '--warmup',
        type=int,
        default=50,
        help='통계에서 제외할 warmup iteration 수')
    parser.add_argument(
        '--iters',
        type=int,
        default=200,
        help='측정할 iteration 수')
    parser.add_argument(
        '--data-root',
        default=None,
        help='Config 내부의 데이터셋 루트 경로 오버라이드')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=0,
        help='중간 로깅을 위한 interval (iteration 단위, 0이면 비활성화)')
    parser.add_argument(
        '--work-dir',
        default='runs/benchmark_latency',
        help='벤치마크 결과를 저장할 디렉토리')
    return parser.parse_args()


def build_test_loader(cfg: Config, batch_size: int):
    """테스트 데이터로더를 생성합니다."""
    dataset = build_dataset(cfg.data.train, dict(test_mode=True)) # 임시 train 사용
    test_loader_cfg = dict(
        samples_per_gpu=batch_size,
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
    
    # 인자 유효성 검사
    if args.batch_size <= 0:
        raise ValueError('--batch-size는 양수여야 합니다.')
    if args.iters <= 0:
        raise ValueError('--iters는 양수여야 합니다.')
    if args.warmup < 0:
        raise ValueError('--warmup은 0 이상이어야 합니다.')

    # Config 로드
    cfg = Config.fromfile(args.config)
    
    # 데이터 루트 오버라이드
    if args.data_root:
        override_data_root(cfg, args.data_root)
    
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.work_dir = args.work_dir or cfg.get('work_dir',
                                            'work_dirs/vitpose_maruhan')
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    
    setup_multi_processes(cfg)
    
    # 데이터로더 생성
    print(f'테스트 데이터셋 로딩 중... (batch_size={args.batch_size})')
    dataset, data_loader = build_test_loader(cfg, args.batch_size)
    print(f'데이터셋 크기: {len(dataset)} 샘플')
    
    data_iter = iter(data_loader)
    try:
        data = next(data_iter)
    except StopIteration:
        raise RuntimeError('테스트 데이터셋이 비어 있습니다.')
    
    # 모델 생성 및 체크포인트 로드
    print(f'모델 로딩 중: {args.checkpoint}')
    model = build_posenet(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # 디바이스 설정
    device_id = parse_device_id(args.device)
    is_cuda = device_id is not None
    
    if is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA를 사용할 수 없습니다.')
        torch.cuda.set_device(device_id)
        model = MMDataParallel(model.cuda(device_id), device_ids=[device_id])
        device = torch.device(f'cuda:{device_id}')
        print(f'디바이스: {device}')
    else:
        model = model.cpu()
        device = torch.device('cpu')
        print('디바이스: CPU')
    
    model.eval()
    
    def synchronize():
        if is_cuda:
            torch.cuda.synchronize(device=device)
    
    def run_forward():
        with torch.no_grad():
            model(return_loss=False, **data)
    
    # Warmup iterations
    print(f'\nWarmup 시작 ({args.warmup} iterations)...')
    for i in range(args.warmup):
        synchronize()
        run_forward()
        synchronize()
        if args.log_interval > 0 and (i + 1) % args.log_interval == 0:
            print(f'  Warmup iteration {i + 1}/{args.warmup}')
    
    # 실제 latency 측정
    print(f'\nLatency 측정 시작 ({args.iters} iterations)...')
    latencies = []
    for i in range(args.iters):
        synchronize()
        start = time.perf_counter()
        run_forward()
        synchronize()
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        if args.log_interval > 0 and (i + 1) % args.log_interval == 0:
            print(f'  Iter {i + 1}/{args.iters}: {elapsed * 1000:.3f} ms')
    
    # 통계 계산
    lat_tensor = torch.tensor(latencies, dtype=torch.float64, device='cpu')
    avg = lat_tensor.mean().item()
    median = lat_tensor.median().item()
    std = lat_tensor.std().item()
    p90 = torch.quantile(lat_tensor, 0.9).item()
    p95 = torch.quantile(lat_tensor, 0.95).item()
    p99 = torch.quantile(lat_tensor, 0.99).item()
    min_lat = lat_tensor.min().item()
    max_lat = lat_tensor.max().item()
    
    # 결과 출력
    print('\n' + '=' * 60)
    print('Latency 벤치마크 결과')
    print('=' * 60)
    print(f'Config       : {args.config}')
    print(f'Checkpoint   : {args.checkpoint}')
    print(f'Device       : {device}')
    print(f'Batch size   : {args.batch_size}')
    print(f'Warmup iters : {args.warmup}')
    print(f'Measured iters: {args.iters}')
    print('-' * 60)
    print(f'평균 latency   : {avg * 1000:.3f} ms')
    print(f'중앙값 latency : {median * 1000:.3f} ms')
    print(f'표준편차      : {std * 1000:.3f} ms')
    print(f'최소 latency  : {min_lat * 1000:.3f} ms')
    print(f'최대 latency  : {max_lat * 1000:.3f} ms')
    print(f'P90 latency   : {p90 * 1000:.3f} ms')
    print(f'P95 latency   : {p95 * 1000:.3f} ms')
    print(f'P99 latency   : {p99 * 1000:.3f} ms')
    print('-' * 60)
    if avg > 0:
        throughput = args.batch_size / avg
        print(f'처리량 (throughput): {throughput:.2f} samples/s')
        print(f'배치당 처리 시간   : {avg * 1000:.3f} ms')
        print(f'샘플당 처리 시간   : {avg * 1000 / args.batch_size:.3f} ms')
    print('=' * 60)
    
    # 결과를 파일로 저장
    result_file = os.path.join(cfg.work_dir, 'latency_benchmark.txt')
    with open(result_file, 'w') as f:
        f.write('Latency 벤치마크 결과\n')
        f.write('=' * 60 + '\n')
        f.write(f'Config       : {args.config}\n')
        f.write(f'Checkpoint   : {args.checkpoint}\n')
        f.write(f'Device       : {device}\n')
        f.write(f'Batch size   : {args.batch_size}\n')
        f.write(f'Warmup iters : {args.warmup}\n')
        f.write(f'Measured iters: {args.iters}\n')
        f.write('-' * 60 + '\n')
        f.write(f'평균 latency   : {avg * 1000:.3f} ms\n')
        f.write(f'중앙값 latency : {median * 1000:.3f} ms\n')
        f.write(f'표준편차      : {std * 1000:.3f} ms\n')
        f.write(f'최소 latency  : {min_lat * 1000:.3f} ms\n')
        f.write(f'최대 latency  : {max_lat * 1000:.3f} ms\n')
        f.write(f'P90 latency   : {p90 * 1000:.3f} ms\n')
        f.write(f'P95 latency   : {p95 * 1000:.3f} ms\n')
        f.write(f'P99 latency   : {p99 * 1000:.3f} ms\n')
        f.write('-' * 60 + '\n')
        if avg > 0:
            throughput = args.batch_size / avg
            f.write(f'처리량 (throughput): {throughput:.2f} samples/s\n')
            f.write(f'배치당 처리 시간   : {avg * 1000:.3f} ms\n')
            f.write(f'샘플당 처리 시간   : {avg * 1000 / args.batch_size:.3f} ms\n')
        f.write('=' * 60 + '\n')
    
    print(f'\n결과가 저장되었습니다: {result_file}')


if __name__ == '__main__':
    main()
