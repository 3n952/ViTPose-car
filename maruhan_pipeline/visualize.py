# usages
# usage: python visualize.py --config configs/maruhan/vitpose_maruhan_256x192.py --checkpoint work_dirs/vitpose_maruhan/best_AP.pth --json-file ../Maruhan-car-kp/val/_annotations.coco.json

"""Visualize Maruhan predictions using a trained checkpoint."""
import argparse
import os

import cv2
import numpy as np
from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def visualize_wheels(image_path, pose_results, out_file):
    """visualize wheels and ground contact"""

    img = cv2.imread(image_path)
    if img is None:
        return
    
    point_reorder = [0, 3, 1, 2] # to draw polygon

    for car in pose_results:
        kpts = car['keypoints']  # shape: (num_keypoints, 3) - (x, y, score)
        wheel_kpts = kpts[:4]  # 0: front_right_wheel, 1: rear_left_wheel, 2: rear_right_wheel, 3: front_left_wheel
        
        # filter kpt with confidence threshold
        valid_wheels = [None] * 4
        for i, (x, y, score) in enumerate(wheel_kpts):
            valid_wheels[i] = (int(x), int(y), score)
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
        
        if all(valid_wheels):
            points = np.array([valid_wheels[i][:2] for i in point_reorder], dtype=np.int32)
            overlay = img.copy()

            cv2.fillPoly(overlay, [points], (255, 200, 100))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.line(img, valid_wheels[0][:2], valid_wheels[3][:2], (0, 0, 255), 2)  # front_left -> front_right
            cv2.line(img, valid_wheels[0][:2], valid_wheels[2][:2], (0, 0, 255), 2)  # front_right -> rear_right
            cv2.line(img, valid_wheels[3][:2], valid_wheels[1][:2], (0, 0, 255), 2)  # rear_right -> rear_left
            cv2.line(img, valid_wheels[1][:2], valid_wheels[2][:2], (0, 0, 255), 2)  # rear_left -> front_left
    
    cv2.imwrite(out_file, img)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run inference + visualization on Maruhan images.')
    parser.add_argument(
        '--config',
        default='configs/maruhan/vitpose_maruhan_256x192.py',
        help='Config path used during training.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file.')
    parser.add_argument(
        '--out-dir',
        default='runs/benchmark/vis',
        help='Directory to store visualized images.')
    parser.add_argument(
        '--device',
        default='cuda:2',
        help='Device for inference (cuda:0 or cpu).')
    return parser.parse_args()


def main():
    args = parse_args()

    pose_model = init_pose_model(
        args.config, args.checkpoint, device=args.device.lower())

    coco = COCO(pose_model.cfg.data['benchmark']['ann_file'])  # option: use test or benchmark annotation file
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = pose_model.cfg.data['benchmark']['type']  # option: use test or benchmark dataset
    dataset_info_cfg = pose_model.cfg.data['benchmark'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info_cfg) if dataset_info_cfg else None

    img_ids = list(coco.imgs.keys())
    for idx, image_id in enumerate(img_ids):
        image_meta = coco.loadImgs(image_id)[0]
        image_path = os.path.join(pose_model.cfg.data['benchmark']['img_prefix'], image_meta['file_name'])
        ann_ids = coco.getAnnIds(image_id)
        ann_info = coco.loadAnns(ann_ids)
        dets = [{
            'bbox': ann['bbox'],
        } for ann in ann_info]

        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            image_path,
            dets,
            bbox_thr=None,
            format='xywh',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False)

        out_file = os.path.join(args.out_dir, f'{idx:04d}_{image_meta["file_name"]}')

        visualize_wheels(
            image_path,
            pose_results,
            out_file,
            )
        
        print(f"[{idx+1}/{len(img_ids)}] Processed: {image_meta['file_name']}")


if __name__ == '__main__':
    main()
