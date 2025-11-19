# usages
# usage: python visualize.py --config configs/maruhan/vitpose_maruhan_256x192.py --checkpoint work_dirs/vitpose_maruhan/best_AP.pth --json-file ../Maruhan-car-kp/val/_annotations.coco.json

"""Visualize Maruhan predictions using a trained checkpoint."""
import argparse
import os

from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run inference + visualization on Maruhan images.')
    parser.add_argument(
        '--config',
        default='configs/maruhan/vitpose_maruhan_256x192.py',
        help='Config path used during training.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file.')
    parser.add_argument(
        '--json-file',
        required=True,
        help='COCO-style annotation file providing image paths and boxes.')
    parser.add_argument(
        '--img-root',
        default='../Maruhan-car-kp/train',
        help='Root directory containing the images referenced in the JSON.')
    parser.add_argument(
        '--out-dir',
        default='runs/exp1/vis',
        help='Directory to store visualized images.')
    parser.add_argument(
        '--device',
        default='cuda:2',
        help='Device for inference (cuda:0 or cpu).')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.4,
        help='Keypoint confidence threshold for drawing.')
    parser.add_argument(
        '--max-images',
        type=int,
        default=150,
        help='Optional limit on how many images to render.')
    return parser.parse_args()


def main():
    args = parse_args()
    coco = COCO(args.json_file)
    os.makedirs(args.out_dir, exist_ok=True)

    pose_model = init_pose_model(
        args.config, args.checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info_cfg = pose_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info_cfg) if dataset_info_cfg else None

    img_ids = list(coco.imgs.keys())
    if args.max_images is not None:
        img_ids = img_ids[:args.max_images]

    for idx, image_id in enumerate(img_ids):
        image_meta = coco.loadImgs(image_id)[0]
        image_path = os.path.join(args.img_root, image_meta['file_name'])
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
        vis_pose_result(
            pose_model,
            image_path,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False,
            out_file=out_file)


if __name__ == '__main__':
    main()
