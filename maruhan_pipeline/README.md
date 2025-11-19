# Maruhan car keypoint pipeline

The files under `maruhan_pipeline/` trim the original MMPose entry scripts
down to the minimum required to train, evaluate, and visualize ViTPose on the
`Maruhan-car-kp` dataset.

## 1. Dataset layout

By default the config expects the Roboflow export to live at
`../Maruhan-car-kp/train` (relative to the repo root). Set the environment
variable `MARUHAN_DATA_ROOT` or pass `--data-root` to the scripts if your copy
resides somewhere else. The config currently points train/val/test to the same
JSON; export or create dedicated splits when you are ready.

## 2. Training

```bash
python maruhan_pipeline/train.py \
  --config configs/maruhan/vitpose_maruhan_256x192.py \
  --work-dir work_dirs/vitpose_maruhan \
  --pretrained checkpoints/ViTPose/vitpose-b.pth
```

The wrapper only exposes the arguments that are typically needed:

- `--data-root` rewrites the annotation/image paths inside the config.
- `--device` chooses the CUDA device or CPU (defaults to `cuda:0`).
- `--resume-from`, `--pretrained`, `--seed`, `--no-validate` behave as expected.

## 3. Evaluation

```bash
python maruhan_pipeline/eval.py \
  --config configs/maruhan/vitpose_maruhan_256x192.py \
  --checkpoint work_dirs/vitpose_maruhan/latest.pth \
  --metric mAP
```

The script builds a dataloader from the config, loads the checkpoint, performs
single-GPU inference, and prints the metrics specified via `--metric`
(defaults to whatever is defined under `evaluation.metric` in the config). Use
`--out results.pkl` if you need the raw predictions.

## 4. Visualization

```bash
python maruhan_pipeline/visualize.py \
  --config configs/maruhan/vitpose_maruhan_256x192.py \
  --checkpoint work_dirs/vitpose_maruhan/best_AP.pth \
  --json-file ../Maruhan-car-kp/train/_annotations.coco.json \
  --img-root ../Maruhan-car-kp/train \
  --out-dir work_dirs/vitpose_maruhan/vis
```

The visualization helper reuses the COCO annotations to obtain bounding boxes,
runs `inference_top_down_pose_model`, and dumps rendered images to `--out-dir`.
Use `--kpt-thr` to adjust the confidence threshold or `--max-images` to limit
how many samples are processed.
