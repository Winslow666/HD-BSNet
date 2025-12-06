#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SR-TOD RH Inference Demo (using real model)

This script loads the SRTOD Cascade R-CNN model defined in the repo, runs a
forward extract on an input image, computes the reconstructed image via RH,
and the difference map between the reconstruction and the (normalized) source
image provided by the custom data preprocessor. Results are saved to disk.

Example:
  python demo/srtod_rh_infer_demo.py \
    --config ./srtod_project/srtod_cascade_rcnn/config/srtod-cascade-rcnn_r50_fpn_1x_coco.py \
    --img demo/demo.jpg \
    --checkpoint /path/to/your_model.pth \
    --device cuda:0 \
    --out-dir outputs
"""

import argparse
import os
import sys
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdet.apis import init_detector
from mmdet.utils import register_all_modules

# Explicitly import custom datasets to ensure they are registered
import mmdet.datasets.visdrone  # This will trigger the @DATASETS.register_module() decorator


def load_image_as_tensor(img_path: str) -> torch.Tensor:
    """Load image via OpenCV (BGR), return CHW uint8 torch tensor without batch."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f'Failed to read image: {img_path}')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
    return img_tensor


def tensor_to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """Convert CHW float tensor in [0,1] to HWC uint8 RGB numpy image."""
    t = t.detach().cpu().clamp(0, 1)
    img = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return img


def save_gray_heatmap(arr: np.ndarray, save_path: str) -> None:
    """Save a single-channel array (float64/float32) as a heatmap PNG."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 使用 'Blues_r' 来获得深蓝色效果：低值显示为深蓝色，高值显示为亮蓝色
    plt.imsave(save_path, arr, cmap='Blues_r')


def run_demo(config: str, checkpoint: str, img_path: str, device: str, out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    # Register all modules to ensure custom datasets are available
    register_all_modules()

    # 1) Build model
    model = init_detector(config, checkpoint if checkpoint else None, device=device)
    model.eval()

    # 2) Prepare data via the model's custom data_preprocessor
    #    We only pass inputs; the preprocessor will also produce 'src_inputs' in [0,1]
    img_tensor_chw = load_image_as_tensor(img_path)  # CHW, uint8
    data = {'inputs': [img_tensor_chw], 'data_samples': None}
    with torch.no_grad():
        processed = model.data_preprocessor(data, training=False)

        batch_inputs = processed['inputs']            # normalized by mean/std
        src_inputs = processed['src_inputs']          # only scaled to [0,1]

        # 3) Feature extraction and RH reconstruction (as in model.predict/loss)
        feats = model.extract_feat(batch_inputs)
        r_img = model.rh(feats[0].clone())            # [B,3,H,W], sigmoid -> [0,1]

        # 4) Compute difference map against preprocessed source image in [0,1]
        #    Use mean over channels (same as code: sum(abs)/3)
        diff_map = torch.mean(torch.abs(r_img - src_inputs), dim=1, keepdim=True)  # [B,1,H,W]

    # 5) Save outputs
    rec_img = tensor_to_uint8_image(r_img[0])
    rec_path = os.path.join(out_dir, 'rh_reconstructed.png')
    Image.fromarray(rec_img).save(rec_path)

    diff_np = diff_map[0, 0].detach().cpu().numpy()
    diff_path = os.path.join(out_dir, 'rh_difference_map.png')
    save_gray_heatmap(diff_np, diff_path)

    return rec_path, diff_path


def main():
    parser = argparse.ArgumentParser(description='SR-TOD RH inference demo (real model)')
    parser.add_argument('--config', type=str, required=False,
                        default=os.path.join('srtod_project', 'srtod_cascade_rcnn', 'config',
                                             'srtod-cascade-rcnn_r50_fpn_1x_coco.py'),
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint (.pth). Optional')
    parser.add_argument('--img', type=str, required=False, default=os.path.join('demo', 'test.jpg'),
                        help='Path to input image')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device, e.g., cuda:0 or cpu')
    parser.add_argument('--out-dir', type=str, default='outputs', help='Directory to save outputs')

    args = parser.parse_args()

    if not os.path.exists(args.img):
        print(f"[ERROR] Image not found: {args.img}")
        sys.exit(1)

    print('[INFO] Building model...')
    print(f'       config:     {args.config}')
    print(f'       checkpoint: {args.checkpoint or "<none> (random init)"}')
    print(f'       device:     {args.device}')

    try:
        rec_path, diff_path = run_demo(
            config=args.config,
            checkpoint=args.checkpoint,
            img_path=args.img,
            device=args.device,
            out_dir=args.out_dir,
        )
        print('[OK] Saved:')
        print(f'     Reconstructed image: {rec_path}')
        print(f'     Difference map:      {diff_path}')
    except Exception as e:
        print(f'[ERROR] {e}')
        raise


if __name__ == '__main__':
    main()


