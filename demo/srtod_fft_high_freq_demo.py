#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SR-TOD FFT High-Frequency Difference Map Demo

This script implements an innovative approach using Fast Fourier Transform (FFT)
to separate high-frequency components from original and reconstructed images,
then constructs a high-frequency difference map for better small object detection.

Key Innovation:
1. Apply FFT to both original and reconstructed images
2. Extract high-frequency components using frequency domain filtering
3. Compute difference map in frequency domain
4. Transform back to spatial domain for visualization
5. Binarize the difference map using learnable threshold (DGFE approach)

Example:
  python demo/srtod_fft_high_freq_demo.py \
    --config ./srtod_project/srtod_cascade_rcnn/config/srtod-cascade-rcnn_r50_fpn_1x_coco.py \
    --img demo/drone.jpg \
    --checkpoint /path/to/your_model.pth \
    --device cuda:0 \
    --out-dir outputs \
    --freq-threshold 0.1 \
    --binary-threshold 0.5 \
    --adaptive-threshold
"""

import argparse
import os
import sys
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
# Set matplotlib to use non-interactive backend to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# Ensure project root is importable and prioritize local mmdet over system installation
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)  # Insert at beginning to prioritize local modules

# Debug: Print sys.path to verify correct order
print(f"[DEBUG] Project root: {project_root}")
print(f"[DEBUG] First few sys.path entries: {sys.path[:3]}")

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


def apply_fft_high_pass_filter(image_tensor: torch.Tensor, freq_threshold: float = 0.1) -> torch.Tensor:
    """
    Apply FFT high-pass filter to extract high-frequency components.
    
    Args:
        image_tensor: Input image tensor [B, C, H, W] in range [0, 1]
        freq_threshold: Frequency threshold for high-pass filtering (0.0 to 1.0)
    
    Returns:
        High-frequency components tensor [B, C, H, W]
    """
    batch_size, channels, height, width = image_tensor.shape
    
    # Convert to frequency domain using FFT
    # Shift zero frequency to center
    fft_shifted = torch.fft.fftshift(torch.fft.fft2(image_tensor), dim=(-2, -1))
    
    # Create high-pass filter mask
    # Center coordinates
    center_h, center_w = height // 2, width // 2
    
    # Create frequency grid
    h_indices = torch.arange(height, device=image_tensor.device)
    w_indices = torch.arange(width, device=image_tensor.device)
    
    # Distance from center (normalized)
    h_dist = (h_indices - center_h) / (height // 2)
    w_dist = (w_indices - center_w) / (width // 2)
    
    # Create 2D distance grid
    h_grid, w_grid = torch.meshgrid(h_dist, w_dist, indexing='ij')
    distance_grid = torch.sqrt(h_grid**2 + w_grid**2)
    
    # High-pass filter mask (Butterworth filter)
    order = 2  # Filter order
    mask = 1.0 / (1.0 + (freq_threshold / (distance_grid + 1e-8))**order)
    
    # Apply mask to frequency domain
    filtered_fft = fft_shifted * mask.unsqueeze(0).unsqueeze(0)
    
    # Transform back to spatial domain
    filtered_image = torch.fft.ifft2(torch.fft.ifftshift(filtered_fft, dim=(-2, -1)))
    
    # Take real part and normalize
    filtered_image = torch.real(filtered_image)
    
    return filtered_image


def compute_high_frequency_difference_map(original: torch.Tensor, reconstructed: torch.Tensor, 
                                        freq_threshold: float = 0.1) -> torch.Tensor:
    """
    Compute high-frequency difference map using FFT.
    
    Args:
        original: Original image tensor [B, C, H, W]
        reconstructed: Reconstructed image tensor [B, C, H, W]
        freq_threshold: Frequency threshold for filtering
    
    Returns:
        High-frequency difference map [B, 1, H, W]
    """
    # Extract high-frequency components
    original_high_freq = apply_fft_high_pass_filter(original, freq_threshold)
    reconstructed_high_freq = apply_fft_high_pass_filter(reconstructed, freq_threshold)
    
    # Compute difference in high-frequency domain
    high_freq_diff = torch.abs(original_high_freq - reconstructed_high_freq)
    
    # Average across channels
    diff_map = torch.mean(high_freq_diff, dim=1, keepdim=True)
    
    # Normalize to [0, 1]
    diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    
    return diff_map


def binarize_difference_map(difference_map: torch.Tensor, threshold: float = 0.0156862, 
                          adaptive: bool = False) -> torch.Tensor:
    """
    Binarize difference map using learnable threshold, following DGFE module approach.
    
    Args:
        difference_map: Input difference map tensor [B, 1, H, W] in range [0, 1]
        threshold: Threshold value for binarization (default from DGFE module)
        adaptive: If True, use adaptive threshold based on statistics
    
    Returns:
        Binary mask tensor [B, 1, H, W] with values 0 or 1
    """
    if adaptive:
        # Use adaptive threshold: mean + 2*std to focus on high-value regions
        diff_np = difference_map[0, 0].detach().cpu().numpy()
        adaptive_threshold = diff_np.mean() + 2 * diff_np.std()
        print(f"[INFO] Using adaptive threshold: {adaptive_threshold:.6f}")
        threshold = adaptive_threshold
    
    # Apply the same binarization method as in DGFE module
    # torch.sign(difference_map - threshold) returns -1, 0, or 1
    # Adding 1 converts to 0, 1, or 2
    # Multiplying by 0.5 converts to 0, 0.5, or 1
    # This effectively creates a binary mask where values > threshold become 1, others become 0
    binary_mask = (torch.sign(difference_map - threshold) + 1) * 0.5
    
    return binary_mask


def save_high_freq_heatmap(arr: np.ndarray, save_path: str, title: str = "High-Frequency Difference Map") -> None:
    """Save high-frequency difference map as a heatmap with enhanced visualization."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create figure with enhanced visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Use 'hot' colormap for better visualization of high-frequency differences
        im = ax.imshow(arr, cmap='Blues_r', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('High-Frequency Difference Intensity', rotation=270, labelpad=15)
        
        # Set title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save with lower DPI to avoid hanging issues
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure
        print(f"[INFO] Saved heatmap: {save_path}")
        
    except Exception as e:
        print(f"[WARNING] Failed to save heatmap {save_path}: {e}")
        # Fallback: save as simple numpy array
        try:
            np.save(save_path.replace('.png', '.npy'), arr)
            print(f"[INFO] Saved as numpy array: {save_path.replace('.png', '.npy')}")
        except Exception as e2:
            print(f"[ERROR] Failed to save numpy array: {e2}")


def save_binary_mask(arr: np.ndarray, save_path: str, title: str = "Binary Mask") -> None:
    """Save binary mask as a black and white image."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create figure for binary visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Use binary colormap (black and white)
        im = ax.imshow(arr, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Binary Value (0 or 1)', rotation=270, labelpad=15)
        
        # Set title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save with lower DPI to avoid hanging issues
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure
        print(f"[INFO] Saved binary mask: {save_path}")
        
    except Exception as e:
        print(f"[WARNING] Failed to save binary mask {save_path}: {e}")
        # Fallback: save as simple numpy array
        try:
            np.save(save_path.replace('.png', '.npy'), arr)
            print(f"[INFO] Saved as numpy array: {save_path.replace('.png', '.npy')}")
        except Exception as e2:
            print(f"[ERROR] Failed to save numpy array: {e2}")


def run_fft_high_freq_demo(config: str, checkpoint: str, img_path: str, device: str, 
                          out_dir: str, freq_threshold: float = 0.1, 
                          binary_threshold: float = 0.5, adaptive_threshold: bool = False) -> Tuple[str, str, str, str]:
    """
    Run FFT-based high-frequency difference map demo with binarization.
    
    Returns:
        Tuple of (reconstructed_path, high_freq_diff_path, traditional_diff_path, binary_mask_path)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Register all modules to ensure custom datasets are available
    register_all_modules()

    # 1) Build model
    print("[INFO] Building model...")
    model = init_detector(config, checkpoint if checkpoint else None, device=device)
    model.eval()

    # 2) Prepare data
    print("[INFO] Loading and preprocessing image...")
    img_tensor_chw = load_image_as_tensor(img_path)
    data = {'inputs': [img_tensor_chw], 'data_samples': None}
    
    with torch.no_grad():
        processed = model.data_preprocessor(data, training=False)
        batch_inputs = processed['inputs']
        src_inputs = processed['src_inputs']

        # 3) Feature extraction and RH reconstruction
        print("[INFO] Extracting features and reconstructing image...")
        feats = model.extract_feat(batch_inputs)
        r_img = model.rh(feats[0].clone())

        # 4) Compute traditional difference map
        print("[INFO] Computing traditional difference map...")
        traditional_diff = torch.mean(torch.abs(r_img - src_inputs), dim=1, keepdim=True)

        # 5) Compute high-frequency difference map using FFT
        print("[INFO] Computing FFT-based high-frequency difference map...")
        high_freq_diff = compute_high_frequency_difference_map(
            src_inputs, r_img, freq_threshold
        )

        # 6) Binarize high-frequency difference map
        print("[INFO] Binarizing high-frequency difference map...")
        
        # Debug: Print statistics of high-frequency difference map
        high_freq_diff_np = high_freq_diff[0, 0].detach().cpu().numpy()
        print(f"[DEBUG] High-freq diff map stats: min={high_freq_diff_np.min():.6f}, max={high_freq_diff_np.max():.6f}, mean={high_freq_diff_np.mean():.6f}")
        print(f"[DEBUG] Using binary threshold: {binary_threshold}")
        
        # If threshold is too low, suggest a better one
        if binary_threshold < high_freq_diff_np.mean():
            suggested_threshold = high_freq_diff_np.mean() + 2 * high_freq_diff_np.std()
            print(f"[WARNING] Threshold {binary_threshold} might be too low. Suggested threshold: {suggested_threshold:.6f}")
        
        binary_mask = binarize_difference_map(high_freq_diff, binary_threshold, adaptive_threshold)
        
        # Debug: Print binary mask statistics
        binary_mask_np = binary_mask[0, 0].detach().cpu().numpy()
        white_pixels = np.sum(binary_mask_np > 0.5)
        total_pixels = binary_mask_np.size
        print(f"[DEBUG] Binary mask: {white_pixels}/{total_pixels} pixels are white ({white_pixels/total_pixels*100:.2f}%)")

    # 7) Save outputs
    print("[INFO] Saving results...")
    
    # Save reconstructed image
    print("[INFO] Saving reconstructed image...")
    rec_img = tensor_to_uint8_image(r_img[0])
    rec_path = os.path.join(out_dir, 'rh_reconstructed_fft.png')
    try:
        Image.fromarray(rec_img).save(rec_path)
        print(f"[INFO] Saved reconstructed image: {rec_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save reconstructed image: {e}")

    # Save traditional difference map
    print("[INFO] Saving traditional difference map...")
    traditional_diff_np = traditional_diff[0, 0].detach().cpu().numpy()
    traditional_diff_path = os.path.join(out_dir, 'traditional_difference_map.png')
    save_high_freq_heatmap(traditional_diff_np, traditional_diff_path, "Traditional Difference Map")

    # Save high-frequency difference map
    print("[INFO] Saving high-frequency difference map...")
    high_freq_diff_np = high_freq_diff[0, 0].detach().cpu().numpy()
    high_freq_diff_path = os.path.join(out_dir, 'high_frequency_difference_map.png')
    save_high_freq_heatmap(high_freq_diff_np, high_freq_diff_path, "High-Frequency Difference Map")

    # Save binary mask
    print("[INFO] Saving binary mask...")
    binary_mask_np = binary_mask[0, 0].detach().cpu().numpy()
    binary_mask_path = os.path.join(out_dir, 'binary_mask.png')
    save_binary_mask(binary_mask_np, binary_mask_path, f"Binary Mask (threshold={binary_threshold})")

    print("[INFO] All files saved successfully!")
    return rec_path, high_freq_diff_path, traditional_diff_path, binary_mask_path


def main():
    parser = argparse.ArgumentParser(description='SR-TOD FFT High-Frequency Difference Map Demo')
    parser.add_argument('--config', type=str, required=False,
                        default=os.path.join('srtod_project', 'srtod_cascade_rcnn', 'config',
                                             'srtod-cascade-rcnn_r50_fpn_1x_coco.py'),
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint (.pth). Optional')
    parser.add_argument('--img', type=str, required=False, default=os.path.join('demo', 'test.jpg'),
                        help='Path to input image')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device, e.g., cuda:0 or cpu')
    parser.add_argument('--out-dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--freq-threshold', type=float, default=0.1, 
                        help='Frequency threshold for high-pass filtering (0.0 to 1.0)')
    parser.add_argument('--binary-threshold', type=float, default=0.1,
                        help='Threshold for binarizing difference map (default 0.5 for cleaner results)')
    parser.add_argument('--adaptive-threshold', action='store_true',
                        help='Use adaptive threshold (mean + 2*std) instead of fixed threshold')

    args = parser.parse_args()

    if not os.path.exists(args.img):
        print(f"[ERROR] Image not found: {args.img}")
        sys.exit(1)

    print('[INFO] FFT High-Frequency Difference Map Demo')
    print(f'       config:           {args.config}')
    print(f'       checkpoint:       {args.checkpoint or "<none> (random init)"}')
    print(f'       image:            {args.img}')
    print(f'       device:           {args.device}')
    print(f'       frequency threshold: {args.freq_threshold}')
    print(f'       binary threshold: {args.binary_threshold}')
    print(f'       adaptive threshold: {args.adaptive_threshold}')
    print(f'       output directory: {args.out_dir}')

    try:
        rec_path, high_freq_diff_path, traditional_diff_path, binary_mask_path = run_fft_high_freq_demo(
            config=args.config,
            checkpoint=args.checkpoint,
            img_path=args.img,
            device=args.device,
            out_dir=args.out_dir,
            freq_threshold=args.freq_threshold,
            binary_threshold=args.binary_threshold,
            adaptive_threshold=args.adaptive_threshold
        )
        
        print('[OK] Demo completed successfully!')
        print(f'     Reconstructed image:        {rec_path}')
        print(f'     High-frequency diff map:    {high_freq_diff_path}')
        print(f'     Traditional diff map:       {traditional_diff_path}')
        print(f'     Binary mask:                {binary_mask_path}')
        print('\n[INFO] The high-frequency difference map should better highlight small objects and edges!')
        print('[INFO] The binary mask shows regions where high-frequency differences exceed the threshold!')
        
    except Exception as e:
        print(f'[ERROR] {e}')
        raise


if __name__ == '__main__':
    main()
