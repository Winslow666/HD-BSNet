[中文版](README_HD_BSNet.md) | English  （The YOLO code is under the branch HD_BSNet_yolo.）

## HD-BSNet Training and Data Preparation Guide

This document describes how to train and evaluate the **HD-BSNet** models in this repository, and gives recommended dataset directory structures.

---

### 1. Environment

- **PyTorch**: 1.12.0  
- **torchvision**: 0.13.0  
- **MMDetection**: 3.1  
- **MMCV**: 2.0.1  

For installation and basic usage of MMDetection, please refer to the official docs:  
`https://mmdetection.readthedocs.io/en/latest/get_started.html`

Extra dependency for AI-TOD evaluation:

```bash
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"
```

---

### 2. Repository Structure Overview

Directories directly related to HD-BSNet (only key parts):

- `hd_bsnet_project/`: Definitions and configs of HD-BSNet Detectors / Faster R-CNN / Cascade R-CNN  
  - `hd_bsnet_detectors/`
  - `hd_bsnet_faster_rcnn/`
  - `hd_bsnet_cascade_rcnn/`
- `configs/`: Common dataset and model configs (AITOD / VisDrone, etc.)
  - `_base_/datasets/coco_detection_aitod.py`
  - `_base_/datasets/coco_detection_visdrone.py`
  - Task configs such as `srtod/`, `cascade_rcnn/`, etc.
- `tools/`: Training and testing scripts
  - `train.py`
  - `test.py`
- `work_dirs/`: Default directory for training logs and model checkpoints

**Recommended data root directories:**

- `data/AI-TOD/` (AI-TOD dataset)
- `data/VisDrone-DET2019/` (VisDrone 2019 DET dataset)

In the corresponding config files, change `data_root` to these relative paths (see examples below).

---

### 3. Dataset Directory Structure

#### 3.1 AI-TOD Dataset

Config file: `configs/_base_/datasets/coco_detection_aitod.py`

Recommended directory layout:

```text
data/
  AI-TOD/
    annotations/
      aitod_trainval_v1.json
      aitod_test_v1.json
    trainval/
      images/
        xxx1.jpg
        xxx2.jpg
        ...
    test/
      images/
        yyy1.jpg
        yyy2.jpg
        ...
```

#### 3.2 VisDrone-DET2019 Dataset

Config file: `configs/_base_/datasets/coco_detection_visdrone.py`

Recommended setting:

```python
data_root = 'data/VisDrone-DET2019/'
```

Recommended directory layout:

```text
data/
  VisDrone-DET2019/
    VisDrone2019-DET-train/
      images/
        000001.jpg
        000002.jpg
        ...
      visdrone_train_coco.json
    VisDrone2019-DET-val/
      images/
        000001.jpg
        000002.jpg
        ...
      visdrone_val_coco.json
```

---

### 4. Training and Testing Commands

All commands below assume you are in the repository root `HD_BSNet/`.

#### 4.1 Single-GPU Training (Windows / Linux)

- **HD-BSNet Cascade R-CNN on VisDrone**:

```bash
python tools/train.py hd_bsnet_project/hd_bsnet_cascade_rcnn/config/hd-bsnet-cascade-rcnn-jwpm-bgonly_r50_fpn_visdrone.py
```

- **HD-BSNet Faster R-CNN on VisDrone**:

```bash
python tools/train.py hd_bsnet_project/hd_bsnet_faster_rcnn/config/hd-bsnet-faster-rcnn-jwpm-bgonly_r50_fpn_visdrone.py
```

- **HD-BSNet Detectors on AI-TOD**:

```bash
python tools/train.py hd_bsnet_project/hd_bsnet_detectors/config/hd-bsnet-detectors-jwpm-bgonly_r50_fpn_aitod.py
```

> **Note**:  
> - Make sure all dataset paths in the configs are updated to your local paths.  
> - Logs and checkpoints are saved under `work_dirs/xxx/` by default.

#### 4.2 Multi-GPU Distributed Training (Linux Recommended)

Use MMDetection’s standard distributed script, e.g. (8 GPUs):

```bash
./tools/dist_train.sh \
  hd_bsnet_project/hd_bsnet_cascade_rcnn/config/hd-bsnet-cascade-rcnn-jwpm-bgonly_r50_fpn_visdrone.py \
  8
```

Here `8` is the number of GPUs; adjust it according to your setup.

#### 4.3 Testing / Inference

Use the checkpoints saved in `work_dirs/` for evaluation or visualization:

```bash
python tools/test.py \
  hd_bsnet_project/hd_bsnet_cascade_rcnn/config/hd-bsnet-cascade-rcnn-jwpm-bgonly_r50_fpn_visdrone.py \
  work_dirs/hd-bsnet-cascade-rcnn-jwpm-bgonly_r50_fpn_visdrone/latest.pth
```

---

### 5. Checklist

- **Is `data_root` correct?**  
  - In all dataset configs (AITOD / VisDrone), check that points to an existing directory.  

- **Do annotation files exist and have the correct format?**  
  - AI-TOD: `annotations/aitod_trainval_v1.json`, `annotations/aitod_test_v1.json` (COCO format + AITODMetric).  
  - VisDrone: `VisDrone2019-DET-train/visdrone_train_coco.json`, `VisDrone2019-DET-val/visdrone_val_coco.json` (COCO format).

- **Are the library versions compatible?**  
  - Check that the versions of PyTorch, MMCV, and MMDetection are mutually compatible and match those listed above.
