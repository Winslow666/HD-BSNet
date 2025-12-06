## HD-BSNet 训练与数据准备说明

本文档说明如何在本仓库中运行 **HD-BSNet** 系列模型的训练与测试，并给出推荐的数据集目录结构。

---

### 1. 环境准备

- **PyTorch**: 1.12.0  
- **torchvision**: 0.13.0  
- **MMDetection**: 3.1  
- **MMCV**: 2.0.1  

MMDetection 的安装与基础使用可参考官方文档：  
`https://mmdetection.readthedocs.io/en/latest/get_started.html`

额外依赖（AI-TOD 评估）：

```bash
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"
```

---

### 2. 仓库目录结构概览

与 HD-BSNet 直接相关的目录（仅列出关键部分）：

- `hd_bsnet_project/`：HD-BSNet 相关 detector / Faster R-CNN / Cascade R-CNN 定义与配置  
  - `hd_bsnet_detectors/`
  - `hd_bsnet_faster_rcnn/`
  - `hd_bsnet_cascade_rcnn/`
- `configs/`：通用数据集和模型配置（AITOD / VisDrone 等）
  - `_base_/datasets/coco_detection_aitod.py`
  - `_base_/datasets/coco_detection_visdrone.py`
  - `srtod/`、`cascade_rcnn/` 等任务配置
- `tools/`：训练与测试脚本
  - `train.py`
  - `test.py`
- `work_dirs/`：训练日志与模型权重的默认输出目录

**建议自行创建的数据根目录：**

- `data/AI-TOD/`（AI-TOD 数据集）
- `data/VisDrone-DET2019/`（VisDrone 2019 DET 数据集）

在相应配置文件中，将 `data_root` 修改为上述相对路径（见下面示例）。

---

### 3. 数据集目录结构

#### 3.1 AI-TOD 数据集

配置文件：`configs/_base_/datasets/coco_detection_aitod.py`

对应推荐目录结构：
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

#### 3.2 VisDrone-DET2019 数据集

配置文件：`configs/_base_/datasets/coco_detection_visdrone.py`

建议设置：

```python
data_root = 'data/VisDrone-DET2019/'
```

对应推荐目录结构：

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

### 4. 训练与测试命令示例

以下命令默认在仓库根目录 `HD_BSNet/` 下执行。

#### 4.1 单卡训练（Windows / Linux 通用）

- **VisDrone 上的 HD-BSNet Cascade R-CNN**：

```bash
python tools/train.py hd_bsnet_project/hd_bsnet_cascade_rcnn/config/hd-bsnet-cascade-rcnn-jwpm-bgonly_r50_fpn_visdrone.py
```

- **VisDrone 上的 HD-BSNet Faster R-CNN**：

```bash
python tools/train.py hd_bsnet_project/hd_bsnet_faster_rcnn/config/hd-bsnet-faster-rcnn-jwpm-bgonly_r50_fpn_visdrone.py
```

- **AI-TOD 上的 HD-BSNet Detectors 版本**：

```bash
python tools/train.py hd_bsnet_project/hd_bsnet_detectors/config/hd-bsnet-detectors-jwpm-bgonly_r50_fpn_aitod.py
```

> **注意**：  
> - 请确认对应配置中数据集路径已经改为你本机的实际路径。  
> - 训练日志和权重会默认保存在 `work_dirs/xxx/` 目录中。

#### 4.2 多卡分布式训练（Linux 推荐）

使用 MMDetection 标准分布式脚本，示例（8 卡训练）：

```bash
./tools/dist_train.sh \
  hd_bsnet_project/hd_bsnet_cascade_rcnn/config/hd-bsnet-cascade-rcnn-jwpm-bgonly_r50_fpn_visdrone.py \
  8
```

其中 `8` 为 GPU 数量，可根据实际情况修改。

#### 4.3 测试 / 推理

使用在 `work_dirs/` 中训练好的权重进行评估或可视化：

```
python tools/test.py \
  hd_bsnet_project/hd_bsnet_cascade_rcnn/config/hd-bsnet-cascade-rcnn-jwpm-bgonly_r50_fpn_visdrone.py \
  work_dirs/hd-bsnet-cascade-rcnn-jwpm-bgonly_r50_fpn_visdrone/latest.pth 
```


### 5. 常见检查项

- **data_root 路径是否正确**  
  - 所有使用到的数据集配置（AITOD / VisDrone）中，是否指向一个实际存在的目录。  

- **标注文件是否存在且格式正确**  
  - AI-TOD：`annotations/aitod_trainval_v1.json`、`annotations/aitod_test_v1.json`（COCO 格式 + AITODMetric）。  
  - VisDrone：`VisDrone2019-DET-train/visdrone_train_coco.json`、`VisDrone2019-DET-val/visdrone_val_coco.json`（COCO 格式）。

- **环境版本是否匹配**  
  - PyTorch / MMCV / MMDetection 版本之间的关系是否匹配兼容。



