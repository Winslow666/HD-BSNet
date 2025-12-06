# SR-TOD FFT High-Frequency Difference Map Demo

## 概述

这是一个基于快速傅里叶变换(FFT)的高频差异图生成demo，专门用于增强小目标和边缘特征的检测。通过分离原始图像和重建图像的高频分量，可以生成更敏感的差异图。

## 核心创新点

### 1. **FFT高频分量分离**
- 使用FFT将图像转换到频域
- 应用高通滤波器提取高频成分（边缘、纹理、小目标）
- 抑制低频成分（平滑区域、大目标）

### 2. **频域差异计算**
- 在频域中计算原始图像与重建图像的高频差异
- 避免空间域直接相减可能带来的噪声干扰
- 更好地突出小目标的特征信息

### 3. **自适应阈值控制**
- 可调节的频率阈值参数
- 支持不同场景下的最佳滤波效果

## 使用方法

### 基本命令
```bash
python demo/srtod_fft_simple_demo.py \
  --checkpoint work_dirs/srtod-cascade-rcnn_r50_fpn_1x_aitod/epoch_9.pth \
  --img demo/drone.jpg \
  --freq-threshold 0.15
```

### 参数说明
- `--checkpoint`: 训练好的模型权重文件路径
- `--img`: 输入图像路径
- `--freq-threshold`: 频率阈值 (0.0-1.0)，控制高通滤波的强度
- `--out-dir`: 输出目录 (默认: outputs)

### 频率阈值调节建议
- **0.05-0.1**: 强滤波，突出边缘和小目标
- **0.15-0.2**: 中等滤波，平衡细节和噪声
- **0.25-0.3**: 弱滤波，保留更多细节信息

## 输出结果

运行成功后，会在输出目录生成三个文件：

1. **`reconstructed_fft.png`**: RH重建的图像
2. **`traditional_difference_map.png`**: 传统差异图（直接相减）
3. **`high_frequency_difference_map.png`**: FFT高频差异图

## 技术原理

### FFT高通滤波过程
```
输入图像 → FFT变换 → 频域 → 高通滤波 → 逆FFT → 高频分量
```

### 高频差异图计算
```
原始图像高频分量 - 重建图像高频分量 = 高频差异图
```

### 滤波器设计
- 使用Butterworth高通滤波器
- 二阶滤波器提供平滑的过渡
- 中心频率可调节

## 优势特点

1. **小目标增强**: 高频分量更好地保留小目标信息
2. **噪声抑制**: 频域滤波减少低频噪声干扰
3. **边缘突出**: 高频差异图更突出边缘特征
4. **参数可控**: 可调节的频率阈值适应不同场景

## 应用场景

- **小目标检测**: 无人机、小动物等小目标识别
- **边缘检测**: 建筑物边缘、道路边界等
- **异常检测**: 图像重建异常区域识别
- **质量评估**: 图像重建质量评估

## 注意事项

1. **计算复杂度**: FFT计算比直接相减稍慢
2. **内存使用**: 需要额外的频域存储空间
3. **参数调优**: 频率阈值需要根据具体场景调整
4. **图像尺寸**: 建议使用2的幂次尺寸以获得最佳FFT性能

## 扩展功能

### 多尺度FFT
可以扩展为多尺度FFT分析，在不同频率范围生成差异图：

```python
# 低频差异图 (0.0-0.1)
low_freq_diff = create_high_frequency_difference_map(src, rec, 0.05)

# 中频差异图 (0.1-0.3)  
mid_freq_diff = create_high_frequency_difference_map(src, rec, 0.2)

# 高频差异图 (0.3-1.0)
high_freq_diff = create_high_frequency_difference_map(src, rec, 0.4)
```

### 自适应阈值
可以根据图像内容自动调整频率阈值：

```python
# 基于图像梯度自动调整阈值
gradient_magnitude = compute_image_gradient(src_inputs)
adaptive_threshold = gradient_magnitude.mean() * 0.5
```

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch size或使用CPU
2. **FFT计算错误**: 检查图像尺寸是否为2的幂次
3. **差异图过暗**: 降低频率阈值
4. **差异图过亮**: 提高频率阈值

### 性能优化
1. **使用torch.fft**: 确保PyTorch版本支持FFT
2. **批处理**: 支持批量图像处理
3. **GPU加速**: 利用GPU加速FFT计算

## 参考文献

- Fast Fourier Transform (FFT) algorithms
- Butterworth filter design
- Frequency domain image processing
- High-frequency component analysis
