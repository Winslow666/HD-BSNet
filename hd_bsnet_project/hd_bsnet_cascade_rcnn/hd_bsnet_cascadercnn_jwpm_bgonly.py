# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .hd_bsnet_cascadercnn import HD_BSNet_CascadeRCNN
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Union

from mmdet.structures import SampleList

import copy
import warnings

import torch.nn.functional as F

# 导入高频增强相关模块
# 为了避免重复注册 HD_BSNet_TwoStageDetector，直接在这里定义需要的类

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))


class Multi_scale_conv(nn.Module):
    def __init__(self, dim, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.conv2 = Conv(dim, dim, k=1, s=1)
        self.conv3 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.conv4 = Conv(dim, dim, 1, 1)
        self.conv5 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = x5 + x
        return x6


class HF_Extractor(nn.Module):
    """
    高频信息提取器：采用"原图下采样"方案
    """
    def __init__(self, mode='bilinear'):
        super(HF_Extractor, self).__init__()
        self.mode = mode
        # 预定义卷积层，避免每次前向传播都创建新的
        self.img_proj = nn.Conv2d(3, 256, kernel_size=1)  # 假设P2通道数为256
        
    def forward(self, original_image, p2_feat_map):
        """
        参数:
            original_image: 原始输入图像 [B, 3, H_orig, W_orig]
            p2_feat_map: P2层特征图 [B, C, H_p2, W_p2]
        返回:
            hf_map: 高频信息特征图 [B, C, H_p2, W_p2]
        """
        # 1. 获取P2特征图的尺寸
        B, C, H_p2, W_p2 = p2_feat_map.shape
        
        # 2. 将原图下采样到P2特征图的尺寸
        if original_image.size(1) != C:
            # 如果原图通道(3)与P2特征通道(C)不同，先进行投影
            projected_img = F.interpolate(original_image, size=(H_p2, W_p2), mode=self.mode)
            # 使用预定义的1x1卷积将3通道投影到C通道
            projected_img = self.img_proj(projected_img)
        else:
            # 如果通道数相同，直接下采样
            projected_img = F.interpolate(original_image, size=(H_p2, W_p2), mode=self.mode)
        
        # 3. 计算差分得到高频信息
        hf_map = projected_img - p2_feat_map
        
        return hf_map


class Multi_scale_HF(nn.Module):
    """
    改进的Multi-scale HF模块：融合P2特征和高频信息特征
    使用Multi_scale_conv替代MultiKernelPerception进行多尺度感知
    """
    def __init__(self, p2_channels, hf_channels=None, kernel_sizes=[3, 5, 7]):
        super(Multi_scale_HF, self).__init__()
        
        if hf_channels is None:
            hf_channels = p2_channels
            
        # 1. 特征对齐与预处理
        self.p2_align = nn.Sequential(
            nn.Conv2d(p2_channels, p2_channels, kernel_size=1),
            nn.BatchNorm2d(p2_channels),
            nn.ReLU(inplace=True)
        )
        
        self.hf_align = nn.Sequential(
            nn.Conv2d(hf_channels, p2_channels, kernel_size=1),
            nn.BatchNorm2d(p2_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 双分支多尺度感知 - 使用Multi_scale_conv替代MultiKernelPerception
        self.p2_multiscale = Multi_scale_conv(p2_channels)  # P2特征的多尺度感知
        self.hf_multiscale = Multi_scale_conv(p2_channels)  # 高频特征的多尺度感知
        
        # 3. 高频引导的注意力融合
        self.attention_gate = nn.Sequential(
            nn.Conv2d(p2_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 4. 跨尺度特征整合
        self.fusion = nn.Sequential(
            nn.Conv2d(p2_channels, p2_channels, kernel_size=1),
            nn.BatchNorm2d(p2_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, p2_feat, hf_feat):
        """
        前向传播
        参数:
            p2_feat: P2层特征图 [B, C, H, W]
            hf_feat: 高频信息特征图 [B, C_hf, H, W]
        返回:
            增强后的P2'特征图 [B, C, H, W]
        """
        # 1. 特征对齐
        p2_aligned = self.p2_align(p2_feat)
        hf_aligned = self.hf_align(hf_feat)
        
        # 2. 多尺度感知
        p2_multi = self.p2_multiscale(p2_aligned)  # 提取多尺度语义上下文
        hf_multi = self.hf_multiscale(hf_aligned)  # 提取多尺度细节特征
        
        # 3. 高频引导的注意力融合
        attention_map = self.attention_gate(hf_multi)  # 生成注意力权重
        enhanced_feat = p2_multi * attention_map + p2_multi # 应用注意力
        
        # 4. 特征整合
        output = self.fusion(enhanced_feat)
        
        return output


class Pred_Layer(nn.Module):
    def __init__(self, in_c=256):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x = self.enlayer(x)
        x1 = self.outlayer(x)
        return x, x1


class Dilated_conv(nn.Module):
    def __init__(self, in_c):
        super(Dilated_conv, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c, 256, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c, 256, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c, 256, 3, 1, padding=7, dilation=7),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class DetailAggregateLoss(nn.Module):
    """Implementation of DetailAggregateLoss for boundary supervision"""
    def __init__(self):
        super(DetailAggregateLoss, self).__init__()
        
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Prediction boundary map
            target (Tensor): Ground truth boundary map
        """
        # For simplicity, using MSE loss for the detail supervision
        return F.mse_loss(pred, target)


class segmenthead(nn.Module):
    """
    Segmentation head for semantic segmentation tasks
    
    Args:
        in_chan (int): Number of input channels
        mid_chan (int): Number of middle channels
        n_classes (int): Number of output classes
    """
    def __init__(self, in_chan, mid_chan, n_classes):
        super(segmenthead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, n_classes, H, W)
        """
        return self.conv(x)


class PMFM(nn.Module):
    def __init__(self, num_levels=2, theta=0.5):
        super(PMFM, self).__init__()
        self.num_levels = num_levels  # 金字塔层数
        self.theta = torch.tensor(theta)  # 衰减因子
        # 添加层级权重参数，可学习不同层级的重要性
        self.level_weights = nn.Parameter(torch.ones(num_levels + 1) / (num_levels + 1))
        self.softmax = nn.Softmax(dim=0)

    def patch_in(self, f, kh, kw, sh, sw):
        # 提取并处理特征图的patch
        B, C, H, W = f.shape
        # 计算每个维度的 patch 数量
        num_patches_h = (H - kh) // sh + 1
        num_patches_w = (W - kw) // sw + 1

        # 使用 reshape 和 permute 手动实现滑动窗口提取
        # 步骤1：将特征图调整为 (B, C, num_patches_h, kh, num_patches_w, kw)
        patches = f[:, :, :num_patches_h * sh, :num_patches_w * sw]
        patches = patches.view(B, C, num_patches_h, sh, num_patches_w, sw)

        # 步骤2：调整形状为 (B, C, num_patches_h, num_patches_w, kh, kw)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches = patches.view(B, C, num_patches_h, num_patches_w, kh, kw)

        # 将 patches 变换成适合 instance normalization 的形状
        patches = patches.contiguous().view(B * C * num_patches_h * num_patches_w, 1, kh, kw)

        # 对每个 patch 进行 instance normalization
        norm_patches = F.instance_norm(patches)

        # 重新变换成原来形状 (B, C, num_patches_h, num_patches_w, kh, kw)
        norm_patches = norm_patches.view(B, C, num_patches_h, num_patches_w, kh, kw)

        # 使用 sigmoid 函数，将归一化后的 patch 激活值压缩到 [0, 1] 范围内
        attention_scores = torch.sigmoid(norm_patches)
        H = f.size()[2]
        W = f.size()[3]

        # 调整 attention_scores 的形状以便使用 F.fold
        attention_scores = attention_scores.permute(0, 1, 4, 5, 2, 3).contiguous()
        attention_scores = attention_scores.view(B, C * kh * kw, num_patches_h * num_patches_w)

        # 使用 fold 恢复原始特征图大小
        attention_map = F.fold(attention_scores, output_size=(int(H), int(W)), kernel_size=(kh, kw), stride=(sh, sw))

        return attention_map

    def calculate_attention_map(self, Fa, p):
        B, C, H, W = Fa.shape

        # 计算当前层的 patch 大小
        sp_square_h = int(H // (2 ** p))
        sp_square_w = int(W // (2 ** p))

        # 计算注意力图
        M_square = self.patch_in(Fa, sp_square_h, sp_square_w, sp_square_h, sp_square_w)
        M_total = M_square

        return M_total
    
    def process_level(self, x, p):
        """处理单个层级的特征"""
        if p > 0:
            # 计算p>0层级的注意力图
            w_p = self.calculate_attention_map(x, p)
        else:
            # p=0层级直接对特征图进行实例归一化和Sigmoid
            w_p = torch.sigmoid(F.instance_norm(x))
        
        # 计算增强后的特征
        enhanced_feature = self.theta * x + (1 - self.theta) * w_p * x
        return enhanced_feature

    def forward(self, x):
        # 并行处理所有层级
        enhanced_features = []
        
        # 对每个层级并行处理
        for p in range(self.num_levels, -1, -1):
            enhanced_feature = self.process_level(x, p)
            enhanced_features.append(enhanced_feature)
        
        # 使用可学习权重融合不同层级的结果
        level_weights = self.softmax(self.level_weights)  # 确保权重和为1
        
        # 加权融合各层级结果
        output = torch.zeros_like(x)
        for i, feature in enumerate(enhanced_features):
            output += level_weights[i] * feature
            
        return output


class BSMM(nn.Module):
    """
    背景提取器：直接从P6特征图生成背景语义图
    """
    def __init__(self, smooth_kernel_size=3, smooth_channels=256):
        super(BSMM, self).__init__()
        
        # P6特征处理层
        self.p6_process = nn.Sequential(
            nn.Conv2d(smooth_channels, smooth_channels, kernel_size=smooth_kernel_size, 
                      padding=smooth_kernel_size//2, bias=False),
            nn.BatchNorm2d(smooth_channels),
            nn.ReLU(inplace=True)
        )
        
        # 背景特征生成层
        self.bg_process = nn.Sequential(
            nn.Conv2d(smooth_channels, smooth_channels, kernel_size=1),
            nn.BatchNorm2d(smooth_channels),
            nn.Sigmoid()  # 直接输出0-1之间的背景图
        )
        
    def forward(self, p6_feat):
        """
        直接从P6特征生成背景语义图
        
        Args:
            p6_feat (Tensor): P6特征图 [B, C, H_p6, W_p6]
            
        Returns:
            bg_map (Tensor): 背景语义图 [B, C, H_p6, W_p6]
        """
        # 处理P6特征
        smooth_p6 = self.p6_process(p6_feat)
        
        # 生成背景图
        bg_map = self.bg_process(smooth_p6)
        
        return bg_map


@MODELS.register_module()
class HD_BSNet_CascadeRCNN_JWPM_BGOnly(HD_BSNet_CascadeRCNN):
    """
    直接使用P6生成背景语义图的Cascade R-CNN检测器，适配SR-TOD框架
    先进行高频增强和背景提取，最后用PMFM增强特征
    """
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 loss_res: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 learnable_thresh: float = 0.0117647,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 jwpm_levels: int = 2,
                 jwpm_theta: float = 0.5) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            loss_res=loss_res,
            rpn_head=rpn_head,
            learnable_thresh=learnable_thresh,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        # 初始化损失函数和模块
        self.boundary_loss_func = DetailAggregateLoss()
        self.rgb_global = Pred_Layer(256)
        self.seghead = segmenthead(256, 256, 1)
        
        # 初始化高频信息提取器和增强模块
        self.hf_extractor = HF_Extractor(mode='bilinear')
        self.multi_scale_hf = Multi_scale_HF(p2_channels=256)
        self.bsmm = BSMM(smooth_channels=256)
        
        # 新增: 初始化PMFM模块，作为最终特征增强
        self.pmfm = PMFM(num_levels=jwpm_levels, theta=jwpm_theta)
    
    def extract_feat(self, img: Tensor) -> Tuple[Tensor]:
        """
        提取特征并融合高频信息，直接使用P6生成背景图，然后应用JWAM和PMFM增强特征
        Args:
            img: 输入图像 [B, 3, H, W]
        Returns:
            融合高频信息，背景增强，最后用PMFM增强后的特征列表 (P2-P6)
        """
        # 1. 提取骨干网络特征
        x = self.backbone(img)
        
        # 2. 应用特征金字塔网络
        if self.with_neck:
            x = self.neck(x)
            
        # 3. 转换为列表，便于处理
        x1 = list(x)  # [P2, P3, P4, P5, P6] 对应 x1[0] 到 x1[4]
        
        # 4. 原图与P2差分得到高频特征图
        hf_features = self.hf_extractor(img, x1[0])  # 输入原图和P2特征，输出高频特征
        
        # 5. 使用Multi_scale_HF模块融合P2特征和高频特征
        enhanced_p2 = self.multi_scale_hf(x1[0], hf_features)  # 输入P2和高频特征，输出增强的P2特征
        
        # 6. 直接使用P6特征生成背景图
        bg_map = self.bsmm(x1[4])  # 输入P6特征，输出背景图
        
        # 7. 将背景图上采样至P2的尺寸
        if bg_map.shape[2:] != enhanced_p2.shape[2:]:
            bg_map = F.interpolate(
                bg_map, 
                size=enhanced_p2.shape[2:], 
                mode='bilinear',
                align_corners=True
            )
        
        # 8. 使用背景图抑制前景噪声并进行残差融合
        foreground_weights = 1 - bg_map
        enhanced_p2 = enhanced_p2 * foreground_weights + x1[0]
        
        # 9. 最后应用PMFM模块进行最终增强
        x1[0] = self.pmfm(enhanced_p2)
        
        return tuple(x1)
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             img_inputs: Tensor) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            img_inputs (Tensor): Original input images without preprocessing.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)
        
        losses = dict()
        
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                            self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses
    
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                img_inputs: Tensor,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            img_inputs (Tensor): Original input images without preprocessing.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(batch_inputs)

        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


