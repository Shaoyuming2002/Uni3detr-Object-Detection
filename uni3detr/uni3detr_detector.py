# projects/Uni3DETR/uni3detr/uni3detr_detector.py
# Copyright (c) OpenMMLab. All rights reserved.
"""PointPillars + Uni3DETR Detector.

* 复用 VoxelNet 特征提取（Pillar → Scatter → SECOND → FPN）
* 输出单层 BEV feature `[B, C, H, W]`
* 将原始点云整理成 `[B, num_query, 3]` 供 UniCrossAtten 使用
"""
from __future__ import annotations

import torch
from typing import Dict, List, Optional
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors import VoxelNet


@MODELS.register_module()
class Uni3DETR3DDetector(VoxelNet):
    """Detector = PointPillars backbone + Uni3DETR transformer head."""

    # ------------------------------------------------------------------
    # constructor —— 与 VoxelNet 参数保持一致，仅显式转发
    # ------------------------------------------------------------------
    def __init__(
        self,
        voxel_encoder: Dict,
        middle_encoder: Dict,
        backbone: Dict,
        neck: Optional[Dict] = None,
        bbox_head: Optional[Dict] = None,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        data_preprocessor: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None
    ) -> None:
        super().__init__(
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )

    # ---------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------
    def forward(
        self,
        inputs: Dict[str, Tensor],   # Pack3DDetInputs 输出的 batch dict
        data_samples: Optional[List] = None,
        mode: str = 'loss'
    ):
        """整体前向流程。"""
        # 1️⃣ 提取 BEV 特征
        x = self.extract_feat(inputs)
        bev_feat = x[0] if isinstance(x, (list, tuple)) else x  # [B, 384, H, W]

        # 2️⃣ 生成参考点 (B, num_query, 3)
        fpsbpts = self._build_query_points(inputs, bev_feat.size(0))

        # 3️⃣ Transformer head 前向
        preds_dicts = self.bbox_head(bev_feat, data_samples, fpsbpts)
        print(f"inputs keys: {list(inputs.keys())}")


        # gt_bboxes_list = inputs['gt_bboxes_3d']
        # gt_labels_list = inputs['gt_labels_3d']
        # print(f'gt_bboxes_list: {gt_bboxes_list}')
        # print(f'gt_labels_list: {gt_labels_list}')

        # 4️⃣ 分支：loss or predict
       # projects/Uni3DETR/uni3detr/uni3detr_detector.py
        # ...
        # 在 forward 开头临时插入
        print(data_samples[0])
        print(dir(data_samples[0].gt_instances_3d))
        # 如果能看出哪些属性存在，比如 bboxes_3d、bboxes、labels_3d、labels，就按实际的名字来取

        if mode == 'loss':
            # —— 正确地从 data_samples 拿 GT —— #
            device = bev_feat.device

            # 对每个样本 ds，取出 ds.gt_instances_3d.bboxes_3d.tensor
            gt_bboxes_list = [
                ds.gt_instances_3d.bboxes_3d.to(device)
                for ds in data_samples
            ]
            # labels_3d 本身就是 Tensor
            gt_labels_list = [
                ds.gt_instances_3d.labels_3d.to(device)
                for ds in data_samples
            ]

            return self.bbox_head.loss(
                gt_bboxes_list,
                gt_labels_list,
                preds_dicts
            )
        else:
            return self.bbox_head.predict(bev_feat, data_samples, preds_dicts)

    # ---------------------------------------------------------------
    # helper —— 把 list[points] → Tensor[B, num_query, 3]
    # ---------------------------------------------------------------
    def _build_query_points(self, inputs: Dict[str, Tensor], bs: int) -> Optional[Tensor]:
        """随机采样每个 batch `num_query` 个点的 xyz，返回 `[B, num_query, 3]`."""
        pts_list = inputs.get('points', None)
        if pts_list is None:
            return None
        if not isinstance(pts_list, (list, tuple)):
            pts_list = [pts_list]

        num_query = self.bbox_head.num_query
        out: List[Tensor] = []
        for item in pts_list:
            # 支持 Tensor 或任何带 `.tensor` 属性的 Points 对象
            if isinstance(item, Tensor):
                pts = item
            elif hasattr(item, 'tensor'):
                pts = item.tensor  # type: ignore
            else:
                raise TypeError(f'Unsupported points type: {type(item)}')

            # 仅取 xyz（三列）
            xyz = pts[:, :3]
            n = xyz.size(0)
            if n >= num_query:
                choice = torch.randperm(n, device=xyz.device)[:num_query]
            else:
                extra = torch.randint(0, n, (num_query - n,), device=xyz.device)
                choice = torch.cat([torch.arange(n, device=xyz.device), extra])
            out.append(xyz[choice])  # (num_query, 3)

        # ↩ 复制一份 → (B, 2*num_query, 3)，与 head 中 2700 行对齐
        ref_xyz = torch.stack(out, 0)              # (B, num_query, 3)
        return torch.cat([ref_xyz, ref_xyz], dim=1)  # (B, 2*num_query, 3)
