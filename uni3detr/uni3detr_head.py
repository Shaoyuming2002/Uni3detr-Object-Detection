#--------
import copy  # 用于对象复制（拷贝），it is divided into two parts: shallow copy and deep copy
# functools提供了一些高阶函数和函数式编程的工具
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmengine.model.weight_init import bias_init_with_prob
# 这个老版本的
# from mmcv.runner import force_fp32, auto_fp16
from mmengine.runner import amp
                        
# from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean

# from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.layers.transformer import inverse_sigmoid
# from mmdet.models import HEADS
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet.registry import MODELS as DETR_MODELS
from mmdet.models.dense_heads import DETRHead

# from mmdet3d.core.bbox.coders import build_bbox_coder
# from mmdet3d.models.builder import build_loss

# from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from projects.PETR.petr.utils import normalize_bbox, denormalize_bbox
from projects.PETR.petr.hungarian_assigner_3d import HungarianAssigner3D

# from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import bbox_overlaps_3d, bbox_overlaps_nearest_3d
from .match_cost import bbox_overlaps_3d, bbox_overlaps_nearest_3d
# from projects.mmdet3d_plugin.core.bbox.util import get_rdiou
# from mmdet3d.core.bbox import AxisAlignedBboxOverlaps3D
# from mmcv.ops import nms3d, nms_bev
from mmcv.ops import nms3d

from mmdet3d.models.task_modules.samplers import PseudoSampler
from mmengine.structures import InstanceData


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class BatchNormDim1Swap(nn.BatchNorm1d):
    """
    继承自nn.BatchNorm1d的自定义批归一化层
    用于处理Transformer中HW x N x C格式的特征图
    HW: 特征图的高度*宽度
    N: batch size 
    C: 通道数
    """

    def forward(self, x):
        """
        输入x: 形状为(HW, N, C)的张量
        输出x: 形状为(HW, N, C)的张量,在通道维度上进行了批归一化
        
        处理步骤:
        1. 将输入张量从(HW, N, C)转置为(N, C, HW)以适配BatchNorm1d
        2. 在通道维度上进行批归一化
        3. 将结果转置回(HW, N, C)格式
        """
        # 获取输入张量的维度信息
        hw, n, c = x.shape  # hw:特征图尺寸, n:批大小, c:通道数
        
        # 第一次维度转置:(HW, N, C) -> (N, C, HW)
        x = x.permute(1, 2, 0)  
        
        # 调用父类BatchNorm1d的forward方法进行批归一化
        x = super(BatchNormDim1Swap, self).forward(x)
        
        # 第二次维度转置:(N, C, HW) -> (HW, N, C)
        x = x.permute(2, 0, 1)
        
        return x


NORM_DICT = {
    "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}


class GenericMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name=None,
        activation="relu",
        use_conv=False,
        dropout=None,
        hidden_use_bias=False,
        output_use_bias=True,
        output_use_activation=False,
        output_use_norm=False,
        weight_init_name=None,
    ):
        super().__init__()
        activation = ACTIVATION_DICT[activation]
        norm = None
        if norm_fn_name is not None:
            norm = NORM_DICT[norm_fn_name]
        if norm_fn_name == "ln" and use_conv:
            norm = lambda x: nn.GroupNorm(1, x)  # easier way to use LayerNorm

        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]

        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            if use_conv:
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
            layers.append(layer)
            if norm:
                layers.append(norm(x))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = x
        if use_conv:
            layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
        else:
            layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
        layers.append(layer)

        if output_use_norm:
            layers.append(norm(output_dim))

        if output_use_activation:
            layers.append(activation())

        self.layers = nn.Sequential(*layers)

        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for (_, param) in self.named_parameters():
            if param.dim() > 1:  # skips batchnorm/layernorm
                func(param)

    def forward(self, x):
        output = self.layers(x)
        return output
# input (B, N, 3).  output (B, N, 3)
# ----------------------------------------
# Simple Point manipulations
# ----------------------------------------
# 用于将点云数据从源范围(src_range)缩放到目标范围(dst_range) 
# 点云坐标归一化
def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
            ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                return self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                return self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += (
                f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
            )
        return st



@MODELS.register_module()
class Uni3DETRHead(DETRHead):
    """Head of UVTR. 
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    # def __init__(self,
    #              in_channels,    
    #              num_query=100,
    #              *args,
    #              with_box_refine=False,
    #              as_two_stage=False,
    #              transformer=None,
    #              positional_encoding=dict(
    #                  type='SinePositionalEncoding',
    #                  num_feats=128,
    #                  normalize=True),
    #              bbox_coder=None,
    #              num_cls_fcs=2,
    #              code_weights=None,
    #              loss_bbox=dict(type='RotatedIoU3DLoss', loss_weight=1.0),
    #              loss_iou=dict(type='RotatedIoU3DLoss', loss_weight=1.0),
    #              post_processing=None,
    #              gt_repeattimes=1,
    #              **kwargs):

    #     self.num_query = num_query
    #     self.embed_dims = 256
    #     self.in_channels = in_channels
    #     assert 'num_feats' in positional_encoding
    #     num_feats = positional_encoding['num_feats']
    #     assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
    #         f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
    #         f' and {num_feats}.'
    #     self.with_box_refine = with_box_refine
    #     self.as_two_stage = as_two_stage
    #     if self.as_two_stage:
    #         transformer['as_two_stage'] = self.as_two_stage
    #     if 'code_size' in kwargs:
    #         self.code_size = kwargs['code_size']
    #     else:
    #         self.code_size = 10
    #     if code_weights is not None:
    #         self.code_weights = code_weights
    #     else:
    #         self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
    #     # if 'train_cfg' in kwargs:
    #     #     self.train_cfg = kwargs['train_cfg']
    #     # else:
    #     #     self.train_cfg = None
        
    #     self.bbox_coder = TASK_UTILS.build(bbox_coder)
    #     self.pc_range = self.bbox_coder.pc_range
    #     self.num_cls_fcs = num_cls_fcs - 1

    #     super(Uni3DETRHead, self).__init__(
    #         *args, loss_bbox=loss_bbox, loss_iou=loss_iou, **kwargs)
        
    #     self.positional_encoding = TASK_UTILS.build(positional_encoding)
    #     self.transformer = MODELS.build(transformer) if transformer else None
    #     # self.loss_bbox = build_loss(loss_bbox)
    #     # self.loss_iou = build_loss(loss_iou)
    #     self.loss_bbox = MODELS.build(loss_bbox)
    #     self.loss_iou = MODELS.build(loss_iou)

    #     self.code_weights = nn.Parameter(torch.tensor(
    #         self.code_weights, requires_grad=False), requires_grad=False)
        
    #     self.fp16_enabled = False
    #     self.post_processing = post_processing
    #     self.gt_repeattimes = gt_repeattimes
    
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 sync_cls_avg_factor=True,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 bbox_coder=None,
                 loss_cls=dict(type='SoftFocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=0.25),
                 loss_iou=dict(type='IoU3DLoss', loss_weight=1.2),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D1',
                         cls_cost=dict(type='FocalLossCost', weight=2.0),
                         reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                         iou_cost=dict(type='IoU3DLoss', weight=1.2))),
                 test_cfg=dict(max_per_img=100),
                # train_cfg = dict(
                # assigner=dict(
                #     type='HungarianAssigner3D1',
                #     match_costs=[
                #         dict(type='ClassificationCost', weight=1.),
                #         dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                #         dict(type='IoUCost', iou_mode='giou', weight=2.0)
                #     ])),
                 num_cls_fcs=2,
                 code_weights=None,
                 post_processing=None,
                 gt_repeattimes=1,
                 init_cfg=None,
                 **kwargs):
        
        # if train_cfg is not None:
        #     pts_train_cfg = train_cfg.get('pts', None)
        #     if pts_train_cfg is not None:
        #         assert 'assigner' in pts_train_cfg, 'assigner should be provided when train_cfg is set.'
        #         assigner_cfg = pts_train_cfg['assigner']
        #         # 注意：你 config 中 loss_cls 的 loss_weight 为 1.5，
        #         # 但 assigner_cfg['cls_cost']['weight'] 为 2.0，
        #         # 如果两者不一致，可以打印警告而不是强制断言。
        #         if loss_cls['loss_weight'] != assigner_cfg['cls_cost']['weight']:
        #             print("Warning: loss_cls weight (%s) is not equal to assigner cls_cost weight (%s)." %
        #                 (loss_cls['loss_weight'], assigner_cfg['cls_cost']['weight']))
        #         # 对于回归损失，两者应一致
        #         assert loss_bbox['loss_weight'] == assigner_cfg['reg_cost']['weight'], \
        #             'The regression L1 weight for loss and matcher should be exactly the same.'
        #         self.assigner = TASK_UTILS.build(assigner_cfg)
        #         # 使用 PseudoSampler 构造采样器
        #         sampler_cfg = dict(type='PseudoSampler')
        #         self.sampler = TASK_UTILS.build(sampler_cfg)

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            # assert loss_iou['loss_weight'] == assigner['iou_cost'][
            #   'weight'], \
            # 'The regression iou weight for loss and matcher should be' \
            # 'exactly the same.'
            self.assigner = TASK_UTILS.build(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = TASK_UTILS.build(sampler_cfg)


        # 保存本类参数
        self.num_query = num_query
        self.embed_dims = 256
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage and transformer is not None:
            transformer['as_two_stage'] = self.as_two_stage
        self.sampler = PseudoSampler()

        # code_size 固定为 10（或可根据需求修改）
        self.code_size = 10  
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            # 默认权重，与 config 中一致
            self.code_weights = [1.0] * 8 + [1.0, 1.0]
        # 记录分类分支层数（减 1，因为后面会构造 FC 层）
        self.num_cls_fcs = num_cls_fcs - 1

        # 使用 TASK_UTILS 构建 bbox_coder，并保存 pc_range
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_pred = 6

        # 调用父类 DETRHead 的初始化（参考 PETRHead 的写法）
        super(Uni3DETRHead, self).__init__(
            num_classes=num_classes,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_iou=loss_iou,
            init_cfg=init_cfg
        )

        # 构建分类损失模块
        self.loss_cls = TASK_UTILS.build(loss_cls)
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        # 构建其他模块：位置编码、Transformer 模块、损失模块
        self.positional_encoding = TASK_UTILS.build(positional_encoding)
        self.transformer = MODELS.build(transformer) 
        self.loss_bbox = TASK_UTILS.build(loss_bbox)
        self.loss_iou = TASK_UTILS.build(loss_iou)
        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, dtype=torch.float32, requires_grad=False),
                                         requires_grad=False)
        self.fp16_enabled = False
        self.post_processing = post_processing
        self.gt_repeattimes = gt_repeattimes

        # 初始化头部层（分类、回归、IoU 分支）
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        iou_branch = []
        for _ in range(self.num_reg_fcs):
            iou_branch.append(Linear(self.embed_dims, self.embed_dims))
            iou_branch.append(nn.ReLU())
        iou_branch.append(Linear(self.embed_dims, 1))
        iou_branch = nn.Sequential(*iou_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        # num_pred = (self.transformer.decoder.num_layers + 1) if \
        #     self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, self.num_pred)
            self.reg_branches = _get_clones(reg_branch, self.num_pred)
            self.iou_branches = _get_clones(iou_branch, self.num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred)])
            self.iou_branches = nn.ModuleList(
                [iou_branch for _ in range(self.num_pred)])

        if not self.as_two_stage:
            self.tgt_embed = nn.Embedding(self.num_query * 2, self.embed_dims)
            self.refpoint_embed = nn.Embedding(self.num_query, 3)
        self.input_proj = nn.Conv2d(self.in_channels, self.embed_dims, 1)
            

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    # @auto_fp16(apply_to=("pts_feats",))
    
    def forward(self, pts_feats, img_metas, fpsbpts):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        tgt_embed = self.tgt_embed.weight           # nq, 256
        refanchor = self.refpoint_embed.weight      # nq, 6
        #query_embeds = torch.cat((tgt_embed, refanchor), dim=1)

        bs = fpsbpts.shape[0]
        # with amp.autocast(enabled=True):
        #     pts_feats = pts_feats.half()
        
        # Ensure pts_feats is a Tensor, not a list
        if isinstance(pts_feats, (list, tuple)):
            try:
                pts_feats = torch.cat(pts_feats, dim=0)
            except Exception:
                pts_feats = pts_feats[0]

        # 保证通道对齐，(B,256,H,W)        
        pts_feats = self.input_proj(pts_feats)
        B, C, H, W = pts_feats.shape
        # pts_value = pts_feats.flatten(2).permute(0, 2, 1)   # (B, H*W, 256)

        if pts_feats.requires_grad:
            tgt_embed = torch.cat([tgt_embed[0:self.num_query], tgt_embed[self.num_query:], tgt_embed[self.num_query:]])
            query_embeds = torch.cat([tgt_embed.unsqueeze(0).expand(bs, -1, -1), torch.cat([refanchor.unsqueeze(0).expand(bs, -1, -1), inverse_sigmoid(fpsbpts)], 1)], -1)
        else:
            random_point = torch.rand(fpsbpts.shape, device=fpsbpts.device)[:, :self.num_query, :]
            tgt_embed = torch.cat([tgt_embed[0:self.num_query], tgt_embed[self.num_query:], tgt_embed[self.num_query:], tgt_embed[self.num_query:]])
            query_embeds = torch.cat([tgt_embed.unsqueeze(0).expand(bs, -1, -1), torch.cat([refanchor.unsqueeze(0).expand(bs, -1, -1), inverse_sigmoid(fpsbpts), inverse_sigmoid(random_point)], 1)], -1)
        # random_point = torch.rand(fpsbpts.shape, device=fpsbpts.device)[:, :self.num_query, :]
        # tgt_embed = torch.cat([tgt_embed[0:self.num_query], tgt_embed[self.num_query:], tgt_embed[self.num_query:], tgt_embed[self.num_query:]])
        # query_embeds = torch.cat([tgt_embed.unsqueeze(0).expand(bs, -1, -1), torch.cat([refanchor.unsqueeze(0).expand(bs, -1, -1), inverse_sigmoid(fpsbpts), inverse_sigmoid(random_point)], 1)], -1)


        # shape: (N, L, C, D, H, W)
        if len(pts_feats.shape) == 5:
            pts_feats = pts_feats.unsqueeze(1)
        print("pts_feats", pts_feats)


        # hs, init_reference, inter_references = self.transformer(
        #     pts_feats,
        #     query_embeds,
        #     self.num_query,
        #     reg_branches=self.reg_branches if self.with_box_refine else None,
        #     img_metas=img_metas,
        # )
      # -------- 修正调用 -----------------------------
        # Uni3DETRHead.__init__
        
        hs, init_reference, inter_references = self.transformer(
            pts_feats,                # ① pts_value
            query_embeds,             # ② query_embed
            self.num_query,           # ③ num_query
            reg_branches=self.reg_branches if self.with_box_refine else None,
            img_metas=img_metas,
        )
        # ----------------------------------------------



        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_ious = []

        #for lvl in range(hs.shape[0]):
        for lvl in range(len(hs)):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            outputs_iou = self.iou_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3 
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            # transfer to lidar system
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_ious.append(outputs_iou)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_ious = torch.stack(outputs_ious)

        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_iou_preds': outputs_ious
        }

        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)

        # assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
        #                                         gt_labels, self.num_query, gt_bboxes_ignore, self.gt_repeattimes)
        assign_result = self.assigner.assign(
                                    bbox_pred,
                                    cls_score,
                                    gt_bboxes,
                                    gt_labels,
                                    gt_bboxes_ignore=gt_bboxes_ignore  # 可选项
                                )


        # sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        sampling_result = self.sampler.sample(
                        assign_result,
                        pred_instances=InstanceData(priors=bbox_pred),
                        gt_instances=InstanceData(bboxes_3d=gt_bboxes, labels_3d=gt_labels)
                    )

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        # bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_targets = torch.zeros_like(bbox_pred)[..., :7]  #######
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    iou_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        #loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        bboxes3d = denormalize_bbox(bbox_preds, self.pc_range) 
        bboxes3d = bboxes3d[:, :7]
        bbox_weights = bbox_weights[:, :7]

        print("bbox_targets.shape:", bbox_targets.shape)
        print("bbox3d.shape:", bboxes3d.shape)

        iou3d = bbox_overlaps_nearest_3d(bboxes3d, bbox_targets, is_aligned=True, coordinate='depth')
        z1, z2, z3, z4 = self._bbox_to_loss(bboxes3d)[:, 2], self._bbox_to_loss(bboxes3d)[:, 5], self._bbox_to_loss(bbox_targets)[:, 2], self._bbox_to_loss(bbox_targets)[:, 5]
        iou_z = torch.max(torch.min(z2, z4) - torch.max(z1, z3), z1.new_zeros(z1.shape)) / (torch.max(z2, z4) - torch.min(z1, z3) )
        iou3d_dec = (iou3d + iou_z)/2

        loss_cls = self.loss_cls(cls_scores, [labels, iou3d_dec], label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        
        loss_bbox = self.loss_bbox(bbox_preds[isnotnan, :7], normalized_bbox_targets[isnotnan, :7], bbox_weights[isnotnan, :7], avg_factor=num_total_pos)

        loss_iou_z = 1 - iou_z[isnotnan]
        loss_iou = self.loss_iou(bboxes3d[isnotnan, :7], bbox_targets[isnotnan, :7], bbox_weights[isnotnan, :7], avg_factor=num_total_pos)
        loss_iou += torch.sum(loss_iou_z * bbox_weights[isnotnan, 0]) / num_total_pos


        iou_preds = iou_preds.reshape(-1)
        iou3d_true = torch.diag(bbox_overlaps_3d(bboxes3d, bbox_targets, coordinate='lidar')).detach()
        # loss_iou_pred = torch.sum( F.binary_cross_entropy_with_logits(iou_preds, iou3d_true, reduction='none') * bbox_weights[isnotnan, 0] ) / num_total_pos * 1.2 
        loss_iou_pred = torch.sum(
                    F.binary_cross_entropy_with_logits(iou_preds[isnotnan], iou3d_true[isnotnan], reduction='none') * bbox_weights[isnotnan, 0]
                ) / num_total_pos * 1.2


        return loss_cls, loss_bbox, loss_iou, loss_iou_pred
    
    @staticmethod
    def _bbox_to_loss(bbox):
        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)
    
    @staticmethod
    def _loss_to_bbox(bbox):
        return torch.stack(
        ( (bbox[..., 0] + bbox[..., 3]) / 2, (bbox[..., 1] + bbox[..., 4]) / 2, (bbox[..., 2] + bbox[..., 5]) / 2,
            bbox[..., 3] - bbox[..., 0], bbox[..., 4] - bbox[..., 1], bbox[..., 5] - bbox[..., 2], bbox[..., -1] ),
            dim=-1)
    
    # @force_fp32(apply_to=('preds_dicts'))
    # @amp.autocast(enabled=False)
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_iou_preds = preds_dicts['all_iou_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.tensor[:, :3], gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # calculate class and box loss
        losses_cls, losses_bbox, losses_iou, losses_iou_pred = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_iou_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_iou_pred'] = losses_iou_pred[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_iou_pred_i in zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1], losses_iou_pred[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_iou_pred'] = loss_iou_pred_i
            num_dec_layer += 1
            

        return loss_dict
    

    def soft_nms(self, boxes, scores, gaussian_sigma=0.3, prune_threshold=1e-3):
        boxes = boxes.clone()
        scores = scores.clone()
        idxs = torch.arange(scores.size()[0]).to(boxes.device)

        idxs_out = []
        scores_out = []

        while scores.numel() > 0:
            top_idx = torch.argmax(scores)
            idxs_out.append(idxs[top_idx].item())
            scores_out.append(scores[top_idx].item())

            top_box = boxes[top_idx]
            ious = bbox_overlaps_3d(top_box.unsqueeze(0), boxes, coordinate='lidar')[0]

            decay = torch.exp(-torch.pow(ious, 2) / gaussian_sigma)

            scores *= decay
            keep = scores > prune_threshold
            keep[top_idx] = False

            # print(keep.device, boxes.device)
            boxes = boxes[keep]
            scores = scores[keep]
            idxs = idxs[keep]

        return torch.tensor(idxs_out).to(boxes.device), torch.tensor(scores_out).to(scores.device)



    # @force_fp32(apply_to=('preds_dicts'))
    # @amp.autocast(enabled=True)
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.shape[-1])
            scores = preds['scores']
            labels = preds['labels']
            ious = preds['ious']
            if self.post_processing is not None:
                if self.post_processing['type'] == 'nms' or self.post_processing['type'] == 'soft_nms':
                    nc = self.num_classes
                    nmsbboxes = []
                    nmslabels = []
                    nmsscores = []
                    nmsious = []
                    for j in range(nc):
                        ind = (labels == j)
                        bboxest, labelst, scorest, iousest = bboxes.tensor[ind], labels[ind], scores[ind], ious[ind]
                        if ind.sum() == 0:
                            continue
                        
                        if self.post_processing['type'] == 'nms':
                            nmsind = nms3d(bboxest[:, :7], scorest, self.post_processing['nms_thr'])
                            nmsbboxes.append(bboxest[nmsind])
                            nmsscores.append(scorest[nmsind])
                            nmsious.append(iousest[nmsind])
                            nmslabels.extend([j] * nmsind.shape[0])
                        else:
                            nmsind, scores_soft = self.soft_nms(bboxest[:, :7], scorest, self.post_processing['gaussian_sigma'], self.post_processing['prune_threshold'])
                            nmsbboxes.append(bboxest[nmsind])
                            nmsscores.append(scores_soft)
                            nmsious.append(iousest[nmsind])
                            nmslabels.extend([j] * nmsind.shape[0])
                    if len(nmsbboxes) == 0:
                        nmsbboxes.append(bboxes.tensor.new_zeros((0, bboxes.tensor.shape[-1])))
                        nmsscores.append(bboxes.tensor.new_zeros((0)))
                        nmsious.append(bboxes.tensor.new_zeros((0)))
                    nmsbboxes = torch.cat(nmsbboxes)
                    bboxes = img_metas[i]['box_type_3d'](nmsbboxes, bboxes.tensor.shape[-1])
                    scores = torch.cat(nmsscores)
                    labels = torch.tensor(nmslabels)
                    ious = torch.cat(nmsious)
                elif self.post_processing['type'] == 'box_merging':
                    from . import bbox_merging as bbox_merging
                    class_labels, detection_boxes_3d, detection_scores, nms_indices = bbox_merging.nms_boxes_3d_merge_only(
                        labels.cpu().numpy(), bboxes.tensor.cpu().numpy(), scores.cpu().numpy(),
                        overlapped_fn=bbox_merging.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=0.1, 
                        appr_factor=1e6, top_k=-1,
                        attributes=np.arange(len(labels)))
                    bboxes = img_metas[i]['box_type_3d'](torch.tensor(detection_boxes_3d), bboxes.tensor.shape[-1])
                    scores = torch.tensor(detection_scores)
                    labels = torch.tensor(class_labels)
                    ious = torch.tensor(nms_indices)
                else:
                    raise(self.post_processing['type'] +' not implemented.')

                if 'score_thr' in self.post_processing:
                    if type(self.post_processing['score_thr']) is list:
                        assert len(self.post_processing['score_thr']) == self.num_classes
                        ind = (scores < -1)
                        for j in range(self.num_classes):
                            ind = torch.logical_or(ind, torch.logical_and(labels==j, scores > self.post_processing['score_thr'][j]))
                        bboxes = img_metas[i]['box_type_3d'](bboxes.tensor[ind], bboxes.tensor.shape[-1])
                        scores = torch.tensor(scores[ind])
                        labels = torch.tensor(labels[ind])
                    else:
                        ind = (scores > self.post_processing['score_thr'])
                        bboxes = img_metas[i]['box_type_3d'](bboxes.tensor[ind], bboxes.tensor.shape[-1])
                        scores = torch.tensor(scores[ind])
                        labels = torch.tensor(labels[ind])

                if 'num_thr' in self.post_processing:
                    ind = torch.argsort(-scores)[0:self.post_processing['num_thr']]
                    bboxes = img_metas[i]['box_type_3d'](bboxes.tensor[ind], bboxes.tensor.shape[-1])
                    scores = torch.tensor(scores[ind])
                    labels = torch.tensor(labels[ind])

            ret_list.append([bboxes, scores, labels])
        return ret_list
