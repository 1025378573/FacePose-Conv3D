from pyskl.smp import *
from pyskl.models import build_model
import torch
from mmcv import load, dump
import copy as cp
from collections import OrderedDict

model_cfg = dict(
    type='MMRecognizer3D',
    backbone=dict(
        type='RGBPoseConv3D',
        speed_ratio=4, 
        channel_ratio=4, 
        pose_pathway=dict(
            num_stages=3, 
            stage_blocks=(4, 6, 3),
            lateral=True,
            lateral_inv=True,
            lateral_infl=16,
            lateral_activate=(0, 1, 1),
            in_channels=17,
            base_channels=32,
            out_indices=(2, ),
            conv1_kernel=(1, 7, 7),
            conv1_stride=(1, 1),
            pool1_stride=(1, 1),
            inflate=(0, 1, 1),
            spatial_strides=(2, 2, 2),
            temporal_strides=(1, 1, 1))),
    cls_head=dict(
        type='RGBPoseHead',
        in_channels=(2048, 512),
        num_classes=21,
        dropout=0.5),
    test_cfg = dict(average_clips='prob'))
model = build_model(model_cfg)

rgb_ckpt = torch.load('/home/chenyanting/pyskl/work_dirs/rgbpose_conv3d/face_only/best_top1_acc_epoch_24.pth', map_location='cpu')['state_dict']
pose_ckpt = torch.load('/home/chenyanting/pyskl/work_dirs/rgbpose_conv3d/new_pose_only/best_top1_acc_epoch_2.pth', map_location='cpu')['state_dict']

rgb_ckpt = {k.replace('backbone', 'backbone.rgb_path').replace('fc_cls', 'fc_rgb'): v for k, v in rgb_ckpt.items()}
pose_ckpt = {k.replace('backbone', 'backbone.pose_path').replace('fc_cls', 'fc_pose'): v for k, v in pose_ckpt.items()}

old_ckpt = {}
old_ckpt.update(rgb_ckpt)
old_ckpt.update(pose_ckpt)

# The difference is in dim-1
def padding(weight, new_shape):
    new_weight = weight.new_zeros(new_shape)
    new_weight[:, :weight.shape[1]] = weight
    return new_weight

ckpt = cp.deepcopy(old_ckpt)
name = 'backbone.rgb_path.layer3.0.conv1.conv.weight'
ckpt[name] = padding(ckpt[name], (256, 640, 3, 1, 1))
name = 'backbone.rgb_path.layer3.0.downsample.conv.weight'
ckpt[name] = padding(ckpt[name], (1024, 640, 1, 1, 1))
name = 'backbone.rgb_path.layer4.0.conv1.conv.weight'
ckpt[name] = padding(ckpt[name], (512, 1280, 3, 1, 1))
name = 'backbone.rgb_path.layer4.0.downsample.conv.weight'
ckpt[name] = padding(ckpt[name], (2048, 1280, 1, 1, 1))
name = 'backbone.pose_path.layer2.0.conv1.conv.weight'
ckpt[name] = padding(ckpt[name], (64, 160, 3, 1, 1))
name = 'backbone.pose_path.layer2.0.downsample.conv.weight'
ckpt[name] = padding(ckpt[name], (256, 160, 1, 1, 1))
name = 'backbone.pose_path.layer3.0.conv1.conv.weight'
ckpt[name] = padding(ckpt[name], (128, 320, 3, 1, 1))
name = 'backbone.pose_path.layer3.0.downsample.conv.weight'
ckpt[name] = padding(ckpt[name], (512, 320, 1, 1, 1))
ckpt = OrderedDict(ckpt)
torch.save({'state_dict': ckpt}, './weights/facepose_conv3d_init.pth')

model.load_state_dict(ckpt, strict=False)