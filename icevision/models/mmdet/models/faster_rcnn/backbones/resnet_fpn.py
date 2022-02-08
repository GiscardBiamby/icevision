__all__ = [
    "resnet50_caffe_dc5_1x",
    "resnet50_caffe_fpn_1x",
    "resnet50_fpn_1x",
    "resnet50_fpn_2x",
    "resnet101_caffe_fpn_1x",
    "resnet101_fpn_1x",
    "resnet101_fpn_2x",
    "resnext101_32x4d_fpn_1x",
    "resnext101_32x4d_fpn_2x",
    "resnext101_64x4d_fpn_1x",
    "resnext101_64x4d_fpn_2x",
    "resnest50_fpn",
    "resnest101_fpn",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetFasterRCNNBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="faster_rcnn", **kwargs)


base_config_path = mmdet_configs_path / "faster_rcnn"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn"

resnet50_caffe_dc5_1x = MMDetFasterRCNNBackboneConfig(
    config_path=base_config_path / "faster_rcnn_r50_caffe_dc5_1x_coco.py",
    weights_url=f"{base_weights_url}/faster_rcnn_r50_caffe_dc5_1x_coco/faster_rcnn_r50_caffe_dc5_1x_coco_20201030_151909-531f0f43.pth",
)

resnet50_caffe_fpn_1x = MMDetFasterRCNNBackboneConfig(
    config_path=base_config_path / "faster_rcnn_r50_caffe_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth",
)

resnet50_fpn_1x = MMDetFasterRCNNBackboneConfig(
    config_path=base_config_path / "faster_rcnn_r50_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
)

resnet50_fpn_2x = MMDetFasterRCNNBackboneConfig(
    config_path=base_config_path / "faster_rcnn_r50_fpn_2x_coco.py",
    weights_url=f"{base_weights_url}/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth",
)

resnet101_caffe_fpn_1x = MMDetFasterRCNNBackboneConfig(
    config_path=base_config_path / "faster_rcnn_r101_caffe_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/faster_rcnn_r101_caffe_fpn_1x_coco/faster_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.398_20200504_180057-b269e9dd.pth",
)

resnet101_fpn_1x = MMDetFasterRCNNBackboneConfig(
    config_path=base_config_path / "faster_rcnn_r101_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth",
)

resnet101_fpn_2x = MMDetFasterRCNNBackboneConfig(
    config_path=base_config_path / "faster_rcnn_r101_fpn_2x_coco.py",
    weights_url=f"{base_weights_url}/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth",
)

resnext101_32x4d_fpn_1x = MMDetFasterRCNNBackboneConfig(
    config_path=base_config_path / "faster_rcnn_x101_32x4d_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/faster_rcnn_x101_32x4d_fpn_1x_coco/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth",
)

resnext101_32x4d_fpn_2x = MMDetFasterRCNNBackboneConfig(
    config_path=base_config_path / "faster_rcnn_x101_32x4d_fpn_2x_coco.py",
    weights_url=f"{base_weights_url}/faster_rcnn_x101_32x4d_fpn_2x_coco/faster_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.412_20200506_041400-64a12c0b.pth",
)

resnext101_64x4d_fpn_1x = MMDetFasterRCNNBackboneConfig(
    config_path=base_config_path / "faster_rcnn_x101_64x4d_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth",
)

resnext101_64x4d_fpn_2x = MMDetFasterRCNNBackboneConfig(
    config_path=base_config_path / "faster_rcnn_x101_64x4d_fpn_2x_coco.py",
    weights_url=f"{base_weights_url}/faster_rcnn_x101_64x4d_fpn_2x_coco/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033-5961fa95.pth",
)

###
###
# MMDetBackboneConfig

# from mmdet.models.backbones.resnest import ResNeSt

resnest_base_config_path = mmdet_configs_path / "resnest"
resnest_base_weights_url = "https://download.openmmlab.com/mmdetection/v2.0/resnest"

resnest50_fpn = MMDetFasterRCNNBackboneConfig(
    config_path=resnest_base_config_path
    / "faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py",
    weights_url=f"https://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20200926_125502-20289c16.pth",
)

resnest101_fpn = MMDetFasterRCNNBackboneConfig(
    config_path=resnest_base_config_path
    / "faster_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py",
    weights_url=f"https://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201006_021058-421517f1.pth",
)
