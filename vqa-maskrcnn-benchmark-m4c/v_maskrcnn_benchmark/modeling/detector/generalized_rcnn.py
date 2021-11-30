# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from v_maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.return_feats = cfg.MODEL.ROI_BOX_HEAD.RETURN_FC_FEATS

    def forward(self, images, targets=None, input_boxes=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        
        features = self.backbone(images.tensors)
        
        # 110 ms?
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        if input_boxes is not None:
            assert not self.training
            load_boxes_for_feature_extraction(proposals, input_boxes)
        
        # 200 ms?
        if self.roi_heads:
            # We are here
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        if self.return_feats and not self.training:
            return (x, result)

        return result


def load_boxes_for_feature_extraction(proposals, input_boxes):
    assert len(proposals) == 1, \
        'only supporting single image feature extraction for now'
    bbox_num = input_boxes.shape[0]
    bbox = torch.zeros(
        (bbox_num, 4),
        dtype=proposals[0].bbox.dtype,
        device=proposals[0].bbox.device
    )
    bbox[...] = input_boxes
    proposals[0].bbox = bbox
    proposals[0].extra_fields.pop('objectness')