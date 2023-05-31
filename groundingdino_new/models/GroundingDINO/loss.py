import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms

from groundingdino_new.util import box_ops
from groundingdino_new.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from groundingdino_new.models.GroundingDINO.matcher import build_matcher
from groundingdino_new.models.GroundingDINO.utils import sigmoid_focal_loss, MLP
from maskrcnn_benchmark.layers import SigmoidFocalLoss, IOULoss, TokenSigmoidFocalLoss

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, cfg):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        # self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = {'loss_ce': cfg.GROUNDINGDINO.loss_ce_coef,'loss_bbox': cfg.GROUNDINGDINO.loss_bbox_coef,'loss_giou': cfg.GROUNDINGDINO.loss_giou_coef}
        self.losses = ['labels', 'boxes']
        self.token_loss_func = TokenSigmoidFocalLoss(cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_ALPHA,
                                                         cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_GAMMA)
        # self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, text_mask, positive_map):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        positive_map_per_image = positive_map.split([len(t) for t in targets])

        idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        positive_map_per_image_o = torch.cat([pos_map[J] for pos_map, (_, J) in zip(positive_map_per_image, indices)])
        target_classes = torch.zeros(src_logits.shape, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes[idx]=positive_map_per_image_o

        unmatched_labels = torch.zeros(target_classes.shape[-1], device=target_classes.device)
        unmatched_labels[-1] = 1.
        target_classes[target_classes.sum(-1)==0] = unmatched_labels

        dot_product_token_loss = self.token_loss_func(src_logits,
                                                          target_classes, text_masks=text_mask,
                                                          version="binary") / num_boxes


        losses = {'loss_ce': dot_product_token_loss}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t.get_field('normed_cxcy_boxes')[i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes


        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False, text_mask=None, positive_map=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets, positive_map)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = len(positive_map)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        for loss in self.losses:
            if 'labels' in loss:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, text_mask=text_mask, positive_map=positive_map))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, positive_map)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    
                    if 'labels' in loss:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, text_mask=text_mask, positive_map=positive_map, **kwargs)
                    else:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        new_losses = {}
        for k,v in losses.items():
            for name, weight in self.weight_dict.items():
                if name in k:
                    new_losses[k] = v * weight
        losses.update(new_losses)


        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


