# Adapted from https://github.com/microsoft/GLIP. The original liscense is:
#   Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized VL R-CNN framework
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.poolers import CustomPooler, Pooler

from ..backbone import build_backbone
from ..rpn import build_rpn
from ..roi_heads import build_roi_heads
from ..query_selector import build_query_selector

from ..language_backbone import build_language_backbone
from transformers import AutoTokenizer

import random
import timeit
import pdb
from copy import deepcopy
from pathlib import Path
from maskrcnn_benchmark.structures.bounding_box import BoxList


def expand_bbox(box_list, expand_ratio=1.5):
    new_box_list=[]
    for boxes in box_list:
        assert boxes.mode == "xyxy"
        bbox=boxes.bbox
        image_size=boxes.size
        box_w, box_h = bbox[:,2] - bbox[:,0], bbox[:,3] - bbox[:,1]
        new_box_w, new_box_h = box_w*expand_ratio, box_h*expand_ratio
        diff_w=(new_box_w-box_w)/2
        diff_h=(new_box_h-box_h)/2
        diff=torch.stack([-diff_w, -diff_h, diff_w, diff_h], dim=1)
        new_bbox=bbox+diff
        new_boxes=BoxList(new_bbox, image_size, mode="xyxy")
        labels=boxes.get_field('labels')
        new_boxes.add_field('labels', labels)
        new_boxes=new_boxes.clip_to_image(remove_empty=True)
        new_box_list.append(new_boxes)
    return new_box_list



def random_word(input_ids, mask_token_id, vocabs, padding_token_id, greenlight_map):
    """
    greenlight_map, batch_size x 256 (seq_len):
        0 means this location cannot be calculated in the MLM loss
        -1 means this location cannot be masked!!
        1 means this location can be masked and can be calculated in the MLM loss
    """
    output_label = deepcopy(input_ids)
    for j in range(input_ids.size(0)):
        for i in range(input_ids.size(1)):
            prob = random.random()
            # mask token with probability
            ratio = 0.15
            if greenlight_map is not None and greenlight_map[j,i] == -1:
                output_label[j,i] = -100
                continue

            if (not input_ids[j,i] == padding_token_id) and prob < ratio:
                prob /= ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    input_ids[j,i] = mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    input_ids[j,i] = random.choice(vocabs)

            else:
                # no masking token (will be ignored by loss function later)
                output_label[j,i] = -100
            
            if greenlight_map is not None and greenlight_map[j,i] != 1:
                output_label[j,i] = -100 # If this location should not be masked
    return input_ids, output_label


class GeneralizedVLRCNN_New(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg, **kwargs):
        super(GeneralizedVLRCNN_New, self).__init__()
        self.cfg = cfg

        # visual encoder
        self.backbone = build_backbone(cfg)

        # box pooler for extracting cache
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        if cfg.VISION_QUERY.SELECT_FPN_LEVEL:
            self.pooler = Pooler(
            output_size= (resolution, resolution) ,
            scales=cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES,
            sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
            use_v2=True,
            )
        else:
            self.pooler = CustomPooler(
                output_size= (resolution, resolution) ,
                scales=cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES,
                sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
                use_v2=True,
            )
            
        # query selector
        if cfg.VISION_QUERY.DISABLE_SELECTOR:
            self.query_selector = None
        else:
            self.query_selector = build_query_selector(cfg)

        self.pool=nn.AvgPool2d(2)

        # language encoder
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            # self.tokenizer = build_tokenizer("clip")
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                print("Reuse token 'ðŁĴĳ</w>' (token_id = 49404) for mask token!")
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                            from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                            from_slow=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
            # self.tokenizer.save_pretrained(Path('MODEL/THIRD_PARTIES', cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE))

        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.tokenizer_vocab_ids = [item for key, item in self.tokenizer_vocab.items()]

        self.language_backbone = build_language_backbone(cfg)

        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.DEBUG = cfg.MODEL.DEBUG

        self.freeze_backbone = cfg.MODEL.BACKBONE.FREEZE
        self.freeze_fpn = cfg.MODEL.FPN.FREEZE
        self.freeze_rpn = cfg.MODEL.RPN.FREEZE
        self.add_linear_layer = cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER

        self.force_boxes = cfg.MODEL.RPN.FORCE_BOXES

        if cfg.MODEL.LINEAR_PROB:
            assert cfg.MODEL.BACKBONE.FREEZE, "For linear probing, backbone should be frozen!"
            if hasattr(self.backbone, 'fpn'):
                assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
        self.linear_prob = cfg.MODEL.LINEAR_PROB
        self.freeze_cls_logits = cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # disable cls_logits
            if hasattr(self.rpn.head, 'cls_logits'):
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False

        self.freeze_language_backbone = self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE
        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            for p in self.language_backbone.parameters():
                p.requires_grad = False
        
        self.use_mlm_loss = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS 
        self.mlm_loss_for_only_positives = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_FOR_ONLY_POSITIVES

        if self.cfg.GLIPKNOW.KNOWLEDGE_FILE:
            from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
            self.class_name_to_knowledge = load_from_yaml_file(self.cfg.GLIPKNOW.KNOWLEDGE_FILE)
            self.class_name_list = sorted([k for k in self.class_name_to_knowledge])

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GeneralizedVLRCNN_New, self).train(mode)
        if self.freeze_backbone:
            self.backbone.body.eval()
            for p in self.backbone.body.parameters():
                p.requires_grad = False
        if self.freeze_fpn:
            self.backbone.fpn.eval()
            for p in self.backbone.fpn.parameters():
                p.requires_grad = False
        if self.freeze_rpn:
            if hasattr(self.rpn, 'head'):
                self.rpn.head.eval()
            for p in self.rpn.parameters():
                p.requires_grad = False
        if self.linear_prob:
            if self.rpn is not None:
                for key, value in self.rpn.named_parameters():
                    if not ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                        value.requires_grad = False
            if self.roi_heads is not None:
                for key, value in self.roi_heads.named_parameters():
                    if not ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                        value.requires_grad = False
        if self.freeze_cls_logits:
            if hasattr(self.rpn.head, 'cls_logits'):
                self.rpn.head.cls_logits.eval()
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False
        if self.add_linear_layer:
            if self.rpn is not None:
                for key, p in self.rpn.named_parameters():
                    if 'tunable_linear' in key:
                        p.requires_grad = True

        if self.freeze_language_backbone:
            self.language_backbone.eval()
            for p in self.language_backbone.parameters():
                p.requires_grad = False

    def load_query_bank(self, query_path):
        self.query_selector.load_query_bank(query_path)

    @torch.no_grad()
    def extract_query(self, 
        images=None,
        targets=None,
        query_images=None, # default_dict(list) ,list[tensors] num_classes: (num_queries, num_scales, num_channels)
        visual_features=None,
        exclude_similar=False,
        device = None,
        max_query_number = None,
        ):
        device = device if device else images.tensors.device

        targets = [target.to(device)
                    for target in targets if target is not None]
        targets=expand_bbox(targets, expand_ratio=self.cfg.VISION_QUERY.EXPAND_RATIO)
        
        if visual_features is None:
            images = to_image_list(images)
            assert 'vl' not in self.cfg.MODEL.SWINT.VERSION, 'Only support vision inputs now'
            visual_features = self.backbone(images.tensors)
        else:
            visual_features = [v.to(device) for v in visual_features]

        if self.cfg.VISION_QUERY.SELECT_FPN_LEVEL:
            query_feats=self.pooler(visual_features, targets) # num_boxes, num_channels, pooler_size, pooler_size
            query_feats=query_feats[None, ...] # 1, num_boxes, num_channels, pooler_size, pooler_size
        else:
            query_feats=self.pooler(visual_features, targets) # num_scales, num_boxes, num_channels, pooler_size, pooler_size
        
        # average different fpn levels
        if not self.cfg.VISION_QUERY.SELECT_FPN_LEVEL:
            assert len(visual_features) == len(query_feats) == 5 # TODO: query flexible level numbers
        query_feats = query_feats.mean(dim=[-2,-1]).permute(1, 0, 2) # num_boxes, num_scales, num_channels

        labels=torch.cat([t.get_field('labels') for t in targets])
        assert len(labels)==len(query_feats)

        max_query_number = self.cfg.VISION_QUERY.MAX_QUERY_NUMBER if max_query_number is None else max_query_number
        for label, feat in zip(labels, query_feats):
            label=label.item()
            num_queries=len(query_images[label])
            if num_queries >= max_query_number:
                continue
            if exclude_similar and num_queries > 0:
                assert feat.shape[0] == 1 # TODO: enable all-level and spacial features
                bank_features = F.normalize(query_images[label], p=2, dim=-1) # N, 1, C
                new_features = F.normalize(feat, p=2, dim=-1) # 1, C
                similarity = einsum('b n d, n d -> b n', bank_features, new_features)
                has_similar_in_bank = (similarity > self.cfg.VISION_QUERY.SIMILARITY_THRESHOLD).sum() > 0
                if has_similar_in_bank:
                    continue

            if num_queries==0:
                query_images[label] = feat[None, ...]
            else:
                query_images[label] = torch.cat([query_images[label], feat[None, ...]])
        return query_images


    def flatten_fpn_features(self, features):
        # downsample and flat fpn features for pre-select in language backbone
        return torch.cat([self.pool(f).flatten(-2,-1) for i, f in enumerate(features)], dim=2).permute(0,2,1)

    @torch.no_grad()
    def get_labels_and_maps_from_positive_map(self, positive_map, dtype=torch.float):
        # Only for inference
        labels_in_caption=[k for k,v in positive_map.items() if len(v) !=0]
        num_labels=len(labels_in_caption)
        all_map = torch.zeros((num_labels, self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN), dtype=dtype, device=self.cfg.MODEL.DEVICE)
        for j, label in enumerate(labels_in_caption):
            position=positive_map[label]
            all_map[j, position] = 1 # inplace
        all_map = all_map / (all_map.sum(-1)[:, None] + 1e-6)
        return labels_in_caption, all_map

    def forward(self, 
        images, 
        targets=None, 
        captions=None, 
        positive_map=None,
        greenlight_map=None,
        return_backbone_features=False,
        ):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

            mask_black_list: batch x 256, indicates whether or not a certain token is maskable or not

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        # batch_size = images.tensors.shape[0]
        device = images.tensors.device

        # visual embedding
        swint_feature_c4 = None
        if 'vl' in self.cfg.MODEL.SWINT.VERSION:
            # the backbone only updates the "hidden" field in language_dict_features
            inputs = {"img": images.tensors, "lang": language_dict_features}
            visual_features, language_dict_features, swint_feature_c4 = self.backbone(inputs)
        else:
            visual_features = self.backbone(images.tensors)

        # query embedding
        if self.cfg.VISION_QUERY.ENABLED:
            if self.training:
                batched_labels_in_caption=[t.get_field('labels_in_caption') for t in targets]
                batched_all_map=[t.get_field('all_map') for t in targets]
                batched_pos_category_map=[t.get_field('positive_category_map') for t in targets]
                ################ BUG: batched_pos_category_map is not binary ######################
                batched_pos_labels = [t.get_field('labels') for t in targets]
            else:
                assert images.tensors.shape[0]==1 # TODO: Only query batch size = 1 for test
                labels_in_caption, all_map = self.get_labels_and_maps_from_positive_map(positive_map, dtype=visual_features[0].dtype)
                batched_labels_in_caption = [labels_in_caption]
                batched_all_map = [all_map]
                batched_pos_category_map = None
                batched_pos_labels = None


            query_features, query_attetion_masks, batched_has_vision_query=self.query_selector(batched_labels_in_caption, batched_all_map, batched_pos_labels)
 
            vision_inputs_in_language_backbone={'vision': query_features, 'images': self.flatten_fpn_features(visual_features), 'vision_attention_mask': query_attetion_masks, 'batched_pos_category_map': batched_pos_category_map}
        else:
            vision_inputs_in_language_backbone={'vision': None, 'images': None, 'vision_attention_mask': None, 'batched_pos_category_map': None}

        # language embedding
        if self.cfg.GLIPKNOW.PARALLEL_LANGUAGE_INPUT:
            language_dict_features, positive_map = self._forward_language_parallel(
                    captions=captions, targets=targets, device=device,
                    positive_map=positive_map, vision_inputs=vision_inputs_in_language_backbone)
        else:
            # language embedding
            language_dict_features = {}
            if captions is not None:
                #print(captions[0])
                tokenized = self.tokenizer.batch_encode_plus(captions,
                                                            max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                            padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True,
                                                            return_tensors='pt',
                                                            truncation=True).to(device)
                if self.use_mlm_loss:
                    if not self.mlm_loss_for_only_positives:
                        greenlight_map = None
                    input_ids, mlm_labels = random_word(
                        input_ids=tokenized.input_ids, 
                        mask_token_id=self.tokenizer.mask_token_id,
                        vocabs=self.tokenizer_vocab_ids,
                        padding_token_id=self.tokenizer.pad_token_id,
                        greenlight_map=greenlight_map)
                else:
                    input_ids = tokenized.input_ids
                    mlm_labels = None
                
                if (self.cfg.VISION_QUERY.ENABLED) and (self.cfg.VISION_QUERY.TEXT_DROPOUT > 0.) and (self.training or self.cfg.VISION_QUERY.MASK_DURING_INFERENCE):
                    if self.cfg.VISION_QUERY.MASK_DURING_INFERENCE:
                        assert self.cfg.VISION_QUERY.PURE_TEXT_RATE == 0. # TODO: enable part text part image
                    if not self.cfg.VISION_QUERY.NEW_MASK_TOKEN:
                        maps = batched_all_map if (self.cfg.VISION_QUERY.MASK_DURING_INFERENCE and not self.training) else batched_pos_category_map
                        for i, (pos_label_position, has_vision_query) in enumerate(zip(maps, batched_has_vision_query)):
                            pos_label_position=pos_label_position.to(torch.bool)
                            for j, position in enumerate(pos_label_position):
                                if (random.random() < self.cfg.VISION_QUERY.TEXT_DROPOUT):
                                    if has_vision_query[j] == 1: # only mask text tokens with vision queries
                                        input_ids[i, position] = self.tokenizer.mask_token_id

                
                tokenizer_input = {"input_ids": input_ids,
                                "attention_mask": tokenized.attention_mask,
                                "vision_inputs": vision_inputs_in_language_backbone}

                # if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
                #     with torch.no_grad():
                #         language_dict_features = self.language_backbone(tokenizer_input)
                # else:
                language_dict_features = self.language_backbone(tokenizer_input)
                
                # ONE HOT
                if self.cfg.DATASETS.ONE_HOT:
                    new_masks = torch.zeros_like(language_dict_features['masks'],
                                                device=language_dict_features['masks'].device)
                    new_masks[:, :self.cfg.MODEL.DYHEAD.NUM_CLASSES] = 1
                    language_dict_features['masks'] = new_masks

                # MASK ALL SPECIAL TOKENS
                if self.cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL:
                    language_dict_features["masks"] = 1 - tokenized.special_tokens_mask
                
                language_dict_features["mlm_labels"] = mlm_labels

        # rpn force boxes
        if targets:
            targets = [target.to(device)
                       for target in targets if target is not None]

        if self.force_boxes:
            proposals = []
            for t in targets:
                tb = t.copy_with_fields(["labels"])
                tb.add_field("scores", torch.ones(tb.bbox.shape[0], dtype=torch.bool, device=tb.bbox.device))
                proposals.append(tb)
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                _, proposal_losses, fused_visual_features = self.rpn(
                    images, visual_features, targets, language_dict_features,
                    positive_map, captions, swint_feature_c4)
            elif self.training:
                null_loss = 0
                for key, param in self.rpn.named_parameters():
                    null_loss += 0.0 * param.sum()
                proposal_losses = {('rpn_null_loss', null_loss)}
        else:
            proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets, language_dict_features, positive_map,
                                              captions, swint_feature_c4)
        if self.roi_heads:
            if self.cfg.MODEL.ROI_MASK_HEAD.PREDICTOR.startswith("VL"):
                if self.training:
                    # "Only support VL mask head right now!!"
                    assert len(targets) == 1 and len(targets[0]) == len(positive_map), "shape match assert for mask head!!"
                    # Not necessary but as a safe guard:
                    # use the binary 0/1 positive map to replace the normalized positive map
                    targets[0].add_field("positive_map", positive_map)
            # TODO: make sure that this use of language_dict_features is correct!! Its content should be changed in self.rpn
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                x, result, detector_losses = self.roi_heads(
                    fused_visual_features, proposals, targets,
                    language_dict_features=language_dict_features,
                    positive_map_label_to_token=positive_map if not self.training else None
                )
            else:
                x, result, detector_losses = self.roi_heads(
                    visual_features, proposals, targets,
                    language_dict_features=language_dict_features,
                    positive_map_label_to_token=positive_map if not self.training else None
                )
        else:
            # RPN-only models don't have roi_heads
            x = visual_features
            result = proposals
            detector_losses = {}


        if self.training:
            #### gate loss #####
            if self.cfg.VISION_QUERY.ENABLED:
                # concatenate all gates
                gates = []
                for _ ,g in language_dict_features['vision_query_gates'].items():
                    gates = gates + g

                num_gates=len(gates)
                loss_gate=0
                for g in gates:
                    loss_gate=loss_gate+(1-torch.abs(g[0]))
                loss_gate= self.cfg.VISION_QUERY.GATE_REGULARIZATION_SCALE * loss_gate / num_gates
                if self.cfg.VISION_QUERY.GATE_REGULARIZATION:
                    gate_losses = {'loss_gate': loss_gate.sum()}
                else:
                    loss_gate = loss_gate.sum().detach() # Only for analysis
                    gate_losses = {'loss_gate': loss_gate}
                ####################
            else:
                gate_losses = {'loss_gate': images.tensors.new_zeros(1)[0]}
            
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(gate_losses)
            return losses

        if self.cfg.VISION_QUERY.RETURN_ATTN_GATE_VALUE:
            gate_value = language_dict_features['vision_query_gates']['attn_gates']
            return result, gate_value

        if return_backbone_features:
            return result, visual_features
        else:
            return result

    def _forward_language_parallel(self, captions=None, targets=None,
            device=None, positive_map=None, vision_inputs=None):
        ktype = self.cfg.GLIPKNOW.KNOWLEDGE_TYPE
        def _construct_captions_from_class_names(class_names):
            captions = []
            for c in class_names:
                try:
                    info = self.class_name_to_knowledge[c]
                    cap = info['clean_name']

                    # combine wiki and gpt3 knowledge
                    if self.cfg.GLIPKNOW.WIKI_AND_GPT3:
                        ktype = 'def_wiki'
                        know_seq = info[ktype]

                        ktype = 'gpt3'
                        if ktype == 'gpt3' or type(info[ktype]) == list:
                            know_seq += ' '.join([seq for seq in info[ktype][:self.cfg.GLIPKNOW.GPT3_NUM] ])

                        cap += ': ' + know_seq

                    # only one knoweldge source is used        
                    else:
                        if ktype and ktype in info and info[ktype]:
                            if ktype == 'gpt3' or type(info[ktype]) == list:
                                know_seq = ' '.join([seq for seq in info[ktype][:self.cfg.GLIPKNOW.GPT3_NUM] ])
                            else: 
                                know_seq = info[ktype]
                            cap += ': ' + know_seq
                except:
                    cap = c
                    print(f'cap {cap}, c {c}')
                    
                    
                captions.append(cap)
            return captions

        if self.training:
            assert captions is None
            assert targets is not None

            max_classes_per_batch = self.cfg.GLIPKNOW.MAX_NUM_CLASSES_PER_BATCH_TRAIN
            if max_classes_per_batch >= len(self.class_name_list):
                shuffled_class_names = self.class_name_list.copy()
                random.shuffle(shuffled_class_names)
                if max_classes_per_batch > len(shuffled_class_names):
                    shuffled_class_names.extend(shuffled_class_names[:max_classes_per_batch
                        -len(shuffled_class_names)])
                    random.shuffle(shuffled_class_names)
            else:
                label_list = []
                label_to_idx = {}
                for target_per_im in targets:
                    labels_per_im = target_per_im.get_field('label_names')
                    for label in labels_per_im:
                        if label not in label_to_idx:
                            label_to_idx[label] = len(label_list)
                            label_list.append(label)

                label_list = label_list[:max_classes_per_batch]
                if len(label_list) < max_classes_per_batch:
                    all_neg_classes = [c for c in self.class_name_list if c not
                            in label_to_idx]
                    neg_label_list = random.sample(all_neg_classes,
                            max_classes_per_batch - len(label_list))
                    label_list.extend(neg_label_list)
                random.shuffle(label_list)
                shuffled_class_names = label_list

            label_to_shuffled_idx = {l: i for i, l in
                    enumerate(shuffled_class_names)}
            total_boxes = sum(len(t) for t in targets)
            positive_map = torch.zeros((total_boxes, max_classes_per_batch+1),
                device=device)
            offset = 0
            for target_per_im in targets:
                labels_per_im = target_per_im.get_field('label_names')
                for label in labels_per_im:
                    j = label_to_shuffled_idx.get(label, -1)
                    if j >= 0:
                        positive_map[offset, j] = 1
                    offset += 1
            captions = _construct_captions_from_class_names(shuffled_class_names)
            captions.append('') # onobj at the end, onedet/modeling/rpn/loss.py:719
            batch_size = len(targets)

        else:
            assert captions is not None
            batch_size = 1
            assert len(captions) == 1
            class_names = captions[0]
            max_classes_per_batch = len(class_names)
            captions = _construct_captions_from_class_names(class_names)
            captions.append('') # onobj at the end, onedet/modeling/rpn/loss.py:719

        tokenized = self.tokenizer.batch_encode_plus(captions,
                                                     max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                     padding="longest",
                                                     return_special_tokens_mask=True,
                                                     return_tensors='pt',
                                                     truncation=True).to(device)
        assert not self.use_mlm_loss
        tokenizer_input = {"input_ids": tokenized.input_ids,
                           "attention_mask": tokenized.attention_mask,
                           "vision_inputs": vision_inputs}

        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            with torch.no_grad():
                language_dict_features = self.language_backbone(tokenizer_input)
        else:
            language_dict_features = self.language_backbone(tokenizer_input)

        assert not self.cfg.DATASETS.ONE_HOT
        assert not self.cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL

        agg_type = self.cfg.GLIPKNOW.LAN_FEATURE_AGG_TYPE
        agg_feats = language_dict_features['hidden']
        agg_emb = language_dict_features['embedded']
        if agg_type == 'first':
            agg_feats = agg_feats[:, 0, :]
            agg_emb = agg_emb[:, 0, :]
        elif agg_type == 'mean':
            attn_mask = language_dict_features['masks']
            seq_len = attn_mask.sum(-1).unsqueeze(-1).float()
            agg_feats = agg_feats * attn_mask.unsqueeze(-1).float()
            agg_feats = agg_feats.sum(1) / seq_len
            agg_emb = agg_emb * attn_mask.unsqueeze(-1).float()
            agg_emb = agg_emb.sum(1) / seq_len
        else:
            raise ValueError('not supported GLIPKNOW.LAN_FEATURE_AGG_TYPE: {}'.format(agg_type))

        expanded_features = agg_feats.unsqueeze(0).repeat(batch_size, 1, 1)
        expanded_embedding = agg_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        lang_dict = {}
        lang_dict["mlm_labels"] = None
        lang_dict["aggregate"] = None
        lang_dict["embedded"] = expanded_embedding
        lang_dict['hidden'] = expanded_features
        lang_dict["masks"] = torch.ones((batch_size, max_classes_per_batch+1),
                device=device, dtype=language_dict_features['masks'].dtype)
        # in GLIP setting, the token at the end of seqence is usually [PAD], and is masked out
        # if [noobj] is not masked out, the loss sum is very big, as most
        # anchors are matched to [noobj]
        lang_dict["masks"][:,-1] = 0
        return lang_dict, positive_map

