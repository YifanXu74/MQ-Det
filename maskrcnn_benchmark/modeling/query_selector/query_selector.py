import torch
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random
import os

class QuerySelector(nn.Module):
    def __init__(self, cfg):
        super(QuerySelector, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        if not os.path.exists(cfg.VISION_QUERY.QUERY_BANK_PATH):
            assert cfg.VISION_QUERY.QUERY_BANK_PATH == "", "query bank path {} not exists".format(cfg.VISION_QUERY.QUERY_BANK_PATH)
            assert not cfg.VISION_QUERY.LEARNABLE_BANK
            self.query_bank = None
        else:
            if cfg.VISION_QUERY.LEARNABLE_BANK:
                query_bank = torch.load(cfg.VISION_QUERY.QUERY_BANK_PATH, map_location='cpu')
                # add qv_layer to name only for easier parameter freezing
                self.query_bank = nn.ParameterDict({str(k): nn.Parameter(v) for k, v in query_bank.items()})
                query_dim = query_bank[list(query_bank.keys())[0]].shape[-1]
            else:
                # add qv_layer to name only for easier parameter freezing
                self.query_bank = torch.load(cfg.VISION_QUERY.QUERY_BANK_PATH, map_location=self.device) # default dict: num_classes [num_queries, num_scales, num_channels ]
                query_dim = self.query_bank[list(self.query_bank.keys())[0]].shape[-1]
        if cfg.VISION_QUERY.ADD_VISION_LAYER:
            self.tunable_vision_linear = torch.nn.Linear(query_dim, 1000, bias=False)
            self.tunable_vision_linear.weight.data.fill_(0.0)
        self.pure_text_rate = cfg.VISION_QUERY.PURE_TEXT_RATE
        self.num_query_per_class = cfg.VISION_QUERY.NUM_QUERY_PER_CLASS
        self.cfg = cfg
    
    def load_query_bank(self, bank_path):
        assert not self.cfg.VISION_QUERY.LEARNABLE_BANK
        assert not self.cfg.VISION_QUERY.ADD_VISION_LAYER
        # add qv_layer to name only for easier parameter freezing
        self.query_bank = torch.load(bank_path, map_location=self.device) # default dict: num_classes [num_queries, num_scales, num_channels ]
    
    # @torch.no_grad()
    def forward(self, batched_label_list, batched_location_map, batched_pos_labels = None):
        '''
        Return query features, attention mask

        batched_label_list: [[list]] - batch_size, num_labels
        batched_location_map: [torch.tensor] one-hot -  batch_size, (num_labels, num_text_tokens)
        '''
        if self.query_bank is None:
            return None, None, None

        batched_queries = []
        batched_queries_attn_mask = []
        batched_has_vision_query = []
        for k, (label_list, location_map) in enumerate(zip(batched_label_list, batched_location_map)):
            query_per_image = []
            mask_per_image = []
            has_vision_query = []
            for label, loc_map in zip(label_list, location_map):
                loc_map = loc_map.to(self.device)
                if self.cfg.VISION_QUERY.LEARNABLE_BANK:
                    candidate_queries=self.query_bank[str(label)]
                else:
                    candidate_queries=self.query_bank[label]
                num_total_queries=len(candidate_queries)
                loc_map = loc_map [None, ...] # 1, num_text_tokens

                # num_query_per_class = self.num_query_per_class
                num_query_per_class = np.random.choice(range(1, self.num_query_per_class+1)) if (self.cfg.VISION_QUERY.RANDOM_KSHOT and self.training) else self.num_query_per_class
                num_queries = min(num_total_queries, num_query_per_class)

                if (random.random() < self.pure_text_rate) and self.training:
                    # data augmentation: random select some labels for only text inputs, without vision query
                    num_queries = 0

                idx= np.random.choice(num_total_queries, num_queries, replace=False).tolist()
                if not self.training:
                    idx = sorted(idx)
                if isinstance(candidate_queries, list):
                    assert len(idx) == 0
                else:
                    queries = candidate_queries[idx]
                    num_scale=queries.shape[1]
                    queries=queries.flatten(0,1)
                    queries_attn_mask = loc_map.expand(num_queries*num_scale, -1)
                    query_per_image.append(queries)
                    mask_per_image.append(queries_attn_mask)

                if batched_pos_labels is None:
                    pos_flag = True
                else:
                    pos_flag = (label in batched_pos_labels[k])

                if pos_flag:
                    has_vision_query.append(1 if num_queries > 0 else 0)

            query_per_image=torch.cat(query_per_image)
            mask_per_image=torch.cat(mask_per_image)
            
            if self.cfg.VISION_QUERY.ADD_VISION_LAYER:
                query_per_image = self.tunable_vision_linear.weight[:query_per_image.size(0), :] + query_per_image


            batched_queries.append(query_per_image)
            batched_queries_attn_mask.append(mask_per_image)
            batched_has_vision_query.append(has_vision_query)

        
        batched_queries=pad_sequence(batched_queries, batch_first=True) # TODO: more efficiet implement
        batched_queries_attn_mask=pad_sequence(batched_queries_attn_mask, batch_first=True)
        
        # The batched_location_map averages the scores, for example, 'apple pie' has two tokenized tokens, thus the location map is (0.5, 0.5) rather than (1, 1). 
        # So we reformulate the batched_queries_attn_mask to 0 or 1.
        batched_queries_attn_mask[batched_queries_attn_mask!=0] = 1



        return batched_queries, batched_queries_attn_mask, batched_has_vision_query

        



        

