import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from maskrcnn_benchmark.modeling.language_backbone.clip_model import QuickGELU, LayerNorm, DropPath
from maskrcnn_benchmark.modeling.utils import cat, concat_box_prediction_layers, permute_and_flatten
import os

class CrossMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(CrossMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim


        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.cache_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_cache_proj = nn.Linear(self.v_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)

        self.clamp_min_for_underflow = cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW
        self.clamp_max_for_overflow = cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)

    def forward(self, v, cache, attention_mask_cache=None):
        bsz, tgt_len, embed_dim = v.size()
        cache = cache.to(v.dtype)

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.cache_proj(cache), -1, bsz)
        value_cache_states = self._shape(self.values_cache_proj(cache), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_cache_states = value_cache_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)
        
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        if attention_mask_cache is not None:
            assert (attention_mask_cache.dim() == 2)
            attention_mask_cache=attention_mask_cache.to(torch.float)
            attention_mask = attention_mask_cache.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_cache_states)


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)

        return attn_output_v



class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, v_dim, embed_dim, num_heads, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(GatedCrossAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_c = nn.LayerNorm(v_dim)
        self.attn = CrossMultiHeadAttention(v_dim=v_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         cfg=cfg)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.CONDITIONAL_GATE:
            self.conditioned_gamma_v = nn.Sequential(nn.Linear(v_dim, int(v_dim/2)), nn.ReLU(), nn.Linear(int(v_dim/2), v_dim))
            # zero init
            nn.init.xavier_uniform_(self.conditioned_gamma_v[0].weight)
            self.conditioned_gamma_v[0].bias.data.fill_(0)
            self.conditioned_gamma_v[2].weight.data.fill_(0)
            self.conditioned_gamma_v[2].bias.data.fill_(0)
        else:
            self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)

        self.cfg = cfg

    def forward(self, q0, q1, q2, q3, q4, cache=None, attention_mask_cache=None):
        # aggrage scales
        visu_feat = []
        size_per_level, visual_features_flatten = [], []
        for ii, feat_per_level in enumerate([q0, q1, q2, q3, q4]):
            bs, c, h, w = feat_per_level.shape
            size_per_level.append([h, w])
            feat = permute_and_flatten(feat_per_level, bs, 1, c, h, w)
            visual_features_flatten.append(feat)
        visual_features_flatten = cat(visual_features_flatten, dim=1)


        new_v = self.single_attention_call(visual_features_flatten, cache, attention_mask_cache=attention_mask_cache)

        # [bs, N, C] -> [bs, C, N]
        new_v = new_v.transpose(1, 2).contiguous()

        # recover scales
        start = 0
        for (h, w) in size_per_level:
            new_v_per_level = new_v[:, :, start:start + h * w].view(bs, -1, h, w).contiguous()
            visu_feat.append(new_v_per_level)
            start += h * w

        return visu_feat[0], visu_feat[1], visu_feat[2], visu_feat[3], visu_feat[4]

    def single_attention_call(self, v, cache, attention_mask_cache=None):
        delta_v = self.attn(self.layer_norm_v(v), self.layer_norm_c(cache), attention_mask_cache=attention_mask_cache)
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CONDITIONAL_GATE:
            gate=self.conditioned_gamma_v(v).tanh()
        else:
            gate=self.gamma_v.tanh()
        v = v + self.drop_path(gate * delta_v)
        return v

class QVFuse(torch.nn.Module):
    """
    Early Fusion Module
    """

    def __init__(self, cfg):
        super(QVFuse, self).__init__()
        self.init_configs(cfg)
        self.cfg = cfg

        self.use_checkpoint = False
        if hasattr(cfg.MODEL.DYHEAD, 'USE_CHECKPOINT'):
            self.use_checkpoint = cfg.MODEL.DYHEAD.USE_CHECKPOINT

        # early fusion module
        self.cross_attn = GatedCrossAttentionBlock(v_dim=self.joint_embedding_size,
                    embed_dim=self.embed_dim,
                    num_heads=self.n_head,
                    dropout=0.1,
                    drop_path=.0,
                    init_values=0.,
                    cfg=cfg
                    )
    def init_configs(self, cfg):
        # common params
        self.joint_embedding_size = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE
        # mha params
        self.n_head = 4
        self.embed_dim = 512

    def forward(self, x):
        visual_features = x["visual"]
        cache = x["cache"]
        fused_visual_features = None

        if self.use_checkpoint:
            q0, q1, q2, q3, q4 = checkpoint.checkpoint(self.cross_attn,
                visual_features[0], visual_features[1],
                visual_features[2], visual_features[3],
                visual_features[4], cache['cache'], cache['attention_mask']
            )
        else:
            q0, q1, q2, q3, q4 = self.cross_attn(
                visual_features[0], visual_features[1],
                visual_features[2], visual_features[3],
                visual_features[4], cache['cache'], cache['attention_mask']
            )

        fused_visual_features = [q0, q1, q2, q3, q4]

        # return features_dict
        x.update({"visual": fused_visual_features})
        return x