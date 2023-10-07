import torch
import torch.utils.checkpoint
from torch import nn, einsum
from einops import rearrange
from einops_exts import rearrange_many

from typing import List, Optional, Tuple, Union
from maskrcnn_benchmark.utils.torch_dropout import Dropout1d

import random
from collections import OrderedDict


from transformers.models.bert.modeling_bert import BertModel, BertEncoder, BertEmbeddings,\
BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions,\
logger, \
add_start_docstrings_to_model_forward, add_code_sample_docstrings, \
BERT_INPUTS_DOCSTRING, _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC


# import torch.nn.utils.rnn as rnn_utils

# def get_index_with_padding_batch(a, padding_value=0):
#     '''
#     Given an attention mask, which only contains 0 and 1, return a tensor that contains the index of non-zero elements. Pad each row of output tensor with given padding_value to the same length.
#     Inputs:
#         a - (B, M, N)
#     Outputs:
#         torch.tensor - (B, M, K) , K is the max length of non-zero elements in the N-dim of a.
#     '''
#     # Compute the indices of non-zero elements
#     indices = [torch.nonzero(row)[:, 0] for row in a.reshape(-1, a.shape[-1])]
    
#     # Pad sequences and reshape back to the original shape
#     padded_indices = rnn_utils.pad_sequence(indices, batch_first=True, padding_value=padding_value)
#     padded_indices = padded_indices.view(a.shape[0], a.shape[1], -1)
    
#     return padded_indices

@torch.no_grad()
def get_index_with_padding_batch(a, padding_value=None):
    '''
    Given an attention mask, which only contains 0 and 1, return a tensor that contains the index of non-zero elements. Pad each row of output tensor with given padding_value to the same length.
    Inputs:
        a - (B, M, N)
    Outputs:
        torch.tensor - (B, M, K) , K is the max length of non-zero elements in the N-dim of a.
    Note!!!
        padding_value == N, namely, concat a zero vector at the end of vision query as a candidate padding token.
    '''
    if padding_value is None:
        padding_value = a.shape[-1]
    else:
        assert padding_value == a.shape[-1]

    # Get the indices of non-zero elements, then insert the indices into a new tensor with all padding value.
    non_zero = (a != 0)
    max_length = non_zero.sum(-1).max()
    indices = torch.where(non_zero, torch.arange(a.shape[-1], dtype=torch.long, device=a.device), torch.tensor(padding_value, dtype=torch.long, device=a.device))
    
    # make valid indices at the begining of the tensor, and then split them out.
    padded_indices = indices.topk(k=max_length, dim=-1, largest=False).values
    return padded_indices[:, :, :max_length]

# def get_index_with_padding_batch(a, padding_value=0):
#     # TODO: more efficient implement
#     '''
#     Given an attention mask, which only contains 0 and 1, return a tensor that contains the index of non-zero elements. Pad each row of output tensor with given padding_value to the same length.
#     Inputs:
#         a - (B, M, N)
#     Outputs:
#         torch.tensor - (B, M, K) , K is the max length of non-zero elements in the N-dim of a.
#     '''
#     B, M, N = a.shape
#     index_list = []
#     max_length = 0
#     for i in range(B):
#         row_indices = []
#         for j in range(M):
#             row_index = torch.nonzero(a[i, j]).squeeze().tolist()
#             row_indices.append(row_index)
#             if len(row_index) > max_length:
#                 max_length = len(row_index)
#         index_list.append(row_indices)

#     for i in range(len(index_list)):
#         for j in range(len(index_list[i])):
#             diff = max_length - len(index_list[i][j])
#             index_list[i][j] += [padding_value] * diff
#         diff = M - len(index_list[i])
#         index_list[i] += [[padding_value] * max_length] * diff

#     return torch.tensor(index_list, device=a.device)

def easy_gather(x, indices):
    # x: B,N,C; indices: B,N
    B, N, C = x.shape
    N_new = indices.shape[1]
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
    indices = indices + offset
    out = x.flatten(0,1)[indices.view(-1)].view(B, N_new, C)
    return out

# gated cross attention

def exists(val):
    if val is not None:
        if len(val) > 0:
            return True
        else:
            return False
    else:
        return False

def FeedForward(dim, mult = 4, out_dim = None):
    inner_dim = int(dim * mult)
    if out_dim is None:
        out_dim = dim
    return nn.Sequential(
                OrderedDict([
                    ('norm', nn.LayerNorm(dim)),
                    ('linear1', nn.Linear(dim, inner_dim, bias = False)),
                    ('gelu', nn.GELU()),
                    ('linear2', nn.Linear(inner_dim, out_dim, bias = False))
                    ])
                )

class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        output_dim = None,
        dim_head = 64,
        heads = 8,
        norm_kv = False,
        share_kv=False,
        cfg=None,
        spase_forward=False, 
    ):
        super().__init__()
        self.spase_forward=spase_forward
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.share_kv=share_kv
        inner_dim = dim_head * heads
        if output_dim is None:
            output_dim = input_dim

        self.norm = nn.LayerNorm(input_dim)
        self.norm_kv = None
        if norm_kv:
            self.norm_kv = nn.LayerNorm(input_dim)

        self.to_q = nn.Linear(input_dim, inner_dim, bias = False)
        if share_kv:
            self.to_kv = nn.Linear(input_dim, inner_dim, bias = False)
        else:
            self.to_kv = nn.Linear(input_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, output_dim, bias = False)

    @classmethod
    def _construct_sparse_inputs(cls, x, vision, attention_mask):
        '''
        Make each text token only attends to a fix number of query vision tokens (typically a small number).
        Inputs:
            x - (batch, text, dim)
            vision - (batch, vision, dim)
            attention_mask - (batch, vision, text)
        Outputs:
            x - (batch * text, 1, dim)
            vision - (batch * text, num_suport_per_class, dim)  e.g., num_suport_per_class = 5
            attention_mask:  mask padding tokens -  (batch * text, 1, num_suport_per_class)
        '''
        B, V, C = vision.shape # batch, vision, dim
        vision=torch.cat([vision, vision.new_zeros(B, 1, C)], dim=1) # B, N+1, C
        padding_index=V
        index = get_index_with_padding_batch(attention_mask.transpose(2,1), padding_value=padding_index)
        B, T, S = index.shape # batch, text, num_querys
        vision=easy_gather(vision, index.flatten(1,2)).reshape(B, T, S, C)
        x = x[:,:,None,...]
        new_mask=(index[:,:,None,...] != padding_index) # batch, vision, text
        new_mask=new_mask.transpose(-2,-1) # batch, vision, text
        return x.flatten(0,1), vision.flatten(0,1), new_mask.flatten(0,1)
    
    def forward(
        self,
        x,      # (batch, text, dim)
        vision, # (batch, vision, dim)
        attention_mask = None, # (batch, vision, text)
    ):
        if self.spase_forward:
            batch_size = x.shape[0]
            x, vision, attention_mask = self._construct_sparse_inputs(x, vision, attention_mask)

        vision = vision.to(x.dtype)
        b, v, d = vision.shape
        h = self.heads

        x = self.norm(x)
        if self.norm_kv:
            vision = self.norm_kv(vision)

        q = self.to_q(x)
        # vision = rearrange(vision, 'b s v d -> b (s v) d')

        if self.share_kv:
            k = v = self.to_kv(vision)
        else:
            k, v = self.to_kv(vision).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k) # (batch, heads, sequence, vision)
        if exists(attention_mask):
            sim=rearrange(sim, 'b h t v -> b h v t')

            mask = sim.new_zeros(attention_mask.shape) # (b, v, t)
            mask[attention_mask==0] = -1e4 # for half
            mask=mask[:, None, ...] # (b, 1, v, t)
            sim = sim + mask
            sim=rearrange(sim, 'b h v t -> b h t v')

        attn = sim.softmax(dim = -1)

        if exists(attention_mask):
            attn=rearrange(attn, 'b h t v -> b v t h')
            attn = attn * attention_mask[..., None] # make sure some ignored tokens got all zero attention
            # attn[attention_mask==0] = 0
            attn=rearrange(attn, 'b v t h -> b h t v')

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if self.spase_forward:
            assert out.shape[1]==1
            out = rearrange(out, '(b t) n d -> b t (n d)', b=batch_size)

        out = self.to_out(out)

        # # self-update: update those with all zero attention masks by themselves, for option 1!
        # if exists(attention_mask):
        #     update_mask = (attention_mask.sum(1)==0) # (b,t)
        #     # assert out[update_mask].sum()==0
        #     out = x * update_mask[..., None] + out

        return out

class GatedCrossAttentionBlock(nn.Module):
    '''
    For each target category, extract one roi feature on each scale, i.e., (batch, scales, latents, dim_v), latents always = k shot.
    "latents" denotes the total length of all vison tokens at each scale.
    If the attention mask of vision v to all text t is False, return the original text embedding.
    '''
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        share_kv=False,
        cfg=None,
        enable_ffn = True
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(input_dim = dim, dim_head = dim_head, heads = heads, share_kv=share_kv, cfg=cfg, norm_kv=True, spase_forward=True)
        if cfg.VISION_QUERY.FIX_ATTN_GATE == -1.0:
            if cfg.VISION_QUERY.CONDITION_GATE:
                if cfg.VISION_QUERY.NONLINEAR_GATE:
                    if cfg.VISION_QUERY.NO_CAT:
                        self.attn_gate = FeedForward(dim=dim, mult=0.5, out_dim = 1)
                        torch.nn.init.constant_(self.attn_gate.linear2.weight, 0)
                    else:
                        self.attn_gate = FeedForward(dim=dim*2, mult=0.5, out_dim = 1)
                        torch.nn.init.constant_(self.attn_gate.linear2.weight, 0)
                else:
                    self.attn_gate = nn.Linear(dim, 1, bias=False)
                    torch.nn.init.constant_(self.attn_gate.weight, 0)
            else:
                self.attn_gate = nn.Parameter(torch.tensor([0.]))
        # if cfg.VISION_QUERY.TEXT_DROPOUT > 0.:
        #     self.mask_token = nn.Parameter(torch.randn(dim))
        self.enable_ffn = enable_ffn
        if enable_ffn:
            self.ff = FeedForward(dim, mult = ff_mult)
            if cfg.VISION_QUERY.FIX_ATTN_GATE == -1.0:
                self.ff_gate = nn.Parameter(torch.tensor([0.]))

        if cfg.VISION_QUERY.ADD_ADAPT_LAYER:
            self.adaptor = FeedForward(dim, mult = 2)

        # self.text_dropout=Dropout1d(p=cfg.VISION_QUERY.TEXT_DROPOUT)
        self.cfg=cfg
        self.attn_gate_value = 0.

    def forward(
        self,
        x,                       # text tensor - (batch, text, dim_t)
        vision,                  # vision query tensor - (batch, vision, dim_v)
        attention_mask = None,   # boolean tensor indicating masks of media - (batch, vision, text)
        batched_positive_label_position = None, # batch: {label: (positions)}
    ):
        # assert exists(attention_mask)

        ## do not drop pure text or padding
        # dropped_x = x
        # if exists(attention_mask):
            # dropped_x = self.text_dropout(x)
            # drooped_mask = (dropped_x.sum(-1)==0) # (b,t)
            # update_mask = (attention_mask.sum(1)==0) # (b,t)
            # mask = drooped_mask * update_mask
            # dropped_x = dropped_x + x * mask

        # # do not drop pure text or padding, for option 2!
        # dropped_x = self.text_dropout(x)
        # drooped_mask = (dropped_x.sum(-1)==0) # (b,t)
        # update_mask = (attention_mask.sum(1)==0) # (b,t)
        # mask = drooped_mask * update_mask
        # dropped_x = dropped_x + x * mask

        # option1: (1-a)*x1 + a*x2, a \in (0,1)

        # option2: x1 + a*x2, a \in (-1,1)
        ## if option2, text drop may be conducted here. Not test yet.
        ## if option1, text drop may be conducted in MaskedCrossAttention


        # # Only mask text with vision query 
        # # Only mask text with positive categories
        # if self.cfg.VISION_QUERY.TEXT_DROPOUT > 0. and self.training:
        #     mask=x.new_zeros(x.shape[:2], dtype=torch.bool) # (batch, text)
        #     pure_text_mask=attention_mask.sum(1) # (batch, text)
        #     for i, pos_label_position in enumerate(batched_positive_label_position):
        #         pos_label_position=pos_label_position.to(torch.bool)
        #         for position in pos_label_position:
        #             text_with_vision_query = (pure_text_mask[i, position].sum()!=0)
        #             if (random.random() < self.cfg.VISION_QUERY.TEXT_DROPOUT) and text_with_vision_query:
        #                 mask[i, position] = True
        #     if self.training:
        #         dropped_x = x.clone()
        #     dropped_x[mask] = self.mask_token
        # else:
        #     dropped_x = x

        if self.cfg.VISION_QUERY.ADD_ADAPT_LAYER:
            vision = self.adaptor(vision) + vision

        dropped_x = x
        supported_x = self.attn(x, vision, attention_mask = attention_mask)

        # dropped_x = self.text_dropout(x)
        if self.cfg.VISION_QUERY.FIX_ATTN_GATE != -1.0:
            attn_gate = self.cfg.VISION_QUERY.FIX_ATTN_GATE
        else:
            if self.cfg.VISION_QUERY.CONDITION_GATE:
                if self.cfg.VISION_QUERY.NO_CAT or not (self.cfg.VISION_QUERY.NONLINEAR_GATE):
                    attn_gate = self.attn_gate(supported_x).tanh()
                else:
                    attn_gate = self.attn_gate(torch.cat([supported_x, dropped_x], dim = -1)).tanh()
            else:
                attn_gate = self.attn_gate.tanh()
        if self.cfg.VISION_QUERY.RETURN_ATTN_GATE_VALUE:
            with torch.no_grad():
                self.attn_gate_value = attn_gate.mean().item()

        x = supported_x * attn_gate + dropped_x
        if self.enable_ffn:
            if self.cfg.VISION_QUERY.FIX_ATTN_GATE != -1.0:
                x = self.ff(x) * self.cfg.VISION_QUERY.FIX_ATTN_GATE  + x
            else:
                x = self.ff(x) * self.ff_gate.tanh()  + x
        return x


class PreSelectBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        out_dim = None,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        share_kv=False,
        cfg=None,
    ):
        super().__init__()
        self.image_condition = MaskedCrossAttention(input_dim = dim, output_dim = out_dim, dim_head = dim_head, heads = heads, norm_kv=True, share_kv=share_kv, cfg=cfg, spase_forward=False)
        self.ff = FeedForward(out_dim, mult = ff_mult)

        if dim != out_dim:
            self.res_mapping = nn.Linear(in_features=dim, out_features=out_dim, bias=False)
        else:
            self.res_mapping = nn.Identity()

    def forward(
        self,
        x,                  
    ):
        vision=x['vision'] # vision query tensor - (batch, vision, dim_v)
        image=x['image'] # query images - (batch, image, dim_v)
        # b, s, v, d = vision.shape
        # vision = rearrange(vision, 'b s v d -> b (s v) d')
        vision = self.image_condition(vision, image) + self.res_mapping(vision)
        vision = self.ff(vision)  + vision
        # vision = rearrange(vision, 'b (s v) d -> b s v d', s=s)
        return {'vision': vision, 'image': image}
    
class PreSelectModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        out_dim,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        num_layers = 2,
        share_kv=False,
        cfg=None
    ):
        super().__init__()
        layers = [PreSelectBlock(dim=dim, out_dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, share_kv=share_kv, cfg=cfg) for _ in range(num_layers-1)]
        layers.append(PreSelectBlock(dim=dim, out_dim=out_dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, share_kv=share_kv, cfg=cfg))
        self.layers = nn.Sequential(*layers)
        self.scale=cfg.VISION_QUERY.VISION_SCALE
        self.augment_image_with_query = cfg.VISION_QUERY.AUGMENT_IMAGE_WITH_QUERY 
        if self.augment_image_with_query:
            assert len(self.layers) > 1

    def forward(
        self,
        vision,                  # vision query tensor - (batch, scales, vision, dim_v)
        image,                   # query images - (batch, scales, image, dim_v)
    ):
        vision = vision * self.scale
        image = image * self.scale
        if self.augment_image_with_query:
            x = self.layers[0]({'vision': image, 'image': vision})
            x = {'vision': x['image'], 'image': x['vision']}
            for layer in self.layers[1:]:
                x = layer(x)
            return x
        else:
            x = {'vision': vision, 'image': image}
            return self.layers(x)
    
class QVBertEmbeddings(BertEmbeddings):
    def __init__(self, config, cfg):
        super().__init__(config)
        self.cfg=cfg
        if (self.cfg.VISION_QUERY.TEXT_DROPOUT > 0.) and (cfg.VISION_QUERY.NEW_MASK_TOKEN):
            self.mask_tok_qv_layer = nn.Parameter(torch.randn(config.hidden_size)) # add qv_layer to name only for easier paramter freezing
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        batched_pos_category_map=None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        if (self.cfg.VISION_QUERY.TEXT_DROPOUT > 0.) and (batched_pos_category_map is not None) and (self.cfg.VISION_QUERY.NEW_MASK_TOKEN) and (self.training):
            raise NotImplementedError
            mask_tok_qv_layer = self.mask_tok_qv_layer.to(inputs_embeds.dtype)
            
            # inputs_embeds_ = []
            # for emb, pos_label_position in zip(inputs_embeds, batched_pos_category_map):
            #     pos_label_position=pos_label_position.to(torch.bool)
            #     for position in pos_label_position:
            #         if (random.random() < self.cfg.VISION_QUERY.TEXT_DROPOUT):
            #             emb=torch.scatter(emb, dim=0, index=position.nonzero()[0][..., None], src=mask_tok_qv_layer[None, ...])
            #     inputs_embeds_.append(emb)
            # inputs_embeds = torch.stack(inputs_embeds_)

            inputs_embeds = inputs_embeds.clone()
            for i, pos_label_position in enumerate(batched_pos_category_map):
                pos_label_position=pos_label_position.to(torch.bool)
                for position in pos_label_position:
                    if (random.random() < self.cfg.VISION_QUERY.TEXT_DROPOUT):
                        inputs_embeds[i, position] = mask_tok_qv_layer


        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class QVBertEncoder(BertEncoder):
    '''
    add qv_layer at each bert_layer that deeper than start_qv_layer_index
    '''
    def __init__(self, 
        config, 
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        start_qv_layer_index = 6, # which layer to start fusing vision
        share_kv=False,
        cfg=None,
        ):
        super().__init__(config=config)
        self.start_qv_layer_index = start_qv_layer_index
        num_hidden_layers = config.num_hidden_layers
        assert start_qv_layer_index < num_hidden_layers
        num_qv_layers = num_hidden_layers - start_qv_layer_index

        self.qv_layer = nn.ModuleList([GatedCrossAttentionBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, share_kv=share_kv, cfg=cfg) 
                                       for _ in range(num_qv_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        vision: Optional[torch.Tensor] = None, 
        vision_attention_mask: Optional[torch.Tensor] = None, 
        batched_pos_category_map=None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if i >= self.start_qv_layer_index and exists(vision):
                qv_index = i - self.start_qv_layer_index
                hidden_states = self.qv_layer[qv_index](hidden_states, vision, vision_attention_mask, batched_pos_category_map)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

        
class QVBertModel(BertModel):
    def __init__(self, 
        config,
        dim_t,
        dim_v,
        dim_head_t = 64,
        dim_head_v = 32,
        heads = 8,
        ff_mult = 4,
        num_pre_select_layers = 2,
        share_kv = False,
        cfg=None,
        **kwargs):
        super().__init__(config=config, **kwargs)
        self.cfg=cfg
        self.embeddings = QVBertEmbeddings(config, cfg)
        self.encoder = QVBertEncoder(config=config, dim=dim_t, dim_head=dim_head_t, heads=heads, ff_mult=ff_mult, share_kv=share_kv, cfg=cfg)
        self.pre_select = PreSelectModule(dim=dim_v, out_dim=dim_t, dim_head=dim_head_v, heads=heads,ff_mult=ff_mult, num_layers=num_pre_select_layers, share_kv=share_kv, cfg=cfg)
        # if cfg.VISION_QUERY.NEW_MASK_TOKEN:
        #     self.mask_tok_qv_layer = nn.Parameter(torch.randn(config.hidden_size)) # add qv_layer to name only for easier paramter freezing
    
    def get_gate_value(self):
        attn_gates=[]
        ff_gates=[]
        for blk in self.encoder.qv_layer:
            # try:
            if self.cfg.VISION_QUERY.FIX_ATTN_GATE != -1.0:
                attn_gates.append(torch.tensor([self.cfg.VISION_QUERY.FIX_ATTN_GATE], device=self.embeddings.word_embeddings.weight.device))
                ff_gates.append(torch.tensor([self.cfg.VISION_QUERY.FIX_ATTN_GATE], device=self.embeddings.word_embeddings.weight.device))
            else:
                if not self.cfg.VISION_QUERY.CONDITION_GATE:
                    attn_gates.append(blk.attn_gate)
                else:
                    if self.cfg.VISION_QUERY.RETURN_ATTN_GATE_VALUE:
                        attn_gates.append(blk.attn_gate_value)
                    else:
                        pass
                # except:
                #     pass
                ff_gates.append(blk.ff_gate)
        return {'attn_gates': attn_gates, 'ffn_gates': ff_gates}

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision: Optional[torch.Tensor] = None, # (batch, vision, dim)
        images: Optional[torch.Tensor] = None,  # (batch, image, dim)
        vision_attention_mask: Optional[torch.Tensor] = None,
        batched_pos_category_map = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            batched_pos_category_map = batched_pos_category_map,
        )

        # if self.cfg.VISION_QUERY.TEXT_DROPOUT > 0. and batched_pos_category_map is not None and self.training:
        #     if self.cfg.VISION_QUERY.NEW_MASK_TOKEN:
        #         # embedding_output = embedding_output.clone()
        #         mask_tok_qv_layer = self.mask_tok_qv_layer.to(embedding_output.dtype)
        #         for i, pos_label_position in enumerate(batched_pos_category_map):
        #             pos_label_position=pos_label_position.to(torch.bool)
        #             for position in pos_label_position:
        #                 if (random.random() < self.cfg.VISION_QUERY.TEXT_DROPOUT):
        #                     embedding_output[i, position] = mask_tok_qv_layer

        augmented_vision = None
        if (exists(images) and exists(vision)):
            vision = self.pre_select(vision, images)['vision']
            augmented_vision = vision


        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            vision=vision,
            vision_attention_mask=vision_attention_mask,
            batched_pos_category_map=batched_pos_category_map,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        out=BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        # if self.cfg.VISION_QUERY.GATE_REGULARIZATION:
        out['vision_query_gates'] = self.get_gate_value()
        if self.cfg.VISION_QUERY.QUERY_FUSION:
            out['augmented_vision'] = augmented_vision
            out['vision_attention_mask'] = vision_attention_mask
        return out


    
    
