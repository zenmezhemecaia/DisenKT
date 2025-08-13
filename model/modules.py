import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from . import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


###################################
#       Feed Forward Module       #
###################################

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        if isinstance(config.hidden_act, str):
            self.feedforward_act_fn = ACT2FN[config.hidden_act]
        else:
            self.feedforward_act_fn = config.hidden_act
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.feedforward_act_fn(
            self.dropout1(self.conv1(inputs.transpose(-1, -2))))
        outputs = self.dropout2(self.conv2(outputs))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class HierarchicalAttention(nn.Module):
    def __init__(self, num_problems, args):
        super(HierarchicalAttention, self).__init__()
        self.num_problems = num_problems
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

        self.self_attention_layers = nn.ModuleList([
            TransformerLayer(d_model=config.hidden_size,
                             d_feature=config.hidden_size // config.num_heads,
                             d_ff=config.hidden_size * 4,
                             n_heads=config.num_heads,
                             dropout=config.dropout_rate,
                             kq_same=True)
            for _ in range(config.num_layers)
        ])

        self.cross_attention_layers = nn.ModuleList([
            TransformerLayer(d_model=config.hidden_size,
                             d_feature=config.hidden_size // config.num_heads,
                             d_ff=config.hidden_size * 4,
                             n_heads=config.num_heads,
                             dropout=config.dropout_rate,
                             kq_same=True)
            for _ in range(config.num_layers * 2)
        ])

        self.last_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-8)

    def forward(self, q_seqs, qa_seqs, seqs_data):
        batch_size, seq_len, _ = q_seqs.shape
        timeline_mask = (seqs_data == self.num_problems).to(self.device)
        q_seqs = q_seqs.masked_fill(timeline_mask.unsqueeze(-1), 0)
        qa_seqs = qa_seqs.masked_fill(timeline_mask.unsqueeze(-1), 0)

        y = qa_seqs
        for layer in self.self_attention_layers:
            y = layer(mask=1, query=y, key=y, values=y)


        x = q_seqs
        flag_first = True
        for layer in self.cross_attention_layers:
            if flag_first:
                x = layer(mask=1, query=x, key=x, values=x, apply_pos=False)
                flag_first = False
            else:
                x = layer(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True

        x = self.last_layernorm(x)
        return x


###################################
#       Transformer Layer         #
###################################

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super(TransformerLayer, self).__init__()
        kq_same = (kq_same == 1)
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
          mask: 1 means peeking is allowed (current timestep visible),
                0 means peeking is not allowed (current timestep masked, only past visible)
          query, key, values: tensors of shape (batch_size, seq_len, d_model)
          apply_pos: whether to apply the feed-forward layer (can be turned off if needed)
        """
        seqlen, batch_size = query.size(1), query.size(0)
        k_val = 1 if mask == 1 else -1
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen), dtype=np.uint8), k=k_val)
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        if mask == 0:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)
        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


###################################
#       MultiHead Attention       #
###################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)
        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad, gammas)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output


###################################
#       Attention Function        #
###################################

def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (bs, h, seq_len, seq_len)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)
        dist_scores = torch.clamp((disttotal_scores - distcum_scores) * position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)
    total_effect = torch.clamp(((dist_scores * gamma).exp()), min=1e-5, max=1e5)
    scores = scores * total_effect
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


###################################
#       Positional Embedding      #
###################################

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(LearnablePositionalEmbedding, self).__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(CosinePositionalEmbedding, self).__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]




###################################
#         Dimension Enum          #
###################################

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
