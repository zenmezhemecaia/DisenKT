# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from . import config
from .gnn import GCNLayer
from .modules import HierarchicalAttention


class Encoder(nn.Module):
    def __init__(self, num_problems, args):
        super(Encoder, self).__init__()
        self.encoder_mu = HierarchicalAttention(num_problems, args)
        self.encoder_logvar = HierarchicalAttention(num_problems, args)

    def forward(self, q_embed_data, qa_embed_data, seqs_data):
        """
        seqs: (batch_size, seq_len, hidden_size)
        seqs_data: (batch_size, seq_len)
        """
        mu = self.encoder_mu(q_embed_data, qa_embed_data, seqs_data)
        logvar = self.encoder_logvar(q_embed_data, qa_embed_data, seqs_data)
        return mu, logvar


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPEncoder, self).__init__()
        self.encoder_mu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, prob_emb, interact_emb, *args, **kwargs):
        x = torch.cat([prob_emb, interact_emb], dim=-1)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar


class DisenKT(nn.Module):
    def __init__(self, num_problems, args):
        super(DisenKT, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.problem_emb_s = nn.Embedding(
            num_problems + 1, config.hidden_size, padding_idx=num_problems)  # share_emb
        self.problem_emb_e = nn.Embedding(
            num_problems + 1, config.hidden_size, padding_idx=num_problems)  # exclusive_emb
        self.interact_emb_e = nn.Embedding(
            num_problems * 2 + 1, config.interact_size, padding_idx=num_problems * 2)
        self.interact_emb_s = nn.Embedding(
            num_problems * 2 + 1, config.interact_size, padding_idx=num_problems * 2)
        self.pos_emb_s = nn.Embedding(args.max_seq_len, config.hidden_size)
        self.pos_emb_e = nn.Embedding(args.max_seq_len, config.hidden_size)
        self.GNN_encoder_s = GCNLayer(args)
        self.GNN_encoder_e = GCNLayer(args)

        self.encoder_s = Encoder(num_problems, args)
        self.encoder_e = Encoder(num_problems, args)
        self.linear = nn.Linear(config.hidden_size, 1)

        self.LayerNorm_s = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm_e = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.mlp_1 = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        )

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]  # (2, 3)->[2, 3, -1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def graph_convolution(self, adj):
        self.problem_index_s = torch.arange(0, self.problem_emb_s.num_embeddings, 1).to(self.device)
        self.problem_index_e = torch.arange(0, self.problem_emb_e.num_embeddings, 1).to(self.device)
        problem_embs_s = self.my_index_select_embedding(self.problem_emb_s,
                                                        self.problem_index_s)  # (num_problems, embedding_dim)
        problem_embs_e = self.my_index_select_embedding(self.problem_emb_e, self.problem_index_e)
        self.problem_graph_embs_s = self.GNN_encoder_s(problem_embs_s, adj)
        self.problem_graph_embs_e = self.GNN_encoder_e(problem_embs_e, adj)

    def get_position_ids(self, seqs):
        seq_length = seqs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=seqs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seqs)
        return position_ids

    def add_position_embedding_s(self, seqs, seq_embeddings):
        position_ids = self.get_position_ids(seqs)
        position_embeddings = self.pos_emb_s(position_ids)
        seq_embeddings += position_embeddings
        seq_embeddings = self.LayerNorm_s(seq_embeddings)
        seq_embeddings = self.dropout(seq_embeddings)
        return seq_embeddings  # (batch_size, seq_len, hidden_size)

    def add_position_embedding_e(self, seqs, seq_embeddings):
        position_ids = self.get_position_ids(seqs)
        position_embeddings = self.pos_emb_e(position_ids)
        seq_embeddings += position_embeddings
        seq_embeddings = self.LayerNorm_e(seq_embeddings)
        seq_embeddings = self.dropout(seq_embeddings)
        return seq_embeddings

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            res = mu + std * eps
        else:
            res = mu
        return res

    def forward(self, prob_seqs, interact_seqs, neg_seqs=None, neg_interact=None, aug_seqs=None, aug_interact=None):
        # Here we need to select the embeddings of problems appearing in the sequence
        # input_seqs(batch_size, seq_len), prob_graph_embs(num_problems,emb_dim=256)
        prob_emb_s = self.my_index_select(self.problem_graph_embs_s, prob_seqs) + self.problem_emb_s(prob_seqs)  # q_emb
        interact_emb_s = self.interact_emb_s(interact_seqs)  # qa_emb

        prob_emb_e = self.my_index_select(self.problem_graph_embs_e, prob_seqs) + self.problem_emb_e(prob_seqs)
        interact_emb_e = self.interact_emb_e(interact_seqs)

        prob_emb_s *= self.problem_emb_s.embedding_dim ** 0.5
        prob_emb_e *= self.problem_emb_e.embedding_dim ** 0.5
        prob_emb_s = self.add_position_embedding_s(prob_seqs, prob_emb_s)
        prob_emb_e = self.add_position_embedding_e(prob_seqs, prob_emb_e)

        interact_emb_s *= self.interact_emb_s.embedding_dim ** 0.5
        interact_emb_e *= self.interact_emb_e.embedding_dim ** 0.5
        interact_emb_s = self.add_position_embedding_s(interact_seqs, interact_emb_s)
        interact_emb_e = self.add_position_embedding_e(interact_seqs, interact_emb_e)

        if self.training:
            neg_interact_emb = self.interact_emb_s(neg_interact)
            neg_prob_emb = self.my_index_select(self.problem_graph_embs_s, neg_seqs) + self.problem_emb_s(neg_seqs)

            aug_interact_emb = self.interact_emb_e(aug_interact)
            aug_prob_emb = self.my_index_select(self.problem_graph_embs_e, aug_seqs) + self.problem_emb_e(aug_seqs)

            neg_prob_emb *= self.problem_emb_s.embedding_dim ** 0.5
            aug_prob_emb *= self.problem_emb_e.embedding_dim ** 0.5
            neg_prob_emb = self.add_position_embedding_s(neg_seqs, neg_prob_emb)
            aug_prob_emb = self.add_position_embedding_e(aug_seqs, aug_prob_emb)

            neg_interact_emb *= self.interact_emb_s.embedding_dim ** 0.5
            aug_interact_emb *= self.interact_emb_e.embedding_dim ** 0.5
            neg_interact_emb = self.add_position_embedding_s(neg_interact, neg_interact_emb)
            aug_interact_emb = self.add_position_embedding_e(aug_interact, aug_interact_emb)

        mu_s, logvar_s = self.encoder_s(prob_emb_s, interact_emb_s, prob_seqs)
        z_s = self.reparameterization(mu_s, logvar_s)

        mu_e, logvar_e = self.encoder_e(prob_emb_e, interact_emb_e, prob_seqs)
        z_e = self.reparameterization(mu_e, logvar_e)

        if self.training:
            neg_mu_s, neg_logvar_s = self.encoder_s(neg_prob_emb, neg_interact_emb, neg_seqs)
            neg_z_s = self.reparameterization(neg_mu_s, neg_logvar_s)

            aug_mu_e, aug_logvar_e = self.encoder_e(aug_prob_emb, aug_interact_emb, aug_seqs)
            aug_z_e = self.reparameterization(aug_mu_e, aug_logvar_e)

        hidden_representation = self.mlp_1(torch.cat([z_s, z_e], dim=-1))
        result = self.linear(hidden_representation)

        hidden_representation_exclusive = self.mlp_2(z_e)
        result_exclusive = self.linear(hidden_representation_exclusive)

        if self.training:
            return result, result_exclusive, \
                mu_s, logvar_s, z_s, mu_e, logvar_e, z_e, neg_z_s, aug_z_e
        else:
            return result, z_s, z_e
