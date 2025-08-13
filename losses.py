# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, hidden_size, max_seq_len):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_size * 2 * max_seq_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, input1, input2, ground_mask):  # z_s, z_g, ground_mask
        input1 *= ground_mask.unsqueeze(-1)  # broadcast in last dim
        input2 *= ground_mask.unsqueeze(-1)  # broadcast in last dim
        # (batch_size, seq_len * hidden_size)
        input1 = torch.flatten(input1, start_dim=1)  # 从第1维开始展平
        # (batch_size, seq_len * hidden_size)
        input2 = torch.flatten(input2, start_dim=1)

        input = torch.cat((input1, input2), dim=-1)
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output




class NCELoss(nn.Module):
    """Noise-Contrastive Estimation (infoNCE).
    """

    def __init__(self, temperature, num_problems, args):
        super(NCELoss, self).__init__()
        self.cs_criterion = nn.CrossEntropyLoss(reduction="none")
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.num_problems = num_problems
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

    # (batch_size, seq_len, hidden_size) (batch_size, seq_len, hidden_size) (batch_size, seq_len) (batch_size, seq_len)
    def forward(self, batch_sample_one, batch_sample_two, seq_one, seq_two):
        timeline_mask_one = torch.BoolTensor(seq_one.cpu() == self.num_problems).to(self.device)
        batch_sample_one = (batch_sample_one * ~timeline_mask_one.unsqueeze(-1)).sum(1) / ~timeline_mask_one.sum(
            -1).unsqueeze(-1)  # (batch_size, hidden_size)

        timeline_mask_two = torch.BoolTensor(seq_two.cpu() == self.num_problems).to(self.device)
        batch_sample_two = (batch_sample_two * ~timeline_mask_two.unsqueeze(-1)).sum(1) / ~timeline_mask_two.sum(
            -1).unsqueeze(-1)  # (batch_size, hidden_size)

        # (batch_size, batch_size)
        sim11 = self.cossim(batch_sample_one.unsqueeze(
            1), batch_sample_one.unsqueeze(0)) / self.temperature
        # (batch_size, batch_size)
        sim22 = self.cossim(batch_sample_two.unsqueeze(
            1), batch_sample_two.unsqueeze(0)) / self.temperature
        # (batch_size, batch_size)
        sim12 = self.cossim(batch_sample_one.unsqueeze(
            1), batch_sample_two.unsqueeze(0)) / self.temperature

        d = sim12.shape[-1]
        # (batch_size, batch_size)，values on the diagonal are set to -inf
        sim11[range(d), range(d)] = float("-inf")
        # (batch_size, batch_size)，values on the diagonal are set to -inf
        sim22[range(d), range(d)] = float("-inf")
        # (batch_size, 2 * batch_size)
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        # (batch_size, 2 * batch_size)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        # (2 * batch_size, 2 * batch_size)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long,
                              device=logits.device)  # (2 * batch_size, )
        nce_loss = self.cs_criterion(logits, labels)  # (2 * batch_size, )
        return nce_loss


class HingeLoss(nn.Module):
    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, pos, neg):
        pos = torch.sigmoid(pos)
        neg = torch.sigmoid(neg)
        gamma = torch.tensor(self.margin).to(pos.device)
        return F.relu(gamma - pos + neg)


class JSDLoss(torch.nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, pos, neg):
        pos = -F.softplus(-pos)
        neg = F.softplus(neg)
        return neg - pos
