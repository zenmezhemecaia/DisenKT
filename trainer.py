# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.disenKT_model import DisenKT
from model import config
from losses import Discriminator, NCELoss, HingeLoss, JSDLoss
from sklearn import metrics
from sklearn.metrics import mean_squared_error


class Trainer(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def train_batch(self, *args, **kwargs):
        raise NotImplementedError

    def test_batch(self, *args, **kwargs):
        raise NotImplementedError


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


class ModelTrainer(Trainer):
    def __init__(self, args, num_problems, max_seq_len):
        self.args = args
        self.method = args.method
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        if self.method == "DisenKT":
            self.model = DisenKT(num_problems, args).to(self.device)
            self.z_s, self.z_g = [None], [None]  # 单元素列表
            self.discri = Discriminator(config.hidden_size, max_seq_len).to(self.device)

        self.bce_criterion = nn.BCEWithLogitsLoss(reduction="none").to(self.device)
        self.cl_criterion = NCELoss(temperature=0.7, num_problems=num_problems, args=args).to(self.device)
        self.hinge_criterion = HingeLoss(margin=0.3).to(self.device)
        self.jsd_criterion = JSDLoss().to(self.device)
        if args.method == "DisenKT":
            self.params = list(self.model.parameters()) + list(self.discri.parameters())
        self.optimizer = self.get_optimizer(args.optimizer, self.params, args.lr)
        self.step = 0

    def train_batch(self, input_seqs, adj, num_problems, args, global_params=None):
        """Trains the model for one batch.

        Args:
            input_seqs: Input user sequences.
            adj: Adjacency matrix of the local graph.
            num_problems: Number of problems in the current domain.
            args: Other arguments for training.
        """
        self.optimizer.zero_grad()
        self.model.graph_convolution(adj)
        input_seqs = [torch.LongTensor(x).to(self.device) for x in input_seqs]
        problem_seqs, interact_seqs, ground_truths, ground_mask, neg_problems, neg_interacts, aug_problems, aug_interacts = input_seqs

        result, result_exclusive, \
            mu_s, logvar_s, self.z_s[0], mu_e, logvar_e, z_e, neg_z_s, aug_z_e = self.model(problem_seqs, interact_seqs,
                                                                                            neg_seqs=neg_problems,
                                                                                            neg_interact=neg_interacts,
                                                                                            aug_seqs=aug_problems,
                                                                                            aug_interact=aug_interacts)
        self.z_s[0] *= ground_mask.unsqueeze(-1)

        loss = self.disenKT_loss_fn(problem_seqs, aug_problems, result, result_exclusive, mu_s,
                                    logvar_s, mu_e, logvar_e,
                                    ground_truths, self.z_s[0], self.z_g[0],
                                    z_e, neg_z_s, aug_z_e, ground_mask, self.step)
        loss.backward()
        self.optimizer.step()
        self.step += 1
        return loss.item()

    def test_batch(self, input_seqs, save_path=None):
        """Tests the model for one batch.
        Args:
            input_seqs: Input user sequences.
        """
        input_seqs = [torch.LongTensor(x).to(self.device) for x in input_seqs]
        # seq: (batch_size, seq_len), ground_truth: (batch_size, seq_len)
        problem_seqs, interact_seqs, ground_truths, ground_mask = input_seqs
        # result: (batch_size, seq_len, 1)
        result, z_s, z_e = self.model(problem_seqs, interact_seqs)
        # preds: (batch_size, seq_len)
        preds = torch.sigmoid(result).squeeze(-1)

        flattened_preds = preds[ground_mask.bool()].detach().cpu().numpy()
        flattened_truths = ground_truths[ground_mask.bool()].detach().cpu().numpy()

        accuracy = compute_accuracy(flattened_truths, flattened_preds)
        auc = compute_auc(flattened_truths, flattened_preds)
        rmse = mean_squared_error(flattened_truths, flattened_preds, squared=False)

        z_s_np = z_s[ground_mask.bool()].detach().cpu().numpy()
        z_e_np = z_e[ground_mask.bool()].detach().cpu().numpy()

        return preds, accuracy, auc, rmse, z_s_np, z_e_np

    def disenKT_loss_fn(self, seqs, aug_seqs, result, result_exclusive, mu_s, logvar_s,
                        mu_e, logvar_e, ground, z_s, z_g, z_e, neg_z_s,
                        aug_z_e, ground_mask, step):
        """Overall loss function of DisenKT."""
        def sim_loss_fn(self, z_s, z_g, neg_z_s, ground_mask):
            pos = self.discri(z_s, z_g, ground_mask)
            neg = self.discri(neg_z_s, z_g, ground_mask)
            sim_loss = self.hinge_criterion(pos, neg)
            sim_loss = sim_loss.mean()
            return sim_loss

        recons_loss = self.bce_criterion(result.view(-1), ground.view(-1).float())
        recons_loss = (recons_loss * ground_mask.reshape(-1)).mean()

        recons_loss_exclusive = self.bce_criterion(result_exclusive.view(-1),
                                                   ground.view(-1).float())  # (batch_size * seq_len, )
        recons_loss_exclusive = (recons_loss_exclusive * ground_mask.reshape(-1)).mean()

        kld_loss_s = -0.5 * \
                     torch.sum(1 + logvar_s - mu_s ** 2 -
                               logvar_s.exp(), dim=-1).reshape(-1)
        kld_loss_s = (kld_loss_s * (ground_mask.reshape(-1))).mean()

        kld_loss_e = -0.5 * \
                     torch.sum(1 + logvar_e - mu_e ** 2 -
                               logvar_e.exp(), dim=-1).reshape(-1)
        kld_loss_e = (kld_loss_e * (ground_mask.reshape(-1))).mean()

        # If it is the first training round
        if z_g is not None:
            sim_loss = sim_loss_fn(self, z_s, z_g, neg_z_s, ground_mask)
        else:
            sim_loss = 0

        kld_weight = self.kl_anneal_function(self.args.anneal_cap, step, self.args.total_annealing_step)
        alpha, lam, alpha, gamma = self.args.alpha, self.args.lam, self.args.alpha, 0.4

        user_representation1 = z_e
        user_representation2 = aug_z_e
        contrastive_loss = self.cl_criterion(user_representation1, user_representation2, seqs, aug_seqs)
        contrastive_loss = contrastive_loss.mean()
        loss = alpha * (recons_loss + kld_weight * kld_loss_s + kld_weight
                        * kld_loss_e) \
               + lam * sim_loss \
               + alpha * recons_loss_exclusive \
               + gamma * contrastive_loss
        return loss

    def kl_anneal_function(self, anneal_cap, step, total_annealing_step):
        """
        step: increment by 1 for every forward-backward step.
        total annealing steps: pre-fixed parameter control the speed of
        anealing.
        """
        return min(anneal_cap, step / total_annealing_step)

    def get_optimizer(self, name, parameters, lr, l2=0):
        if name == "adam":
            return torch.optim.AdamW(parameters, weight_decay=l2, lr=lr, betas=(0.9, 0.98))

    def get_loss_value(self, loss):
        return loss.item() if isinstance(loss, torch.Tensor) else loss
