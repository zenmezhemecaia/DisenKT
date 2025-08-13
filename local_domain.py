# -*- coding: utf-8 -*-
import os
import gc
import copy
import logging
import numpy as np
import torch
from dataloader import KTDataloader
from tqdm import tqdm
import sys


class LocalDomain:
    def __init__(self, model_fn, domain_id, args, adj, train_dataset, valid_dataset, test_dataset):
        self.num_problems = train_dataset.num_problems
        self.domain_name = train_dataset.domain_name
        self.max_seq_len = args.max_seq_len
        self.trainer = model_fn(args, self.num_problems, self.max_seq_len)
        self.model = self.trainer.model
        self.method = args.method
        self.checkpoint_dir = args.checkpoint_dir
        self.model_id = args.id if len(args.id) > 1 else "0" + args.id
        if args.method == "DisenKT":
            self.z_s = self.trainer.z_s
            self.z_g = self.trainer.z_g
        self.domain_id = domain_id
        self.args = args
        self.adj = adj

        self.train_dataloader = KTDataloader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_dataloader = KTDataloader(
            valid_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_dataloader = KTDataloader(
            test_dataset, batch_size=args.batch_size, shuffle=False)

        # Compute the number of samples for each domain
        self.n_samples_train = len(train_dataset)
        self.n_samples_valid = len(valid_dataset)
        self.n_samples_test = len(test_dataset)

        # The aggretation weight
        self.train_pop, self.valid_weight, self.test_weight = 0.0, 0.0, 0.0

    def train_epoch(self, round, args, global_params=None):
        """Trains one domain with its own training data for one epoch.

        Args:
            round: Training round.
            args: Other arguments for training.
        """
        self.trainer.model.train()
        for _ in range(args.local_epoch):
            loss = 0
            step = 0
            with tqdm(self.train_dataloader, file=sys.stdout) as pbar:
                for _, problems in pbar:
                    if "DisenKT" in args.method:
                        batch_loss = self.trainer.train_batch(
                            problems, self.adj, self.num_problems, args,
                            global_params=global_params)
                    else:
                        batch_loss = self.trainer.train_batch(problems, self.adj, self.num_problems, args)
                    pbar.set_postfix({"Loss": f"{batch_loss:.3f}"})
                    loss += batch_loss
                    step += 1

                gc.collect()
        print("Epoch {}/{} - domain {} -  Training Loss: {:.3f}".format(
            round, args.epochs, self.domain_id, loss / step))
        return self.n_samples_train

    def evaluation(self, round, mode="valid"):
        """Evaluates one domain with its own valid data for one epoch.
        Args:
            mode: `valid`
        """
        if mode == "valid":
            dataloader = self.valid_dataloader

        if mode == "test":
            dataloader = self.test_dataloader

        self.trainer.model.eval()
        self.trainer.model.graph_convolution(self.adj)

        total_acc = 0
        total_rmse = 0
        total_auc = 0
        total_batches = 0

        for _, problems in tqdm(dataloader):
            preds, acc, auc, rmse, z_s_np, z_e_np = self.trainer.test_batch(problems)
            total_acc += acc
            total_rmse += rmse
            total_auc += auc
            total_batches += 1
            # print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, RMSE: {rmse:.4f}")
        avg_acc = total_acc / total_batches if total_batches > 0 else 0
        avg_rmse = total_rmse / total_batches if total_batches > 0 else 0
        avg_auc = total_auc / total_batches if total_batches > 0 else 0
        print(
            f"domain {self.domain_id}, Total Accuracy: {avg_acc:.4f}, Total AUC: {avg_auc:.4f}, Total RMSE: {avg_rmse:.4f}")
        gc.collect()

        return {"AUC": avg_auc, "ACC": avg_acc, "RMSE": avg_rmse}

    def get_params(self):
        if self.method == "DisenKT":
            return copy.deepcopy([self.model.encoder_s.state_dict()])

    def get_reps_shared(self):
        """Returns the user sequence representations that need to be shared
        between domains.
        """
        assert (self.method == "DisenKT")
        return copy.deepcopy(self.z_s[0].detach())

    def set_global_params(self, global_params):
        """Assign the local shared model parameters with global model
        parameters.
        """
        if self.method == "DisenKT":
            self.model.encoder_s.load_state_dict(global_params[0])

    def set_global_reps(self, global_rep):
        """Copy global user sequence representations to local.
        """
        assert (self.method == "DisenKT")
        self.z_g[0] = copy.deepcopy(global_rep)

    def save_params(self):
        method_ckpt_path = os.path.join(self.checkpoint_dir,
                                        "domain_" +
                                        "".join([domain[0]
                                                 for domain
                                                 in self.args.domains]),
                                        self.method + "_" + self.model_id)
        if not os.path.exists(method_ckpt_path):
            print("Directory {} do not exist; creating...".format(method_ckpt_path))
            os.makedirs(method_ckpt_path)
        ckpt_filename = os.path.join(method_ckpt_path, "domain%d.pt" % self.domain_id)
        params = self.trainer.model.state_dict()
        try:
            torch.save(params, ckpt_filename)
            print("Model saved to {}".format(ckpt_filename))
        except IOError:
            print("[ Warning: Saving failed... continuing anyway. ]")

    def load_params(self):
        ckpt_filename = os.path.join(self.checkpoint_dir,
                                     "domain_" +
                                     "".join([domain[0]
                                              for domain in self.args.domains]),
                                     self.method + "_" + self.model_id,
                                     "domain%d.pt" % self.domain_id)
        try:
            checkpoint = torch.load(ckpt_filename)
        except IOError:
            print("[ Fail: Cannot load model from {}. ]".format(ckpt_filename))
            exit(1)
        if self.trainer.model is not None:
            self.trainer.model.load_state_dict(checkpoint)



