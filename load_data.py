# -*- coding: utf-8 -*-
import numpy as np
import torch
from mydataset import KTDataset
from local_graph import LocalGraph


def load_dataset(args):
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for domain in args.domains:
        model = "DisenKT"
        train_dataset = KTDataset(
            domain, model, mode="train", max_seq_len=args.max_seq_len,  # user_ids and sessions
            load_prep=args.load_prep)
        valid_dataset = KTDataset(
            domain, model, mode="valid", max_seq_len=args.max_seq_len,
            load_prep=args.load_prep)
        test_dataset = KTDataset(
            domain, model, mode="test", max_seq_len=args.max_seq_len,
            load_prep=args.load_prep)

        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)
        test_datasets.append(test_dataset)

    adjs = []  # 邻接矩阵
    for train_dataset, domain in zip(train_datasets, args.domains):
        local_graph = LocalGraph(args, domain, train_dataset.num_problems)
        adjs.append(local_graph.adj)
        print("%s graph loaded!" % domain)

    if args.cuda:
        torch.cuda.empty_cache()
        device = "cuda:%s" % args.gpu
    else:
        device = "cpu"

    for idx, adj in enumerate(adjs):
        adjs[idx] = adj.to(device)

    return train_datasets, valid_datasets, test_datasets, adjs




def init_domains_weight(domains):
    domain_n_samples_train = [domain.n_samples_train for domain in domains]

    samples_sum_train = np.sum(domain_n_samples_train)
    for domain in domains:
        domain.train_weight = domain.n_samples_train / samples_sum_train
        domain.valid_weight = 1 / len(domains)