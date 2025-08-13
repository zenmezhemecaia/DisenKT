# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import os


def normalize(mx):
    """Row-normalize sparse matrix.
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class LocalGraph(object):
    """A local graph data structure class reading training data of a certain
    domain from ".txt" files, and preprocess it into a local graph.
    """
    data_dir = "datasets"

    def __init__(self, args, domain, num_problems):
        self.args = args
        self.dataset_dir = os.path.join(self.data_dir, domain)
        self.raw_data = self.read_train_data(self.dataset_dir)
        self.num_problems = num_problems
        self.adj = self.preprocess(self.raw_data)

    def read_train_data(self, dataset_dir):
        with open(os.path.join(dataset_dir, "train_data.txt"),
                  "rt", encoding="utf-8") as infile:
            train_data = []
            for idx, line in enumerate(infile.readlines()):
                if idx % 3 != 2:
                    continue
                problems = []
                line = line.strip().split("\t")
                for problem in line[1:]:  # Start from index 1 to exclude user ID
                    split_problems = problem.split()
                    for p in split_problems:
                        problems.append(int(p))
                train_data.append(problems)
        return train_data

    def preprocess(self, data):
        VV_edges = []
        for problems in data:
            source = -1
            for problem in problems:
                if source != -1:
                    VV_edges.append([source, problem])
                source = problem

        # VV_edges.shape[0] is the number of edges,
        # VV_edges[:, 0] and VV_edges[:, 1] are the source and target nodes of each edge, respectively
        VV_edges = np.array(VV_edges)
        adj = sp.coo_matrix((np.ones(VV_edges.shape[0]), (VV_edges[:, 0],
                                                          VV_edges[:, 1])),
                            shape=(self.num_problems + 1, self.num_problems + 1),
                            dtype=np.float32)

        adj = normalize(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        return adj
