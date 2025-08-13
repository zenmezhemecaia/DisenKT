# -*- coding: utf-8 -*-
"""Customized dataset.
"""
import math
import torch
import random
import os
import copy
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset


class KTDataset(Dataset):
    data_dir = "datasets"
    prep_dir = "prep_data"

    def __init__(self, domain, model="DisenKT", mode="train", max_seq_len=50,
                 load_prep=True):
        self.domain_name = domain
        self.model = model
        self.mode = mode  # train, test, valid
        self.dataset_dir = os.path.join(self.data_dir, self.domain_name)
        self.user_ids, self.problems, self.corrects, self.num_problems = self.read_data(
            self.dataset_dir)
        self.max_seq_len = max_seq_len
        self.prep_data = self.preprocess(self.problems, self.corrects, self.dataset_dir, load_prep)

    def read_data(self, dataset_dir):
        with open(os.path.join(dataset_dir, "problem_num.txt"),
                  "rt", encoding="utf-8") as infile:
            num_problems = int(infile.readline())


        with open(os.path.join(self.data_dir, self.domain_name,
                               "%s_data.txt" % self.mode), "rt",
                  encoding="utf-8") as infile:
            user_ids, problems, corrects = [], [], []
            lines = infile.readlines()
            for i in range(0, len(lines), 3):
                user_id = int(lines[i].strip().split("\t")[0])
                user_ids.append(user_id)

                problem_line = lines[i + 2].strip().split("\t")[1:]
                problems.append([int(p) for problem in problem_line for p in problem.split()])

                correct_line = lines[i + 1].strip().split("\t")[1:]
                corrects.append([int(c) for correct in correct_line for c in correct.split()])


        return user_ids, problems, corrects, num_problems

    def preprocess(self, problems, corrects, dataset_dir, load_prep):
        if not os.path.exists(os.path.join(dataset_dir, self.prep_dir)):
            os.makedirs(os.path.join(dataset_dir, self.prep_dir))

        self.prep_data_path = os.path.join(
            dataset_dir, self.prep_dir, "%s_%s_data.pkl" % (self.model,
                                                            self.mode))
        if os.path.exists(self.prep_data_path) and load_prep:
            with open(os.path.join(self.prep_data_path), "rb") as infile:
                prep_data = pickle.load(infile)

        else:
            prep_data = self.preprocess_disenkt(problems, corrects, mode=self.mode)
            with open(self.prep_data_path, "wb") as infile:
                pickle.dump(prep_data, infile)

        return prep_data

    def preprocess_disenkt(self, problems, corrects, mode="train"):
        prep_data = []
        for i in range(len(problems)):
            temp = []
            problems_input = problems[i]
            ground_truths = corrects[i]
            interact_input = []
            for i in range(len(problems_input)):
                interact_index = int(problems_input[i]) + int(ground_truths[i]) * self.num_problems
                interact_input.append(interact_index)

            if mode == "train":
                js_neg_seq_problems = copy.deepcopy(problems_input)
                js_neg_seq_interact = copy.deepcopy(interact_input)
                for i in range(len(js_neg_seq_interact)):
                    if random.random() < 0.6:
                        neg_correct = 1 - ground_truths[i]
                        js_neg_seq_interact[i] = int(problems_input[i]) + neg_correct * self.num_problems

                contrast_aug_seq_problems = copy.deepcopy(problems_input)
                contrast_aug_seq_interact = copy.deepcopy(interact_input)

                combined_seq = list(zip(contrast_aug_seq_problems, contrast_aug_seq_interact))
                random.shuffle(combined_seq)
                contrast_aug_seq_problems, contrast_aug_seq_interact = map(list, zip(*combined_seq))

            pad_len = self.max_seq_len - len(problems_input)
            problems_input = [self.num_problems] * pad_len + problems_input
            interact_input = [self.num_problems * 2] * pad_len + interact_input

            temp.append(problems_input)
            temp.append(interact_input)

            if mode == "train":
                pad_len1 = self.max_seq_len - len(js_neg_seq_problems)
                pad_len2 = self.max_seq_len - len(contrast_aug_seq_problems)

                ground_mask = [0] * pad_len + [1] * len(ground_truths)
                ground_truths = [2] * pad_len + ground_truths

                js_neg_seq_problems = [self.num_problems] * pad_len1 + js_neg_seq_problems
                js_neg_seq_interact = [self.num_problems * 2] * pad_len1 + js_neg_seq_interact

                contrast_aug_seq_problems = [self.num_problems] * pad_len2 + contrast_aug_seq_problems
                contrast_aug_seq_interact = [self.num_problems * 2] * pad_len2 + contrast_aug_seq_interact

                temp.append(ground_truths)
                temp.append(ground_mask)
                temp.append(js_neg_seq_problems)
                temp.append(js_neg_seq_interact)
                temp.append(contrast_aug_seq_problems)
                temp.append(contrast_aug_seq_interact)
            else:
                # valid
                ground_mask = [0] * pad_len + [1] * len(ground_truths)
                ground_truths = [2] * pad_len + ground_truths
                temp.append(ground_truths)
                temp.append(ground_mask)

            prep_data.append(temp)

        return prep_data

    def __getitem__(self, idx):
        user_ids = self.user_ids[idx]
        data = self.prep_data[idx]
        return user_ids, data


    def __len__(self):
        return len(self.prep_data)

    def __setitem__(self, idx, value):
        """To support shuffle operation.
        """
        self.user_ids[idx] = value[0]
        self.prep_data[idx] = value[1]

    def __add__(self, other):
        """To support concatenation operation.
        """
        user_ids, prep_data = other
        self.user_ids += user_ids
        self.prep_data += prep_data
        return self
