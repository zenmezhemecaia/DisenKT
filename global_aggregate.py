# -*- coding: utf-8 -*-
import math
import numpy as np


class Aggregator(object):
    def __init__(self, args, init_global_params):
        self.args = args
        self.global_params = init_global_params
        if args.method == "DisenKT":
            self.global_reps = None

    def aggregate_params(self, domains, random_dids):
        """Sums up parameters of models shared by all active domains at each
        epoch.

        Args:
            domains: A list of domains instances.
            random_cids: Randomly selected domain ID in each training round.
        """
        # Record the model parameter aggregation results of each branch
        # separately
        num_branchs = len(self.global_params)
        for branch_idx in range(num_branchs):
            domain_params_sum = None
            for d_id in random_dids:
                # Obtain current domain's parameters
                current_domain_params = domains[d_id].get_params()[branch_idx]
                # Sum it up with weights
                if domain_params_sum is None:
                    domain_params_sum = dict((key, value
                                              * domains[d_id].train_weight)
                                             for key, value
                                             in current_domain_params.items())
                else:
                    for key in domain_params_sum.keys():
                        domain_params_sum[key] += domains[d_id].train_weight \
                            * current_domain_params[key]
            self.global_params[branch_idx] = domain_params_sum

    def aggregate_reps(self, domains, random_dids):
        """Sums up representations of user sequences shared by all active
        domains at each epoch.

        Args:
            domains: A list of domains instances.
            random_cids: Randomly selected domain ID in each training round.
        """
        # Record the user sequence aggregation results of each branch
        # separately
        domain_reps_sum = None
        for d_id in random_dids:
            # Obtain current domain's user sequence representations
            current_domain_reps = domains[d_id].get_reps_shared()
            # Sum it up with weights
            if domain_reps_sum is None:
                domain_reps_sum = current_domain_reps * \
                    domains[d_id].train_weight
            else:
                domain_reps_sum += domains[d_id].train_weight * \
                    current_domain_reps
        self.global_reps = domain_reps_sum

    def choose_domains(self, n_domains, ratio=1.0):
        """Randomly chooses some domains.
        """
        choose_num = math.ceil(n_domains * ratio)
        return np.random.permutation(n_domains)[:choose_num]

    def get_global_params(self):
        """Returns a reference to the parameters of the global model.
        """
        return self.global_params

    def get_global_reps(self):
        """Returns a reference to the parameters of the global model.
        """
        return self.global_reps
