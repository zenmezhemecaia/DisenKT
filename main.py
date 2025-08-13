# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import argparse
import torch
from load_data import load_dataset, init_domains_weight
from local_domain import LocalDomain
from trainer import ModelTrainer
from global_aggregate import Aggregator
from run import run_model
from utils import save_config, ensure_dir
import datetime
import logging


def arg_parse():
    parser = argparse.ArgumentParser()

    # Dataset part
    parser.add_argument("--domains", nargs="*", default=["kdd_source","kdd_target"], help="List of domains")
    parser.add_argument("--load_prep", dest="load_prep", action="store_true",
                        default=False,
                        help="Whether need to load preprocessed the data. If "
                             "you want to load preprocessed data, add it")
    parser.add_argument("--max_seq_len", type=int,
                        default=50, help="maximum sequence length")

    # Training part
    parser.add_argument("--method", type=str, default="DisenKT")
    parser.add_argument("--log_dir", type=str,
                        default="log", help="directory of logs")
    parser.add_argument("--cuda", type=bool, default=True, help="Whether to use CUDA")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of total training iterations.")
    parser.add_argument("--local_epoch", type=int, default=1,
                        help="Number of local training epochs.")
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Applies to adam")  # 0.001
    parser.add_argument("--lr_decay", type=float, default=0.9,
                        help="Learning rate decay rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--decay_epoch", type=int, default=0,
                        help="Decay learning rate after this epoch.")
    parser.add_argument("--batch_size", type=int,
                        default=128, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int,
                        default=1, help="Interval of evalution")
    parser.add_argument("--frac", type=float, default=1,
                        help="Fraction of participating domains")
    parser.add_argument("--anneal_cap", type=float, default=0.03,
                        help="KL annealing arguments for variantional method")
    parser.add_argument("--total_annealing_step", type=int, default=50000)
    parser.add_argument("--sim", type=str, default="dot")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoint", help="Checkpoint Dir")
    parser.add_argument("--id", type=str, default="0",
                        help="Model ID under which to save models.")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--es_patience", type=int,
                        default=5, help="Early stop patience.")
    parser.add_argument("--ld_patience", type=int, default=1,
                        help="Learning rate decay patience.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight coefficient for the recon loss.")
    parser.add_argument("--lam", type=float, default=0.3,
                        help="Weight coefficient for the sim loss.")


    args = parser.parse_args()
    return args


def seed_everything(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_logger(args):
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(args.log_dir,
                            "domain_" + "".join([domain[0] for domain
                                                 in args.domains]))
    ensure_dir(log_path, verbose=True)
    model_id = time_str

    log_file = os.path.join(log_path, args.method + "_" + model_id + ".log")

    logging.basicConfig(
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode="w+"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def main():
    args = arg_parse()
    seed_everything(args)
    init_logger(args)
    train_datasets, valid_datasets, test_datasets, adjs = load_dataset(args)
    n_domains = len(args.domains)
    domains = [LocalDomain(ModelTrainer, c_id, args, adjs[c_id],
                      train_datasets[c_id], valid_datasets[c_id], test_datasets[c_id]) for c_id in range(n_domains)]
    init_domains_weight(domains)
    save_config(args)
    aggregator = Aggregator(args, domains[0].get_params())
    run_model(domains, aggregator, args)


if __name__ == "__main__":
    main()
