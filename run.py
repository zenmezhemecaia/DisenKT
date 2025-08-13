# -*- coding: utf-8 -*-
import logging

def evaluation_logging(eval_logs, round, weights, mode="valid"):
    if mode == "valid":
        logging.info("Epoch%d Valid:" % round)
    else:
        logging.info("Test:")

    for domain, eval_log in eval_logs.items():
        logging.info("%s AUC: %.4f \t ACC: %.4f \t RMSE: %.4f"
                     % (domain, eval_log["AUC"], eval_log["ACC"],
                         eval_log["RMSE"]))



def load_and_eval_model(n_domains, domains, args):
    eval_logs = {}
    for d_id in range(n_domains):
        domains[d_id].load_params()
        eval_log = domains[d_id].evaluation(args.epochs, mode="test")
        eval_logs[domains[d_id].domain_name] = eval_log
    weights = dict((domain.domain_name, domain.test_weight) for domain in domains)
    evaluation_logging(eval_logs, args.epochs, weights, mode="test")


def run_model(domains, aggregator, args):
    n_domains = len(domains)
    if args.do_eval:
        load_and_eval_model(n_domains, domains, args)
    else:
        for round in range(1, args.epochs + 1):
            random_dids = aggregator.choose_domains(n_domains, args.frac)

            # Train with these domains
            for d_id in random_dids:
                # Restore global parameters to domain's model
                domains[d_id].set_global_params(aggregator.get_global_params())
                domains[d_id].set_global_reps(aggregator.get_global_reps())

                # Train one domain
                domains[d_id].train_epoch(
                    round, args, global_params=aggregator.global_params)

            aggregator.aggregate_params(domains, random_dids)
            aggregator.aggregate_reps(domains, random_dids)

            if round % args.eval_interval == 0:
                eval_logs = {}
                for d_id in range(n_domains):
                    if "Fed" in args.method:
                        domains[d_id].set_global_params(
                            aggregator.get_global_params())
                    if d_id in random_dids:
                        eval_log = domains[d_id].evaluation(round, mode="valid")
                    eval_logs[domains[d_id].domain_name] = eval_log



        for domain in domains:
            domain.save_params()
            print(f"Saved parameters for domain {domain.domain_name}")
        load_and_eval_model(n_domains, domains, args)

