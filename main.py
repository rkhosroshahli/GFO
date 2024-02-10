import os
import argparse
<<<<<<< HEAD
import warnings
from matplotlib import pyplot as plt
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
=======
import numpy as np
>>>>>>> feed58546091c003099702d2bff8f00e585857db
import torch

from data_loader import *
from model import model_loader
from gfo import GradientFreeOptimization
from block import *
from block_differential_evolution import block_differential_evolution
from cs import cs_3point
<<<<<<< HEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.display.output import Output
from pymoo.util.display.column import Column
=======
>>>>>>> feed58546091c003099702d2bff8f00e585857db


def main(args):
    DEVICE = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    print("Running on device:", DEVICE.upper())

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    sample_loader, train_loader, test_loader, num_classes = data_loader(
        args.dataset.lower(), args.batch_size, args.sample_size, seed=None
    )
<<<<<<< HEAD

=======
>>>>>>> feed58546091c003099702d2bff8f00e585857db
    model_save_path = f"output/models/{args.model}/{args.dataset}/{args.model}_{args.dataset}_epochs{args.epochs}_state_dict"
    model, weights = model_loader(arch=args.model.lower(), dataset=args.dataset.upper())

    seeds = [
        23,
        97232445,
        45689,
        96793335,
        12345679,
        23,
        97232445,
        45689,
        96793335,
        12345679,
    ]
    for i in range(args.completed, args.runs):
        seed_pop = seeds[i]
        seed_block = seeds[i]
        print(f"Run {i}: pop init seed: {seed_pop}, block seed: {seed_block}")

        gfo = GradientFreeOptimization(
            neural_network=model,
            weights=weights,
            num_classes=num_classes,
            data_loader=sample_loader,
            val_loader=test_loader,
            metric=args.metric,
            DEVICE=DEVICE,
        )

        if args.pre_train:
<<<<<<< HEAD
            print("Pre-train is enabled!")
=======
>>>>>>> feed58546091c003099702d2bff8f00e585857db
            gfo.pre_train(
                epochs=args.epochs,
                train_loader=train_loader,
                model_save_path=model_save_path,
            )
        model_params = gfo.get_parameters(gfo.model)

        # blocks_path = ""
        shared_link = f"{args.dir}/{args.model}_{args.dataset}_{args.global_algo}_np{args.np}_{args.strategy}_maxFE{args.global_maxiter*args.np}"
        global_save_link = f"{shared_link}_history_{i}"
        global_plot_link = f"{shared_link}_plot_{i}"

        block = Block(
            scheme=args.block_scheme,
            dims=len(model_params),
            block_size=args.block_size,
<<<<<<< HEAD
            path=f"output/blocks/classic/{args.model}_{args.dataset}_epochs{args.epochs}_{args.block_scheme.split('_')[0]}_maxD{args.max_dims}.pickle",
=======
            path=f"output/blocks/{args.model}_{args.dataset}_epochs{args.epochs}_{args.block_scheme}_maxD{args.max_dims}.pickle",
>>>>>>> feed58546091c003099702d2bff8f00e585857db
        )
        blocks_mask = None
        blocked_dims = None
        blocker = None
        unblocker = None
<<<<<<< HEAD
        if "optimized" in args.block_scheme:
=======
        if args.block_scheme == "optimized":
>>>>>>> feed58546091c003099702d2bff8f00e585857db
            blocks_mask = block.generator(
                max_dims=args.max_dims,
                gfo=gfo,
                train_loader=train_loader,
                test_loader=test_loader,
                seed=seed_block,
            )
            blocker = block.blocker
            unblocker = block.unblocker
<<<<<<< HEAD
            if "merge" in args.block_scheme:
                print("Merging blocks with size less than 2.")
                new_path = f"output/blocks/merged/{args.model}_{args.dataset}_epochs{args.epochs}_{args.block_scheme}_maxD{args.max_dims}.pickle"
                blocks_mask = block.merge_blocks(blocks_mask, new_path)
=======
            new_path = f"output/blocks/{args.model}_{args.dataset}_epochs{args.epochs}_{args.block_scheme}_maxD{args.max_dims}_merged.pickle"
            blocks_mask = block.merge_blocks(new_path)
>>>>>>> feed58546091c003099702d2bff8f00e585857db
        elif args.block_scheme == "randomized":
            blocks_mask = block.generator(
                gfo=gfo,
                max_dims=args.max_dims,
                block_size=args.block_size,
                train_loader=train_loader,
                test_loader=test_loader,
                seed=seed_block,
            )
            blocker = block.blocker
            unblocker = block.randomized_unblocker

        blocked_dims = block.blocked_dims

        init_pop = None
        init_pop_fitness = None
        # if args.global_maxiter == 0 and args.local_maxiter > 0:
        #     init_pop = gfo.random_population_init(popsize=args.np, seed=seed_pop)
        #     init_pop = block.blocker(init_pop)
        #     print("No global but local")
        if args.global_algo != "":
<<<<<<< HEAD
            if "optimized" in args.block_scheme:
=======
            if args.block_scheme == "optimized":
>>>>>>> feed58546091c003099702d2bff8f00e585857db
                init_pop = gfo.optimized_population_init(
                    args.np, blocked_dims, blocks_mask, seed=seed_pop
                )
            elif args.block_scheme == "randomized":
                init_pop = gfo.random_population_init(popsize=args.np, seed=seed_pop)
            else:
                init_pop = gfo.random_population_init(popsize=args.np, seed=seed_pop)

            bounds = np.concatenate(
                [
                    init_pop.min(axis=0).reshape(-1, 1),
                    init_pop.max(axis=0).reshape(-1, 1),
                ],
                axis=1,
            )

        best_solution = model_params
        best_fitness = gfo.evaluate_params(
            model_params, data_loader=train_loader, metric=args.metric
        )
        val_f1 = gfo.validation_func(best_solution)
        print(
            f"Adam, f(x)= {1 - best_fitness:.4f}, 1-f(x)= {(best_fitness):.4f}, g(x)={val_f1:.4f}",
        )
        best_fitness = 1 - best_fitness
        fitness_history = [best_fitness]

        nfe = 0
        if args.global_algo == "DE" and args.global_maxiter > 0:

            mutation_rate = None
            if args.mutation == "vectorized":
                mutation_rate = [
                    np.array([0.1] * init_pop.shape[1]),
                    np.array([1.0] * init_pop.shape[1]),
                ]
            elif args.mutation == "const":
                mutation_rate = 0.5
            elif args.mutation == "random":
                mutation_rate = [[0.1], [1.0]]
            else:
                ValueError("Please enter a valid mutation rate initialization")

            # nfe += args.np

            res = block_differential_evolution(
                gfo.fitness_func,
                bounds,
                mutation=mutation_rate,
                maxiter=args.global_maxiter,
                block_size=args.block_size,
                blocked_dimensions=blocked_dims,
                save_link=global_save_link,
                plot_link=global_plot_link,
                blocks_link=block.blocks_path,
                popsize=args.np,
                callback=None,
                polish=False,
                disp=True,
                updating="deferred",
                strategy=args.strategy,
                init=init_pop,
                val_func=gfo.validation_func,
            )
            best_solution = res.x
            best_fitness = res.fun
            nfe += res.nfev
        else:
            print("Global optimization is skipped.")

        if args.local_algo == "cs3p" and args.local_maxiter > 0:
            # sample_loader, _, _, _ = data_loader(
            #     args.dataset.lower(),
            #     args.batch_size,
            #     args.sample_size,
            #     max_num_call=1,
            #     seed=None,
            # )
            # sample_loader = data_sampler(
            #     dataset=args.dataset.lower(),
            #     batch_size=args.batch_size,
            #     max_num_call=2 * block.blocked_dims + 1,
            #     seed=None,
            # )

            sample_loader = data_fixed_sampler(
                dataset=args.dataset.lower(),
                batch_size=args.batch_size,
                sample_size=args.sample_size,
<<<<<<< HEAD
                seed=59,
=======
                seed=None,
>>>>>>> feed58546091c003099702d2bff8f00e585857db
            )

            gfo.data_loader = sample_loader

            best_solution = model_params
            best_fitness = gfo.evaluate_params(
                best_solution, data_loader=sample_loader, metric=args.metric
            )
            val_f1 = gfo.validation_func(best_solution)
            print(
                f"Adam, f(x)= {1 - best_fitness:.4f}, 1-f(x)= {(best_fitness):.4f}, g(x)={val_f1:.4f}",
            )
            best_fitness = 1 - best_fitness
            fitness_history = [best_fitness]

            var_min = None
            var_max = None
            if init_pop == None:
<<<<<<< HEAD
                if "optimized" in args.block_scheme:
=======
                if args.block_scheme == "optimized":
>>>>>>> feed58546091c003099702d2bff8f00e585857db
                    var_min, var_max = gfo.optimized_local_search_boundaries(
                        blocked_dims, blocks_mask, seed=seed_pop
                    )
                # elif args.block_scheme == "randomized":
                # init_pop = gfo.random_population_init(popsize=args.np, seed=seed_pop)
                # else:
                # init_pop = gfo.random_population_init(popsize=args.np, seed=seed_pop)
            M = 10
            if init_pop_fitness == None and init_pop != None:
                if init_pop.shape[1] != len(model_params):
                    init_pop_fitness = [
                        gfo.fitness_func(p) for p in block.unblocker(init_pop)
                    ]
                    (best_blocked_solution,) = block.blocker(np.array([best_solution]))
                    init_pop[0] = best_blocked_solution
                    init_pop_fitness[0] = best_fitness
                else:
                    init_pop_fitness = [gfo.fitness_func(p) for p in (init_pop)]
                    init_pop[0] = best_solution.copy()
                    init_pop_fitness[0] = best_fitness

                init_pop_fitness = np.asarray(init_pop_fitness)
                fitness_history = [init_pop_fitness.min()]

                var_min = np.min(init_pop[init_pop_fitness.argsort()[:M]], axis=0)
                var_max = np.max(init_pop[init_pop_fitness.argsort()[:M]], axis=0)

            shared_link = f"{args.dir}/{args.model}_{args.dataset}_{args.local_algo}_maxiter{args.local_maxiter}"
            local_save_link = f"{shared_link}_history_{i}"
            local_plot_link = f"{shared_link}_plot_{i}"

<<<<<<< HEAD
            max_nfe = args.local_maxiter * (2 * block.blocked_dims + 1) + nfe
            if args.max_nfe:
                max_nfe = args.max_nfe

=======
>>>>>>> feed58546091c003099702d2bff8f00e585857db
            res = cs_3point(
                fitness=gfo.fitness_func,
                best_solution=best_solution,
                best_fitness=best_fitness,
                fitness_history=fitness_history,
                var_min=var_min,
                var_max=var_max,
                nit=args.local_maxiter,
                nfe=nfe,
<<<<<<< HEAD
                max_nfe=max_nfe,
=======
                max_nfe=args.local_maxiter * (2 * block.blocked_dims + 1) + nfe,
>>>>>>> feed58546091c003099702d2bff8f00e585857db
                block=block,
                plot_link=local_plot_link,
                history_link=local_save_link,
                validation_func=gfo.validation_func,
            )
        else:
            print("Local search is skipped.")

<<<<<<< HEAD
        if args.moo_algo == "nsga2" and args.moo_maxiter > 0:
            sample_loader = data_fixed_sampler(
                dataset=args.dataset.lower(),
                batch_size=args.batch_size,
                sample_size=args.sample_size,
                seed=59,
            )
            gfo.data_loader = sample_loader

            if "optimized" in args.block_scheme:
                init_pop = gfo.optimized_population_init(
                    args.np, blocked_dims, blocks_mask, seed=seed_pop
                )
            elif args.block_scheme == "randomized":
                init_pop = gfo.random_population_init(popsize=args.np, seed=seed_pop)

            init_pop[0] = block.blocker(np.array([model_params]))

            problem = gfo.moo_objective_func(dims=np.size(init_pop, 1), block=block)

            algorithm = NSGA2(pop_size=args.np, sampling=init_pop)

            class MyOutput(Output):
                def __init__(self):
                    super().__init__()
                    self.precision = Column("Precision", width=13)
                    self.recall = Column("Recall", width=13)
                    self.columns += [self.precision, self.recall]

                def update(self, algorithm):
                    super().update(algorithm)
                    self.precision.set(np.mean(algorithm.pop.get("F")))
                    self.recall.set(np.std(algorithm.pop.get("F")))

            # Handle the warning
            warnings.filterwarnings("ignore")
            res = minimize(
                problem,
                algorithm,
                ("n_gen", args.moo_maxiter),
                seed=1,
                verbose=True,
                save_history=True,  # output=MyOutput(),
            )

            shared_link = f"{args.dir}/{args.model}_{args.dataset}_{args.moo_algo}_maxiter{args.moo_maxiter}"
            moo_save_link = f"{shared_link}_history_{i}"
            moo_plot_link = f"{shared_link}_plot_{i}"

            # import dill
            # with open(moo_save_link, "wb") as f:
            #     dill.dump(res.history, f)

            X, F = res.opt.get("X", "F")

            n_evals = []  # corresponding number of function evaluations
            hist_F = []  # the objective space values in each generation

            for algo in res.history:
                # store the number of function evaluations
                n_evals.append(algo.evaluator.n_eval)
                # retrieve the optimum from the algorithm
                hist_F.append(algo.opt.get("F"))

            approx_ideal = F.min(axis=0)
            approx_nadir = F.max(axis=0)

            from pymoo.indicators.hv import Hypervolume

            metric = Hypervolume(
                ref_point=np.array([1.0, 1.0]),
                norm_ref_point=False,
                zero_to_one=True,
                ideal=approx_ideal,
                nadir=approx_nadir,
            )

            hv = [metric.do(_F) for _F in hist_F]

            plt.figure(figsize=(7, 5))
            plt.plot(n_evals, hv, color="black", lw=0.7, label="Block NSGA2")
            plt.scatter(n_evals, hv, facecolor="none", edgecolor="black", marker="p")
            plt.title(shared_link)
            plt.xlabel("FEs")
            plt.ylabel("Hypervolume (HV)")
            plt.legend()
            plt.savefig(moo_plot_link + "_hv.png")
            plt.show()

            plt.figure(figsize=(7, 5))
            plt.scatter(
                res.pop.get("F")[:, 0],
                res.pop.get("F")[:, 1],
                color="black",
                label="Population",
            )
            plt.scatter(
                F[:, 0],
                F[:, 1],
                facecolor="none",
                edgecolor="red",
                marker="p",
                label="Pareto Front",
            )
            plt.title(shared_link)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.savefig(moo_plot_link + "_paretofront.png")
            plt.show()

            np.savez(moo_save_link, pareto_front=X, fitness_history=hist_F)

=======
>>>>>>> feed58546091c003099702d2bff8f00e585857db

if __name__ == "__main__":
    # --------------------------------------------------
    # SETUP INPUT PARSER
    # --------------------------------------------------
    parser = argparse.ArgumentParser(description="Setup variables")

    # Neural Network model
    parser.add_argument("--model", type=str, default="ANN", help="Model to use")
    parser.add_argument(
        "--pre-train",
        type=bool,
        default=False,
        help="The model to be pre-trained or use random weights",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs the model to be trained/fine-tuned",
    )

    # dataset
    parser.add_argument(
        "--dataset", type=str, default="MNIST", help="Dataset to be evlauated"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size of data loader"
    )
    parser.add_argument(
        "--sample-size", type=int, default=10000, help="Sample size of data loader"
    )
    parser.add_argument("--cuda", type=bool, default=True, help="Whether to use cuda")

    # grad-free training params
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--completed", type=int, default=0, help="Start run number")
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        help="Metric used in optimization [f1, top1, top5]",
    )

    # algorithm
    parser.add_argument(
        "--global-algo", type=str, default="", help="Optimization methods"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="rand1bin",
        help="Mutation and Crossover strategy",
    )
    parser.add_argument(
        "--mutation", type=str, default="const", help="Mutation and Crossover strategy"
    )

    parser.add_argument("--np", type=int, default=100, help="Number of population")

    parser.add_argument(
        "--global-maxiter", type=int, default=0, help="Max number of iterations"
    )

    parser.add_argument(
        "--local-algo", type=str, default="", help="Optimization methods"
    )

    parser.add_argument(
        "--local-maxiter", type=int, default=0, help="Max number of iterations"
    )

<<<<<<< HEAD
    parser.add_argument("--moo-algo", type=str, default="", help="Optimization methods")

    parser.add_argument(
        "--moo-maxiter", type=int, default=0, help="Max number of iterations"
    )

    parser.add_argument(
        "--block-scheme", type=str, default="random", help="A hyper-paramater in BDE"
    )

=======
    parser.add_argument(
        "--block-scheme", type=str, default="random", help="A hyper-paramater in BDE"
    )
>>>>>>> feed58546091c003099702d2bff8f00e585857db
    parser.add_argument(
        "--block-size", type=int, default=10, help="Expected dimensions"
    )
    parser.add_argument(
        "--max-dims", type=int, default=10000, help="Expected dimensions"
    )

<<<<<<< HEAD
    parser.add_argument(
        "--max-nfe", type=int, default=100000, help="Expected dimensions"
    )

=======
>>>>>>> feed58546091c003099702d2bff8f00e585857db
    # dir
    parser.add_argument("--dir", type=str, default="./output/", help="Output directory")
    # parser.add_argument('--model-dir', type=str, default='./models/', help='Save directory')
    parser.add_argument(
        "--other-info",
        type=str,
        default=None,
        help="Output file middle name which contains setting",
    )

    args = parser.parse_args()

    main(args)
