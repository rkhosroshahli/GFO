import os
import argparse
import numpy as np
import torch

from data_loader import *
from model import model_loader
from gfo import GradientFreeOptimization
from block import *
from block_differential_evolution import block_differential_evolution
from cs import cs_3point


def main(args):
    DEVICE = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    print("Running on device:", DEVICE.upper())

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    sample_loader, train_loader, test_loader, num_classes = data_loader(
        args.dataset.lower(), args.batch_size, args.sample_size, seed=None
    )
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
            path=f"output/blocks/{args.model}_{args.dataset}_epochs{args.epochs}_{args.block_scheme}_maxD{args.max_dims}.pickle",
        )
        blocks_mask = None
        blocked_dims = None
        blocker = None
        unblocker = None
        if args.block_scheme == "optimized":
            blocks_mask = block.generator(
                max_dims=args.max_dims,
                gfo=gfo,
                train_loader=train_loader,
                test_loader=test_loader,
                seed=seed_block,
            )
            blocker = block.blocker
            unblocker = block.unblocker
            new_path = f"output/blocks/{args.model}_{args.dataset}_epochs{args.epochs}_{args.block_scheme}_maxD{args.max_dims}_merged.pickle"
            blocks_mask = block.merge_blocks(new_path)
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
            if args.block_scheme == "optimized":
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
                seed=None,
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
                if args.block_scheme == "optimized":
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

            res = cs_3point(
                fitness=gfo.fitness_func,
                best_solution=best_solution,
                best_fitness=best_fitness,
                fitness_history=fitness_history,
                var_min=var_min,
                var_max=var_max,
                nit=args.local_maxiter,
                nfe=nfe,
                max_nfe=args.local_maxiter * (2 * block.blocked_dims + 1) + nfe,
                block=block,
                plot_link=local_plot_link,
                history_link=local_save_link,
                validation_func=gfo.validation_func,
            )
        else:
            print("Local search is skipped.")


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

    parser.add_argument(
        "--block-scheme", type=str, default="random", help="A hyper-paramater in BDE"
    )
    parser.add_argument(
        "--block-size", type=int, default=10, help="Expected dimensions"
    )
    parser.add_argument(
        "--max-dims", type=int, default=10000, help="Expected dimensions"
    )

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
