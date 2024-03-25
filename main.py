import argparse
import os

import numpy as np
import torch

from custom.callback import moo_callback
from custom.display import moo_display
from custom.selection import gfo_rankandcrowding
from block import Block
from data_loader import GFO_data
from gfo import GradientFreeOptimization
from model import model_loader
from optimizer_handler import handle_moo_optimizers


def main(args):
    DEVICE = args.device
    if not torch.cuda.is_available() and args.device == 'gpu':
        DEVICE = 'cpu'
    print("Running on device:", DEVICE.upper())

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Dataloader object
    data = GFO_data(dataset=args.dataset, num_samples=args.sample_size)
    # Model object
    model, weights = model_loader(arch=args.model.lower(), dataset=args.dataset)

    model_save_path = f"output/models/{args.model}/{args.dataset}/{args.model}_{args.dataset}_epochs{args.epochs}_state_dict"
    # Gradient Free Optimization (gfo) object
    gfo = GradientFreeOptimization(
        network=model,
        weights=weights,
        data=data,
        metric=args.metric,
        DEVICE=DEVICE,
        model_save_path=model_save_path,
        dataset=args.dataset.lower(),
        model_name=args.model.lower(),
    )
    if args.pre_train:
        if not os.path.exists(f"output/models/{args.model}/{args.dataset}"):
            os.makedirs(f"output/models/{args.model}/{args.dataset}")
        gfo.pre_train(epochs=args.epochs,
                      train_loader=data.train_loader,
                      model_save_path=model_save_path)
    model_params = gfo.get_parameters(gfo.model)
    dims = len(model_params)
    print("Total number of trainable params:", dims)

    # Block
    block = None
    if args.block:
        block_path = None
        if args.block_path:
            block_path = args.block_path
        else:
            block_path = f"{args.model}_{args.dataset}_epochs{args.epochs}_{args.block_scheme.split('_')[0]}_maxD{args.max_dims}.pickle",
        block = Block(
            scheme=args.block_scheme,
            gfo=gfo,
            dims=dims,
            block_file=block_path,
            save_dir=f"output/blocks",
            arch=args.model.lower(),
            dataset=args.dataset.upper(),
            num_blocks=args.num_blocks,
            num_bins=args.bins,
        )
        block_model_params, = block.blocker(np.array([model_params.copy()]))
        dims = len(block_model_params)
        # print(dims)

    optimizer_name = args.optimizer.upper()
    if args.algorithm == "single":
        shared_link = f"{args.dir}/{args.model}_{args.dataset}_{args.global_algo}_np{args.np}_{args.strategy}_maxFE{args.global_maxiter * args.np}"
        global_save_link = f"{shared_link}_history_{args.run}"
        global_plot_link = f"{shared_link}_plot_{args.run}"

    elif args.algorithm == "multi":

        init_pop = None
        if args.init_pop == "random":
            init_pop = gfo.random_population_init(dims=dims, pop_size=args.np)
        if args.init_pop == "random+best":
            init_pop = gfo.random_population_init(dims=dims, pop_size=args.np)
            init_pop[0, :] = block_model_params.copy()
        elif args.init_pop == "block":
            init_pop = gfo.block_population_init(pop_size=args.np, block=block)
        elif args.init_pop == "block+best":
            init_pop = gfo.block_population_init(pop_size=args.np, block=block)
            init_pop[0, :] = block_model_params.copy()

        handle_moo_optimizers(optimizer=optimizer_name, gfo=gfo, block=block, callback=moo_callback, display=moo_display,
                              dimensions=dims, output_dir=args.output_dir,
                              pop_size=args.np, sampling=init_pop, nfe=args.nfe, survival=gfo_rankandcrowding())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup variables")
    # Add arguments for algorithm selection
    parser.add_argument('--algorithm', choices=['single', 'multi'], required=True,
                        help="Select the type of optimization algorithm (single or multi).")
    # Add argument for optimizer selection
    parser.add_argument('--optimizer', choices=['DE', 'PSO', 'NSGA2', 'NSGA3'], required=True,
                        help="Select the optimization algorithm.")
    # Add argument for selecting deep neural network architecture
    parser.add_argument('--model', choices=['resnet', 'vgg', 'alexnet', 'lenet'], required=True,
                        help="Select the deep neural network architecture model.")
    # Add argument for choosing dataset
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'svhn', 'mnist'],
                        help="Choose the dataset for training and evaluation.")
    # Add arguments for training parameters
    parser.add_argument('--pre_train', action='store_true', help="Use pre-trained")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epochs for training.")
    # Add argument for specifying sample size
    parser.add_argument('--sample_size', type=int, default=1000,
                        help="Sample size for the optimization problem.")
    # Add argument for specifying device (CPU or GPU) for CUDA
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                        help="Specify device (CPU or GPU) for CUDA.")
    # Add argument for evaluation metric
    parser.add_argument('--metric', choices=['f1', 'top1', 'precision_recall'], default='f1',
                        help="Select the evaluation metric (f1-score, top1, precision_recall).")
    # Add arguments for EA algorithms
    parser.add_argument('--np', type=int, default=100, help="Number of population")
    parser.add_argument('--init_pop', type=str, default='random', choices=['random', 'block', 'random+best', 'block+best'])

    parser.add_argument('--nfe', type=int, default=1000000, help="Maximum number of function evaluation (nfe)")

    parser.add_argument('--output_dir', type=str, default='output',
                        help="Directory to save the output of optimization process.")

    # Add argument for specifying strategy for DE optimizer
    parser.add_argument('--de_strategy',
                        choices=['order1bin', 'rand1bin', 'best2bin', 'rand2bin', 'currenttobest1bin',
                                 'randtobest1bin'],
                        default='order1bin', help="Specify the strategy for Differential Evolution optimizer.")
    # Add argument for specifying mutation type for DE optimizer
    parser.add_argument('--de_mutation', choices=['constant', 'random', 'vectorized'],
                        default='constant', help="Specify the mutation type for Differential Evolution optimizer.")

    # Add arguments for local search algorithm
    parser.add_argument('--local_search', action='store_false', help="Use local search algorithm (Coordinate Search)")
    parser.add_argument('--ls_iter', type=int, default=3, help="Number of local search iterations")

    # Add arguments for block related parameters
    parser.add_argument('--block', action='store_false', help="Enable block")
    parser.add_argument('--block_scheme', type=str, default='search', choices=['search', '1bin', 'random'],
                        help="Specify the block scheme")
    parser.add_argument('--bins', type=int, default=1000, help="Number of bins in histogram for Optimized Blocking")
    parser.add_argument('--num_blocks', type=int, default=100, help="Number of blocks for Random Blocking")
    parser.add_argument('--block_path', type=str, help='Path to the block file (pickle)')

    parser.add_argument('--run', type=int, default=0, help="Run index")

    args = parser.parse_args()
    main(args)
