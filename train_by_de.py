import os
import argparse
import numpy as np
import torch

from block_differential_evolution import block_differential_evolution

from model import NeuralNetwork
from data_loader import *
from gfo import GradientFreeOptimization


def optimal_block_generator(dimensions, blocked_dimensions,
                                     block_size, seed=None):
    blocks_data = None
    rng = np.random.default_rng(seed)
    return blocks_data

def random_block_generator(dimensions, blocked_dimensions,
                                     block_size, seed=None):
    blocks_data = None
    rng = np.random.default_rng(seed)
    tries = 0
    while True:
                blocks_data=f"./data/blocks/block_b{block_size}_s{seed}_t{tries}_data"
                blocks = np.arange(dimensions + ((block_size-(dimensions%block_size))))
                for i in range(10):
                    rng.shuffle(blocks)
                print(blocks[0])
                blocks = blocks.reshape((blocked_dimensions, block_size))
                # print(np.sum(dimensions > blocks[:, 0]))
                if np.sum(dimensions > blocks[:, 0]) == blocked_dimensions:
                    np.save(blocks_data, blocks)
                    break
                tries+=1
    return blocks_data

def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    print("Running on device:", DEVICE.upper())

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_loader = load_mnist_train(samples_per_class=100, seed=args.seed_data, batch_size=args.batch_size)
    # test_loader = load_mnist_test(batch_size=args.batch_size)

    block_size = args.block_size
    max_iterations = args.max_iter
    popsize = args.np

    start = args.completed # how many runs are completed?
    j = 0 # iterator for block shuffling tries counter

    seeds = [23, 97232447, 45689, 96793335, 12345679, 23, 97232447, 45689, 96793335, 12345679]
    for i in range(start, args.runs):
        seed_pop = args.seed_pop if args.seed_pop != None else seeds[i]
        seed_block = args.seed_block if args.seed_block != None else seeds[i]

        print(f"run {i}: pop init seed: {seed_pop}, block seed: {seed_block}")

        gfo = GradientFreeOptimization(NeuralNetwork, train_loader, DEVICE)
        initial_population = gfo.population_initializer(popsize, seed=seed_pop)
        print(gfo.fitness_func(initial_population[-1]))
        dimensions = initial_population.shape[1]
        print("Number of parameters:", dimensions)

        bounds = np.concatenate([initial_population.min(axis=0).reshape(-1, 1), initial_population.max(axis=0).reshape(-1,1)], axis=1)
        mutation_rate = []

        file_mid = "_"
        algorithm = args.algorithm
        blocks_data = None
        mut_dims = dimensions           
        if block_size > 0:
            file_mid += f"b{block_size}_"

            blocked_dimensions = dimensions//block_size
            if dimensions//block_size % 10 != 0:
                blocked_dimensions +=1
            mut_dims = blocked_dimensions

            # blocks_data = optimal_block_generator(dimensions=dimensions, blocked_dimensions=blocked_dimensions,
            #                          block_size=block_size, seed=seed_block, )
            blocks_data = random_block_generator(dimensions=dimensions, blocked_dimensions=blocked_dimensions,
                                     block_size=block_size, seed=seed_block)
        else:
            ValueError("Please enter a valid block size")


        if args.mut_rate == "vector":
                mutation_rate = [np.array([0.1] * mut_dims), np.array([1.0] * mut_dims)]
        elif args.mut_rate == "const":
                mutation_rate = 0.5
        elif args.mut_rate == "rand":
                mutation_rate = [[0.1], [1.0]]
        else:
                ValueError("Please enter a valid mutation rate initialization")

        if args.other_info != None:
            file_mid += args.other_info+"_"
        save_link = f'{args.output_dir}ann_{algorithm}_np{popsize}_{args.strategy}{file_mid}maxFE{max_iterations*popsize}_mnist_training_history_{i}'
        plot_link = f'{args.output_dir}ann_{algorithm}_np{popsize}_{args.strategy}{file_mid}maxFE{max_iterations*popsize}_mnist_training_plot_{i}.png'
        print(save_link)
        if os.path.exists(save_link+'.npz'):
            continue
        res = block_differential_evolution(gfo.fitness_func, bounds, 
                                                mutation=mutation_rate, maxiter=max_iterations, block_size=block_size,
                                                save_link=save_link, plot_link=plot_link, blocks_link=blocks_data,
                                                popsize=popsize, callback=None, polish=False, local_search=args.local_search,
                                                disp=True, updating='deferred', strategy=args.strategy, init=initial_population)

if __name__ == '__main__':
    # --------------------------------------------------
    # SETUP INPUT PARSER
    # --------------------------------------------------
    parser = argparse.ArgumentParser(description='Setup variables')

    # dir
    parser.add_argument('--output-dir', type=str, default='./output/', help='Output directory')
    # parser.add_argument('--model-dir', type=str, default='./models/', help='Save directory')
    parser.add_argument('--other-info', type=str, default=None, help='Output file middle name which contains setting')

    # dataset
    # parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset')
    # parser.add_argument('--download', type=bool, default=False, help='Whether to download the dataset')
    parser.add_argument('--seed-data', type=int, default=42, help='seed for Reproducibility in dataset')
    parser.add_argument('--seed-pop', type=int, default=None, help='seed for Reproducibility in population initilization')
    parser.add_argument('--seed-block', type=int, default=None, help='seed for Reproducibility in block initialization')

    # algorithm
    parser.add_argument('--algorithm', type=str, default='de', help='Optimization methods')
    parser.add_argument('--block-size', type=int, default=0, help='A hyper-paramater in BDE')
    parser.add_argument('--local-search', type=bool, default=False, help='Coordiante Descent enable')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use cuda')
    parser.add_argument('--strategy', type=str, default="rand1bin", help="Mutation and Crossover strategy")
    parser.add_argument('--mut-rate', type=str, default="const", help="Mutation and Crossover strategy")

    # grad-free training params
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--completed', type=int, default=0, help='Start run number')
    parser.add_argument('--np', type=int, default=100, help='Number of population')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size of tasks to update meta-parameters')
    parser.add_argument('--max-iter', type=int, default=100000, help='Max number of iterations')
    parser.add_argument('--save-iter', type=int, default=2000, help='Save iter')

    args = parser.parse_args()

    main(args)