import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score

from block_differential_evolution import block_differential_evolution

from model import *
from data_loader import *
from gfo import GradientFreeOptimization


def pre_train_model(gfo, epochs=10, train_loader=None, model_save_path=None):
    model = gfo.model
    DEVICE = gfo.DEVICE
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_f1_history = []
    train_loss_history = []
    val_f1_history = []
    # Step 5: Train the network
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        true_labels = []
        predicted_labels = []

        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            loss = criterion(output, label)
            # log_probs = nn.functional.log_softmax(output, dim=1)
            # loss = criterion(log_probs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, pred = torch.max(output.data, 1)

            true_labels.extend(label.tolist())
            predicted_labels.extend(pred.tolist())

        train_loss = running_loss / len(train_loader)
        train_loss_history.append(train_loss)
        train_f1score = f1_score(true_labels, predicted_labels, average="macro")
        val_f1score = gfo.validation_func(gfo.get_parameters(model))
        train_f1_history.append(train_f1score)
        val_f1_history.append(val_f1score)

        print(
            f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_f1score: .5f}| Val acc: {val_f1score: .5f}"
        )
        # print(
        #     f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, F1-score: {train_f1score*100:.2f}%, Validation"
        # )
    import matplotlib.pyplot as plt

    plt.plot(train_f1_history, label="train")
    plt.plot(val_f1_history, label="val")
    plt.ylabel("f1-score")
    plt.xlabel("epoch")
    plt.savefig(model_save_path + ".png")
    plt.show()
    plt.close()

    np.savez(
        model_save_path + ".npz",
        train_loss_history=train_loss_history,
        train_f1_history=train_f1_history,
        val_f1_history=val_f1_history,
    )

    gfo.model = model
    torch.save(model.state_dict(), model_save_path + ".pth")
    # params = gfo.get_parameters(model)
    print("Model is saved to:", model_save_path + ".pth")
    return model


def optimal_block_generator(
    dimensions,
    exp_dimensions,
    gfo=None,
    train_loader=None,
    test_loader=None,
    blocks_path=None,
):
    print("Block path:", blocks_path)
    params = gfo.get_parameters(gfo.model)
    if os.path.exists(blocks_path):
        import pickle

        with open(blocks_path, "rb") as f:
            blocks_mask = pickle.load(f)
        new_blocked_dims = len(blocks_mask)
        print("Optimal blocked dimensions:", new_blocked_dims)

    else:
        # Define the number of bins
        num_bins = exp_dimensions
        # Calculate the bin edges
        bin_edges = np.linspace(params.min(), params.max(), num_bins)
        # Split the data into bins
        binned_data = np.digitize(params, bin_edges)

        blocks_mask = []
        for i in range(exp_dimensions):
            b_i = np.where(binned_data == i)[0]
            if len(b_i) != 0:
                blocks_mask.append(b_i)
        new_blocked_dims = len(blocks_mask)
        print("Optimal blocked dimensions:", new_blocked_dims)

        import pickle

        with open(blocks_path, "wb") as f:
            pickle.dump(blocks_mask, f)

    # blocking
    params_blocked = np.zeros(new_blocked_dims)
    for i in range(new_blocked_dims):
        block_params = params[blocks_mask[i]]
        if len(block_params) != 0:
            params_blocked[i] = np.mean(block_params)

    # unblocking
    params_unblocked = np.ones(dimensions)
    for i in range(new_blocked_dims):
        params_unblocked[blocks_mask[i]] *= params_blocked[i]

    print("-" * 50)
    print("Optimized by Adam:")
    f1_train = gfo.evaluate_params(params, train_loader)
    print(f"Training data f1-score {100*f1_train:.2f}%")
    f1_test = gfo.evaluate_params(params, test_loader)
    print(f"Test data f1-score {100*f1_test:.2f}%")

    print("After blocking and unblocking...")
    f1_train = gfo.evaluate_params(params_unblocked, train_loader)
    print(f"Training data f1-score {100*f1_train:.2f}%")
    f1_test = gfo.evaluate_params(params_unblocked, test_loader)
    print(f"Test data f1-score {100*f1_test:.2f}%")
    print("-" * 50)

    return blocks_mask, new_blocked_dims


def random_block_generator(
    dimensions, exp_dimensions, block_size, seed=None, blocks_path=None
):
    # blocks_path = f"{blocks_path_p1}_"
    rng = np.random.default_rng(seed)
    tries = 0
    while True:
        # blocks_path = f"./data/blocks/block_b{block_size}_s{seed}_t{tries}_data"
        blocks_mask = np.arange(dimensions + ((block_size - (dimensions % block_size))))
        # for i in range(10):
        rng.shuffle(blocks_mask)
        blocks_mask = blocks_mask.reshape((exp_dimensions, block_size))
        print(np.sum(dimensions > blocks_mask[:, 0]))
        if np.sum(dimensions > blocks_mask[:, 0]) == exp_dimensions:
            import pickle

            with open(blocks_path, "wb") as f:
                pickle.dump(blocks_mask, f)
            # np.save(blocks_path, blocks)
            break
        tries += 1
    return blocks_mask


def unblocker_optimal(pop_blocked, true_dimensions, blocked_dimensions, blocks_mask):
    pop_unblocked = np.ones((pop_blocked.shape[0], true_dimensions))
    for i_p in range(pop_blocked.shape[0]):
        for i in range(blocked_dimensions):
            pop_unblocked[i_p, blocks_mask[i]] *= pop_blocked[i_p, i]
    return pop_unblocked


def main(args):
    DEVICE = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    print("Running on device:", DEVICE.upper())

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_classes = None
    data_shape = None
    full_train_loader = None
    sample_train_loader = None
    if args.dataset == "MNIST":
        num_classes = 10
        data_shape = (28, 28, 1)
        full_train_loader = load_mnist_train_full(batch_size=args.batch_size)
        sample_train_loader = load_mnist_train_each_step(
            num_samples=args.data_size,
            seed=args.seed_data,
            batch_size=args.batch_size,
        )
        test_loader = load_mnist_test(batch_size=args.batch_size)
    elif args.dataset == "CIFAR10":
        print("Correct dataset")
        num_classes = 10
        data_shape = (32, 32, 3)
        full_train_loader = load_cifar10_train_full(batch_size=args.batch_size)
        sample_train_loader = load_cifar10_train_each_step(
            num_samples=args.data_size,
            seed=args.seed_data,
            batch_size=args.batch_size,
        )
        test_loader = load_cifar10_test(batch_size=args.batch_size)
    elif args.dataset == "CIFAR100":
        num_classes = 100
        data_shape = (32, 32, 3)
        full_train_loader = load_cifar100_train_full(batch_size=args.batch_size)
        sample_train_loader = load_cifar100_train_each_step(
            num_samples=args.data_size,
            seed=args.seed_data,
            batch_size=args.batch_size,
        )
        test_loader = load_cifar100_test(batch_size=args.batch_size)
    else:
        ValueError(
            "Please enter a valid dataset, choose between:",
            ["MNIST", "CIFAR10", "CIFAR100"],
        )
    # test_loader = load_mnist_test(batch_size=args.batch_size)

    model = None
    weights = None
    if args.model.lower() == "ann":
        if args.dataset == "MNIST":
            model = ANN1H1D
        elif args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
            model = ANN1H3D
        else:
            model = ANN2H1D
    elif args.model.lower() == "resnet18":
        from torchvision.models import resnet18

        model = resnet18
        weights = "DEFAULT"
    elif args.model.lower() == "vgg16":
        from torchvision.models import vgg16

        model = vgg16
        weights = "DEFAULT"
    elif args.model.lower() == "lenet5":
        if args.dataset == "MNIST":
            model = LeNet5V1
        else:
            model = LeNet5
    else:
        ValueError("Please enter a valid model, choose between:", ["ANN", "resnet18"])

    max_iterations = args.max_iter
    popsize = args.np

    start = args.completed  # how many runs are completed?
    blocks_path = None

    # Golden seed 97232447
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
    for i in range(start, args.runs):
        block_size = args.block_size
        blocked_dimensions = None
        seed_pop = args.seed_pop if args.seed_pop != None else seeds[i]
        seed_block = args.seed_block if args.seed_block != None else seeds[i]

        print(f"Run {i}: pop init seed: {seed_pop}, block seed: {seed_block}")

        gfo = GradientFreeOptimization(
            network=model,
            weights=weights,
            num_classes=num_classes,
            data_loader=sample_train_loader,
            val_loader=test_loader,
            data_shape=data_shape,
            DEVICE=DEVICE,
        )
        gfo.find_param_sizes()
        initial_population = None
        model_params = gfo.get_parameters(gfo.model)
        dimensions = len(model_params)
        print("Number of parameters:", dimensions)
        mut_dims = dimensions

        print("Pre training:", args.pre_train)
        model_save_dir = f"{args.output_dir}models"
        model_save_path = f"{model_save_dir}/{args.model.lower()}_{args.dataset}_epochs{args.epochs}_state_dict"

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        if os.path.isfile(model_save_path + ".pth"):
            gfo.model.load_state_dict(torch.load(model_save_path + ".pth"))
            gfo.model.to(gfo.DEVICE)
            print("Saved model is loaded from:", model_save_path + ".pth")
        elif args.pre_train:
            model = pre_train_model(
                gfo,
                epochs=args.epochs,
                train_loader=full_train_loader,
                model_save_path=model_save_path,
            )
            # print(gfo.get_parameters(model)[0:10])
            gfo.model = model
            print("Model is pre-trained and saved to:", model_save_path + ".pth")
        else:
            print("Model is loaded with random parameters")

        mutation_rate = []
        file_mid = "_"
        algorithm = args.algorithm
        if block_size != None:
            if args.exp_dimensions != None:
                exp_dimensions = args.exp_dimensions
            elif args.block_size > 0:
                exp_dimensions = dimensions // block_size
                if dimensions % block_size != 0:
                    exp_dimensions += 1

            blocks_dir = f"{args.output_dir}blocks"
            if not os.path.exists(blocks_dir):
                os.makedirs(blocks_dir)

            if args.block_size < 0:  # Optimal Blocks
                blocks_path = f"{blocks_dir}/{args.model.lower()}_{args.dataset}_epochs{args.epochs}_optimal_blocks_maxD{exp_dimensions}_data.pickle"

                blocks_mask, blocked_dimensions = optimal_block_generator(
                    dimensions=dimensions,
                    exp_dimensions=exp_dimensions,
                    gfo=gfo,
                    train_loader=full_train_loader,
                    test_loader=test_loader,
                    blocks_path=blocks_path,
                )
                gfo.model.load_state_dict(torch.load(model_save_path + ".pth"))
                gfo.model.to(gfo.DEVICE)
                print("Saved model is loaded from:", model_save_path + ".pth")
                initial_population = gfo.population_initializer_by_blocked_intervals(
                    popsize, blocked_dimensions, blocks_mask, seed=seed_block
                )
                unblocked_pop = unblocker_optimal(
                    initial_population, dimensions, blocked_dimensions, blocks_mask
                )
                bounds = np.concatenate(
                    [
                        unblocked_pop.min(axis=0).reshape(-1, 1),
                        unblocked_pop.max(axis=0).reshape(-1, 1),
                    ],
                    axis=1,
                )
            else:  # Shuffle random blocks
                blocks_path = f"{blocks_dir}/{args.model.lower()}_{args.dataset}_random_blocks_bd{exp_dimensions}_data.pickle"
                blocks_mask = random_block_generator(
                    dimensions=dimensions,
                    exp_dimensions=exp_dimensions,
                    block_size=block_size,
                    seed=seed_block,
                    blocks_path=blocks_path,
                )
                blocked_dimensions = exp_dimensions
                initial_population = gfo.population_initializer(popsize, seed=seed_pop)
                if args.pre_train:
                    gfo.model.load_state_dict(torch.load(model_save_path + ".pth"))
                    gfo.model.to(gfo.DEVICE)
                    print(
                        "Saved pre-trained model is loaded from:",
                        model_save_path + ".pth",
                    )
                    # adam_opt_params = gfo.get_parameters(gfo.model)
                    # print(adam_opt_params[0:10])
                    initial_population[-1] = gfo.get_parameters(gfo.model)

                bounds = np.concatenate(
                    [
                        initial_population.min(axis=0).reshape(-1, 1),
                        initial_population.max(axis=0).reshape(-1, 1),
                    ],
                    axis=1,
                )

            # if blocked_dimensions != exp_dimensions:
            #     file_mid += f"bd{blocked_dimensions}_"
            # else:
            #     file_mid += f"bs{block_size}_bd{blocked_dimensions}_"
            mut_dims = blocked_dimensions
        else:
            ValueError("Please enter a valid block size")

        if initial_population is None:
            initial_population = gfo.population_initializer(popsize, seed=seed_pop)
            if args.pre_train:
                gfo.model.load_state_dict(torch.load(model_save_path + ".pth"))
                gfo.model.to(gfo.DEVICE)
                print(
                    "Saved pre-trained model is loaded from:", model_save_path + ".pth"
                )
                initial_population[-1] = gfo.get_parameters(gfo.model)

            bounds = np.concatenate(
                [
                    initial_population.min(axis=0).reshape(-1, 1),
                    initial_population.max(axis=0).reshape(-1, 1),
                ],
                axis=1,
            )

        if args.mut_rate == "vectorized":
            mutation_rate = [np.array([0.1] * mut_dims), np.array([1.0] * mut_dims)]
        elif args.mut_rate == "const":
            mutation_rate = 0.5
        elif args.mut_rate == "random":
            mutation_rate = [[0.1], [1.0]]
        else:
            ValueError("Please enter a valid mutation rate initialization")

        if args.other_info != None:
            file_mid += args.other_info + "_"

        shared_link = f"{args.output_dir}{args.model}_{args.dataset}_{algorithm}_np{popsize}_{args.strategy}{file_mid}maxFE{max_iterations*popsize}_training_"
        save_link = f"{shared_link}history_{i}"
        plot_link = f"{shared_link}plot_{i}.png"

        res = block_differential_evolution(
            gfo.fitness_func,
            bounds,
            mutation=mutation_rate,
            maxiter=max_iterations,
            block_size=block_size,
            blocked_dimensions=blocked_dimensions,
            save_link=save_link,
            plot_link=plot_link,
            blocks_link=blocks_path,
            popsize=popsize,
            callback=None,
            polish=False,
            local_search=args.local_search,
            disp=True,
            updating="deferred",
            strategy=args.strategy,
            init=initial_population,
            val_func=gfo.validation_func,
        )


if __name__ == "__main__":
    # --------------------------------------------------
    # SETUP INPUT PARSER
    # --------------------------------------------------
    parser = argparse.ArgumentParser(description="Setup variables")

    # dir
    parser.add_argument(
        "--output-dir", type=str, default="./output/", help="Output directory"
    )
    # parser.add_argument('--model-dir', type=str, default='./models/', help='Save directory')
    parser.add_argument(
        "--other-info",
        type=str,
        default=None,
        help="Output file middle name which contains setting",
    )

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
    # parser.add_argument('--download', type=bool, default=False, help='Whether to download the dataset')
    parser.add_argument(
        "--seed-data", type=int, default=42, help="seed for Reproducibility in dataset"
    )
    parser.add_argument(
        "--data-size", type=int, default=1000, help="Training dataset size balanced"
    )
    parser.add_argument(
        "--seed-pop",
        type=int,
        default=None,
        help="seed for Reproducibility in population initilization",
    )
    parser.add_argument(
        "--seed-block",
        type=int,
        default=None,
        help="seed for Reproducibility in block initialization",
    )

    # algorithm
    parser.add_argument(
        "--algorithm", type=str, default="de", help="Optimization methods"
    )
    parser.add_argument(
        "--block-size", type=int, default=None, help="A hyper-paramater in BDE"
    )
    parser.add_argument(
        "--exp-dimensions", type=int, default=None, help="Expected dimensions"
    )
    parser.add_argument(
        "--local-search", type=bool, default=False, help="Coordiante Descent enable"
    )
    parser.add_argument("--cuda", type=bool, default=True, help="Whether to use cuda")
    parser.add_argument(
        "--strategy",
        type=str,
        default="rand1bin",
        help="Mutation and Crossover strategy",
    )
    parser.add_argument(
        "--mut-rate", type=str, default="const", help="Mutation and Crossover strategy"
    )

    # grad-free training params
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--completed", type=int, default=0, help="Start run number")
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        help="Metric used in optimization [f1, top1, top5]",
    )
    parser.add_argument("--np", type=int, default=100, help="Number of population")
    parser.add_argument(
        "--max-iter", type=int, default=100000, help="Max number of iterations"
    )
    parser.add_argument("--save-iter", type=int, default=2000, help="Save iter")

    args = parser.parse_args()

    main(args)
