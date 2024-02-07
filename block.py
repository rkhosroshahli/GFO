import os
import numpy as np
import pickle


class Block:
    def __init__(self, scheme, dims, block_size, path):
        self.scheme = scheme
        self.blocker = None
        self.unblocker = None
        self.generator = None
        if scheme == "optimized":
            self.blocker = self.optimized_blocker
            self.unblocker = self.optimized_unblocker
            self.generator = self.optimized_block_generator
        elif scheme == "randomized":
            self.blocker = self.randomized_blocker
            self.unblocker = self.randomized_unblocker
            self.generator = self.randomized_block_generator
        else:
            self.blocker = None
            self.unblocker = None
            self.generator = None

        self.path = path
        self.dims = dims
        self.blocked_dims = None
        self.block_size = block_size

    def load_mask(self):
        blocks_mask = None
        with open(self.path, "rb") as f:
            blocks_mask = pickle.load(f)
        return blocks_mask

    def optimized_block_generator(
        self, gfo, max_dims, train_loader=None, test_loader=None, seed=None
    ):
        if os.path.exists(self.path):
            blocks_mask = self.load_mask()
            new_blocked_dims = len(blocks_mask)
            print("Optimized blocked dimensions:", new_blocked_dims)
            self.blocked_dims = new_blocked_dims
            return blocks_mask

        params = gfo.get_parameters(gfo.model)
        # dims = len(params)
        # Define the number of bins
        num_bins = max_dims
        # Calculate the bin edges
        bin_edges = np.linspace(params.min(), params.max(), num_bins)
        # Split the data into bins
        binned_data = np.digitize(params, bin_edges)

        blocks_mask = []
        for i in range(max_dims):
            b_i = np.where(binned_data == i)[0]
            if len(b_i) != 0:
                blocks_mask.append(b_i)
        new_blocked_dims = len(blocks_mask)
        print("Optimal blocked dimensions:", new_blocked_dims)
        self.blocked_dims = new_blocked_dims

        with open(self.path, "wb") as f:
            pickle.dump(blocks_mask, f)

        (params_unblocked,) = self.unblocker(np.array([params]))

        print("-" * 50)
        print("Optimized by Adam:")
        f1_train = gfo.evaluate_params(params, train_loader)
        print(f"Training data f1-score {f1_train:.4f}%")
        f1_test = gfo.evaluate_params(params, test_loader)
        print(f"Test data f1-score {f1_test:.4f}%")

        print("After blocking and unblocking...")
        f1_train = gfo.evaluate_params(params_unblocked, train_loader)
        print(f"Training data f1-score {f1_train:.4f}%")
        f1_test = gfo.evaluate_params(params_unblocked, test_loader)
        print(f"Test data f1-score {f1_test:.4f}%")
        print("-" * 50)

        return blocks_mask

    def randomized_block_generator(
        self,
        gfo=None,
        max_dims=0,
        train_loader=None,
        test_loader=None,
        blocks_path=None,
        seed=None,
    ):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                blocks_mask = pickle.load(f)
            new_blocked_dims = len(blocks_mask)
            print("Randomized blocked dimensions:", new_blocked_dims)
            self.blocked_dims = new_blocked_dims
            return blocks_mask

        params = gfo.get_parameters(gfo.model)
        dims = len(params)
        rng = np.random.default_rng(seed)
        tries = 0
        while True:
            blocks_mask = np.arange(
                dims + ((self.block_size - (dims % self.block_size)))
            )

            rng.shuffle(blocks_mask)
            blocks_mask = blocks_mask.reshape((max_dims, self.block_size))
            print(np.sum(dims > blocks_mask[:, 0]))
            if np.sum(dims > blocks_mask[:, 0]) == max_dims:
                with open(self.path, "wb") as f:
                    pickle.dump(blocks_mask, f)
                break
            tries += 1
        self.blocked_dims = max_dims
        # with open(self.path, "wb") as f:
        #     pickle.dump(blocks_mask, f)
        (params_unblocked,) = self.unblocker([params])

        print("-" * 50)
        print("Optimized by Adam:")
        f1_train = gfo.evaluate_params(params, train_loader)
        print(f"Training data f1-score {f1_train:.4f}%")
        f1_test = gfo.evaluate_params(params, test_loader)
        print(f"Test data f1-score {f1_test:.4f}%")

        print("After blocking and unblocking...")
        f1_train = gfo.evaluate_params(params_unblocked, train_loader)
        print(f"Training data f1-score {f1_train:.4f}%")
        f1_test = gfo.evaluate_params(params_unblocked, test_loader)
        print(f"Test data f1-score {f1_test:.4f}%")
        print("-" * 50)

        return blocks_mask

    def optimized_unblocker(self, pop_blocked):
        blocks_mask = self.load_mask()
        pop_unblocked = np.ones((pop_blocked.shape[0], self.dims))
        for i_p in range(pop_blocked.shape[0]):
            for i in range(self.blocked_dims):
                pop_unblocked[i_p, blocks_mask[i]] *= pop_blocked[i_p, i]
        return pop_unblocked

    def randomized_unblocker(self, pop_blocked):
        blocks_mask = self.load_mask()
        pop_unblocked = np.ones(
            (
                len(pop_blocked),
                self.dims + ((self.block_size - (self.dims % self.block_size))),
            )
        )

        for i in range(self.blocked_dims):
            for j in range(self.block_size):
                pop_unblocked[:, blocks_mask[i, j]] = pop_blocked[:, i]

        return pop_unblocked[:, : self.dims]

    def randomized_blocker(self, pop):
        blocks_mask = self.load_mask()
        return pop[:, blocks_mask[:, 0]].copy()

    def optimized_blocker(self, pop=None):
        blocks_mask = self.load_mask()
        params_blocked = np.zeros((pop.shape[0], self.blocked_dims))
        for i_p in range(pop.shape[0]):
            for i in range(self.blocked_dims):
                block_params = pop[i_p, blocks_mask[i]]
                if len(block_params) != 0:
                    params_blocked[i_p, i] = np.mean(block_params)

        return params_blocked

    def merge_blocks(self, new_path, gfo=None):
        if os.path.exists(self.path):
            blocks_mask = self.load_mask()
            new_blocked_dims = len(blocks_mask)
            print("Optimized blocked dimensions:", new_blocked_dims)
            self.blocked_dims = new_blocked_dims

        # params = gfo.get_parameters(gfo.model)

        merged_blocks_mask = []
        i = 0
        while i < self.blocked_dims - 1:
            if len(blocks_mask[i]) < 2:
                merged_blocks_mask.append(
                    np.concatenate([blocks_mask[i], blocks_mask[i + 1]])
                )
                # print(np.concatenate([blocks_mask[i], blocks_mask[i + 1]]).shape)
                i += 2
            else:
                merged_blocks_mask.append(blocks_mask[i])
                i += 1

        self.blocked_dims = len(merged_blocks_mask)
        print("New blocked dimensions", self.blocked_dims)

        with open(new_path, "wb") as f:
            pickle.dump(merged_blocks_mask, f)

        self.path = new_path

        return merged_blocks_mask
