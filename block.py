import math
import os
from matplotlib import pyplot as plt
import numpy as np
import pickle

import pandas as pd
from bi_obj_optimization import CustomProblem, moo_callback

from gfo import GradientFreeOptimization


class Block:
    def __init__(
            self,
            scheme,
            gfo,
            dims,
            block_file=None,
            save_dir=None,
            arch=None,
            dataset=None,
            **kwargs,
    ):

        self.scheme = scheme
        self.gfo = gfo
        self.dir = save_dir
        self.file = save_dir + '/' + block_file + '.pickle'
        self.block_file = block_file
        self.dims = dims
        self.blocked_dims = None
        self.arch = arch
        self.dataset = dataset
        self.blocker = None
        self.unblocker = None
        self.generator = None
        if "search" in scheme:
            self.blocker = self.optimized_blocker
            self.unblocker = self.optimized_unblocker
            self.generator = self.optimized_block_generator
            self.find_optimal_dimensions(gfo=gfo)
        elif "random" in scheme:
            self.blocker = self.randomized_blocker
            self.unblocker = self.randomized_unblocker
            self.generator = self.randomized_block_generator
            num_blocks = kwargs["num_blocks"]
            self.randomized_block_generator(gfo=gfo, max_dims=num_blocks)
        elif "1bin" in scheme:
            self.blocker = self.optimized_blocker
            self.unblocker = self.optimized_unblocker
            self.generator = self.optimized_block_generator
            num_bins = kwargs["num_bins"]
            self.optimized_block_generator(gfo=gfo, num_bins=num_bins)
        else:
            raise ValueError("The block scheme is not recognized!")
        self.scheme = scheme

    def load_mask(self, path=None):
        if path == None:
            path = self.file
        blocks_mask = None
        with open(path, "rb") as f:
            blocks_mask = pickle.load(f)
        return blocks_mask

    def save_mask(self, blocks_mask, max_dims=None):
        blocks_mask_path = f"{self.dir}/opt/codebook_{self.arch}_{self.dataset}_optimal_maxD{max_dims}.pickle"
        with open(
                blocks_mask_path,
                "wb",
        ) as f:
            pickle.dump(blocks_mask, f)
        return blocks_mask_path

    def find_optimal_dimensions(self, gfo: GradientFreeOptimization):

        model_params = gfo.get_parameters(gfo.model).tolist()
        adam_f = gfo.evaluate_params(np.array(model_params), gfo.data_loader)
        adam_test_f = gfo.evaluate_params(np.array(model_params), gfo.val_loader)
        print("Adam score w/o block on samples:", adam_f)
        print("Adam score w/o block on test:", adam_test_f)

        opt_fs = []
        opt_dims = []
        org_fs = []
        org_dims = []
        nbins_paths = []
        nbins = []
        rng = [
            pow(base=10, exp=exp) for exp in range(1, int(math.log10(self.dims)) - 1)
        ]
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize

        if not os.path.exists(f"{self.dir}/{gfo.model_name}_{gfo.dataset}"):
            os.makedirs(f"{self.dir}/{gfo.model_name}_{gfo.dataset}")
            os.makedirs(f"{self.dir}/{gfo.model_name}_{gfo.dataset}/solutions")

        problem = CustomProblem(n_var=1, xl=10, xu=300, params=model_params, gfo=gfo, block=self)
        algorithm = NSGA2(pop_size=10)

        pf_X, pf_F, pop_X, pop_F = None, None, None, None
        if os.path.isfile(f"{self.dir}/{gfo.model_name}_{gfo.dataset}/last_pf.npz"):
            pop_npz = np.load(f"{self.dir}/{gfo.model_name}_{gfo.dataset}/last_pop.npz")
            pop_X, pop_F = pop_npz["X"], pop_npz["F"]
            pf_npz = np.load(f"{self.dir}/{gfo.model_name}_{gfo.dataset}/last_pf.npz")
            pf_X, pf_F = pf_npz["X"], pf_npz["F"]
        else:
            df = pd.DataFrame(
                {
                    "Gen": [0],
                    "Bins": [0],
                    "F(x)": [adam_f],
                    "D": [len(model_params)],
                }
            )
            df.to_csv(
                f"{self.dir}/{gfo.model_name}_{gfo.dataset}/current_pf_state.csv",
                index=False,
            )
            res = minimize(problem, algorithm, ("n_gen", 10), seed=1, verbose=True,
                           callback=moo_callback(block=self, gfo=gfo))
            pf_X = res.X
            pf_F = res.F
            pop_X = res.pop.get("X")
            pop_F = res.pop.get("F")
            np.savez(
                f"{self.dir}/{gfo.model_name}_{gfo.dataset}/last_pf", X=pf_X, F=pf_F
            )
            np.savez(
                f"{self.dir}/{gfo.model_name}_{gfo.dataset}/last_pop", X=pop_X, F=pop_F
            )

        opt_dims = []
        opt_fs = []
        opt_test_fs = []
        for j in range(len(pop_X)):
            nbins.append(int(pop_X[j]))
            mask_path = f"{self.dir}/{gfo.model_name}_{gfo.dataset}/solutions/solution_{int(pop_X[j])}bins.pickle"
            nbins_paths.append(mask_path)
            x_merged_blocks = self.load_mask(mask_path)
            opt_dims.append(len(x_merged_blocks))

            (x_merged_params,) = self.optimized_unblocker(
                self.optimized_blocker(np.array([model_params]), x_merged_blocks),
                x_merged_blocks,
            )
            x_f = gfo.evaluate_params(x_merged_params, data_loader=gfo.data_loader)
            opt_fs.append(x_f)
            x_test_f = gfo.evaluate_params(x_merged_params, data_loader=gfo.val_loader)
            opt_test_fs.append(x_test_f)

        df = pd.DataFrame(
            {
                "Bins": nbins,
                "D": opt_dims,
                "Sample F(x)": opt_fs,
                "Test F(x)": opt_test_fs,
            }
        )
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "Bins": [0],
                        "D": [self.dims],
                        "Sample F(x)": [adam_f],
                        "Test F(x)": [adam_test_f],
                    }
                ),
            ]
        )
        # df.to_excel(f"{self.dir}/opt/optimization_history_excel.xlsx", index=False)
        df.to_csv(
            f"{self.dir}/{gfo.model_name}_{gfo.dataset}/optimization_history.csv",
            index=False,
        )

        plt.scatter(x=opt_dims, y=opt_fs, label=f"train")
        plt.scatter(x=opt_dims, y=opt_test_fs, label="test")
        plt.xlabel("#Parameters")
        plt.ylabel("F1-score")
        plt.legend()
        plt.savefig(
            f"{self.dir}/{gfo.model_name}_{gfo.dataset}/optimization_pareto_plot.png"
        )
        # plt.show()
        plt.close()

        df.to_excel(
            f"{self.dir}/{gfo.model_name}_{gfo.dataset}/optimization_history_excel.xlsx",
            index=False,
        )

        argmax = np.argmax(opt_test_fs)
        best_block = self.load_mask(path=nbins_paths[argmax])
        self.file = nbins_paths[argmax]
        np.savez(
            f"{self.dir}/{gfo.model_name}_{gfo.dataset}/optimization_fs_dims",
            fs=opt_fs,
            dims=opt_dims,
            paths=nbins_paths,
        )

        return best_block

    def optimized_block_generator(self, gfo, num_bins, seed=None):
        params = gfo.get_parameters(gfo.model)
        if os.path.exists(self.file):
            blocks_mask = self.load_mask()
            new_blocked_dims = len(blocks_mask)
            print("Optimized blocked dimensions:", new_blocked_dims)
            self.blocked_dims = new_blocked_dims
            # print("Merging until no improvement...")
            # blocks_mask = self.merge_till_no_improvement(self.file, gfo, params)
            return blocks_mask

        # Calculate the bin edges
        bin_edges = np.linspace(params.min(), params.max(), num_bins)
        # Split the data into bins
        binned_data = np.digitize(params, bin_edges)

        blocks_mask = []
        for i in range(num_bins):
            b_i = np.where(binned_data == i)[0]
            if len(b_i) != 0:
                blocks_mask.append(b_i)
        new_blocked_dims = len(blocks_mask)
        print("Optimal blocked dimensions:", new_blocked_dims)
        self.blocked_dims = new_blocked_dims

        with open(self.file, "wb") as f:
            pickle.dump(blocks_mask, f)

        return blocks_mask

    def randomized_block_generator(
            self,
            gfo=None,
            max_dims=0,
            seed=None,
    ):
        if os.path.exists(self.file):
            with open(self.file, "rb") as f:
                blocks_mask = pickle.load(f)
            new_blocked_dims = len(blocks_mask)
            print("Randomized blocked dimensions:", new_blocked_dims)
            self.blocked_dims = new_blocked_dims
            return blocks_mask

        params = gfo.get_parameters(gfo.model)
        dims = len(params)
        block_size = int(math.ceil(dims / max_dims))
        rng = np.random.default_rng(seed)
        tries = 0
        while True:
            blocks_mask = np.arange(
                dims + ((block_size - (dims % block_size)))
            )

            rng.shuffle(blocks_mask)
            blocks_mask = blocks_mask.reshape((max_dims, block_size))
            print(np.sum(dims > blocks_mask[:, 0]))
            if np.sum(dims > blocks_mask[:, 0]) == max_dims:
                break
            tries += 1
        self.blocked_dims = max_dims

        with open(self.file, "wb") as f:
            pickle.dump(blocks_mask, f)

        return blocks_mask

    def optimized_unblocker(self, pop_blocked, blocks_mask=None):
        if blocks_mask == None:
            blocks_mask = self.load_mask()
        blocked_dims = len(blocks_mask)
        pop_unblocked = np.ones((pop_blocked.shape[0], self.dims))
        for i_p in range(pop_blocked.shape[0]):
            for i in range(blocked_dims):
                # print(blocks_mask[i].shape)
                pop_unblocked[i_p, blocks_mask[i]] *= pop_blocked[i_p, i]
        return pop_unblocked

    def randomized_unblocker(self, pop_blocked):
        blocks_mask = self.load_mask()
        block_size = int(math.ceil(self.dims / self.blocked_dims))
        pop_unblocked = np.ones(
            (
                len(pop_blocked),
                self.dims + ((block_size - (self.dims % block_size))),
            )
        )

        for i in range(self.blocked_dims):
            pop_unblocked[:, blocks_mask[i, :]] *= pop_blocked[:, i]

        return pop_unblocked[:, : self.dims]

    def randomized_blocker(self, pop):
        blocks_mask = self.load_mask()
        dim_to_block = 0
        return pop[:, blocks_mask[:, dim_to_block]].copy()

    def optimized_blocker(self, pop=None, blocks_mask=None):
        if blocks_mask == None:
            blocks_mask = self.load_mask()
        blocked_dims = len(blocks_mask)
        params_blocked = np.zeros((pop.shape[0], blocked_dims))
        for i_p in range(pop.shape[0]):
            for i in range(blocked_dims):
                block_params = pop[i_p, blocks_mask[i]]
                if len(block_params) != 0:
                    params_blocked[i_p, i] = np.mean(block_params)

        return params_blocked

    def merge_blocks(self, blocks_mask, new_path):
        if os.path.exists(new_path):
            self.file = new_path
            blocks_mask = self.load_mask()
            new_blocked_dims = len(blocks_mask)
            print("Optimized and merged blocked dimensions:", new_blocked_dims)
            self.blocked_dims = new_blocked_dims
            return blocks_mask

        merged_blocks_mask = []
        i = 0
        while i < self.blocked_dims - 1:
            if len(blocks_mask[i]) < 2:
                merged_blocks_mask.append(
                    np.concatenate([blocks_mask[i], blocks_mask[i + 1]])
                )
                i += 2
            else:
                merged_blocks_mask.append(blocks_mask[i])
                i += 1

        self.blocked_dims = len(merged_blocks_mask)
        print("New blocked dimensions", self.blocked_dims)

        with open(new_path, "wb") as f:
            pickle.dump(merged_blocks_mask, f)

        self.file = new_path

        return merged_blocks_mask

    def merge_till_no_improvement(self, path, gfo, params):
        blocks_mask = self.load_mask(path)
        org_dims = len(blocks_mask)
        mask_size = org_dims

        best_f = gfo.evaluate_params(
            self.optimized_unblocker(
                self.optimized_blocker(
                    np.array([params.copy()]), blocks_mask=blocks_mask
                ),
                blocks_mask=blocks_mask,
            )[0],
            data_loader=gfo.data_loader,
        )

        improvement = True
        while improvement:
            improvement = False
            i = 1
            while i < mask_size - 1:
                # Left
                left_merged_blocks = (
                        blocks_mask[0: i - 1]
                        + [np.concatenate([blocks_mask[i - 1], blocks_mask[i]])]
                        + blocks_mask[i + 1:]
                )

                (left_merged_params,) = self.optimized_unblocker(
                    self.optimized_blocker(
                        np.array([params.copy()]), left_merged_blocks
                    ),
                    left_merged_blocks,
                )

                left_f = gfo.evaluate_params(
                    left_merged_params, data_loader=gfo.data_loader
                )
                # Right
                right_merged_blocks = (
                        blocks_mask[0:i]
                        + [np.concatenate([blocks_mask[i], blocks_mask[i + 1]])]
                        + blocks_mask[i + 2:]
                )

                (right_merged_params,) = self.optimized_unblocker(
                    self.optimized_blocker(
                        np.array([params.copy()]), right_merged_blocks
                    ),
                    right_merged_blocks,
                )
                right_f = gfo.evaluate_params(
                    right_merged_params, data_loader=gfo.data_loader
                )
                print(
                    f"Bins:{org_dims}, D:{i + 1}/{org_dims}", [left_f, right_f, best_f]
                )
                argmax_f = np.argmax([left_f, right_f, best_f])
                if argmax_f == 0:
                    blocks_mask = left_merged_blocks
                    best_f = left_f
                    mask_size -= 1
                    improvement = True
                    # i -= 1
                elif argmax_f == 1:
                    blocks_mask = right_merged_blocks
                    best_f = right_f
                    mask_size -= 1
                    improvement = True
                    # i -= 1
                else:
                    i += 1

        with open(self.dir + '/' + self.block_file + '_remerged'+'.pickle', "wb") as f:
            pickle.dump(blocks_mask, f)

        (merged_params,) = self.optimized_unblocker(
            self.optimized_blocker(
                np.array([params.copy()]), blocks_mask
            ),
            blocks_mask,
        )

        best_test_f = gfo.evaluate_params(
            merged_params, data_loader=gfo.val_loader
        )
        df = pd.read_csv(f"{self.dir}/{gfo.model_name}_{gfo.dataset}/optimization_history.csv")
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "Bins": [org_dims],
                        "D": [len(blocks_mask)],
                        "Sample F(x)": [best_f],
                        "Test F(x)": [best_test_f],
                    }
                ),
            ]
        )
        # df.to_excel(f"{self.dir}/opt/optimization_history_excel.xlsx", index=False)
        df.to_csv(
            f"{self.dir}/{gfo.model_name}_{gfo.dataset}/optimization_history.csv",
            index=False,
        )

        return blocks_mask