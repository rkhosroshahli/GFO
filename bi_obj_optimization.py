import os
import pickle
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback

class CustomProblem(Problem):
    def __init__(self, n_var=1, xl=10, xu=1000, params=None, gfo=None, block=None):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            vtype=int,
        )
        self.params = params
        self.gfo = gfo
        self.block = block

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        # return np.array([x, 1 - np.power(x, 2)]).T
        return np.ones((n_pareto_points, 2))

    def _evaluate(self, X, out, *args, **kwargs):

        NP = X.shape[0]
        f1 = np.zeros(NP)
        f2 = np.zeros(NP)
        nbins = np.zeros(NP)
        for j in range(NP):
            blocks_mask = None
            B = int(X[j])
            nbins[j] = B
            x_path = f"{self.block.dir}/{self.gfo.model_name}_{self.gfo.dataset}/solutions/solution_{B}bins.pickle"
            # condition to check if a file exists
            if os.path.exists(x_path):
                with open(
                        x_path,
                        "rb",
                ) as f:
                    blocks_mask = pickle.load(f)

                best_f = self.gfo.evaluate_params(
                    self.block.optimized_unblocker(
                        self.block.optimized_blocker(
                            np.array([self.params]), blocks_mask=blocks_mask
                        ),
                        blocks_mask=blocks_mask,
                    )[0],
                    data_loader=self.gfo.data_loader,
                )
                f1[j] = best_f
                f2[j] = len(blocks_mask)
                continue

            bin_edges = np.linspace(np.min(self.params), np.max(self.params), B)
            # Split the data into bins
            binned_data = np.digitize(self.params, bin_edges)
            blocks_mask = []
            for i in range(B):
                b_i = np.where(binned_data == i)[0]
                if len(b_i) != 0:
                    blocks_mask.append(b_i)

            best_f = self.gfo.evaluate_params(
                self.block.optimized_unblocker(
                    self.block.optimized_blocker(
                        np.array([self.params]), blocks_mask=blocks_mask
                    ),
                    blocks_mask=blocks_mask,
                )[0],
                data_loader=self.gfo.data_loader,
            )

            new_dims = len(blocks_mask)
            mask_size = new_dims
            i = 1
            while i < mask_size - 1:
                # Left
                left_merged_blocks = (
                        blocks_mask[0: i - 1]
                        + [np.concatenate([blocks_mask[i - 1], blocks_mask[i]])]
                        + blocks_mask[i + 1:]
                )

                (left_merged_params,) = self.block.optimized_unblocker(
                    self.block.optimized_blocker(
                        np.array([self.params.copy()]), left_merged_blocks
                    ),
                    left_merged_blocks,
                )

                left_f = self.gfo.evaluate_params(
                    left_merged_params, data_loader=self.gfo.data_loader
                )
                # Right
                right_merged_blocks = (
                        blocks_mask[0:i]
                        + [np.concatenate([blocks_mask[i], blocks_mask[i + 1]])]
                        + blocks_mask[i + 2:]
                )

                (right_merged_params,) = self.block.optimized_unblocker(
                    self.block.optimized_blocker(
                        np.array([self.params.copy()]), right_merged_blocks
                    ),
                    right_merged_blocks,
                )
                right_f = self.gfo.evaluate_params(
                    right_merged_params, data_loader=self.gfo.data_loader
                )
                # print(
                #     f"Bins:{B}, D:{i+1}/{new_dims}", [left_f, right_f, best_f]
                # )
                argmax_f = np.argmax([left_f, right_f, best_f])
                if argmax_f == 0:
                    blocks_mask = left_merged_blocks
                    best_f = left_f
                    mask_size -= 1
                    # i -= 1
                elif argmax_f == 1:
                    blocks_mask = right_merged_blocks
                    best_f = right_f
                    mask_size -= 1
                    # i -= 1
                else:
                    i += 1

            x_path = f"{self.block.dir}/{self.gfo.model_name}_{self.gfo.dataset}/solutions/solution_{B}bins.pickle"
            # condition to check if a file exists
            if not os.path.exists(x_path):
                with open(
                        x_path,
                        "wb",
                ) as f:
                    pickle.dump(blocks_mask, f)

            f1[j] = best_f
            f2[j] = len(blocks_mask)

        out["F"] = np.column_stack([1 - f1, f2])
        return out


class moo_callback(Callback):

    def __init__(self, block, gfo) -> None:
        super().__init__()
        self.block = block
        self.gfo = gfo

    def notify(self, algorithm):
        X = algorithm.opt.get("X")
        F = algorithm.opt.get("F")
        n_gen = algorithm.n_gen

        df_path = f"{self.block.dir}/{self.gfo.model_name}_{self.gfo.dataset}/current_pf_state.csv"
        df = pd.read_csv(df_path)
        df_pop = pd.DataFrame(
            {
                "Gen": [n_gen] * X.shape[0],
                "Bins": np.concatenate(X).astype(int),
                "F(x)": 1 - F[:, 0],
                "D": F[:, 1],
            }
        )
        df = pd.concat([df, df_pop])
        # df.to_excel(f"{self.dir}/opt/optimization_history_excel.xlsx", index=False)
        df.to_csv(df_path, index=False)
