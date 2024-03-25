import numpy as np
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import (
    get_crowding_function,
)
from pymoo.util.randomized_argsort import randomized_argsort


class gfo_rankandcrowding(Survival):

    def __init__(self, nds=None, crowding_func="cd"):
        """
        A generalization of the NSGA-II survival operator that ranks individuals by dominance criteria
        and sorts the last front by some user-specified crowding metric. The default is NSGA-II's crowding distances
        although others might be more effective.

        For many-objective problems, try using 'mnn' or '2nn'.

        For Bi-objective problems, 'pcd' is very effective.

        Parameters
        ----------
        nds : str or None, optional
            Pymoo type of non-dominated sorting. Defaults to None.

        crowding_func : str or callable, optional
            Crowding metric. Options are:

                - 'cd': crowding distances
                - 'pcd' or 'pruning-cd': improved pruning based on crowding distances
                - 'ce': crowding entropy
                - 'mnn': M-Neaest Neighbors
                - '2nn': 2-Neaest Neighbors

            If callable, it has the form ``fun(F, filter_out_duplicates=None, n_remove=None, **kwargs)``
            in which F (n, m) and must return metrics in a (n,) array.

            The options 'pcd', 'cd', and 'ce' are recommended for two-objective problems, whereas 'mnn' and '2nn' for many objective.
            When using 'pcd', 'mnn', or '2nn', individuals are already eliminated in a 'single' manner.
            Due to Cython implementation, they are as fast as the corresponding 'cd', 'mnn-fast', or '2nn-fast',
            although they can singnificantly improve diversity of solutions.
            Defaults to 'cd'.
        """

        crowding_func_ = get_crowding_function(crowding_func)

        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = crowding_func_

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        pop_X = pop.get("X")
        algo = kwargs["algorithm"]

        # pop.apply(lambda ind: ind.evaluated.update(out.keys()))
        # algorihtm.evaluator.eval(
        #     problem,
        #     pop,
        #     algorithm=algorithm,
        #     evaluate_values_of=range(0, 100),
        # )
        # get the objective space values and objects

        if len(pop) != algo.pop_size:
            out = {"F": None}
            problem._evaluate(pop_X[0: algo.pop_size], out, reset=True)
            # print(out["F"])
            pop[0: algo.pop_size].set("F", out["F"])

            algo.evaluator.n_eval += algo.pop_size

        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            I = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(I) > n_survive:

                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = self.crowding_func.do(
                    F[front, :], n_remove=n_remove
                )

                I = randomized_argsort(
                    crowding_of_front, order="descending", method="numpy"
                )
                I = I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = self.crowding_func.do(
                    F[front, :], n_remove=0
                )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]