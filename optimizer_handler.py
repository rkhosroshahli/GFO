import warnings

import numpy as np
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.survival import Survival
from pymoo.operators.survival.rank_and_crowding.metrics import (
    get_crowding_function,
)

from block import Block
from gfo import GradientFreeOptimization, PercisionRecallProblem


def save_result(res, problem, algorithm, gfo, output_dir, run, gb_f1, gb_f2):
    X, F = res.opt.get("X", "F")
    n_evals = res.algorithm.callback.data["n_evals"]  # corresponding number of function evaluations
    hist_opt_F = res.algorithm.callback.data["opt_F"]  # the objective space values in each generation
    hist_pop_F = res.algorithm.callback.data["pop_F"]  # the objective space values in each generation

    np.savez(f"{output_dir}/last_pf_{run}", X=X, F=F)
    np.savez(f"{output_dir}/last_pop_{run}", X=res.pop.get("X"), F=res.pop.get("F"))

    import pickle

    with open(f"{output_dir}/pop_history_{run}.pickle", "wb") as f:
        pickle.dump(hist_pop_F, f)

    with open(f"{output_dir}/pf_history_{run}.pickle", "wb") as f:
        pickle.dump(hist_opt_F, f)


    from pymoo.indicators.hv import Hypervolume
    # approx_ideal = F.min(axis=0)
    # approx_nadir = F.max(axis=0)
    metric = Hypervolume(
        pf=problem.pareto_front(),
        zero_to_one=True
    )
    hv = [metric.do(_F) for _F in hist_opt_F]

    n_gens = np.arange(1, (len(hist_opt_F) + 1))
    plt.figure(figsize=(12, 5))
    algo_label = f"{algorithm}"
    algo_label = "MHB-" + algo_label
    plt.plot(n_gens, hv, color="black", lw=0.7, label=algo_label)
    plt.scatter(n_gens, hv, edgecolor="black", marker="o")
    plt.xlabel("Generations")
    plt.ylabel("Hyper volume (HV)")
    plt.title(gfo.data.dataset)
    plt.legend()
    plt.savefig(f"{output_dir}/plot_hv.png")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.scatter(
        1 - res.pop.get("F")[:, 0],
        1 - res.pop.get("F")[:, 1],
        # color="black",
        label="Population",
        facecolor="none",
        edgecolor="black",
        marker="s",
        s=45,
    )
    plt.scatter(
        1 - F[:, 0],
        1 - F[:, 1],
        color="red",
        label="Pareto Front",
        s=20,
    )
    plt.scatter(
        gb_f1[0],
        gb_f2[0],
        color="blue",
        label="Adam",
        s=20,
    )
    plt.scatter(
        gb_f1[1],
        gb_f2[1],
        color="green",
        label="Blocked Adam",
        s=20,
    )
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.grid()
    plt.title(gfo.data.dataset)
    # plt.title(args.dataset)
    plt.legend()
    plt.savefig( f"{output_dir}/plot_pf.png")
    plt.show()


def handle_moo_optimizers(optimizer=None, gfo: GradientFreeOptimization = None, dimensions: int = None, nfe: int = 300,
                          block: Block = None, callback=None, display=None, run: int = 0, output_dir: str = None,
                          *args, **kwargs):
    algorithm = None
    if optimizer == 'NSGA2':
        algorithm = NSGA2(
            **kwargs
        )
    elif optimizer == 'NSGA3':
        algorithm = NSGA3(
            **kwargs
        )
    else:
        raise ValueError("Optimizer not recognized")

    plot_link = f"{output_dir}/plot_{run}"

    problem = PercisionRecallProblem(n_var=dimensions, sample_size=gfo.data.num_samples, gfo=gfo, block=block,
                                     data=gfo.data)

    model_params = gfo.get_parameters(
        gfo.model
    )
    out = {"F": None}
    problem._evaluate(np.array([model_params]), out)
    F = out["F"]
    gb_precision, gb_recall = 1 - F[0, 0], 1 - F[0, 1]


    (block_solution,) = block.blocker(np.array([model_params]))
    out = {"F": None}
    problem._evaluate(np.array([block_solution]), out)
    F = out["F"]
    block_gb_precision, block_gb_recall = 1 - F[0, 0], 1 - F[0, 1]

    warnings.filterwarnings("ignore")
    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_eval", nfe),
        seed=1,
        verbose=True,
        save_history=False,
        callback=callback(),
        output=display(),
        gb_f1=[gb_precision, block_gb_precision],
        gb_f2=[gb_recall, block_gb_recall],
        plot_link=plot_link,
        test_validator=gfo.validation_func,
        train_validator=gfo.train_func,
        unblocker=block.unblocker
    )

    save_result(res, problem, algorithm, gfo, output_dir, run, gb_f1=[gb_precision, block_gb_precision],
                gb_f2=[gb_recall, block_gb_recall])

