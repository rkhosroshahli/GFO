from matplotlib import pyplot as plt
from pymoo.core.callback import Callback


class moo_callback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["opt_F"] = []
        self.data["pop_F"] = []
        self.data["n_evals"] = []

    def notify(self, algorithm):
        self.data["opt_F"].append(algorithm.opt.get("F"))
        self.data["pop_F"].append(algorithm.pop.get("F"))
        self.data["n_evals"].append(algorithm.evaluator.n_eval)

        if algorithm.n_iter % 2:
            plt.figure(figsize=(5, 5))
            plt.scatter(
                1 - algorithm.pop.get("F")[:, 0],
                1 - algorithm.pop.get("F")[:, 1],
                # color="black",
                label="Population",
                facecolor="none",
                edgecolor="black",
                marker="s",
                s=45,
            )
            plt.scatter(
                1 - algorithm.opt.get("F")[:, 0],
                1 - algorithm.opt.get("F")[:, 1],
                color="red",
                label="Pareto Front",
                s=20,
            )

            gb_precision, gb_recall = (
                algorithm.gb_f1[0],
                algorithm.gb_f2[0],
            )
            plt.scatter(
                gb_precision,
                gb_recall,
                color="blue",
                label="Adam",
                s=20,
            )
            gb_precision, gb_recall = (
                algorithm.gb_f1[1],
                algorithm.gb_f2[1],
            )
            plt.scatter(
                gb_precision,
                gb_recall,
                color="green",
                label="Adam Blocked",
                s=20,
            )
            # plt.title(shared_link)
            plt.grid()
            plt.ylabel("Recall")
            plt.xlabel("Precision")
            plt.legend(
                bbox_to_anchor=(0, 1.02, 1, 0.2),
                loc="lower left",
                mode="expand",
                borderaxespad=0,
                ncol=3,
            )

            plt.savefig(algorithm.plot_link + "_paretofront.png")
            plt.close()