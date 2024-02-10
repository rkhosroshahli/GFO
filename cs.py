from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from block import *


def cs_3point(
    fitness,
    best_solution,
    best_fitness,
    fitness_history,
    var_max,
    var_min,
    nit,
    nfe,
    max_nfe,
    block=None,
    plot_link=None,
    history_link=None,
    validation_func=None,
):
    dimensions = len(best_solution)
    if dimensions != block.dims:
        print("Dimensions is not equal to block object dimensions.")

    if block and len(best_solution) == block.dims:
        (best_solution,) = block.blocker(np.array([best_solution]))
        dimensions = len(best_solution)

        (ubs,) = block.unblocker(np.array([best_solution]))
        best_fitness = fitness(ubs)
        val_f1 = validation_func(ubs)
        print(
            f"Adam blocked, f(x)= {best_fitness:.4f}, 1-f(x)= {(1-best_fitness):.4f}, g(x)={val_f1:.4f}",
        )

    nfe += 1

    iteration_col = []
    dims_col = []
    error_col = []
    f1s_col = []
    val_col = []

    # fitness_history = [best_fitness]

    shuffled_dims = np.arange(dimensions)
    for i in range(nit):
        if nfe >= max_nfe:
            status_message = "Maximum number of function evaluations has been reached."
            print(status_message)
            break
        np.random.shuffle(shuffled_dims)
        it = 0
        for d in shuffled_dims:
            l_d = var_max[d] - var_min[d]
            c1 = best_solution.copy()
            c2 = best_solution.copy()
            c1[d] = var_min[d] + l_d / 4
            c2[d] = var_max[d] - l_d / 4
            # print(var_min[d], var_max[d], l_d, c1[d], c2[d])
            c1_f = None
            c2_f = None
            cands = np.array([c1, c2])
            if block:
                ubc1, ubc2 = block.unblocker(cands)
                c1_f, c2_f = [fitness(c) for c in [ubc1, ubc2]]
            else:
                c1_f, c2_f = [fitness(c) for c in (cands)]
            # print(best_fitness, c1_f, c2_f)
            # print(best_fitness - c1_f, best_fitness - c2_f)
            if c1_f < best_fitness and c1_f < c2_f:
                if c2_f < best_fitness:
                    fitness_history.append(c2_f)
                best_solution = c1.copy()
                best_fitness = c1_f
                var_max[d] -= l_d / 2
                fitness_history.append(best_fitness)
                val_f1 = validation_func(ubc1)
                print(
                    f"CS, iteration{i}, D{d}, f(x)= {best_fitness:.4f}, 1-f(x)= {(1 - best_fitness):.4f}, g(x)={val_f1:.4f}",
                )
                iteration_col.append(i)
                dims_col.append(d)
                error_col.append(best_fitness)
                f1s_col.append(1 - best_fitness)
                val_col.append(val_f1)
            elif c2_f < best_fitness and c2_f < c1_f:
                if c1_f < best_fitness:
                    fitness_history.append(c1_f)
                best_solution = c2.copy()
                best_fitness = c2_f
                var_min[d] += l_d / 2
                fitness_history.append(best_fitness)
                val_f1 = validation_func(ubc2)
                print(
                    f"CS, iteration{i}, D{d}, f(x)= {best_fitness:.4f}, 1-f(x)= {(1 - best_fitness):.4f}, g(x)={val_f1:.4f}",
                )
                iteration_col.append(i)
                dims_col.append(d)
                error_col.append(best_fitness)
                f1s_col.append(1 - best_fitness)
                val_col.append(val_f1)
            else:
                var_min[d] += l_d / 4
                var_max[d] -= l_d / 4
                fitness_history.append(best_fitness)
                fitness_history.append(best_fitness)

                iteration_col.append(i)
                dims_col.append(d)
                error_col.append(c1_f)
                f1s_col.append(1 - c1_f)
                val_col.append(0)

                iteration_col.append(i)
                dims_col.append(d)
                error_col.append(c2_f)
                f1s_col.append(1 - c2_f)
                val_col.append(0)

            if plot_link and nfe % 9 == 0:
                # Plot codes come here
                plt.figure(figsize=(12, 5))
                plt.plot(1 - np.asarray(fitness_history), label="F1-score")
                plt.xlabel("FEs")
                plt.ylabel("Fitness")
                plt.legend()
                plt.grid()
                if block:
                    plt.title("Block CS")
                else:
                    plt.title("CS")
                plt.savefig(plot_link)
                plt.close()
            if history_link and nfe % 9 == 0:
                np.savez(
                    history_link,
                    fitness_history=fitness_history,
                    best_solution=best_solution,
                    var_min=var_min,
                    var_max=var_max,
                )
                df = pd.DataFrame(
                    {
                        "iteration": iteration_col,
                        "dimension": dims_col,
                        "error": error_col,
                        "1-error": f1s_col,
                        "val": val_col,
                    }
                )
                df.to_csv(history_link + ".csv", index=False)
            nfe += 2

        cands = np.array([best_solution])
        if block:
            (ubs,) = block.unblocker(cands)
            (best_fitness,) = [fitness(c) for c in [ubs]]
        else:
            (best_fitness,) = [fitness(c) for c in (cands)]

    if history_link:
        np.savez(
            history_link,
            fitness_history=fitness_history,
            best_solution=best_solution,
            var_min=var_min,
            var_max=var_max,
        )
