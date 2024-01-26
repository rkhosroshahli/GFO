"""
differential_evolution: The differential evolution global optimization algorithm
Added by Andrew Nelson 2014
"""
import warnings
import os
import numpy as np
from scipy.optimize import OptimizeResult, minimize

# from scipy.optimize._optimize import _status_message
# from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper

from scipy.optimize._constraints import (
    Bounds,
    new_bounds_to_old,
    NonlinearConstraint,
    LinearConstraint,
)
from scipy.sparse import issparse

__all__ = ["differential_evolution"]


_MACHEPS = np.finfo(np.float64).eps

_status_message = {
    "success": "Optimization terminated successfully.",
    "maxfev": "Maximum number of function evaluations has " "been exceeded.",
    "maxiter": "Maximum number of iterations has been " "exceeded.",
    "pr_loss": "Desired error not necessarily achieved due " "to precision loss.",
    "nan": "NaN result encountered.",
    "out_of_bounds": "The result is outside of the provided " "bounds.",
}

import matplotlib.pyplot as plt


def block_differential_evolution(
    func,
    bounds,
    args=(),
    strategy="rand1bin",
    maxiter=30000,
    popsize=100,
    tol=0,
    mutation=0.5,
    recombination=0.9,
    seed=None,
    callback=None,
    disp=True,
    polish=False,
    local_search=False,
    init="random",
    atol=0,
    updating="deferred",
    workers=1,
    constraints=(),
    x0=None,
    *,
    integrality=None,
    vectorized=False,
    block_size=None,
    blocked_dimensions=None,
    save_link=None,
    plot_link=None,
    blocks_link=None,
    val_func=None,
):
    # using a context manager means that any created Pool objects are
    # cleared up.
    with DifferentialEvolutionSolver(
        func,
        bounds,
        args=args,
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        polish=polish,
        local_search=local_search,
        callback=callback,
        disp=disp,
        init=init,
        atol=atol,
        updating=updating,
        workers=workers,
        constraints=constraints,
        x0=x0,
        integrality=integrality,
        vectorized=vectorized,
        block_size=block_size,
        blocked_dimensions=blocked_dimensions,
        save_link=save_link,
        plot_link=plot_link,
        blocks_link=blocks_link,
        val_func=val_func,
    ) as solver:
        ret = solver.solve()

    return ret


class DifferentialEvolutionSolver:
    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {
        "best1bin": "_best1",
        "randtobest1bin": "_randtobest1",
        "currenttobest1bin": "_currenttobest1",
        "best2bin": "_best2",
        "rand2bin": "_rand2",
        "rand1bin": "_rand1",
        "order1bin": "_order1",
    }
    _exponential = {
        "best1exp": "_best1",
        "rand1exp": "_rand1",
        "randtobest1exp": "_randtobest1",
        "currenttobest1exp": "_currenttobest1",
        "best2exp": "_best2",
        "rand2exp": "_rand2",
    }

    __init_error_msg = (
        "The population initialization method must be one of "
        "'latinhypercube' or 'random', or an array of shape "
        "(S, N) where N is the number of parameters and S>5"
    )

    def __init__(
        self,
        func,
        bounds,
        args=(),
        strategy="rand1bin",
        maxiter=1000,
        popsize=100,
        tol=0,
        mutation=0.5,
        recombination=0.9,
        seed=None,
        maxfun=None,
        callback=None,
        disp=False,
        polish=False,
        local_search=False,
        init="random",
        atol=0,
        updating="deferred",
        workers=1,
        constraints=(),
        x0=None,
        *,
        integrality=None,
        vectorized=False,
        block_size=None,
        blocked_dimensions=None,
        true_dimensions=None,
        save_link=None,
        plot_link=None,
        blocks_link=None,
        val_func=None,
    ):
        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy

        self.callback = callback
        self.polish = polish
        self.local_search = local_search

        # set the updating / parallelisation options
        if updating in ["immediate", "deferred"]:
            self._updating = updating

        self.vectorized = vectorized

        # want to use parallelisation, but updating is immediate
        if workers != 1 and updating == "immediate":
            warnings.warn(
                "differential_evolution: the 'workers' keyword has"
                " overridden updating='immediate' to"
                " updating='deferred'",
                UserWarning,
                stacklevel=2,
            )
            self._updating = "deferred"

        if vectorized and workers != 1:
            warnings.warn(
                "differential_evolution: the 'workers' keyword"
                " overrides the 'vectorized' keyword",
                stacklevel=2,
            )
            self.vectorized = vectorized = False

        if vectorized and updating == "immediate":
            warnings.warn(
                "differential_evolution: the 'vectorized' keyword"
                " has overridden updating='immediate' to updating"
                "='deferred'",
                UserWarning,
                stacklevel=2,
            )
            self._updating = "deferred"

        # an object with a map method.
        if vectorized:

            def maplike_for_vectorized_func(func, x):
                # send an array (N, S) to the user func,
                # expect to receive (S,). Transposition is required because
                # internally the population is held as (S, N)
                return np.atleast_1d(func(x.T))

            workers = maplike_for_vectorized_func

        self._mapwrapper = MapWrapper(workers)

        # relative and absolute tolerances for convergence
        self.tol, self.atol = tol, atol

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.scale = mutation
        if (
            not np.all(np.isfinite(mutation))
            or np.any(np.array(mutation) >= 2)
            or np.any(np.array(mutation) < 0)
        ):
            raise ValueError(
                "The mutation constant must be a float in "
                "U[0, 2), or specified as a tuple(min, max)"
                " where min < max and min, max are in U[0, 2)."
            )

        self.dither = None
        if hasattr(mutation, "__iter__") and len(mutation) > 1:
            self.dither = mutation

        self.cross_over_probability = recombination

        # we create a wrapped function to allow the use of map (and Pool.map
        # in the future)
        self.func = _FunctionWrapper(func, args)
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        if isinstance(bounds, Bounds):
            self.limits = np.array(
                new_bounds_to_old(bounds.lb, bounds.ub, len(bounds.lb)), dtype=float
            ).T
        else:
            self.limits = np.array(bounds, dtype="float").T

        if np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits)):
            raise ValueError(
                "bounds should be a sequence containing "
                "real valued (min, max) pairs for each value"
                " in x"
            )

        if maxiter is None:  # the default used to be None
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:  # the default used to be None
            maxfun = (maxiter) * (popsize)
        self.maxfun = maxfun
        self.nit_cd = 10
        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.parameter_count = np.size(self.limits, 1)

        self.random_number_generator = check_random_state(seed)

        # Which parameters are going to be integers?
        if np.any(integrality):
            # # user has provided a truth value for integer constraints
            integrality = np.broadcast_to(integrality, self.parameter_count)
            integrality = np.asarray(integrality, bool)
            # For integrality parameters change the limits to only allow
            # integer values lying between the limits.
            lb, ub = np.copy(self.limits)

            lb = np.ceil(lb)
            ub = np.floor(ub)
            if not (lb[integrality] <= ub[integrality]).all():
                # there's a parameter that doesn't have an integer value
                # lying between the limits
                raise ValueError(
                    "One of the integrality constraints does not"
                    " have any possible integer values between"
                    " the lower/upper bounds."
                )
            nlb = np.nextafter(lb[integrality] - 0.5, np.inf)
            nub = np.nextafter(ub[integrality] + 0.5, -np.inf)

            self.integrality = integrality
            self.limits[0, self.integrality] = nlb
            self.limits[1, self.integrality] = nub
        else:
            self.integrality = False

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.
        # the minimum is 5 because 'best2bin' requires a population that's at
        # least 5 long
        self.num_population_members = max(5, popsize)
        self.population_shape = (self.num_population_members, self.parameter_count)

        self.block_size = block_size
        self.blocked_dimensions = blocked_dimensions

        if block_size != None:
            if blocked_dimensions == None:
                self.blocked_dimensions = self.parameter_count // self.block_size
                if self.parameter_count // self.block_size % 10 != 0:
                    self.blocked_dimensions += 1

            if blocks_link == None:
                mask1 = np.ones(
                    (self.num_population_members, self.parameter_count), dtype=bool
                )
                mask2 = np.zeros(
                    (
                        self.num_population_members,
                        self.block_size - (self.parameter_count % self.block_size),
                    ),
                    dtype=bool,
                )
                self.blocks_mask = np.concatenate([mask1, mask2], axis=1)
            elif block_size > 0:
                import pickle

                with open(blocks_link, "rb") as f:
                    self.blocks_mask = pickle.load(f)
            else:
                import pickle

                with open(blocks_link, "rb") as f:
                    self.blocks_mask = pickle.load(f)

        self._nfev = 0

        # infrastructure for constraints
        self.constraints = constraints
        self._wrapped_constraints = []

        if hasattr(constraints, "__len__"):
            # sequence of constraints, this will also deal with default
            # keyword parameter
            for c in constraints:
                self._wrapped_constraints.append(_ConstraintWrapper(c, self.x))
        else:
            self._wrapped_constraints = [_ConstraintWrapper(constraints, self.x)]
        self.total_constraints = np.sum(
            [c.num_constr for c in self._wrapped_constraints]
        )
        self.constraint_violation = np.zeros((self.num_population_members, 1))
        self.feasible = np.ones(self.num_population_members, bool)

        self.disp = disp
        self.save_link = save_link
        self.plot_link = plot_link
        self.best_gens_fitness_history = []  # Best DE steps fitness saves here
        self.local_search_fitness_history = []  # Best local search steps saves here
        self.best_gens_solution = []

        if os.path.exists(save_link + ".npz"):
            npzfile = np.load(save_link + ".npz")
            init = npzfile["last_population"]
            self.best_gens_solution = npzfile["best_solution"].tolist()
            self.best_gens_fitness_history = npzfile["fitness_history"].tolist()
            # self.local_search_fitness_history=npzfile["local_search_fitness_history"]

        self.startiter = len(self.best_gens_fitness_history)
        self.init_population_array(init)
        self.val_func = val_func

    def init_population_array(self, init):
        """
        Initializes the population with a user specified population.

        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (S, N), where N is the number of parameters.
            The population is clipped to the lower and upper bounds.
        """
        # make sure you're using a float array
        popn = np.asfarray(init)

        if np.size(popn, 0) < 5 or len(popn.shape) != 2:
            raise ValueError(
                "The population supplied needs to have shape"
                " (S, len(x)), where S > 4."
            )

        # scale values and clip to bounds, assigning to population
        self.population = popn

        if self.block_size != None:
            if np.size(popn, 1) == self.blocked_dimensions:
                print("Here1")
                self.population_blocked = popn
                # self.population = self._unblocker_optimal(popn)
            elif self.block_size == -1:
                # self.population_blocked = self._blocker_random(self.population)
                # self.population_blocked = self._blocker_random_average(self.population)
                print("Here2")
                self.population_blocked = self._blocker_optimal(self.population)
            else:
                self.population_blocked = self._blocker_random(self.population)

        self.num_population_members = np.size(self.population, 0)

        self.population_shape = (self.num_population_members, self.parameter_count)

        # reset population energies
        self.population_energies = np.full(self.num_population_members, np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver
        """
        return self.population[self.population_energies.argmin()]

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """
        nit, warning_flag = 0, False
        status_message = _status_message["success"]

        # The population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).
        # do the optimization.
        maxdeiter = self.maxiter
        startdeiter = self.startiter
        if self.local_search:
            maxdeiter = (
                (self.maxfun - (self.nit_cd * (self.blocked_dimensions) * 2))
                // self.num_population_members
            ) + 1

        print("Max DE iterations:", maxdeiter)

        for nit in range(startdeiter, maxdeiter):
            # evolve the population by a generation
            try:
                next(self)
                self.best_gens_fitness_history.append(self.population_energies.min())
                # self.best_gens_solution.append(self.population[self.population_energies.argmin()])
                if self.block_size != None:
                    self.best_gens_solution = self.population_blocked[
                        self.population_energies.argmin()
                    ]
                    self.best_gens_solution = self._unblocker_optimal(
                        np.array([self.best_gens_solution])
                    )[0]
                else:
                    self.best_gens_solution = self.population[
                        self.population_energies.argmin()
                    ]
            except StopIteration:
                warning_flag = True
                if self._nfev > self.maxfun:
                    status_message = _status_message["maxfev"]
                elif self._nfev == self.maxfun:
                    status_message = (
                        "Maximum number of function evaluations" " has been reached."
                    )
                break

            if self.disp and nit % 2 == 0:
                self.plot_fitness_save()

            if self.disp and nit % 10 == 1:
                val = self.val_func(self.best_gens_solution)

                print(
                    "differential_evolution step %d: f(x)= %.6f, 1-f(x)= %.6f, 1-f'(x)= %.6f"
                    % (
                        nit,
                        self.population_energies.min(),
                        1 - self.population_energies.min(),
                        val,
                    )
                )
            if nit % 100 == 1:
                pop = None
                if self.block_size == None:
                    pop = self.population
                else:
                    pop = self.population_blocked
                np.savez(
                    self.save_link,
                    best_solution=self.best_gens_solution,
                    fitness_history=self.best_gens_fitness_history,
                    last_population=pop,
                )

        DE_result = OptimizeResult(
            x=self.population[self.population_energies.argmin()],
            fun=self.population_energies.min(),
            nfev=self._nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True),
        )

        # if self.local_search:
        #     best_solution = self.population_blocked[self.population_energies.argmin()]
        #     best_fitness = DE_result.fun
        #     var_max = np.max(self.population_blocked, axis=0)
        #     var_min = np.min(self.population_blocked, axis=0)
        #     for i in range(self.nit_cd):
        #         if self._nfev == self.maxfun:
        #             status_message = (
        #                 "Maximum number of function evaluations has been reached."
        #             )
        #             break
        #         for d in range(self.blocked_dimensions):
        #             l_d = var_max[d] - var_min[d]
        #             c1 = best_solution.copy()
        #             c2 = best_solution.copy()
        #             c1[d] = var_min[d] + l_d / 4
        #             c2[d] = var_max[d] - l_d / 4
        #             c1_f, c2_f = self._calculate_population_energies(
        #                 self._unblocker_random(np.array([c1, c2]))
        #             )

        #             if c1_f < c2_f < best_fitness:
        #                 best_solution = c1.copy()
        #                 best_fitness = c1_f
        #                 var_max[d] -= l_d / 2
        #             elif c2_f < c1_f < best_fitness:
        #                 best_solution = c2.copy()
        #                 best_fitness = c2_f
        #                 var_min[d] += l_d / 2
        #             else:
        #                 var_min[d] += l_d / 4
        #                 var_max[d] -= l_d / 4

        #         if self.disp:
        #             print(
        #                 "local search CD step %d: f(x)= %.6f, 1-f(x)= %.6f"
        #                 % (i, best_fitness, 1 - best_fitness)
        #             )

        #         self.local_search_fitness_history.append(best_fitness)
        #         # self.best_gens_solution.append(self.population[self.population_energies.argmin()])
        #         self.best_gens_solution = self.population[
        #             self.population_energies.argmin()
        #         ]
        #         self.plot_fitness_save()
        #         np.savez(
        #             self.save_link,
        #             best_solution=self.best_gens_solution,
        #             fitness_history=self.best_gens_fitness_history,
        #             local_search_fitness_history=self.local_search_fitness_history,
        #             last_population=self.population,
        #         )

        #     DE_result.nfev = self._nfev
        #     DE_result.fun = best_fitness
        #     #
        #     DE_result.x = self._unblocker_optimal(np.array([best_solution]))[0]
        #     DE_result.message = status_message
        #     # to keep internal state consistent
        #     self.population_energies[0] = best_fitness
        #     self.population[0] = DE_result.x

        #     self.local_search_fitness_history.append(best_fitness)
        #     # self.best_gens_solution.append(self.population[self.population_energies.argmin()])
        #     self.best_gens_solution = self.population[self.population_energies.argmin()]
        #     self.plot_fitness_save()
        #     np.savez(
        #         self.save_link,
        #         best_solution=self.best_gens_solution,
        #         fitness_history=self.best_gens_fitness_history,
        #         local_search_fitness_history=self.local_search_fitness_history,
        #         last_population=self.population,
        #     )

        if self.polish and not np.all(self.integrality):
            # can't polish if all the parameters are integers
            if np.any(self.integrality):
                # set the lower/upper bounds equal so that any integrality
                # constraints work.
                limits, integrality = self.limits, self.integrality
                limits[0, integrality] = DE_result.x[integrality]
                limits[1, integrality] = DE_result.x[integrality]

            polish_method = "L-BFGS-B"

            if self._wrapped_constraints:
                polish_method = "trust-constr"

                constr_violation = self._constraint_violation_fn(DE_result.x)
                if np.any(constr_violation > 0.0):
                    warnings.warn(
                        "differential evolution didn't find a"
                        " solution satisfying the constraints,"
                        " attempting to polish from the least"
                        " infeasible solution",
                        UserWarning,
                    )

            result = minimize(
                self.func,
                np.copy(DE_result.x),
                method=polish_method,
                bounds=self.limits.T,
                constraints=self.constraints,
            )

            self._nfev += result.nfev
            DE_result.nfev = self._nfev

            # Polishing solution is only accepted if there is an improvement in
            # cost function, the polishing was successful and the solution lies
            # within the bounds.
            if (
                result.fun < DE_result.fun
                and result.success
                and np.all(result.x <= self.limits[1])
                and np.all(self.limits[0] <= result.x)
            ):
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)

        if self._wrapped_constraints:
            DE_result.constr = [
                c.violation(DE_result.x) for c in self._wrapped_constraints
            ]
            DE_result.constr_violation = np.max(np.concatenate(DE_result.constr))
            DE_result.maxcv = DE_result.constr_violation
            if DE_result.maxcv > 0:
                # if the result is infeasible then success must be False
                DE_result.success = False
                DE_result.message = (
                    "The solution does not satisfy the "
                    f"constraints, MAXCV = {DE_result.maxcv}"
                )

        self.plot_fitness_save()
        pop = None
        if self.block_size == None:
            pop = self.population
        else:
            pop = self.population_blocked
        np.savez(
            self.save_link,
            best_solution=self.best_gens_solution,
            fitness_history=self.best_gens_fitness_history,
            last_population=pop,
        )

        return DE_result

    def _calculate_population_energies(self, population):
        """
        Calculate the energies of a population.

        Parameters
        ----------
        population : ndarray
            An array of parameter vectors normalised to [0, 1] using lower
            and upper limits. Has shape ``(np.size(population, 0), N)``.

        Returns
        -------
        energies : ndarray
            An array of energies corresponding to each population member. If
            maxfun will be exceeded during this call, then the number of
            function evaluations will be reduced and energies will be
            right-padded with np.inf. Has shape ``(np.size(population, 0),)``
        """
        num_members = np.size(population, 0)
        # S is the number of function evals left to stay under the
        # maxfun budget
        S = min(num_members, self.maxfun - self._nfev)

        energies = np.full(num_members, np.inf)

        # parameters_pop = self._scale_parameters(population)
        calc_energies = []
        try:
            if self.block_size == None:
                calc_energies = list(self._mapwrapper(self.func, population[0:S]))
                calc_energies = np.squeeze(calc_energies)
            else:
                for s in population:
                    unb_s = self._unblocker_optimal(np.array([s]))[0]
                    calc_energies.append(self.func(unb_s))
                calc_energies = np.squeeze(calc_energies)
        except (TypeError, ValueError) as e:
            # wrong number of arguments for _mapwrapper
            # or wrong length returned from the mapper
            raise RuntimeError(
                "The map-like callable must be of the form f(func, iterable), "
                "returning a sequence of numbers the same length as 'iterable'"
            ) from e

        if calc_energies.size != S:
            if self.vectorized:
                raise RuntimeError(
                    "The vectorized function must return an"
                    " array of shape (S,) when given an array"
                    " of shape (len(x), S)"
                )
            raise RuntimeError("func(x, *args) must return a scalar value")

        energies[0:S] = calc_energies

        if self.vectorized:
            self._nfev += 1
        else:
            self._nfev += S

        return energies

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._mapwrapper.__exit__(*args)

    def _accept_trial(
        self,
        energy_trial,
        feasible_trial,
        cv_trial,
        energy_orig,
        feasible_orig,
        cv_orig,
    ):
        """
        Trial is accepted if:
        * it satisfies all constraints and provides a lower or equal objective
          function value, while both the compared solutions are feasible
        - or -
        * it is feasible while the original solution is infeasible,
        - or -
        * it is infeasible, but provides a lower or equal constraint violation
          for all constraint functions.

        This test corresponds to section III of Lampinen [1]_.

        Parameters
        ----------
        energy_trial : float
            Energy of the trial solution
        feasible_trial : float
            Feasibility of trial solution
        cv_trial : array-like
            Excess constraint violation for the trial solution
        energy_orig : float
            Energy of the original solution
        feasible_orig : float
            Feasibility of original solution
        cv_orig : array-like
            Excess constraint violation for the original solution

        Returns
        -------
        accepted : bool

        """
        if feasible_orig and feasible_trial:
            return energy_trial <= energy_orig
        elif feasible_trial and not feasible_orig:
            return True
        elif not feasible_trial and (cv_trial <= cv_orig).all():
            # cv_trial < cv_orig would imply that both trial and orig are not
            # feasible
            return True

        return False

    def __next__(self):
        """
        Evolve the population by a single generation

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        if np.all(np.isinf(self.population_energies)):
            # only need to work out population energies for those that are
            # feasible
            if self.block_size != None:
                # temp_pop = self._unblocker_random(self.population_blocked)
                # if self.block_size < 0:
                #     temp_pop = self._unblocker_optimal(self.population_blocked)
                # else:
                #     temp_pop = self._unblocker_random(self.population_blocked)
                self.population_energies = self._calculate_population_energies(
                    self.population_blocked
                )
            else:
                self.population_energies = self._calculate_population_energies(
                    self.population
                )

            self.best_gens_fitness_history.append(self.population_energies.min())
            # self.best_gens_solution.append(self.population[self.population_energies.argmin()])
            if self.block_size != None:
                self.best_gens_solution = self.population_blocked[
                    self.population_energies.argmin()
                ]
                self.best_gens_solution = self._unblocker_optimal(
                    np.array([self.best_gens_solution])
                )[0]
            else:
                self.best_gens_solution = self.population[
                    self.population_energies.argmin()
                ]

            val = self.val_func(self.best_gens_solution)
            print(
                f"initial population f(x): {self.population_energies.min():.6f}, 1-f(x): {1-self.population_energies.min():.6f}, 1-f'(x)= {val:.6f}"
            )
            # self.plot_fitness_save()

        # if self.dither is not None:
        #     self.scale = self.random_number_generator.uniform(
        #         low=self.dither[0], high=self.dither[1], size=len(self.dither[0])
        #     )

        if self._updating == "immediate":
            # update best solution immediately
            for candidate in range(self.num_population_members):
                if self._nfev > self.maxfun:
                    raise StopIteration

                # create a trial solution
                trial = self._mutate(candidate)

                # ensuring that it's in the range [0, 1)
                self._ensure_constraint(trial)

                # scale from [0, 1) to the actual parameter value
                parameters = self._scale_parameters(trial)

                # determine the energy of the objective function
                if self._wrapped_constraints:
                    cv = self._constraint_violation_fn(parameters)
                    feasible = False
                    energy = np.inf
                    if not np.sum(cv) > 0:
                        # solution is feasible
                        feasible = True
                        energy = self.func(parameters)
                        self._nfev += 1
                else:
                    feasible = True
                    cv = np.atleast_2d([0.0])
                    energy = self.func(parameters)
                    self._nfev += 1

                self.population[candidate] = trial
                self.population_energies[candidate] = np.squeeze(energy)

        elif self._updating == "deferred":
            # update best solution once per generation
            if self._nfev >= self.maxfun:
                raise StopIteration

            # 'deferred' approach, vectorised form.
            # create trial solutions
            trial_pop = np.array(
                [self._mutate(i) for i in range(self.num_population_members)]
            )

            # if self.block_size != None:
            #     trial_pop_blocked = trial_pop.copy()
            #     # trial_pop = self._unblocker_random(trial_pop_blocked)
            #     if self.block_size < 0:
            #         trial_pop = self._unblocker_optimal(trial_pop_blocked)
            #     elif self.block_size != None:
            #         trial_pop = self._unblocker_random(trial_pop_blocked)

            # only calculate for feasible entries
            trial_energies = self._calculate_population_energies(trial_pop)
            # print(
            #     trial_energies[trial_energies.argsort()[:10]],
            #     self.population_energies[self.population_energies.argsort()[:10]],
            # )

            # which solutions are 'improved'?
            loc = trial_energies < self.population_energies
            loc = np.array(loc)
            # self.population = np.where(loc[:, np.newaxis], trial_pop, self.population)
            if self.block_size != None:
                self.population_blocked = np.where(
                    loc[:, np.newaxis], trial_pop, self.population_blocked
                )
            self.population_energies = np.where(
                loc, trial_energies, self.population_energies
            )
        trial_pop = None
        if self.block_size != None:
            trial_pop_blocked = self.population_blocked
            if self.block_size < 0:
                trial_pop = self._unblocker_optimal(trial_pop_blocked)
            elif self.block_size != None:
                trial_pop = self._unblocker_random(trial_pop_blocked)
        else:
            trial_pop = self.population
        return (
            trial_pop[self.population_energies.argmin()],
            self.population_energies.min(),
        )

    def _blocker_optimal(self, pop):
        params_blocked = np.zeros((pop.shape[0], self.blocked_dimensions))
        for i_p in range(pop.shape[0]):
            for i in range(self.blocked_dimensions):
                block_params = pop[i_p, self.blocks_mask[i]]
                if len(block_params) != 0:
                    params_blocked[i_p, i] = np.mean(block_params)
        return params_blocked

    def _blocker_random_average(self, pop):
        pop_blocked = np.zeros((pop.shape[0], self.blocked_dimensions))
        for i in range(pop.shape[0]):
            for j in range(self.blocked_dimensions):
                temp = np.delete(
                    self.blocks_mask[j],
                    np.where(self.blocks_mask[j] > self.parameter_count - 1),
                )
                pop_blocked[i, j] = np.mean(pop[i, temp])

        return pop_blocked

    def _blocker_random(self, pop):
        return pop[:, self.blocks_mask[:, 0]].copy()

    def _blocker_fixed(self):
        return self.population[
            :, np.arange(0, self.parameter_count, self.block_size)
        ].copy()

    def _unblocker_optimal(self, pop_blocked):
        pop_unblocked = np.ones((pop_blocked.shape[0], self.parameter_count))
        for i_p in range(pop_blocked.shape[0]):
            for i in range(self.blocked_dimensions):
                pop_unblocked[i_p, self.blocks_mask[i]] *= pop_blocked[i_p, i]
        return pop_unblocked

    def _unblocker_random(self, pop_blocked):
        pop_unblocked = np.ones(
            (
                len(pop_blocked),
                self.parameter_count
                + ((self.block_size - (self.parameter_count % self.block_size))),
            )
        )

        for i in range(self.blocked_dimensions):
            for j in range(self.block_size):
                pop_unblocked[:, self.blocks_mask[i, j]] = pop_blocked[:, i]

        return pop_unblocked[
            :, : self.parameter_count
        ]  # .reshape((self.num_population_members, self.parameter_count))

    def _unblocker_fixed(self, trial_pop_blocked):
        unblocked_pop_notshaped = np.multiply(
            trial_pop_blocked.flatten().reshape(-1, 1),
            np.ones(
                (self.num_population_members * self.blocked_dimensions, self.block_size)
            ),
        )
        unblocked_pop = unblocked_pop_notshaped.flatten()[
            self.blocks_mask.flatten()
        ].reshape(self.num_population_members, self.parameter_count)

        return unblocked_pop

    def _scale_parameters(self, trial):
        """Scale from a number between 0 and 1 to parameters."""
        # trial either has shape (N, ) or (L, N), where L is the number of
        # solutions being scaled
        return trial
        scaled = self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2
        # if np.any(self.integrality):
        #    i = np.broadcast_to(self.integrality, scaled.shape)
        #    scaled[i] = np.round(scaled[i])
        # return scaled

    def _unscale_parameters(self, parameters):
        """Scale from parameters to a number between 0 and 1."""
        return parameters
        # return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """Make sure the parameters lie between the limits."""
        mask = np.where((trial > 1) | (trial < 0))
        trial[mask] = self.random_number_generator.uniform(size=mask[0].shape)

    def _mutate(self, candidate):
        """Create a trial vector based on a mutation strategy."""
        # trial = np.copy(self.population_blocked[candidate])

        rng = self.random_number_generator

        if self.block_size != None:
            trial = np.copy(self.population_blocked[candidate])
            fill_point = rng.choice(self.blocked_dimensions)
        else:
            trial = np.copy(self.population[candidate])
            fill_point = rng.choice(self.parameter_count)

        if self.strategy in ["currenttobest1exp", "currenttobest1bin"]:
            bprime = self.mutation_func(candidate, self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        if self.strategy in self._binomial:
            if self.block_size != None:
                crossovers = rng.uniform(size=self.blocked_dimensions)
            else:
                crossovers = rng.uniform(size=self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.strategy in self._exponential:
            i = 0
            crossovers = rng.uniform(size=self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            while i < self.parameter_count and crossovers[i]:
                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1

            return trial

    def _best1(self, samples):
        """best1bin, best1exp"""
        r0, r1 = samples[:2]
        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(
                low=self.dither[0], high=self.dither[1], size=len(self.dither[0])
            )
        return self.population[self.population_energies.argmin()] + self.scale * (
            self.population[r0] - self.population[r1]
        )

    def _rand1(self, samples):
        """rand1bin, rand1exp"""
        r0, r1, r2 = samples[:3]
        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(
                low=self.dither[0], high=self.dither[1], size=len(self.dither[0])
            )
        if self.block_size != None:
            return self.population_blocked[r0] + self.scale * (
                self.population_blocked[r1] - self.population_blocked[r2]
            )
        else:
            return self.population[r0] + self.scale * (
                self.population[r1] - self.population[r2]
            )

    def _randtobest1(self, samples):
        """randtobest1bin, randtobest1exp"""
        r0, r1, r2 = samples[:3]
        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(
                low=self.dither[0], high=self.dither[1], size=len(self.dither[0])
            )
        bprime = np.copy(self.population[r0])
        bprime += self.scale * (
            self.population[self.population_energies.argmin()] - bprime
        )
        bprime += self.scale * (self.population[r1] - self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        """currenttobest1bin, currenttobest1exp"""
        r0, r1 = samples[:2]
        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(
                low=self.dither[0], high=self.dither[1], size=len(self.dither[0])
            )
        bprime = self.population[candidate] + self.scale * (
            self.population[self.population_energies.argmin()]
            - self.population[candidate]
            + self.population[r0]
            - self.population[r1]
        )
        return bprime

    def _best2(self, samples):
        """best2bin, best2exp"""
        r0, r1, r2, r3 = samples[:4]
        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(
                low=self.dither[0], high=self.dither[1], size=len(self.dither[0])
            )
        bprime = self.population[self.population_energies.argmin()] + self.scale * (
            self.population[r0]
            + self.population[r1]
            - self.population[r2]
            - self.population[r3]
        )

        return bprime

    def _rand2(self, samples):
        """rand2bin, rand2exp"""
        r0, r1, r2, r3, r4 = samples
        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(
                low=self.dither[0], high=self.dither[1], size=len(self.dither[0])
            )
        bprime = self.population[r0] + self.scale * (
            self.population[r1]
            + self.population[r2]
            - self.population[r3]
            - self.population[r4]
        )

        return bprime

    def _order1(self, samples):
        """order1bin, order1exp"""
        rand1 = np.asarray(samples[:3])
        r0, r1, r2 = rand1[self.population_energies[rand1].argsort()]
        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(
                low=self.dither[0], high=self.dither[1], size=len(self.dither[0])
            )
        if self.block_size != None:
            return self.population_blocked[r0] + self.scale * (
                self.population_blocked[r1] - self.population_blocked[r2]
            )
        else:
            return self.population[r0] + self.scale * (
                self.population[r1] - self.population[r2]
            )

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement. You can't have the original candidate either.
        """
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs

    def plot_fitness_save(self):
        fitness_history = (1 - np.asarray(self.best_gens_fitness_history)) * 100
        if self.local_search:
            fitness_history = (
                np.concatenate(
                    [self.best_gens_fitness_history, self.local_search_fitness_history]
                )
                * -100
            )
        plt.figure(figsize=(12, 5))
        plt.plot(fitness_history, label="F1-score")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid()
        if self.block_size != None:
            plt.title("Block DE")
        else:
            plt.title("DE")
        plt.savefig(self.plot_link)
        plt.close()


class _ConstraintWrapper:
    """Object to wrap/evaluate user defined constraints.

    Very similar in practice to `PreparedConstraint`, except that no evaluation
    of jac/hess is performed (explicit or implicit).

    If created successfully, it will contain the attributes listed below.

    Parameters
    ----------
    constraint : {`NonlinearConstraint`, `LinearConstraint`, `Bounds`}
        Constraint to check and prepare.
    x0 : array_like
        Initial vector of independent variables, shape (N,)

    Attributes
    ----------
    fun : callable
        Function defining the constraint wrapped by one of the convenience
        classes.
    bounds : 2-tuple
        Contains lower and upper bounds for the constraints --- lb and ub.
        These are converted to ndarray and have a size equal to the number of
        the constraints.
    """

    def __init__(self, constraint, x0):
        self.constraint = constraint

        if isinstance(constraint, NonlinearConstraint):

            def fun(x):
                x = np.asarray(x)
                return np.atleast_1d(constraint.fun(x))

        elif isinstance(constraint, LinearConstraint):

            def fun(x):
                if issparse(constraint.A):
                    A = constraint.A
                else:
                    A = np.atleast_2d(constraint.A)
                return A.dot(x)

        elif isinstance(constraint, Bounds):

            def fun(x):
                return np.asarray(x)

        else:
            raise ValueError("`constraint` of an unknown type is passed.")

        self.fun = fun

        lb = np.asarray(constraint.lb, dtype=float)
        ub = np.asarray(constraint.ub, dtype=float)

        x0 = np.asarray(x0)

        # find out the number of constraints
        f0 = fun(x0)
        self.num_constr = m = f0.size
        self.parameter_count = x0.size

        if lb.ndim == 0:
            lb = np.resize(lb, m)
        if ub.ndim == 0:
            ub = np.resize(ub, m)

        self.bounds = (lb, ub)

    def __call__(self, x):
        return np.atleast_1d(self.fun(x))

    def violation(self, x):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        x : array-like
            Vector of independent variables, (N, S), where N is number of
            parameters and S is the number of solutions to be investigated.

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by, for each of the
            constraints specified by `_ConstraintWrapper.fun`.
            Has shape (M, S) where M is the number of constraint components.
        """
        # expect ev to have shape (num_constr, S) or (num_constr,)
        ev = self.fun(np.asarray(x))

        try:
            excess_lb = np.maximum(self.bounds[0] - ev.T, 0)
            excess_ub = np.maximum(ev.T - self.bounds[1], 0)
        except ValueError as e:
            raise RuntimeError(
                "An array returned from a Constraint has"
                " the wrong shape. If `vectorized is False`"
                " the Constraint should return an array of"
                " shape (M,). If `vectorized is True` then"
                " the Constraint must return an array of"
                " shape (M, S), where S is the number of"
                " solution vectors and M is the number of"
                " constraint components in a given"
                " Constraint object."
            ) from e

        v = (excess_lb + excess_ub).T
        return v


from contextlib import contextmanager
import functools
import operator
import sys
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
    Optional,
    Union,
    TYPE_CHECKING,
    TypeVar,
)

import numpy as np

IntNumber = Union[int, np.integer]
DecimalNumber = Union[float, np.floating, np.integer]

# Since Generator was introduced in numpy 1.17, the following condition is needed for
# backward compatibility
if TYPE_CHECKING:
    SeedType = Optional[Union[IntNumber, np.random.Generator, np.random.RandomState]]
    GeneratorType = TypeVar(
        "GeneratorType", bound=Union[np.random.Generator, np.random.RandomState]
    )

try:
    from numpy.random import Generator as Generator
except ImportError:

    class Generator:  # type: ignore[no-redef]
        pass


def float_factorial(n: int) -> float:
    """Compute the factorial and return as a float

    Returns infinity when result is too large for a double
    """
    return float(math.factorial(n)) if n < 171 else np.inf


class DeprecatedImport:
    """
    Deprecated import with redirection and warning.

    Examples
    --------
    Suppose you previously had in some module::

        from foo import spam

    If this has to be deprecated, do::

        spam = DeprecatedImport("foo.spam", "baz")

    to redirect users to use "baz" module instead.

    """

    def __init__(self, old_module_name, new_module_name):
        self._old_name = old_module_name
        self._new_name = new_module_name
        __import__(self._new_name)
        self._mod = sys.modules[self._new_name]

    def __dir__(self):
        return dir(self._mod)

    def __getattr__(self, name):
        warnings.warn(
            "Module %s is deprecated, use %s instead"
            % (self._old_name, self._new_name),
            DeprecationWarning,
        )
        return getattr(self._mod, name)


# copy-pasted from scikit-learn utils/validation.py
# change this to scipy.stats._qmc.check_random_state once numpy 1.16 is dropped
def check_random_state(seed):
    """Turn `seed` into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed

    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )


# Add a replacement for inspect.getfullargspec()/
# The version below is borrowed from Django,
# https://github.com/django/django/pull/4846.

# Note an inconsistency between inspect.getfullargspec(func) and
# inspect.signature(func). If `func` is a bound method, the latter does *not*
# list `self` as a first argument, while the former *does*.
# Hence, cook up a common ground replacement: `getfullargspec_no_self` which
# mimics `inspect.getfullargspec` but does not list `self`.
#
# This way, the caller code does not need to know whether it uses a legacy
# .getfullargspec or a bright and shiny .signature.

FullArgSpec = namedtuple(
    "FullArgSpec",
    [
        "args",
        "varargs",
        "varkw",
        "defaults",
        "kwonlyargs",
        "kwonlydefaults",
        "annotations",
    ],
)


class _FunctionWrapper:
    """
    Object to wrap user's function, allowing picklability
    """

    def __init__(self, f, args):
        self.f = f
        self.args = [] if args is None else args

    def __call__(self, x):
        return self.f(x, *self.args)


class MapWrapper:
    """
    Parallelisation wrapper for working with map-like callables, such as
    `multiprocessing.Pool.map`.

    Parameters
    ----------
    pool : int or map-like callable
        If `pool` is an integer, then it specifies the number of threads to
        use for parallelization. If ``int(pool) == 1``, then no parallel
        processing is used and the map builtin is used.
        If ``pool == -1``, then the pool will utilize all available CPUs.
        If `pool` is a map-like callable that follows the same
        calling sequence as the built-in map function, then this callable is
        used for parallelization.
    """

    def __init__(self, pool=1):
        self.pool = None
        self._mapfunc = map
        self._own_pool = False

        if callable(pool):
            self.pool = pool
            self._mapfunc = self.pool
        else:
            from multiprocessing import Pool

            # user supplies a number
            if int(pool) == -1:
                # use as many processors as possible
                self.pool = Pool()
                self._mapfunc = self.pool.map
                self._own_pool = True
            elif int(pool) == 1:
                pass
            elif int(pool) > 1:
                # use the number of processors requested
                self.pool = Pool(processes=int(pool))
                self._mapfunc = self.pool.map
                self._own_pool = True
            else:
                raise RuntimeError(
                    "Number of workers specified must be -1,"
                    " an int >= 1, or an object with a 'map' "
                    "method"
                )

    def __enter__(self):
        return self

    def terminate(self):
        if self._own_pool:
            self.pool.terminate()

    def join(self):
        if self._own_pool:
            self.pool.join()

    def close(self):
        if self._own_pool:
            self.pool.close()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._own_pool:
            self.pool.close()
            self.pool.terminate()

    def __call__(self, func, iterable):
        # only accept one iterable because that's all Pool.map accepts
        try:
            return self._mapfunc(func, iterable)
        except TypeError as e:
            # wrong number of arguments
            raise TypeError(
                "The map-like callable must be of the" " form f(func, iterable)"
            ) from e
