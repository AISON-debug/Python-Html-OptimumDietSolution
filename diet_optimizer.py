"""Lightweight diet optimisation module.

This module provides a small, dependency‑free solver for choosing food portions
to meet nutrient targets.  It mirrors the optimisation core used in the HTML
web application: an iterative non‑negative least squares routine that attempts
to cover a fraction of the remaining nutrient residual on each iteration.  The
public :class:`DietOptimizer` exposes :meth:`compute_optimal_diet` which searches
over residual fractions and random orderings of products to minimise the
weighted RMSE between achieved and target nutrients.

The code includes a small demo executed when run as a script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import math
import random


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def solve_linear_system(g: List[List[float]], b: List[float]) -> List[float]:
    """Solve G x = b using Gaussian elimination with partial pivoting.

    The function is intentionally lightweight and expects small systems.  If the
    system is singular the solution defaults to zeros.
    """

    n = len(g)
    # Build augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(g)]

    for i in range(n):
        # Find pivot row
        pivot = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[pivot][i]) < 1e-12:
            return [0.0] * n
        if pivot != i:
            aug[i], aug[pivot] = aug[pivot], aug[i]
        # Normalize pivot row
        pivot_val = aug[i][i]
        for j in range(i, n + 1):
            aug[i][j] /= pivot_val
        # Eliminate column
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            for c in range(i, n + 1):
                aug[r][c] -= factor * aug[i][c]
    return [aug[i][n] for i in range(n)]


def compute_rmse(targets: Sequence[float], totals: Sequence[float], weights: Sequence[float]) -> float:
    """Return weighted RMSE between targets and totals."""
    n = len(targets)
    acc = 0.0
    for k in range(n):
        acc += weights[k] * (targets[k] - totals[k]) ** 2
    return math.sqrt(acc / n)


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------


@dataclass
class DietOptimizer:
    """Non-negative least squares diet optimizer with residual scaling."""

    nutrient_matrix: List[List[float]]
    steps: List[float]
    max_portions: List[float]
    weights: List[float]
    targets: List[float]
    calorie_index: int | None = None
    max_iterations: int = 10

    # ------------------------------------------------------------------
    # Internal optimisation routines
    # ------------------------------------------------------------------

    def _run_iterative_optimization(
        self, var_idxs: Sequence[int], resid: List[float], alpha: float
    ) -> Dict[int, float]:
        """Allocate weights to ``var_idxs`` attempting to cover ``alpha`` of
        the residual in each iteration.

        This mirrors the JavaScript ``runIterativeOptimization`` routine in the
        WebApp【F:nutrition_webapp 31.08.2025.html†L569-L705】.
        """

        resid_vec = resid[:]
        step_vals = {i: self.steps[i] for i in var_idxs}
        max_vals = {i: self.steps[i] * self.max_portions[i] for i in var_idxs}
        var_add_map = {i: 0.0 for i in var_idxs}

        iteration = 0
        while iteration < self.max_iterations:
            active = [
                idx
                for idx in var_idxs
                if step_vals[idx] > 0
                and max_vals[idx] - var_add_map[idx] >= step_vals[idx] / 2
            ]
            if not active:
                break

            dim = len(active)
            g = [[0.0 for _ in range(dim)] for _ in range(dim)]
            b = [0.0 for _ in range(dim)]
            for a, idx_a in enumerate(active):
                for j in range(len(resid_vec)):
                    b[a] += (
                        self.weights[j]
                        * self.nutrient_matrix[idx_a][j]
                        * resid_vec[j]
                        * alpha
                    )
                    for bcol, idx_b in enumerate(active):
                        g[a][bcol] += (
                            self.weights[j]
                            * self.nutrient_matrix[idx_a][j]
                            * self.nutrient_matrix[idx_b][j]
                        )

            def nnls_solve(
                gmat: List[List[float]],
                bvec: List[float],
                act: List[int],
            ) -> Tuple[List[float], List[int]]:
                active_idx = act[:]
                G = [row[:] for row in gmat]
                B = bvec[:]
                while active_idx:
                    sol_vec = solve_linear_system(G, B)
                    neg = [i for i, val in enumerate(sol_vec) if val < 0]
                    if not neg:
                        return sol_vec, active_idx
                    active_idx = [
                        active_idx[i]
                        for i in range(len(active_idx))
                        if i not in neg
                    ]
                    B = [B[i] for i in range(len(B)) if i not in neg]
                    G = [
                        [row[j] for j in range(len(row)) if j not in neg]
                        for i, row in enumerate(G)
                        if i not in neg
                    ]
                return [], []

            sol, sol_active = nnls_solve(g, b, active)
            if not sol:
                break

            updated = False
            for grams, idx in zip(sol, sol_active):
                if grams <= 0:
                    continue
                allowed = max_vals[idx] - var_add_map[idx]
                if allowed <= 0:
                    continue
                step = step_vals[idx]
                grams = min(grams, allowed)
                if step:
                    grams = round(grams / step) * step
                    if grams > allowed:
                        grams = math.floor(allowed / step) * step
                    if grams < 0:
                        grams = 0
                if grams <= 0:
                    continue
                updated = True
                var_add_map[idx] += grams
                for k in range(len(resid_vec)):
                    resid_vec[k] -= self.nutrient_matrix[idx][k] * grams

            if not updated:
                break
            iteration += 1

        return var_add_map

    def _evaluate_diet(
        self,
        var_idxs: Sequence[int],
        resid: List[float],
        totals_fixed: List[float],
        fixed_indices: Dict[int, float],
        alpha: float,
    ) -> Dict[str, object]:
        """Evaluate a diet configuration for a given ``alpha``.

        Returns additions for selected variables and for the full product set
        alongside nutrient totals and RMSE values.  Mirrors the WebApp's
        ``evaluateDiet`` function【F:nutrition_webapp 31.08.2025.html†L772-L841】.
        """

        var_add = {i: 0.0 for i in range(len(self.nutrient_matrix))}
        if var_idxs:
            var_map = self._run_iterative_optimization(var_idxs, resid[:], alpha)
            var_add.update(var_map)

        totals_var = [0.0 for _ in self.targets]
        for idx in var_idxs:
            grams = var_add.get(idx, 0.0)
            for k in range(len(totals_var)):
                totals_var[k] += self.nutrient_matrix[idx][k] * grams

        totals_final = [
            totals_fixed[k] + totals_var[k] for k in range(len(totals_var))
        ]
        rmse = compute_rmse(self.targets, totals_final, self.weights)

        all_var_idxs = [
            i for i in range(len(self.nutrient_matrix)) if i not in fixed_indices
        ]
        full_map: Dict[int, float] = {}
        totals_full = totals_fixed[:]
        rmse_full = rmse
        if all_var_idxs:
            full_map = self._run_iterative_optimization(all_var_idxs, resid[:], alpha)
            totals_var_full = [0.0 for _ in self.targets]
            for idx, grams in full_map.items():
                for k in range(len(totals_var_full)):
                    totals_var_full[k] += self.nutrient_matrix[idx][k] * grams
            totals_full = [
                totals_fixed[k] + totals_var_full[k]
                for k in range(len(totals_var_full))
            ]
            rmse_full = compute_rmse(self.targets, totals_full, self.weights)

        return {
            "varAdd": {k: v for k, v in var_add.items() if v > 0},
            "totalsFinal": totals_final,
            "rmse": rmse,
            "totalsFull": totals_full,
            "rmseFull": rmse_full,
            "fullMap": {k: v for k, v in full_map.items() if v > 0},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_optimal_diet(
        self,
        fixed_indices: Dict[int, float] | None = None,
        min_tail_percent: int = 10,
        run_count: int = 1,
    ) -> Dict[str, object]:
        """Search for the best diet configuration over residual fractions.

        ``min_tail_percent`` sets the lower bound (1–100) of the residual
        fraction search space, and ``run_count`` controls how many random
        permutations of the variable products are evaluated for each
        fraction.  The procedure mirrors the web application's
        ``computeOptimalDiet`` routine【F:nutrition_webapp 31.08.2025.html†L879-L976】.
        """

        if fixed_indices is None:
            fixed_indices = {}

        totals_fixed = [0.0 for _ in self.weights]
        for idx, grams in fixed_indices.items():
            for k in range(len(totals_fixed)):
                totals_fixed[k] += self.nutrient_matrix[idx][k] * grams

        resid = [
            self.targets[k] - totals_fixed[k] for k in range(len(totals_fixed))
        ]
        var_idxs = [
            i for i in range(len(self.nutrient_matrix)) if i not in fixed_indices
        ]

        best_sel: Dict[str, object] | None = None
        best_full: Dict[str, object] | None = None
        run_best: List[Dict[str, float] | None] = [None] * run_count

        for perc in range(max(1, min_tail_percent), 101):
            alpha = perc / 100.0
            for run in range(run_count):
                shuffled = var_idxs[:]
                random.shuffle(shuffled)
                res = self._evaluate_diet(
                    shuffled, resid[:], totals_fixed, fixed_indices, alpha
                )
                if run_best[run] is None or res["rmse"] < run_best[run]["rmse"]:
                    run_best[run] = {"run": run + 1, "rf": perc, "rmse": res["rmse"]}
                if best_sel is None or res["rmse"] < best_sel["rmse"]:
                    best_sel = res | {"rf": perc, "run": run + 1}
                if best_full is None or res["rmseFull"] < best_full["rmseFull"]:
                    best_full = res | {"rf": perc, "run": run + 1}

        if best_sel is None:
            best_sel = {
                "varAdd": {},
                "totalsFinal": totals_fixed[:],
                "rmse": compute_rmse(self.targets, totals_fixed, self.weights),
                "rf": 1.0,
                "run": 1,
            }
        if best_full is None:
            best_full = best_sel.copy()

        overall_best = min(run_best, key=lambda x: x["rmse"]) if run_best else None
        avg_rf = (
            sum(r["rf"] for r in run_best if r is not None) / run_count
            if run_best
            else 0
        )

        return {
            "bestSelection": best_sel,
            "bestFull": best_full,
            "runBest": run_best,
            "overallBestRmse": best_sel["rmse"],
            "alphaStar": best_sel["rf"] / 100.0,
            "avgRf": avg_rf,
            "overallBest": overall_best,
        }


# ---------------------------------------------------------------------------
# Demo usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple demonstration with three foods and two nutrients
    nutrient_matrix = [
        [10, 5],   # food 0
        [0, 20],   # food 1
        [30, 0],   # food 2
    ]
    steps = [10, 10, 10]
    max_portions = [10, 10, 10]
    weights = [1.0, 1.0]
    targets = [50, 50]

    optimizer = DietOptimizer(
        nutrient_matrix=nutrient_matrix,
        steps=steps,
        max_portions=max_portions,
        weights=weights,
        targets=targets,
    )

    # Assume 20g of food 0 is fixed
    fixed = {0: 20.0}
    result = optimizer.compute_optimal_diet(fixed_indices=fixed)
    print("Best RMSE:", result["overallBestRmse"])
    print("Best alpha:", result["alphaStar"])
    print("Selected additions:", result["bestSelection"]["varAdd"])
