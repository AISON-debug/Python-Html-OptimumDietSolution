"""Lightweight diet optimisation module.

This module provides a lightweight solver for choosing food portions to meet
nutrient targets.  It mirrors the optimisation core used in the HTML web
application: an iterative non‑negative least squares routine that attempts to
cover a fraction of the remaining nutrient residual on each iteration.  The
public :class:`DietOptimizer` exposes :meth:`compute_optimal_diet` which searches
over residual fractions and random orderings of products to minimise the
weighted RMSE between achieved and target nutrients.  Plotly is used for
optional graph generation when running the demo.

The code includes a small demo executed when run as a script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import math
import random
import plotly.graph_objects as go


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
    product_names: List[str] | None = None
    nutrient_names: List[str] | None = None
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
        heat_data: List[Dict[str, float]] = []

        for perc in range(max(1, min_tail_percent), 101):
            alpha = perc / 100.0
            for run in range(run_count):
                shuffled = var_idxs[:]
                random.shuffle(shuffled)
                res = self._evaluate_diet(
                    shuffled, resid[:], totals_fixed, fixed_indices, alpha
                )
                heat_data.append({"x": perc, "y": run + 1, "rmse": res["rmse"]})
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
            "heatData": heat_data,
            "minPercent": max(1, min_tail_percent),
            "runCount": run_count,
            "overallBestRmse": best_sel["rmse"],
            "alphaStar": best_sel["rf"] / 100.0,
            "avgRf": avg_rf,
            "overallBest": overall_best,
        }

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def format_result_tables(
        self,
        result: Dict[str, object],
        fixed_indices: Dict[int, float] | None = None,
    ) -> str:
        """Return text tables for products and nutrient comparison.

        Mirrors the WebApp's rendering of the *"Оптимизированный рацион"* and
        *"Сравнение целевых показателей и результатов оптимизации"*
        tables【F:nutrition_webapp 31.08.2025.html†L996-L1044】.
        """

        if fixed_indices is None:
            fixed_indices = {}
        products = self.product_names or [f"Product {i}" for i in range(len(self.nutrient_matrix))]
        nutrients = self.nutrient_names or [f"Nutrient {i}" for i in range(len(self.targets))]

        best_sel = result["bestSelection"]
        best_full = result["bestFull"]
        var_add = best_sel.get("varAdd", {})
        full_map = best_full.get("fullMap", {})

        full_items = []
        selected_items = []
        for idx, name in enumerate(products):
            sel_g = fixed_indices.get(idx, var_add.get(idx, 0.0))
            full_g = fixed_indices.get(idx, full_map.get(idx, 0.0))
            if full_g > 0:
                full_items.append((name, full_g))
            if sel_g > 0:
                selected_items.append((name, sel_g))

        full_items.sort(key=lambda x: x[1], reverse=True)
        selected_items.sort(key=lambda x: x[1], reverse=True)
        max_len = max(len(full_items), len(selected_items))

        lines: List[str] = []
        lines.append("Оптимизированный рацион")
        header = (
            f"{'Продукт (Полная база)':<25} {'Вес (г)':>10}    "
            f"{'Продукт (Выбранные)':<25} {'Вес (г)':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for i in range(max_len):
            f_name, f_g = full_items[i] if i < len(full_items) else ("", 0.0)
            s_name, s_g = selected_items[i] if i < len(selected_items) else ("", 0.0)
            f_val = f"{f_g:10.2f}" if f_name else " " * 10
            s_val = f"{s_g:10.2f}" if s_name else " " * 10
            lines.append(
                f"{f_name:<25} {f_val}    {s_name:<25} {s_val}"
            )

        lines.append("")
        lines.append("Сравнение целевых показателей и результатов оптимизации")
        header2 = (
            f"{'Показатель':<30} {'Цель':>10} {'Полная база':>15} {'Выбранные':>15}"
        )
        lines.append(header2)
        lines.append("-" * len(header2))
        for k, name in enumerate(nutrients):
            unit = "" if self.calorie_index is not None and k == self.calorie_index else " г"
            tgt = f"{self.targets[k]:.1f}" if unit == "" else f"{self.targets[k]:.0f}"
            full_val = best_full["totalsFull"][k]
            full_str = f"{full_val:.1f}" if unit == "" else f"{full_val:.0f}"
            sel_val = best_sel["totalsFinal"][k]
            sel_str = f"{sel_val:.1f}" if unit == "" else f"{sel_val:.0f}"
            lines.append(
                f"{name:<30} {tgt + unit:>10} {full_str + unit:>15} {sel_str + unit:>15}"
            )

        lines.append(
            f"{'Доля остатка':<30} {'—':>10} {str(best_full['rf']) + '%':>15} {str(best_sel['rf']) + '%':>15}"
        )
        lines.append(
            f"{'RMSE':<30} {'—':>10} {best_full['rmseFull']:15.3f} {best_sel['rmse']:15.3f}"
        )

        return "\n".join(lines)

    def plot_result_graphs(
        self, result: Dict[str, object], prefix: str = "diet"
    ) -> List[str]:
        """Generate interactive plots mirroring the WebApp graphs.

        The function creates three HTML files using Plotly:
        ``{prefix}_rmse_surface.html`` for the RMSE surface across runs and
        residual fractions, ``{prefix}_rf_runs.html`` showing the best residual
        fraction per run, and ``{prefix}_rmse_best_surface.html`` visualising the
        cumulative best RMSE.  File paths are returned for convenience.
        """

        heat_data = result.get("heatData", [])
        if not heat_data:
            return []

        run_best = result["runBest"]
        best_sel = result["bestSelection"]
        best_full = result["bestFull"]
        avg_rf = result.get("avgRf", 0)
        overall_best = result.get("overallBest", run_best[0])
        run_count = result.get("runCount", len(run_best))
        min_percent = result.get("minPercent", 1)

        percents = sorted({d["x"] for d in heat_data})
        runs = list(range(1, run_count + 1))

        rmse_by_perc = {p: [None] * run_count for p in percents}
        for d in heat_data:
            rmse_by_perc[d["x"]][d["y"] - 1] = d["rmse"]

        values = [d["rmse"] for d in heat_data]
        min_sel = min(values)
        max_sel = max(values)
        denom = max_sel - min_sel if max_sel > min_sel else 1.0
        z_matrix = [
            [
                (rmse_by_perc[p][r - 1] - min_sel + 1)
                if rmse_by_perc[p][r - 1] is not None
                else None
                for p in percents
            ]
            for r in runs
        ]
        color_matrix = [
            [
                math.sqrt((rmse_by_perc[p][r - 1] - min_sel) / denom)
                if rmse_by_perc[p][r - 1] is not None
                else None
                for p in percents
            ]
            for r in runs
        ]

        best_matrix: List[List[float]] = []
        for p in percents:
            arr = rmse_by_perc[p]
            best = float("inf")
            row = []
            for val in arr:
                if val is not None and val < best:
                    best = val
                row.append(best)
            best_matrix.append(row)

        flat_best = [v for row in best_matrix for v in row if math.isfinite(v)]
        min_best = min(flat_best)
        max_best = max(flat_best)
        k_diff = max_best / min_best if min_best > 0 else 1
        base = 2 ** k_diff
        thresholds = [
            (min_best, "#f0fff0"),
            (min_best * base ** (1 / 5), "#00ff00"),
            (min_best * base ** (1 / 4), "#ffff00"),
            (min_best * base ** (1 / 3), "#ffa500"),
            (min_best * base ** (1 / 2), "#ff0000"),
            (min_best * base, "#8b0000"),
        ]
        max_scale = min_best * base
        colorscale_best = [
            ((v - min_best) / (max_scale - min_best), color)
            for v, color in thresholds
        ]

        paths = []
        fig1 = go.Figure(
            data=[
                go.Surface(
                    x=percents,
                    y=runs,
                    z=z_matrix,
                    surfacecolor=color_matrix,
                    colorscale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
                    cmin=0,
                    cmax=1,
                    showscale=False,
                    contours={"x": {"show": True}, "y": {"show": True}},
                )
            ],
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(title="Доля остатка (%)", range=[min_percent, 100]),
                    yaxis=dict(title="Прогон", range=[1, run_count]),
                    zaxis=dict(title="RMSE", type="log"),
                ),
                annotations=[
                    dict(
                        text=f"min RMSE пула: {best_sel['rmse']:.3f}",
                        x=1,
                        y=1,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        xanchor="right",
                        yanchor="top",
                    ),
                    dict(
                        text=f"min RMSE базы: {best_full['rmseFull']:.3f}",
                        x=1,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        xanchor="right",
                        yanchor="top",
                    ),
                ],
            ),
        )
        path1 = f"{prefix}_rmse_surface.html"
        fig1.write_html(path1, include_plotlyjs="cdn", auto_open=False)
        paths.append(path1)

        x_runs = [r["run"] for r in run_best]
        y_rf = [r["rf"] for r in run_best]
        colors = ["red" if r["run"] == overall_best["run"] else "black" for r in run_best]
        sizes = [10 if r["run"] == overall_best["run"] else 6 for r in run_best]
        fig2 = go.Figure(
            data=[
                go.Scatter(
                    x=x_runs,
                    y=y_rf,
                    mode="lines+markers",
                    marker=dict(color=colors, size=sizes),
                    line=dict(color="black"),
                )
            ],
            layout=go.Layout(
                xaxis=dict(title="Прогон", dtick=1),
                yaxis=dict(title="Доля остатка (%)", range=[min_percent, 100]),
                annotations=[
                    dict(
                        text=f"наименьший rmse {overall_best['rmse']:.3f}; средняя доля {avg_rf:.2f}%",
                        x=0,
                        y=1,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        xanchor="left",
                        yanchor="top",
                    )
                ],
            ),
        )
        path2 = f"{prefix}_rf_runs.html"
        fig2.write_html(path2, include_plotlyjs="cdn", auto_open=False)
        paths.append(path2)

        fig3 = go.Figure(
            data=[
                go.Surface(
                    x=runs,
                    y=percents,
                    z=best_matrix,
                    colorscale=colorscale_best,
                    cmin=min_best,
                    cmax=max_scale,
                    showscale=False,
                    contours={"x": {"show": True}, "y": {"show": True}},
                )
            ],
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(title="Прогон", range=[1, run_count]),
                    yaxis=dict(title="Доля остатка (%)", range=[min_percent, 100]),
                    zaxis=dict(title="Лучший RMSE", type="log"),
                ),
                annotations=[],
            ),
        )
        path3 = f"{prefix}_rmse_best_surface.html"
        fig3.write_html(path3, include_plotlyjs="cdn", auto_open=False)
        paths.append(path3)

        return paths


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
    product_names = ["Food A", "Food B", "Food C"]
    nutrient_names = ["Nutrient 1", "Nutrient 2"]

    optimizer = DietOptimizer(
        nutrient_matrix=nutrient_matrix,
        steps=steps,
        max_portions=max_portions,
        weights=weights,
        targets=targets,
        product_names=product_names,
        nutrient_names=nutrient_names,
    )

    # Assume 20g of food 0 is fixed
    fixed = {0: 20.0}
    result = optimizer.compute_optimal_diet(fixed_indices=fixed)
    print(optimizer.format_result_tables(result, fixed_indices=fixed))
    paths = optimizer.plot_result_graphs(result, prefix="demo")
    if paths:
        print("Generated plots:")
        for p in paths:
            print("  ", p)
