"""Flask application serving the nutrition calculator and backend optimizer."""

from __future__ import annotations

from flask import Flask, request, jsonify
from diet_optimizer import DietOptimizer

app = Flask(__name__, static_folder=".")


@app.route("/")
def index() -> str:
    """Serve the main calculator HTML page."""
    with open("nutrition_calculator.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/optimize")
def optimize() -> "Response":
    """Compute optimal diet based on posted JSON data."""
    data = request.get_json(force=True)
    try:
        optimizer = DietOptimizer(
            nutrient_matrix=data["nutrient_matrix"],
            steps=data["steps"],
            max_portions=data["max_portions"],
            weights=data["weights"],
            targets=data["targets"],
            calorie_index=len(data["targets"]) - 1,
        )
        fixed = {int(k): float(v) for k, v in data.get("fixed_indices", {}).items()}
        result = optimizer.compute_optimal_diet(fixed_indices=fixed)
        best = result.get("bestSelection", {})
        return jsonify({
            "varAdd": best.get("varAdd", {}),
            "rmse": result.get("overallBestRmse"),
        })
    except Exception as exc:  # pragma: no cover - safeguard for debug
        return jsonify({"error": str(exc)})


if __name__ == "__main__":
    app.run(debug=True)

