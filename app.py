"""
app.py — Flask Backend & API Server
=====================================
Person 3 (Frontend + Integration Engineer) module.

Serves the dashboard and provides REST API endpoints
for predictions, ranking, scenario analysis, and more.
"""

import os
from flask import Flask, render_template, request, jsonify
from predict import predict_single, predict_batch, get_explainability, get_model_comparison
from search_module import greedy_rank_events, astar_risk_path, analyze_scenario
from utils import validate_input, format_prediction, get_sample_news

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)


# ──────────────────────────────────────────────
# Page Route
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ──────────────────────────────────────────────
# API: Single Prediction
# ──────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    model = data.get("model", "logistic_regression")

    valid, err = validate_input(text)
    if not valid:
        return jsonify({"error": err}), 400

    try:
        raw = predict_single(text, model)
        result = format_prediction(raw)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# API: Batch Prediction + Ranking
# ──────────────────────────────────────────────
@app.route("/api/predict-batch", methods=["POST"])
def api_predict_batch():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    model = data.get("model", "logistic_regression")

    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Provide a list of texts."}), 400
    if len(texts) > 20:
        return jsonify({"error": "Maximum 20 news items at once."}), 400

    # Validate each text
    for i, t in enumerate(texts):
        valid, err = validate_input(t)
        if not valid:
            return jsonify({"error": f"Item {i+1}: {err}"}), 400

    try:
        raw_results = predict_batch(texts, model)
        results = [format_prediction(r) for r in raw_results]
        return jsonify({"success": True, "results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# API: Rank Events (Greedy)
# ──────────────────────────────────────────────
@app.route("/api/rank", methods=["POST"])
def api_rank():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    model = data.get("model", "logistic_regression")

    if not texts or len(texts) < 2:
        return jsonify({"error": "Provide at least 2 news items to rank."}), 400

    try:
        predictions = predict_batch(texts, model)
        events = [format_prediction(p) for p in predictions]
        ranking = greedy_rank_events(events)
        return jsonify({"success": True, "ranking": ranking})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# API: Scenario Analysis
# ──────────────────────────────────────────────
@app.route("/api/scenario", methods=["POST"])
def api_scenario():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    model = data.get("model", "logistic_regression")

    if not texts or len(texts) < 2:
        return jsonify({"error": "Provide at least 2 news items for scenario analysis."}), 400

    try:
        predictions = predict_batch(texts, model)
        events = [format_prediction(p) for p in predictions]
        scenario = analyze_scenario(events)
        return jsonify({"success": True, "scenario": scenario})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# API: A* Risk Path
# ──────────────────────────────────────────────
@app.route("/api/risk-path", methods=["POST"])
def api_risk_path():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    model = data.get("model", "logistic_regression")

    if not texts or len(texts) < 2:
        return jsonify({"error": "Provide at least 2 news items for risk path analysis."}), 400

    try:
        predictions = predict_batch(texts, model)
        events = [format_prediction(p) for p in predictions]
        path = astar_risk_path(events)
        return jsonify({"success": True, "risk_path": path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# API: Model Comparison
# ──────────────────────────────────────────────
@app.route("/api/models/compare", methods=["GET"])
def api_model_compare():
    try:
        report = get_model_comparison()
        return jsonify({"success": True, "report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# API: Explainability
# ──────────────────────────────────────────────
@app.route("/api/explainability", methods=["POST"])
def api_explainability():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    model = data.get("model", "logistic_regression")

    valid, err = validate_input(text)
    if not valid:
        return jsonify({"error": err}), 400

    try:
        result = get_explainability(text, model)
        result["prediction"] = format_prediction(result["prediction"])
        return jsonify({"success": True, "explainability": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# API: Sample News
# ──────────────────────────────────────────────
@app.route("/api/samples", methods=["GET"])
def api_samples():
    n = request.args.get("n", 5, type=int)
    return jsonify({"success": True, "samples": get_sample_news(n)})


# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Crisis Intelligence & Decision Support System")
    print("=" * 60)
    print("  Dashboard: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
