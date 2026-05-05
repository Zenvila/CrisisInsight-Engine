"""
predict.py — Model Inference Module
=====================================
Loads trained models and provides prediction functions
for single and batch news inputs, plus explainability.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from preprocessing import transform_text, transform_batch, get_top_features

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

try:
    from tensorflow.keras.models import load_model as load_keras_model
except Exception:
    load_keras_model = None

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "crisis_news_dataset.csv")
REPORT_PATH = os.path.join(MODELS_DIR, "accuracy_report.json")
LOGISTIC_MODEL_FILES = ("model_logistic_regression.pkl", "logistic_model.pkl")
LINEAR_MODEL_FILES = ("model_linear_regression.pkl", "linear_model.pkl")
VECTORIZER_FILES = ("tfidf_vectorizer.pkl",)
NEURAL_VECTORIZER_FILES = ("tfidf_nn.pkl",)
NEURAL_MODEL_FILES = ("model_neural_network.keras",)
LABEL_ENCODER_FILES = ("label_encoder.pkl",)

# ──────────────────────────────────────────────
# Model Loading (singleton pattern)
# ──────────────────────────────────────────────
_models_cache = {}


def load_models():
    """Load all trained models into memory. Cached after first call."""
    global _models_cache
    if _models_cache:
        return _models_cache

    def _load_joblib_artifact(possible_filenames):
        for filename in possible_filenames:
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                return joblib.load(path)
        raise FileNotFoundError(
            f"None of these artifacts were found in {MODELS_DIR}: {', '.join(possible_filenames)}"
        )

    def _load_keras_artifact(possible_filenames):
        if load_keras_model is None:
            return None, (
                "TensorFlow is not installed, so the neural-network model cannot be loaded. "
                "Install the project requirements to enable `model_neural_network.keras`."
            )

        for filename in possible_filenames:
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                return load_keras_model(path), None

        raise FileNotFoundError(
            f"None of these artifacts were found in {MODELS_DIR}: {', '.join(possible_filenames)}"
        )

    _models_cache["vectorizer"] = _load_joblib_artifact(VECTORIZER_FILES)
    _models_cache["neural_vectorizer"] = _load_joblib_artifact(NEURAL_VECTORIZER_FILES)
    _models_cache["label_encoder"] = _load_joblib_artifact(LABEL_ENCODER_FILES)
    _models_cache["logistic"] = _load_joblib_artifact(LOGISTIC_MODEL_FILES)
    _models_cache["linear"] = _load_joblib_artifact(LINEAR_MODEL_FILES)

    neural_model, neural_error = _load_keras_artifact(NEURAL_MODEL_FILES)
    _models_cache["neural"] = neural_model
    if neural_error:
        _models_cache["neural_error"] = neural_error

    # Load accuracy report if available
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            _models_cache["report"] = json.load(f)

    print(f"[OK] All models loaded from {MODELS_DIR}")
    return _models_cache


def _get_classifier(model_name: str):
    """Get the appropriate classification model."""
    models = load_models()
    if model_name == "neural_network":
        if models.get("neural") is None:
            raise ImportError(models.get("neural_error", "Neural-network model is unavailable."))
        return models["neural"]
    return models["logistic"]  # default


def _get_vectorizer(model_name: str):
    """Get the TF-IDF vectorizer that matches the classifier."""
    models = load_models()
    if model_name == "neural_network":
        return models["neural_vectorizer"]
    return models["vectorizer"]


def _predict_classification(model_name: str, classifier, X):
    """Predict labels and probabilities for either sklearn or Keras classifiers."""
    models = load_models()
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X)
        raw_categories = classifier.predict(X)
        classes = getattr(classifier, "classes_", None)

        if classes is not None:
            classes_array = np.asarray(classes)
            if np.issubdtype(classes_array.dtype, np.number):
                label_encoder = models.get("label_encoder")
                if label_encoder is not None:
                    categories = label_encoder.inverse_transform(np.asarray(raw_categories, dtype=int))
                    classes = label_encoder.classes_
                else:
                    categories = raw_categories
            else:
                categories = raw_categories
        else:
            categories = raw_categories
    else:
        probabilities = np.asarray(classifier.predict(X, verbose=0))
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        label_encoder = models["label_encoder"]
        predicted_indices = np.argmax(probabilities, axis=1)
        categories = label_encoder.inverse_transform(predicted_indices)
        classes = label_encoder.classes_

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)

    return categories, probabilities, classes


def _prediction_from_arrays(text: str, model_name: str, category_X, impact_X) -> dict:
    """Build a prediction dict from precomputed feature matrices."""
    models = load_models()
    classifier = _get_classifier(model_name)
    regressor = models["linear"]

    categories, probabilities, classes = _predict_classification(model_name, classifier, category_X)

    category = categories[0]
    probability_row = probabilities[0]
    confidence = float(np.max(probability_row))
    if classes is None:
        classes = range(len(probability_row))
    all_probs = {
        cls: round(float(prob), 4)
        for cls, prob in zip(classes, probability_row)
    }

    impact_raw = regressor.predict(impact_X)[0]
    impact_score = float(np.clip(impact_raw, 0, 100))

    return {
        "headline": text.strip()[:200],
        "category": category,
        "impact_score": round(impact_score, 1),
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
        "model_used": model_name,
    }


# ──────────────────────────────────────────────
# Single Prediction
# ──────────────────────────────────────────────
def predict_single(text: str, model_name: str = "logistic_regression") -> dict:
    """
    Predict category and impact score for a single news text.

    Args:
        text: raw news headline or article
        model_name: 'logistic_regression' or 'neural_network'

    Returns:
        dict with: category, impact_score, confidence, model_used, headline
    """
    category_vectorizer = _get_vectorizer(model_name)
    impact_vectorizer = _get_vectorizer("logistic_regression")

    category_X = transform_text(text, category_vectorizer)
    if model_name == "neural_network":
        category_X = category_X.toarray().astype(np.float32)
    impact_X = transform_text(text, impact_vectorizer)

    return _prediction_from_arrays(text, model_name, category_X, impact_X)


# ──────────────────────────────────────────────
# Batch Prediction
# ──────────────────────────────────────────────
def predict_batch(texts: list, model_name: str = "logistic_regression") -> list:
    """
    Predict category and impact for multiple news texts.

    Args:
        texts: list of raw news strings
        model_name: classifier to use

    Returns:
        List of prediction dicts
    """
    category_vectorizer = _get_vectorizer(model_name)
    impact_vectorizer = _get_vectorizer("logistic_regression")

    category_X = transform_batch(texts, category_vectorizer)
    if model_name == "neural_network":
        category_X = category_X.toarray().astype(np.float32)
    impact_X = transform_batch(texts, impact_vectorizer)

    models = load_models()
    classifier = _get_classifier(model_name)
    regressor = models["linear"]

    categories, probabilities, classes = _predict_classification(model_name, classifier, category_X)
    impacts = np.clip(regressor.predict(impact_X), 0, 100)

    results = []
    for i, text in enumerate(texts):
        probability_row = probabilities[i]
        confidence = float(np.max(probability_row))
        all_probs = {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probability_row)
        }
        results.append({
            "headline": text.strip()[:200],
            "category": categories[i],
            "impact_score": round(float(impacts[i]), 1),
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs,
            "model_used": model_name,
        })

    return results


# ──────────────────────────────────────────────
# Explainability
# ──────────────────────────────────────────────
def get_explainability(text: str, model_name: str = "logistic_regression") -> dict:
    """
    Get explainability data for a prediction — which words most influenced
    the classification and impact score.

    Returns:
        dict with: prediction, top_features, category_weights
    """
    models = load_models()
    vectorizer = _get_vectorizer(model_name)
    classifier = _get_classifier(model_name)

    # Get prediction
    prediction = predict_single(text, model_name)

    # Top TF-IDF features in this text
    top_features = get_top_features(text, vectorizer, top_n=15)

    # Get classifier feature weights for predicted category
    category_weights = []
    if hasattr(classifier, "coef_"):
        feature_names = vectorizer.get_feature_names_out()
        class_labels = list(classifier.classes_)
        if class_labels and isinstance(class_labels[0], (int, np.integer, float, np.floating)):
            label_encoder = models["label_encoder"]
            class_labels = list(label_encoder.classes_)

        cat_idx = class_labels.index(prediction["category"])
        coefs = classifier.coef_[cat_idx]

        # Find which of the top features have high model weights
        X = transform_text(text, vectorizer)
        dense = X.toarray().flatten()

        for feat in top_features:
            word = feat["word"]
            if word in feature_names:
                feat_idx = list(feature_names).index(word)
                weight = float(coefs[feat_idx])
                contribution = float(dense[feat_idx] * weight)
                category_weights.append({
                    "word": word,
                    "tfidf_score": feat["score"],
                    "model_weight": round(weight, 4),
                    "contribution": round(contribution, 4),
                    "direction": "positive" if contribution > 0 else "negative",
                })

        category_weights.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    return {
        "prediction": prediction,
        "top_features": top_features[:10],
        "category_weights": category_weights[:10],
        "vectorizer_used": "tfidf_nn.pkl" if model_name == "neural_network" else "tfidf_vectorizer.pkl",
        "explanation": _generate_explanation(prediction, category_weights[:5]),
    }


def _generate_explanation(prediction: dict, weights: list) -> str:
    """Generate a human-readable explanation of the prediction."""
    cat = prediction["category"]
    score = prediction["impact_score"]
    conf = prediction["confidence"] * 100

    positive_words = [w["word"] for w in weights if w.get("direction") == "positive"][:3]
    words_str = ", ".join(f'"{w}"' for w in positive_words) if positive_words else "contextual patterns"

    return (
        f"This news was classified as {cat} (confidence: {conf:.1f}%) "
        f"with an impact score of {score:.1f}/100. "
        f"Key words driving this prediction: {words_str}."
    )


# ──────────────────────────────────────────────
# Model Comparison
# ──────────────────────────────────────────────
def get_model_comparison() -> dict:
    """Return the model comparison report, generating it if needed."""
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            report = json.load(f)
        models = load_models()
        models["report"] = report
        return report

    models = load_models()
    report = _build_model_comparison_report(models)
    models["report"] = report
    return report


def _build_model_comparison_report(models: dict) -> dict:
    """Create a fresh comparison report from the dataset and saved models."""
    if not os.path.exists(DATASET_PATH):
        return {"error": f"Dataset not found at {DATASET_PATH}"}

    df = pd.read_csv(DATASET_PATH)
    required_columns = {"headline", "category", "impact_score"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        return {"error": f"Dataset is missing columns: {', '.join(sorted(missing_columns))}"}

    texts = df["headline"].fillna("").astype(str).tolist()
    y_category = df["category"].astype(str).values
    y_impact = df["impact_score"].astype(float).values

    logistic_vectorizer = models["vectorizer"]
    neural_vectorizer = models["neural_vectorizer"]
    X_logistic = transform_batch(texts, logistic_vectorizer)
    X_neural = transform_batch(texts, neural_vectorizer)

    indices = np.arange(len(df))
    train_idx, test_idx, y_cat_train, y_cat_test, y_imp_train, y_imp_test = train_test_split(
        indices,
        y_category,
        y_impact,
        test_size=0.2,
        random_state=42,
        stratify=y_category,
    )

    X_logistic_test = X_logistic[test_idx]
    X_neural_test = X_neural[test_idx]

    label_encoder = models["label_encoder"]

    def _normalize_class_predictions(predictions, classifier):
        classes = getattr(classifier, "classes_", None)
        if classes is not None:
            classes_array = np.asarray(classes)
            if np.issubdtype(classes_array.dtype, np.number):
                return label_encoder.inverse_transform(np.asarray(predictions, dtype=int))
        return np.asarray(predictions)

    logistic = models["logistic"]
    lr_pred = _normalize_class_predictions(logistic.predict(X_logistic_test), logistic)
    lr_report = {
        "type": "classification",
        "accuracy": round(accuracy_score(y_cat_test, lr_pred), 4),
        "precision": round(precision_score(y_cat_test, lr_pred, average="macro", zero_division=0), 4),
        "recall": round(recall_score(y_cat_test, lr_pred, average="macro", zero_division=0), 4),
        "f1_score": round(f1_score(y_cat_test, lr_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_cat_test, lr_pred).tolist(),
    }

    nn_report = {"type": "classification", "architecture": "Input -> 256 -> 128 -> 4", "activation": "ReLU", "optimizer": "Adam"}
    neural = models.get("neural")
    if neural is not None:
        X_test_dense = X_neural_test.toarray().astype(np.float32)
        nn_probs = np.asarray(neural.predict(X_test_dense, verbose=0))
        if nn_probs.ndim == 1:
            nn_probs = nn_probs.reshape(1, -1)
        nn_indices = np.argmax(nn_probs, axis=1)
        nn_pred = label_encoder.inverse_transform(nn_indices)
        nn_report.update({
            "accuracy": round(accuracy_score(y_cat_test, nn_pred), 4),
            "precision": round(precision_score(y_cat_test, nn_pred, average="macro", zero_division=0), 4),
            "recall": round(recall_score(y_cat_test, nn_pred, average="macro", zero_division=0), 4),
            "f1_score": round(f1_score(y_cat_test, nn_pred, average="macro", zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_cat_test, nn_pred).tolist(),
        })
    else:
        nn_report.update({"error": models.get("neural_error", "Neural-network model is unavailable.")})

    ridge = models["linear"]
    ridge_pred = np.clip(ridge.predict(X_logistic_test), 0, 100)
    ridge_report = {
        "type": "regression",
        "target": "impact_score (0-100)",
        "mae": round(mean_absolute_error(y_imp_test, ridge_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_imp_test, ridge_pred)), 2),
        "r2_score": round(r2_score(y_imp_test, ridge_pred), 4),
    }

    report = {
        "models": {
            "logistic_regression": lr_report,
            "neural_network": nn_report,
            "linear_regression": ridge_report,
        },
        "dataset_info": {
            "total_records": int(len(df)),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "features": int(X_logistic.shape[1]),
            "categories": sorted(df["category"].astype(str).unique().tolist()),
        },
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return report


# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test
    test_text = "Major earthquake strikes coastal city causing widespread destruction"
    print("Testing prediction pipeline...\n")
    result = predict_single(test_text)
    print(f"Text: {test_text}")
    print(f"Category: {result['category']}")
    print(f"Impact: {result['impact_score']}")
    print(f"Confidence: {result['confidence']:.2%}")
