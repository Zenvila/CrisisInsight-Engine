"""
utils.py — Shared Utilities
============================
Helper functions used across all modules.
"""

import random
import json
import os

# ──────────────────────────────────────────────
# Risk Level Classification
# ──────────────────────────────────────────────
RISK_LEVELS = {
    "LOW":      {"min": 0,  "max": 40,  "color": "#00d2d3", "emoji": "🟢", "description": "Minimal threat. Situation is under control."},
    "MEDIUM":   {"min": 40, "max": 60,  "color": "#feca57", "emoji": "🟡", "description": "Moderate concern. Monitor closely."},
    "HIGH":     {"min": 60, "max": 80,  "color": "#ff9f43", "emoji": "🟠", "description": "Significant impact expected. Action recommended."},
    "CRITICAL": {"min": 80, "max": 100, "color": "#ff6b6b", "emoji": "🔴", "description": "Severe impact. Immediate response required."},
}

CATEGORY_INFO = {
    "War":         {"icon": "⚔️", "color": "#ff6b6b", "description": "Military conflicts, armed disputes, geopolitical tensions"},
    "Economy":     {"icon": "📉", "color": "#feca57", "description": "Financial crises, market disruptions, trade conflicts"},
    "Disaster":    {"icon": "🌊", "color": "#ff9f43", "description": "Natural disasters, environmental emergencies"},
    "Technology":  {"icon": "💻", "color": "#54a0ff", "description": "Cybersecurity threats, technology disruptions, AI risks"},
    "Tech":        {"icon": "💻", "color": "#54a0ff", "description": "Cybersecurity threats, technology disruptions, AI risks"},
}

# Category interaction multipliers for scenario analysis
INTERACTION_MATRIX = {
    ("Economy", "Technology"): 1.15,
    ("Disaster", "War"):       1.5,
    ("Economy", "War"):        1.4,
}


def get_risk_level(impact_score: float) -> dict:
    """
    Convert a numeric impact score (0-100) to a risk level.

    Returns:
        dict with keys: level, color, emoji, description, score
    """
    score = max(0, min(100, impact_score))
    levels = list(RISK_LEVELS.items())
    for index, (level_name, info) in enumerate(levels):
        is_last_level = index == len(levels) - 1
        if info["min"] <= score < info["max"] or (is_last_level and score >= info["min"]):
            return {
                "level": level_name,
                "color": info["color"],
                "emoji": info["emoji"],
                "description": info["description"],
                "score": round(score, 1),
            }
    # Fallback
    return {"level": "MEDIUM", "color": "#feca57", "emoji": "🟡",
            "description": "Moderate concern.", "score": round(score, 1)}


def get_interaction_multiplier(cat1: str, cat2: str) -> float:
    """Get the compound impact multiplier for two event categories."""
    normalized = ["Technology" if cat == "Tech" else cat for cat in [cat1, cat2]]
    key = tuple(sorted(normalized))
    return INTERACTION_MATRIX.get(key, 1.0)


def format_prediction(prediction: dict) -> dict:
    """Format a raw prediction dict for API response."""
    risk = get_risk_level(prediction.get("impact_score", 0))
    cat = prediction.get("category", "Unknown")
    cat_info = CATEGORY_INFO.get(cat, {"icon": "❓", "color": "#888", "description": ""})

    return {
        "category": cat,
        "category_icon": cat_info["icon"],
        "category_color": cat_info["color"],
        "category_description": cat_info["description"],
        "impact_score": round(prediction.get("impact_score", 0), 1),
        "confidence": round(prediction.get("confidence", 0) * 100, 1),
        "risk": risk,
        "model_used": prediction.get("model_used", "logistic_regression"),
        "headline": prediction.get("headline", ""),
    }


def validate_input(text: str) -> tuple:
    """
    Validate user input text.

    Returns:
        (is_valid: bool, error_message: str or None)
    """
    if not text or not isinstance(text, str):
        return False, "Input text is required."
    text = text.strip()
    if len(text) < 10:
        return False, "Input text must be at least 10 characters."
    if len(text) > 5000:
        return False, "Input text must be under 5000 characters."
    return True, None


# ──────────────────────────────────────────────
# Sample News for Demo / Testing
# ──────────────────────────────────────────────
SAMPLE_NEWS = [
    "Major earthquake strikes coastal city, thousands displaced and infrastructure destroyed",
    "Global stock markets plunge amid fears of widespread economic recession",
    "Military forces launch large-scale offensive near disputed border region",
    "Massive data breach exposes millions of user records from major tech company",
    "Category 5 hurricane approaches populated coastline, evacuations ordered",
    "Trade war escalates as new tariffs imposed on critical imports",
    "Ceasefire negotiations collapse, renewed fighting reported across multiple fronts",
    "Artificial intelligence system malfunction causes widespread service disruption",
    "Devastating floods submerge entire districts after record rainfall",
    "Central bank raises interest rates sharply to combat spiraling inflation",
    "National power grid experiences rolling blackouts after extreme heatwave surge",
    "Major port operations suspended following cyclone damage to cargo terminals",
    "Ransomware attack disrupts emergency hotline and hospital appointment systems",
    "Severe drought conditions trigger food supply shortages in multiple provinces",
    "Currency loses 12 percent value after sovereign credit downgrade",
    "Wildfire spreads near telecom hub, forcing regional internet shutdown",
    "Public transit network halted after cyber intrusion into signaling platform",
    "Fuel distribution delays intensify after refinery fire at strategic facility",
    "Cross-border shelling displaces thousands from rural communities",
    "Water contamination warning issued after floodwater enters treatment plants",
    "Stock exchange invokes circuit breaker after rapid market decline",
    "Cloud service outage impacts banking, aviation, and emergency response tools",
]


def get_sample_news(n: int = 5) -> list:
    """Return n random sample news items for demo purposes."""
    return random.sample(SAMPLE_NEWS, min(n, len(SAMPLE_NEWS)))


# ──────────────────────────────────────────────
# File I/O Helpers
# ──────────────────────────────────────────────
def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str):
    """Save data to a JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
