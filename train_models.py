"""
train_models.py — Dataset Generation & Model Training
======================================================
Person 1 (ML Engineer) module.

Generates synthetic crisis news dataset, trains 3 models:
  1. Logistic Regression (classification)
  2. Neural Network / MLP (classification)
  3. Ridge Regression (impact score prediction)
Saves all models + accuracy report.
"""

import os
import random
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    classification_report
)

from preprocessing import clean_corpus, build_tfidf

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(DATA_DIR, "crisis_news_dataset.csv")


# ══════════════════════════════════════════════
# DATASET GENERATION
# ══════════════════════════════════════════════

WAR_TEMPLATES = {
    "headlines": [
        "{country} launches military {op} in {region}",
        "Armed conflict {verb} near {region} border",
        "{country} imposes new sanctions on {target} amid tensions",
        "Military forces deploy to {region} as tensions escalate",
        "Peace talks between {country} and {target} collapse",
        "{num} casualties reported in {region} {attack_type}",
        "UN condemns military aggression in {region}",
        "Rebel forces capture key {location} in {region}",
        "Missile strikes hit {location} in {region} overnight",
        "Border clashes intensify between {country} and {target}",
        "{country} mobilizes reserve forces amid {region} crisis",
        "Weapons shipment intercepted near {region} conflict zone",
        "Humanitarian corridor in {region} comes under fire",
        "Coalition airstrikes target positions in {region}",
        "Ceasefire violations reported across {region} front lines",
        "Naval blockade imposed on {region} ports by {country}",
        "Drone strikes escalate in {region} conflict",
        "Guerrilla attacks disrupt supply lines in {region}",
        "Military coup attempt reported in {country}",
        "Ethnic violence erupts across {region} provinces",
    ],
    "descriptions": [
        "The situation continues to deteriorate as military operations expand across the region. Civilian casualties are mounting and international pressure grows for a diplomatic resolution.",
        "Reports indicate significant troop movements and artillery exchanges along the front lines. Aid organizations warn of a growing humanitarian crisis in affected areas.",
        "Intelligence sources confirm escalation of hostilities with heavy weaponry deployed. Diplomatic channels remain strained as both sides refuse to back down from their positions.",
        "Security forces have been placed on high alert following the latest developments. Evacuation of civilians from border areas is underway amid fears of further escalation.",
        "The conflict has disrupted critical infrastructure including hospitals and schools. International observers call for immediate ceasefire and humanitarian access to affected populations.",
    ],
    "vars": {
        "country": ["United States", "Russia", "China", "Iran", "Israel", "Turkey", "India", "Pakistan", "North Korea", "Saudi Arabia"],
        "target": ["neighboring state", "opposition forces", "separatist groups", "rival nation", "insurgent groups"],
        "region": ["Eastern Europe", "Middle East", "South Asia", "Korean Peninsula", "North Africa", "Central Asia", "Sub-Saharan Africa", "Southeast Asia"],
        "op": ["offensive", "operation", "campaign", "intervention", "assault"],
        "verb": ["escalates", "intensifies", "spreads", "worsens", "flares up"],
        "attack_type": ["shelling", "airstrike", "ground assault", "bombardment", "ambush"],
        "location": ["military base", "strategic city", "port facility", "government building", "supply depot"],
        "num": ["12", "47", "83", "120", "250", "500"],
    },
    "impact_range": (45, 95),
}

ECONOMY_TEMPLATES = {
    "headlines": [
        "Global markets {verb} amid fears of {crisis}",
        "{region} GDP contracts by {pct}% in latest quarter",
        "Inflation hits {num}-year high across {region}",
        "Central bank {action} interest rates to combat {crisis}",
        "Trade war escalates as {country} imposes tariffs on {goods}",
        "Unemployment surges to {pct}% in {region}",
        "Major {institution} faces insolvency sparking contagion fears",
        "{region} currency crashes against major trading pairs",
        "Oil prices {verb2} after {trigger}",
        "Housing market bubble bursts in {region}",
        "Supply chain disruption causes {goods} shortage globally",
        "Government debt reaches critical levels in {country}",
        "Stock exchange halts trading after {pct}% single-day drop",
        "Foreign investors flee {region} amid economic instability",
        "Consumer confidence plummets to record low in {region}",
        "Banking crisis deepens as deposits withdrawn en masse",
        "Credit rating downgrade hits {country} government bonds",
        "Commodity prices spike disrupting global trade flows",
        "Pension fund losses threaten retirement savings for millions",
        "Economic sanctions create severe shortages in {country}",
    ],
    "descriptions": [
        "Financial analysts warn of prolonged economic downturn as key indicators continue to decline. Markets have shown extreme volatility with investors moving to safe-haven assets.",
        "The economic impact is being felt across multiple sectors with manufacturing and services both contracting. Consumer spending has fallen sharply as confidence hits multi-year lows.",
        "Central banks face difficult policy decisions as traditional monetary tools prove insufficient. Fiscal stimulus packages are being debated but political divisions slow implementation.",
        "International trade flows have been significantly disrupted affecting supply chains globally. Companies report declining revenues and are implementing cost-cutting measures including layoffs.",
        "The crisis threatens to spill over into emerging markets creating a cascade of financial instability. International lending institutions are mobilizing emergency support packages.",
    ],
    "vars": {
        "country": ["United States", "China", "European Union", "Japan", "United Kingdom", "Brazil", "India", "Germany", "South Korea", "Australia"],
        "region": ["Asia-Pacific", "European", "North American", "Latin American", "Global", "Emerging Market"],
        "verb": ["plunge", "tumble", "crash", "decline sharply", "drop significantly"],
        "verb2": ["surge", "collapse", "spike dramatically", "plummet"],
        "crisis": ["recession", "banking collapse", "debt default", "stagflation", "economic downturn"],
        "action": ["raises", "hikes", "slashes", "freezes"],
        "goods": ["semiconductor", "agricultural", "energy", "steel", "technology"],
        "institution": ["investment bank", "commercial lender", "insurance giant", "hedge fund"],
        "trigger": ["OPEC decision", "pipeline disruption", "refinery shutdown", "geopolitical tensions"],
        "pct": ["3.2", "4.7", "6.1", "8.5", "12", "15"],
        "num": ["10", "15", "20", "30", "40"],
    },
    "impact_range": (25, 85),
}

DISASTER_TEMPLATES = {
    "headlines": [
        "Magnitude {mag} earthquake strikes {region} causing widespread destruction",
        "Category {cat} hurricane makes landfall in {region}",
        "Devastating floods hit {region} after record rainfall",
        "Massive wildfire spreads across {region} burning {num} acres",
        "Tsunami warning issued after undersea earthquake near {region}",
        "Volcanic eruption in {region} forces mass evacuation",
        "Severe drought threatens food supply across {region}",
        "Deadly landslide buries homes in {region} after heavy rains",
        "Extreme heat wave kills {deaths} across {region}",
        "Tornado outbreak causes widespread destruction in {region}",
        "Tropical cyclone {name} intensifies threatening {region}",
        "Glacier collapse triggers massive flooding in {region}",
        "Dam failure floods downstream communities in {region}",
        "Avalanche traps {deaths} mountaineers in {region}",
        "Sandstorm engulfs cities across {region} halting transportation",
        "Monsoon flooding displaces millions in {region}",
        "Forest fire smoke blankets {region} causing health emergency",
        "Blizzard paralyzes transportation across {region}",
        "Rising sea levels inundate coastal areas in {region}",
        "Sinkhole swallows buildings in {region} urban area",
    ],
    "descriptions": [
        "Emergency response teams are mobilizing to affected areas as the scale of destruction becomes apparent. Thousands have been displaced and critical infrastructure has been severely damaged.",
        "Rescue operations are underway with military and civilian teams searching for survivors. The disaster has overwhelmed local response capabilities and international assistance has been requested.",
        "The death toll is expected to rise as teams reach isolated communities cut off by the disaster. Hospitals are operating beyond capacity and medical supplies are running critically low.",
        "Satellite imagery reveals the full extent of devastation across the affected region. Agricultural land has been destroyed threatening food security for the coming season.",
        "Climate scientists link the disaster to changing weather patterns and warn of increased frequency of such events. Reconstruction costs are estimated in the billions of dollars.",
    ],
    "vars": {
        "region": ["Southeast Asia", "Caribbean", "Central America", "Pacific Coast", "South Asia", "East Africa", "Mediterranean", "Oceania", "Andes Region", "Central Europe"],
        "mag": ["6.2", "6.8", "7.1", "7.4", "7.9", "8.1"],
        "cat": ["3", "4", "5"],
        "num": ["50000", "100000", "200000", "500000"],
        "deaths": ["23", "47", "89", "156", "300"],
        "name": ["Maria", "Irma", "Harvey", "Katrina", "Haiyan", "Dorian", "Michael"],
    },
    "impact_range": (50, 98),
}

TECH_TEMPLATES = {
    "headlines": [
        "Major data breach exposes {num} million records at {company}",
        "Ransomware attack cripples {target} systems across {region}",
        "AI system failure causes widespread {impact} disruption",
        "Critical vulnerability found in {software} affecting millions",
        "Autonomous {vehicle} incident raises safety concerns globally",
        "Government surveillance program exposed by whistleblower",
        "Tech giant faces antitrust action over {issue}",
        "Deepfake technology used in {attack} operation targeting {target2}",
        "Social media platform suffers {hours}-hour global outage",
        "Cybersecurity attack targets {region} power grid infrastructure",
        "AI-generated misinformation campaign influences {target2}",
        "Critical infrastructure hack exposes {target} vulnerabilities",
        "Quantum computing breakthrough threatens encryption standards",
        "Autonomous weapons deployment sparks international debate",
        "Mass surveillance technology exported to authoritarian regimes",
        "Algorithm bias discovered in {target} decision-making systems",
        "Cloud service provider outage affects {num} million users worldwide",
        "Zero-day exploit actively used against {software} installations",
        "Biometric data leak compromises identity security for millions",
        "AI model training data poisoning attack discovered at scale",
    ],
    "descriptions": [
        "Security researchers have confirmed the severity of the breach affecting systems worldwide. Organizations are scrambling to patch vulnerabilities and assess the extent of data compromise.",
        "The incident highlights growing concerns about cybersecurity preparedness in critical sectors. Government agencies are investigating and coordinating response efforts with private sector partners.",
        "Experts warn this represents a new class of technological threat requiring updated defense strategies. The attack vector exploited known weaknesses that had not been adequately addressed.",
        "The technology implications extend beyond immediate damage raising questions about regulatory oversight. Industry leaders call for updated frameworks to address emerging digital threats.",
        "Users are advised to change credentials and monitor accounts for suspicious activity. The full scope of the impact is still being assessed as forensic analysis continues.",
    ],
    "vars": {
        "company": ["global tech firm", "social media giant", "cloud provider", "financial platform", "healthcare system"],
        "target": ["hospital", "banking", "transportation", "government", "energy", "telecommunications"],
        "target2": ["elections", "financial markets", "public opinion", "government officials", "corporate executives"],
        "region": ["North American", "European", "Asia-Pacific", "global"],
        "software": ["enterprise software", "operating system", "web framework", "database platform", "IoT firmware"],
        "vehicle": ["vehicle", "drone", "delivery robot", "aircraft"],
        "impact": ["service", "transportation", "financial", "communication"],
        "issue": ["market dominance", "data privacy", "anti-competitive practices", "content moderation"],
        "attack": ["fraud", "espionage", "disinformation", "phishing"],
        "hours": ["6", "12", "18", "24", "48"],
        "num": ["2", "5", "15", "50", "100", "200"],
    },
    "impact_range": (20, 75),
}

ALL_TEMPLATES = {
    "War": WAR_TEMPLATES,
    "Economy": ECONOMY_TEMPLATES,
    "Disaster": DISASTER_TEMPLATES,
    "Tech": TECH_TEMPLATES,
}

REGIONS_LIST = [
    "North America", "South America", "Europe", "Middle East",
    "East Asia", "South Asia", "Southeast Asia", "Africa", "Oceania",
]


def _fill_template(template: str, variables: dict) -> str:
    """Fill a template string with random variable choices."""
    result = template
    for key, values in variables.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def generate_dataset(n_per_category: int = 130) -> pd.DataFrame:
    """
    Generate a synthetic crisis news dataset.

    Args:
        n_per_category: number of records per category (~130 -> 520 total)

    Returns:
        DataFrame with columns: headline, description, category, impact_score, region, date
    """
    print("[...] Generating synthetic crisis news dataset...")
    random.seed(42)
    records = []
    base_date = datetime(2024, 1, 1)

    for category, tmpl in ALL_TEMPLATES.items():
        headlines_pool = tmpl["headlines"]
        descriptions_pool = tmpl["descriptions"]
        variables = tmpl["vars"]
        impact_min, impact_max = tmpl["impact_range"]

        for i in range(n_per_category):
            # Pick and fill headline template
            headline_tmpl = headlines_pool[i % len(headlines_pool)]
            headline = _fill_template(headline_tmpl, variables)

            # Pick description
            description = random.choice(descriptions_pool)

            # Impact score with some noise
            base_impact = random.uniform(impact_min, impact_max)
            # Higher severity words -> higher impact
            severity_boost = 0
            high_severity = ["catastrophic", "devastating", "massive", "critical", "extreme", "deadly", "severe", "collapse", "crash"]
            for word in high_severity:
                if word in headline.lower():
                    severity_boost += random.uniform(3, 8)
            impact_score = min(100, base_impact + severity_boost)

            # Random region and date
            region = random.choice(REGIONS_LIST)
            date = base_date + timedelta(days=random.randint(0, 365))

            records.append({
                "headline": headline,
                "description": description,
                "category": category,
                "impact_score": round(impact_score, 1),
                "region": region,
                "date": date.strftime("%Y-%m-%d"),
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"[OK] Dataset saved -> {DATASET_PATH}")
    print(f"    Total records: {len(df)}")
    print(f"    Categories: {dict(df['category'].value_counts())}")
    return df


# ══════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════

def train_all_models():
    """Full training pipeline: generate data -> preprocess -> train -> save -> report."""

    # ── 1. Load or generate dataset ──
    if os.path.exists(DATASET_PATH):
        print(f"[OK] Loading existing dataset from {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
    else:
        df = generate_dataset()

    print(f"\n{'='*60}")
    print(f"  TRAINING PIPELINE")
    print(f"{'='*60}\n")

    # ── 2. Combine headline + description for richer features ──
    df["text"] = df["headline"] + " " + df["description"]

    # ── 3. Clean text ──
    print("[...] Cleaning text corpus...")
    cleaned_texts = clean_corpus(df["text"].tolist())
    print(f"[OK] Cleaned {len(cleaned_texts)} documents")

    # ── 4. Build TF-IDF ──
    print("\n[...] Building TF-IDF vectorizer...")
    vectorizer = build_tfidf(cleaned_texts)
    X = vectorizer.transform(cleaned_texts)

    # ── 5. Prepare labels ──
    y_category = df["category"].values
    y_impact = df["impact_score"].values

    # ── 6. Train/test split ──
    X_train, X_test, y_cat_train, y_cat_test, y_imp_train, y_imp_test = train_test_split(
        X, y_category, y_impact, test_size=0.2, random_state=42, stratify=y_category
    )

    print(f"\n    Train size: {X_train.shape[0]}")
    print(f"    Test size:  {X_test.shape[0]}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    report = {"models": {}, "dataset_info": {
        "total_records": len(df),
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "features": X.shape[1],
        "categories": list(df["category"].unique()),
    }}

    # ════════════════════════════════════════
    # MODEL 1: Logistic Regression (Classification)
    # ════════════════════════════════════════
    print(f"\n{'-'*50}")
    print("  Model 1: Logistic Regression (Classification)")
    print(f"{'-'*50}")

    lr_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
    )
    lr_model.fit(X_train, y_cat_train)
    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)

    lr_acc = accuracy_score(y_cat_test, lr_pred)
    lr_prec = precision_score(y_cat_test, lr_pred, average="macro")
    lr_rec = recall_score(y_cat_test, lr_pred, average="macro")
    lr_f1 = f1_score(y_cat_test, lr_pred, average="macro")
    lr_cm = confusion_matrix(y_cat_test, lr_pred).tolist()

    print(f"  Accuracy:  {lr_acc:.4f}")
    print(f"  Precision: {lr_prec:.4f}")
    print(f"  Recall:    {lr_rec:.4f}")
    print(f"  F1 Score:  {lr_f1:.4f}")

    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_model.pkl"))
    print(f"  [OK] Saved -> logistic_model.pkl")

    report["models"]["logistic_regression"] = {
        "type": "classification",
        "accuracy": round(lr_acc, 4),
        "precision": round(lr_prec, 4),
        "recall": round(lr_rec, 4),
        "f1_score": round(lr_f1, 4),
        "confusion_matrix": lr_cm,
    }

    # ════════════════════════════════════════
    # MODEL 2: MLP Neural Network (Classification)
    # ════════════════════════════════════════
    print(f"\n{'-'*50}")
    print("  Model 2: Neural Network / MLP (Classification)")
    print(f"{'-'*50}")

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=False, # Disabled to avoid validation score nan check issue
        learning_rate="adaptive",
        batch_size=32,
    )
    # MLP often requires dense float arrays
    X_train_dense = X_train.toarray().astype(np.float64)
    X_test_dense = X_test.toarray().astype(np.float64)

    mlp_model.fit(X_train_dense, y_cat_train)
    mlp_pred = mlp_model.predict(X_test_dense)

    mlp_acc = accuracy_score(y_cat_test, mlp_pred)
    mlp_prec = precision_score(y_cat_test, mlp_pred, average="macro")
    mlp_rec = recall_score(y_cat_test, mlp_pred, average="macro")
    mlp_f1 = f1_score(y_cat_test, mlp_pred, average="macro")
    mlp_cm = confusion_matrix(y_cat_test, mlp_pred).tolist()

    print(f"  Accuracy:  {mlp_acc:.4f}")
    print(f"  Precision: {mlp_prec:.4f}")
    print(f"  Recall:    {mlp_rec:.4f}")
    print(f"  F1 Score:  {mlp_f1:.4f}")
    print(f"  Architecture: Input -> 256 -> 128 -> 4 (ReLU, Adam)")

    joblib.dump(mlp_model, os.path.join(MODELS_DIR, "neural_model.pkl"))
    print(f"  [OK] Saved -> neural_model.pkl")

    report["models"]["neural_network"] = {
        "type": "classification",
        "architecture": "Input -> 256 -> 128 -> 4",
        "activation": "ReLU",
        "optimizer": "Adam",
        "accuracy": round(mlp_acc, 4),
        "precision": round(mlp_prec, 4),
        "recall": round(mlp_rec, 4),
        "f1_score": round(mlp_f1, 4),
        "confusion_matrix": mlp_cm,
    }

    # ════════════════════════════════════════
    # MODEL 3: Ridge Regression (Impact Score)
    # ════════════════════════════════════════
    print(f"\n{'-'*50}")
    print("  Model 3: Ridge Regression (Impact Score)")
    print(f"{'-'*50}")

    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train, y_imp_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_pred = np.clip(ridge_pred, 0, 100)  # Clamp to valid range

    ridge_mae = mean_absolute_error(y_imp_test, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_imp_test, ridge_pred))
    ridge_r2 = r2_score(y_imp_test, ridge_pred)

    print(f"  MAE:  {ridge_mae:.2f}")
    print(f"  RMSE: {ridge_rmse:.2f}")
    print(f"  R2:   {ridge_r2:.4f}")

    joblib.dump(ridge_model, os.path.join(MODELS_DIR, "linear_model.pkl"))
    print(f"  [OK] Saved -> linear_model.pkl")

    report["models"]["linear_regression"] = {
        "type": "regression",
        "target": "impact_score (0-100)",
        "mae": round(ridge_mae, 2),
        "rmse": round(ridge_rmse, 2),
        "r2_score": round(ridge_r2, 4),
    }

    # ── Save accuracy report ──
    report_path = os.path.join(MODELS_DIR, "accuracy_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ALL MODELS TRAINED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"\n[OK] Accuracy report saved -> {report_path}")
    print(f"\nFiles saved in {MODELS_DIR}/:")
    for fname in os.listdir(MODELS_DIR):
        fpath = os.path.join(MODELS_DIR, fname)
        size = os.path.getsize(fpath)
        print(f"  - {fname} ({size/1024:.1f} KB)")

    return report


# ──────────────────────────────────────────────
if __name__ == "__main__":
    train_all_models()
