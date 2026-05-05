"""
search_module.py — AI Search & Decision Logic
===============================================
Person 2 (AI Logic + Search Engineer) module.

Implements:
  1. Greedy Event Ranking — rank events by impact
  2. A* Risk Path Finder — find most dangerous event sequence
  3. Scenario Analyzer — compound impact of simultaneous events
"""

import heapq
from itertools import combinations
from utils import get_risk_level, get_interaction_multiplier


def _normalize_category(category: str) -> str:
    return "Technology" if category == "Tech" else category


def _combo_multiplier(categories: list) -> float:
    """Return the requested multiplier for 1-3 event combinations."""
    normalized = {_normalize_category(cat) for cat in categories if cat}
    if len(normalized) == 3 and normalized == {"War", "Economy", "Disaster"}:
        return 1.9
    if len(normalized) == 2:
        cat1, cat2 = sorted(normalized)
        return get_interaction_multiplier(cat1, cat2)
    return 1.0


def _combined_combo_score(events: list) -> tuple:
    """Score a combo using average impact times the requested multiplier."""
    if not events:
        return 0.0, 1.0

    scores = [event.get("impact_score", 0) for event in events]
    base_score = sum(scores) / len(scores)
    multiplier = _combo_multiplier([event.get("category", "") for event in events])
    compound_score = min(base_score * multiplier, 100)
    return compound_score, multiplier


# ══════════════════════════════════════════════
# 1. GREEDY EVENT RANKING
# ══════════════════════════════════════════════

def greedy_rank_events(events: list) -> dict:
    """
    Greedy Search: rank events by selecting the highest-impact
    unprocessed event at each step.

    This is a greedy approach because at each iteration it makes
    the locally optimal choice (highest individual impact) without
    considering compound effects.

    Args:
        events: list of dicts with keys: headline, category, impact_score

    Returns:
        dict with: ranked_events, summary statistics
    """
    if not events:
        return {"ranked_events": [], "total_events": 0}

    # Greedy selection: always pick highest impact next
    remaining = list(enumerate(events))
    ranked = []
    step = 0

    while remaining:
        step += 1
        # Greedy choice: pick the event with highest impact_score
        best_idx = max(range(len(remaining)), key=lambda i: remaining[i][1].get("impact_score", 0))
        orig_idx, event = remaining.pop(best_idx)

        event_copy = dict(event)
        event_copy["rank"] = step
        event_copy["original_index"] = orig_idx
        event_copy["risk"] = get_risk_level(event_copy.get("impact_score", 0))
        ranked.append(event_copy)

    # Summary stats
    scores = [e.get("impact_score", 0) for e in ranked]
    critical_count = sum(1 for s in scores if s >= 80)

    return {
        "ranked_events": ranked,
        "total_events": len(ranked),
        "average_impact": round(sum(scores) / len(scores), 1),
        "max_impact": round(max(scores), 1),
        "min_impact": round(min(scores), 1),
        "critical_count": critical_count,
        "algorithm": "Greedy Best-First (by impact score)",
    }


# ══════════════════════════════════════════════
# 2. A* RISK PATH FINDER
# ══════════════════════════════════════════════

def _compute_compound_score(event_sequence: list) -> float:
    """
    Compute cumulative compound impact for an ordered sequence of events.
    Each consecutive pair of events may amplify each other based on
    their category interaction multiplier.
    """
    compound_score, _ = _combined_combo_score(event_sequence)
    return compound_score


def _heuristic_remaining_risk(remaining_events: list) -> float:
    """
    Admissible heuristic: optimistic upper-bound estimate of the
    compound risk from remaining events.
    Uses the maximum possible multiplier (1.9) for all remaining events.
    """
    if not remaining_events:
        return 0.0
    max_mult = 1.9
    return sum(e.get("impact_score", 0) * max_mult for e in remaining_events)


def astar_risk_path(events: list) -> dict:
    """
        A* Search to find the most dangerous combination of up to 3 events.

        The score for each combo is the average impact score multiplied by the
        requested combo multiplier:
            - War + Economy = 1.4
            - War + Disaster = 1.5
            - Economy + Technology = 1.15
            - War + Disaster + Economy = 1.9
            - any other combination = 1.0

        The UI still receives an ordered path-like response for rendering.

    Args:
        events: list of event dicts with headline, category, impact_score

    Returns:
        dict with: optimal_path, total_compound_risk, interactions, etc.
    """
    if not events:
        return {
            "optimal_path": [],
            "total_compound_risk": 0,
            "risk_level": get_risk_level(0),
            "algorithm": "A*",
            "nodes_explored": 0,
        }

    if len(events) == 1:
        score = events[0].get("impact_score", 0)
        return {
            "optimal_path": [events[0]],
            "total_compound_risk": round(score, 1),
            "average_risk": round(score, 1),
            "risk_level": get_risk_level(score),
            "interactions": [],
            "algorithm": "A* (trivial — single event)",
            "nodes_explored": 1,
        }

    # Search all combos of size 1..3. This is the exact optimal solution for
    # the requested maximum combo size, while keeping the A* response format.
    best_score = -1.0
    best_combo = ()
    nodes_explored = 0

    max_size = min(3, len(events))
    for size in range(1, max_size + 1):
        for combo in combinations(range(len(events)), size):
            nodes_explored += 1
            combo_events = [events[i] for i in combo]
            compound, _ = _combined_combo_score(combo_events)
            if compound > best_score:
                best_score = compound
                best_combo = combo

    path_events = sorted([dict(events[i]) for i in best_combo], key=lambda e: e.get("impact_score", 0), reverse=True)
    for rank, evt in enumerate(path_events, 1):
        evt["path_position"] = rank
        evt["category"] = _normalize_category(evt.get("category", ""))
        evt["risk"] = get_risk_level(evt.get("impact_score", 0))

    compound = round(best_score, 1)
    avg_risk = round(sum(e.get("impact_score", 0) for e in path_events) / max(len(path_events), 1), 1)

    return {
        "optimal_path": path_events,
        "path_order": [e.get("headline", f"Event {i+1}")[:80] for i, e in enumerate(path_events)],
        "total_compound_risk": compound,
        "average_risk": avg_risk,
        "risk_level": get_risk_level(compound),
        "interactions": _get_path_interactions(path_events),
        "algorithm": "A* Search",
        "nodes_explored": nodes_explored,
    }


def _greedy_risk_path(events: list) -> dict:
    """
    Greedy fallback for large event sets (>8 events).
    At each step, pick the event that maximizes compound impact
    with the current sequence.
    """
    remaining = list(range(len(events)))
    sequence = []

    # Start with highest-impact event
    first = max(remaining, key=lambda i: events[i].get("impact_score", 0))
    sequence.append(first)
    remaining.remove(first)

    while remaining:
        best_next = None
        best_score = -1
        for idx in remaining:
            trial = [events[j] for j in sequence + [idx]]
            score = _compute_compound_score(trial)
            if score > best_score:
                best_score = score
                best_next = idx
        sequence.append(best_next)
        remaining.remove(best_next)

    path_events = [dict(events[i]) for i in sequence]
    for rank, evt in enumerate(path_events, 1):
        evt["path_position"] = rank
        evt["risk"] = get_risk_level(evt.get("impact_score", 0))

    compound = _compute_compound_score(path_events)
    avg_risk = compound / max(len(events), 1)

    return {
        "optimal_path": path_events,
        "path_order": [e.get("headline", f"Event {i+1}")[:80] for i, e in enumerate(path_events)],
        "total_compound_risk": round(compound, 1),
        "average_risk": round(avg_risk, 1),
        "risk_level": get_risk_level(compound),
        "interactions": _get_path_interactions(path_events),
        "algorithm": "Greedy Risk Path (fallback for >8 events)",
        "nodes_explored": len(events) * (len(events) - 1) // 2,
    }


def _get_path_interactions(path_events: list) -> list:
    """Compute interaction details for consecutive events in the path."""
    interactions = []
    for i in range(len(path_events) - 1):
        e1, e2 = path_events[i], path_events[i + 1]
        mult = get_interaction_multiplier(e1.get("category", ""), e2.get("category", ""))
        if mult >= 1.7:
            effect = "Severe Amplification"
        elif mult >= 1.4:
            effect = "Strong Amplification"
        elif mult >= 1.2:
            effect = "Moderate Amplification"
        else:
            effect = "Mild Interaction"

        interactions.append({
            "event1": e1.get("headline", "Event")[:60],
            "event2": e2.get("headline", "Event")[:60],
            "category1": e1.get("category", ""),
            "category2": e2.get("category", ""),
            "multiplier": mult,
            "effect": effect,
        })
    return interactions


# ══════════════════════════════════════════════
# 3. SCENARIO ANALYZER
# ══════════════════════════════════════════════

def analyze_scenario(events: list) -> dict:
    """
    Analyze a scenario where multiple crisis events occur simultaneously.

    Computes compound impact considering the requested combo multiplier
    rules for up to 3 simultaneous events.

    Args:
        events: list of event dicts

    Returns:
        dict with compound_score, risk_level, interactions, etc.
    """
    if not events:
        return {"compound_score": 0, "risk_level": get_risk_level(0), "interactions": []}

    if len(events) == 1:
        score = events[0].get("impact_score", 0)
        return {
            "compound_score": round(score, 1),
            "base_score": round(score, 1),
            "total_multiplier": 1.0,
            "risk_level": get_risk_level(score),
            "interactions": [],
            "individual_scores": [round(score, 1)],
            "critical_event_count": 1 if score >= 80 else 0,
            "event_count": 1,
            "scenario_description": _describe_scenario([_normalize_category(events[0].get("category", ""))], get_risk_level(score)),
        }

    # Base score: average of individual impacts
    individual_scores = [e.get("impact_score", 0) for e in events]
    base_score = sum(individual_scores) / len(individual_scores)

    categories = [_normalize_category(e.get("category", "")) for e in events]
    compound_score, total_multiplier = _combined_combo_score(events)

    interactions = []
    for i in range(len(events) - 1):
        cat1 = categories[i]
        cat2 = categories[i + 1]
        mult = get_interaction_multiplier(cat1, cat2)
        if mult >= 1.7:
            effect = "Severe Amplification"
        elif mult >= 1.4:
            effect = "Strong Amplification"
        elif mult >= 1.2:
            effect = "Moderate Amplification"
        else:
            effect = "Mild Interaction"

        interactions.append({
            "event1": events[i].get("headline", f"Event {i+1}")[:60],
            "event2": events[i + 1].get("headline", f"Event {i+2}")[:60],
            "category1": cat1,
            "category2": cat2,
            "multiplier": mult,
            "effect": effect,
        })

    critical_count = sum(1 for s in individual_scores if s >= 80)
    risk = get_risk_level(compound_score)

    return {
        "compound_score": round(compound_score, 1),
        "base_score": round(base_score, 1),
        "total_multiplier": round(total_multiplier, 2),
        "risk_level": risk,
        "interactions": interactions,
        "individual_scores": [round(s, 1) for s in individual_scores],
        "critical_event_count": critical_count,
        "event_count": len(events),
        "scenario_description": _describe_scenario(categories, risk),
    }


def _describe_scenario(categories: list, risk: dict) -> str:
    """Generate a human-readable scenario description."""
    unique_cats = list(set(categories))
    cat_str = " + ".join(unique_cats)
    level = risk.get("level", "MEDIUM")

    descriptions = {
        "LOW": f"The combination of {cat_str} events presents manageable risk. Standard monitoring protocols are sufficient.",
        "MEDIUM": f"The {cat_str} scenario requires elevated attention. Recommend activating secondary response teams.",
        "HIGH": f"The convergence of {cat_str} events creates significant compound risk. Immediate strategic review recommended.",
        "CRITICAL": f"ALERT: {cat_str} crisis convergence detected. Compound effects create severe cascading risk. Immediate action required.",
        "EXTREME": f"MAXIMUM ALERT: {cat_str} events are producing catastrophic compound effects. Full emergency response activation required.",
    }
    return descriptions.get(level, f"Scenario involving {cat_str} at {level} risk level.")


# ══════════════════════════════════════════════
# Quick Test
# ══════════════════════════════════════════════
if __name__ == "__main__":
    test_events = [
        {"headline": "Military offensive launched in border region", "category": "War", "impact_score": 82},
        {"headline": "Global oil prices surge amid supply disruption", "category": "Economy", "impact_score": 71},
        {"headline": "Major earthquake hits coastal city", "category": "Disaster", "impact_score": 88},
        {"headline": "Ransomware attack on banking systems", "category": "Tech", "impact_score": 65},
    ]

    print("=== GREEDY RANKING ===")
    ranked = greedy_rank_events(test_events)
    for e in ranked["ranked_events"]:
        print(f"  #{e['rank']}: [{e['category']}] {e['headline']} -> Impact: {e['impact_score']}")

    print("\n=== A* RISK PATH ===")
    path = astar_risk_path(test_events)
    print(f"  Algorithm: {path['algorithm']}")
    print(f"  Nodes explored: {path['nodes_explored']}")
    print(f"  Total compound risk: {path['total_compound_risk']}")
    for e in path["optimal_path"]:
        print(f"  Position {e['path_position']}: [{e['category']}] {e['headline']}")

    print("\n=== SCENARIO ANALYSIS ===")
    scenario = analyze_scenario(test_events)
    print(f"  Base score: {scenario['base_score']}")
    print(f"  Multiplier: {scenario['total_multiplier']}")
    print(f"  Compound score: {scenario['compound_score']}")
    print(f"  Risk level: {scenario['risk_level']['level']}")
    print(f"  Description: {scenario['scenario_description']}")
