#!/usr/bin/env python3
"""
Add pairwise comparison data to aggregated_leaderboard.json
Reads judgments from all three datasets and aggregates them.

IMPORTANT: All model names use exact names from model-info2.json.
No provider prefixes (anthropic-, google-, openai-, etc.).
Comparison keys also use canonical names.
"""

import json
from pathlib import Path
from collections import defaultdict

from model_names import normalize_model_name

DATASETS = ["msmarco", "pg", "scifact"]


def load_all_judgments():
    """Load and aggregate judgments from all datasets.

    Returns comparisons dict with canonical model names as keys.
    """
    comparisons = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0}))

    for dataset in DATASETS:
        judgments_file = Path(f"../{dataset}-outputs/llm-judge/judgments.jsonl")
        if not judgments_file.exists():
            print(f"Warning: {judgments_file} not found")
            continue

        with open(judgments_file) as f:
            for line in f:
                data = json.loads(line)
                # Normalize model names to canonical format
                model_x = normalize_model_name(data['model_x'])
                model_y = normalize_model_name(data['model_y'])
                judgment = data['judgment']

                if judgment == 'A':
                    comparisons[model_x][model_y]["wins"] += 1
                    comparisons[model_y][model_x]["losses"] += 1
                elif judgment == 'B':
                    comparisons[model_x][model_y]["losses"] += 1
                    comparisons[model_y][model_x]["wins"] += 1
                else:  # TIE
                    comparisons[model_x][model_y]["ties"] += 1
                    comparisons[model_y][model_x]["ties"] += 1

    return comparisons


def add_comparisons_to_leaderboard():
    """Add comparisons to aggregated_leaderboard.json"""

    # Load current leaderboard
    leaderboard_file = Path("../aggregated_leaderboard.json")
    with open(leaderboard_file) as f:
        leaderboard = json.load(f)

    # Load all comparisons (already normalized)
    comparisons = load_all_judgments()

    # Add comparisons to each model
    for model_entry in leaderboard:
        model_name = model_entry["name"]
        model_comps = comparisons.get(model_name, {})

        # Build comparisons dict with win_rate
        # All keys are already canonical names
        model_entry["comparisons"] = {}
        for opponent, stats in model_comps.items():
            total = stats["wins"] + stats["losses"] + stats["ties"]
            model_entry["comparisons"][opponent] = {
                "wins": stats["wins"],
                "losses": stats["losses"],
                "ties": stats["ties"],
                "total": total,
                "win_rate": round(stats["wins"] / total, 4) if total > 0 else 0
            }

        print(f"{model_name}: {len(model_entry['comparisons'])} comparisons added")

    # Save updated leaderboard
    with open(leaderboard_file, 'w') as f:
        json.dump(leaderboard, f, indent=2)

    # Also save as benchmarks.json
    benchmarks_file = Path("../benchmarks.json")
    with open(benchmarks_file, 'w') as f:
        json.dump(leaderboard, f, indent=2)

    print(f"\nUpdated {leaderboard_file}")
    print(f"Updated {benchmarks_file}")
    return leaderboard


if __name__ == "__main__":
    add_comparisons_to_leaderboard()
