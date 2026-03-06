#!/usr/bin/env python3
"""
Aggregate results from all datasets into final benchmarks.json

Combines:
- ELO ratings and win/loss/tie stats from each dataset
- Pairwise comparisons from judgments
- Model name normalization to match model-info.json

Output: ../benchmarks.json (ready to use, no additional processing needed)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# =============================================================================
# MODEL NAME NORMALIZATION
# =============================================================================

CANONICAL_NAMES = [
    "claude-opus-4-5-20251101", "deepseek-r1", "gemini-2.5-pro",
    "gemini-3-flash-preview", "gemini-3-pro-preview", "gpt-5.1", "gpt-5.2",
    "gpt-oss-120b", "qwen3-30b-a3b-thinking-2507", "grok-4-fast", "glm-4.6",
    "claude-opus-4-6", "claude-sonnet-4.6", "claude-sonnet-4.5",
    "gpt-5.4", "gpt-5.4-pro",
]

NAME_MAPPING = {
    "anthropic-claude-opus-4-6": "claude-opus-4-6",
    "anthropic-claude-opus-4.5": "claude-opus-4-5-20251101",
    "anthropic-claude-sonnet-4.6": "claude-sonnet-4.6",
    "anthropic-claude-sonnet-4.5": "claude-sonnet-4.5",
    "google-gemini-2.5-pro": "gemini-2.5-pro",
    "google-gemini-3-flash-preview": "gemini-3-flash-preview",
    "google-gemini-3-pro-preview": "gemini-3-pro-preview",
    "openai-gpt-5.1": "gpt-5.1",
    "openai-gpt-5.2": "gpt-5.2",
    "openai-gpt-oss-120b": "gpt-oss-120b",
    "x-ai-grok-4-fast": "grok-4-fast",
    "z-ai-glm-4.6": "glm-4.6",
    "deepseek-deepseek-r1": "deepseek-r1",
    "qwen-qwen3-30b-a3b-thinking-2507": "qwen3-30b-a3b-thinking-2507",
}


def normalize_name(name: str) -> str:
    """Convert any model name to canonical format from model-info.json"""
    if name in NAME_MAPPING:
        return NAME_MAPPING[name]
    if name in CANONICAL_NAMES:
        return name
    # Try removing common prefixes
    for prefix in ["anthropic-", "google-", "openai-", "x-ai-", "z-ai-", "deepseek-", "qwen-"]:
        if name.startswith(prefix):
            stripped = name[len(prefix):]
            if stripped in CANONICAL_NAMES:
                return stripped
    print(f"  WARNING: Unknown model '{name}'")
    return name


# =============================================================================
# DATA LOADING
# =============================================================================

DATASETS = {
    "msmarco": {"name": "MSMARCO", "path": Path("../msmarco-outputs")},
    "pg": {"name": "PG", "path": Path("../pg-outputs")},
    "scifact": {"name": "SciFact", "path": Path("../scifact-outputs")},
}


def load_elo_ratings(dataset_path: Path) -> dict:
    """Load ELO ratings and win/loss/tie from leaderboard.json"""
    leaderboard_file = dataset_path / "llm-elo" / "leaderboard.json"
    with open(leaderboard_file) as f:
        data = json.load(f)
    return data.get("ratings", {}), data.get("win_loss_tie", {})


def load_metrics(dataset_path: Path) -> dict:
    """Load and average metrics per model"""
    metrics_file = dataset_path / "llm-judge" / "metrics.jsonl"
    if not metrics_file.exists():
        return {}

    model_metrics = defaultdict(lambda: {k: [] for k in
        ["correctness", "faithfulness", "grounding", "relevance", "completeness"]})

    with open(metrics_file) as f:
        for line in f:
            m = json.loads(line)
            for key in model_metrics[m["model"]]:
                model_metrics[m["model"]][key].append(m[key])

    result = {}
    for model, metrics in model_metrics.items():
        result[model] = {k: round(np.mean(v), 2) for k, v in metrics.items() if v}
        result[model]["overall"] = round(np.mean(list(result[model].values())), 2)
    return result


def load_latencies(dataset_path: Path) -> dict:
    """Load latency stats from answer files"""
    rag_dir = dataset_path / "llm-rag"
    if not rag_dir.exists():
        return {}

    latencies = {}
    for f in rag_dir.glob("*_answers.jsonl"):
        model = f.stem.replace("_answers", "")
        vals = []
        with open(f) as file:
            for line in file:
                d = json.loads(line)
                if d.get("latency_ms"):
                    vals.append(d["latency_ms"])
        if vals:
            latencies[model] = {"mean": int(np.mean(vals)), "min": min(vals), "max": max(vals)}
    return latencies


def load_comparisons(dataset_path: Path) -> dict:
    """Load pairwise comparisons from judgments.jsonl"""
    judgments_file = dataset_path / "llm-judge" / "judgments.jsonl"
    if not judgments_file.exists():
        return {}

    comps = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0}))

    with open(judgments_file) as f:
        for line in f:
            j = json.loads(line)
            a, b = normalize_name(j["model_x"]), normalize_name(j["model_y"])
            verdict = j["judgment"]

            if verdict == "A":
                comps[a][b]["wins"] += 1
                comps[b][a]["losses"] += 1
            elif verdict == "B":
                comps[a][b]["losses"] += 1
                comps[b][a]["wins"] += 1
            else:
                comps[a][b]["ties"] += 1
                comps[b][a]["ties"] += 1

    return comps


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate():
    """Aggregate all datasets into final benchmarks.json"""
    print("=" * 60)
    print("GENERATING BENCHMARKS")
    print("=" * 60)

    # Collect data from all datasets
    all_data = defaultdict(lambda: {
        "elos": [], "wins": [], "losses": [], "ties": [],
        "latencies": [], "metrics": [], "by_dataset": {},
        "comparisons": defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
    })

    for key, info in DATASETS.items():
        print(f"\nLoading {info['name']}...")
        path = info["path"]

        ratings, wlt = load_elo_ratings(path)
        metrics = load_metrics(path)
        latencies = load_latencies(path)
        comparisons = load_comparisons(path)

        for raw_name, elo in ratings.items():
            name = normalize_name(raw_name)
            model_wlt = wlt.get(raw_name, {"wins": 0, "losses": 0, "ties": 0})
            model_metrics = metrics.get(raw_name, {})
            model_latency = latencies.get(raw_name, {})

            # Aggregate
            all_data[name]["elos"].append(elo)
            all_data[name]["wins"].append(model_wlt["wins"])
            all_data[name]["losses"].append(model_wlt["losses"])
            all_data[name]["ties"].append(model_wlt["ties"])
            if model_latency.get("mean"):
                all_data[name]["latencies"].append(model_latency["mean"])
            if model_metrics.get("overall"):
                all_data[name]["metrics"].append(model_metrics["overall"])

            # Per-dataset
            total = model_wlt["wins"] + model_wlt["losses"] + model_wlt["ties"]
            all_data[name]["by_dataset"][info["name"]] = {
                "elo": round(elo, 2),
                "wins": model_wlt["wins"],
                "losses": model_wlt["losses"],
                "ties": model_wlt["ties"],
                "win_rate": round(model_wlt["wins"] / total, 4) if total else 0,
                "metrics": model_metrics,
                "latency": {"mean_ms": model_latency.get("mean", 0),
                           "min_ms": model_latency.get("min", 0),
                           "max_ms": model_latency.get("max", 0)}
            }

            # Comparisons
            for opponent, stats in comparisons.get(name, {}).items():
                all_data[name]["comparisons"][opponent]["wins"] += stats["wins"]
                all_data[name]["comparisons"][opponent]["losses"] += stats["losses"]
                all_data[name]["comparisons"][opponent]["ties"] += stats["ties"]

        print(f"  Loaded {len(ratings)} models")

    # Build final output
    benchmarks = []
    for name, data in all_data.items():
        total = sum(data["wins"]) + sum(data["losses"]) + sum(data["ties"])

        # Build comparisons with win_rate
        comparisons = {}
        for opp, stats in data["comparisons"].items():
            t = stats["wins"] + stats["losses"] + stats["ties"]
            comparisons[opp] = {
                "wins": stats["wins"],
                "losses": stats["losses"],
                "ties": stats["ties"],
                "total": t,
                "win_rate": round(stats["wins"] / t, 4) if t else 0
            }

        benchmarks.append({
            "name": name,
            "overall": {
                "elo": round(np.mean(data["elos"]), 2),
                "elo_std": round(np.std(data["elos"]), 2),
                "wins": sum(data["wins"]),
                "losses": sum(data["losses"]),
                "ties": sum(data["ties"]),
                "win_rate": round(sum(data["wins"]) / total, 4) if total else 0,
                "total_judgments": total,
                "avg_latency_ms": round(np.mean(data["latencies"]), 2) if data["latencies"] else 0,
                "avg_metric_overall": round(np.mean(data["metrics"]), 2) if data["metrics"] else 0,
            },
            "by_dataset": data["by_dataset"],
            "comparisons": comparisons,
        })

    # Sort by ELO
    benchmarks.sort(key=lambda x: x["overall"]["elo"], reverse=True)

    # Save
    output = Path("../benchmarks.json")
    with open(output, "w") as f:
        json.dump(benchmarks, f, indent=2)

    # Also save as aggregated_leaderboard.json for compatibility
    with open(Path("../aggregated_leaderboard.json"), "w") as f:
        json.dump(benchmarks, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("LEADERBOARD")
    print("=" * 60)
    for i, m in enumerate(benchmarks, 1):
        print(f"{i:2}. {m['name']:35} ELO: {m['overall']['elo']:7.2f}  WR: {m['overall']['win_rate']:.1%}")

    print(f"\n Saved: {output}")
    print(f" Saved: ../aggregated_leaderboard.json")


if __name__ == "__main__":
    aggregate()
