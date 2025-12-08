#!/usr/bin/env python3
"""
Aggregate results from MS MARCO, Paul Graham, and SciFact datasets
Output: Single JSON with overall and per-dataset stats + visualization
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATASETS = {
    "msmarco": {
        "name": "MSMARCO",
        "path": Path("../msmarco-outputs/llm-elo")
    },
    "pg": {
        "name": "PG",
        "path": Path("../pg-outputs/llm-elo")
    },
    "scifact": {
        "name": "SciFact",
        "path": Path("../scifact-outputs/llm-elo")
    }
}

def load_dataset_results(dataset_path: Path):
    """Load leaderboard.json for a dataset"""
    leaderboard_file = dataset_path / "leaderboard.json"

    if not leaderboard_file.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_file}")

    with open(leaderboard_file) as f:
        data = json.load(f)

    return data['models']

def aggregate_results():
    """Aggregate results from all datasets"""

    print("=" * 70)
    print("AGGREGATING RESULTS FROM 3 DATASETS")
    print("=" * 70)

    # Load all dataset results
    dataset_results = {}
    for key, info in DATASETS.items():
        print(f"\nLoading {info['name']}...")
        models = load_dataset_results(info['path'])
        dataset_results[key] = {model['name']: model for model in models}
        print(f"  ✅ Loaded {len(models)} models")

    # Get all model names (should be same 9 across all datasets)
    all_models = set()
    for models_dict in dataset_results.values():
        all_models.update(models_dict.keys())

    print(f"\n📊 Total models: {len(all_models)}")
    print(f"   Models: {', '.join(sorted(all_models))}")

    # Aggregate for each model
    aggregated = []

    for model_name in sorted(all_models):
        print(f"\n🔄 Processing: {model_name}")

        model_data = {
            "name": model_name,
            "overall": {},
            "by_dataset": {},
            "comparisons": {}
        }

        # Collect stats across datasets
        all_elos = []
        all_wins = []
        all_losses = []
        all_ties = []
        all_latencies = []
        all_metrics = []
        all_comparisons = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})

        for dataset_key, dataset_info in DATASETS.items():
            dataset_name = dataset_info['name']

            if model_name not in dataset_results[dataset_key]:
                print(f"  ⚠️  Missing from {dataset_name}")
                continue

            data = dataset_results[dataset_key][model_name]

            # Collect for overall
            all_elos.append(data['elo'])
            all_wins.append(data['win_loss_tie']['wins'])
            all_losses.append(data['win_loss_tie']['losses'])
            all_ties.append(data['win_loss_tie']['ties'])
            all_latencies.append(data['performance']['avg_latency_ms'])
            all_metrics.append(data['metrics']['overall'])

            # Per-dataset stats
            model_data['by_dataset'][dataset_name] = {
                "elo": round(data['elo'], 2),
                "wins": data['win_loss_tie']['wins'],
                "losses": data['win_loss_tie']['losses'],
                "ties": data['win_loss_tie']['ties'],
                "win_rate": round(data['win_loss_tie']['wins'] /
                                (data['win_loss_tie']['wins'] + data['win_loss_tie']['losses'] + data['win_loss_tie']['ties']), 4),
                "metrics": {
                    "correctness": round(data['metrics']['correctness'], 2),
                    "faithfulness": round(data['metrics']['faithfulness'], 2),
                    "grounding": round(data['metrics']['grounding'], 2),
                    "relevance": round(data['metrics']['relevance'], 2),
                    "completeness": round(data['metrics']['completeness'], 2),
                    "overall": round(data['metrics']['overall'], 2)
                },
                "latency": {
                    "mean_ms": int(data['performance']['avg_latency_ms']),
                    "min_ms": int(data['performance']['min_latency_ms']),
                    "max_ms": int(data['performance']['max_latency_ms'])
                }
            }

            # Aggregate comparisons
            for opponent, comp_data in data['comparisons'].items():
                all_comparisons[opponent]['wins'] += comp_data['wins']
                all_comparisons[opponent]['losses'] += comp_data['losses']
                all_comparisons[opponent]['ties'] += comp_data['ties']

        # Calculate overall stats
        if all_elos:
            model_data['overall'] = {
                "elo": round(np.mean(all_elos), 2),
                "elo_std": round(np.std(all_elos), 2),
                "wins": sum(all_wins),
                "losses": sum(all_losses),
                "ties": sum(all_ties),
                "win_rate": round(sum(all_wins) / (sum(all_wins) + sum(all_losses) + sum(all_ties)), 4),
                "total_judgments": sum(all_wins) + sum(all_losses) + sum(all_ties),
                "avg_latency_ms": round(np.mean(all_latencies), 2),
                "avg_metric_overall": round(np.mean(all_metrics), 2)
            }

            # Add comparisons
            for opponent, comp_stats in all_comparisons.items():
                total = comp_stats['wins'] + comp_stats['losses'] + comp_stats['ties']
                model_data['comparisons'][opponent] = {
                    "wins": comp_stats['wins'],
                    "losses": comp_stats['losses'],
                    "ties": comp_stats['ties'],
                    "total": total,
                    "win_rate": round(comp_stats['wins'] / total, 4) if total > 0 else 0
                }

            aggregated.append(model_data)
            print(f"  ✅ Overall ELO: {model_data['overall']['elo']:.2f} (±{model_data['overall']['elo_std']:.2f})")

    # Sort by overall ELO
    aggregated.sort(key=lambda x: x['overall']['elo'], reverse=True)

    return aggregated

def save_results(aggregated_data, output_file: Path):
    """Save aggregated results to JSON"""
    with open(output_file, 'w') as f:
        json.dump(aggregated_data, f, indent=2)
    print(f"\n✅ Saved aggregated results: {output_file}")

def print_summary(aggregated_data):
    """Print summary of aggregated results"""
    print("\n" + "=" * 70)
    print("AGGREGATED LEADERBOARD (Overall ELO)")
    print("=" * 70)

    for i, model in enumerate(aggregated_data, 1):
        elo = model['overall']['elo']
        elo_std = model['overall']['elo_std']
        win_rate = model['overall']['win_rate']
        metric = model['overall']['avg_metric_overall']

        print(f"{i:2}. {model['name']:<40} "
              f"ELO: {elo:7.2f} (±{elo_std:5.2f})  "
              f"Win Rate: {win_rate:.3f}  "
              f"Metric: {metric:.2f}")

def main():
    # Aggregate results
    aggregated_data = aggregate_results()

    # Print summary
    print_summary(aggregated_data)

    # Save to file
    output_file = Path("../aggregated_leaderboard.json")
    save_results(aggregated_data, output_file)

    print("\n" + "=" * 70)
    print("✅ AGGREGATION COMPLETE!")
    print("=" * 70)
    print(f"\nNext step: Create visualization")
    print(f"  python3 visualize_aggregated.py --input {output_file} --output ../aggregated_leaderboard.png")

if __name__ == "__main__":
    main()
