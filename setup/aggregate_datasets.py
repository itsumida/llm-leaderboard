#!/usr/bin/env python3
"""
Aggregate results from MS MARCO, Paul Graham, and SciFact datasets
Output: Single JSON with overall and per-dataset stats + visualization

IMPORTANT: All model names in output use exact names from model-info2.json.
No provider prefixes (anthropic-, google-, openai-, etc.).
Comparison keys also use canonical names.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from model_names import normalize_model_name, normalize_comparisons

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

def load_latencies(dataset_path: Path):
    """Load latency data from RAG answer files"""
    rag_dir = dataset_path.parent / "llm-rag"
    if not rag_dir.exists():
        return {}

    model_latencies = defaultdict(list)
    for answer_file in rag_dir.glob("*_answers.jsonl"):
        model_name = answer_file.stem.replace("_answers", "")
        with open(answer_file) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if 'latency_ms' in d and d['latency_ms']:
                        model_latencies[model_name].append(d['latency_ms'])
                except Exception:
                    continue
    return {
        model: {
            'avg': round(np.mean(vals), 2),
            'min': round(min(vals), 2),
            'max': round(max(vals), 2)
        }
        for model, vals in model_latencies.items() if vals
    }

def load_metrics(dataset_path: Path):
    """Load metrics.jsonl and aggregate per model"""
    metrics_file = dataset_path.parent / "llm-judge" / "metrics.jsonl"
    if not metrics_file.exists():
        return {}

    model_metrics = defaultdict(lambda: {'correctness': [], 'faithfulness': [], 'grounding': [], 'relevance': [], 'completeness': []})
    with open(metrics_file) as f:
        for line in f:
            m = json.loads(line)
            model = m['model']
            for key in ['correctness', 'faithfulness', 'grounding', 'relevance', 'completeness']:
                model_metrics[model][key].append(m[key])

    # Average the metrics
    result = {}
    for model, metrics in model_metrics.items():
        result[model] = {
            key: round(np.mean(vals), 2) if vals else 0
            for key, vals in metrics.items()
        }
        result[model]['overall'] = round(np.mean([result[model][k] for k in ['correctness', 'faithfulness', 'grounding', 'relevance', 'completeness']]), 2)
    return result


def load_dataset_results(dataset_path: Path):
    """Load leaderboard.json for a dataset"""
    leaderboard_file = dataset_path / "leaderboard.json"

    if not leaderboard_file.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_file}")

    with open(leaderboard_file) as f:
        data = json.load(f)

    # Load metrics and latencies if available
    metrics_data = load_metrics(dataset_path)
    latency_data = load_latencies(dataset_path)

    # Handle both old format (with 'models' key) and new format (with 'ratings' key)
    if 'models' in data:
        return data['models']

    # Convert new format to expected structure
    # Normalize all model names to canonical format from model-info2.json
    models = []
    for raw_model_name, elo in data['ratings'].items():
        # Normalize model name (removes provider prefixes)
        model_name = normalize_model_name(raw_model_name)

        wlt = data.get('win_loss_tie', {}).get(raw_model_name, {'wins': 0, 'losses': 0, 'ties': 0})
        model_metrics = metrics_data.get(raw_model_name, {
            'correctness': 0, 'faithfulness': 0, 'grounding': 0,
            'relevance': 0, 'completeness': 0, 'overall': 0
        })
        models.append({
            'name': model_name,  # Use canonical name
            'elo': elo,
            'win_loss_tie': wlt,
            'metrics': model_metrics,
            'performance': {
                'avg_latency_ms': latency_data.get(raw_model_name, {}).get('avg', 0),
                'min_latency_ms': latency_data.get(raw_model_name, {}).get('min', 0),
                'max_latency_ms': latency_data.get(raw_model_name, {}).get('max', 0)
            },
            'comparisons': {}
        })
    return models

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

    for raw_model_name in sorted(all_models):
        # Normalize model name to canonical format
        model_name = normalize_model_name(raw_model_name)
        print(f"\n🔄 Processing: {raw_model_name} -> {model_name}")

        model_data = {
            "name": model_name,  # Use canonical name
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

            if raw_model_name not in dataset_results[dataset_key]:
                print(f"  ⚠️  Missing from {dataset_name}")
                continue

            data = dataset_results[dataset_key][raw_model_name]

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

            # Aggregate comparisons (normalize opponent names)
            for raw_opponent, comp_data in data['comparisons'].items():
                opponent = normalize_model_name(raw_opponent)
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
