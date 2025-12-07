#!/usr/bin/env python3
"""
Step 6: Generate Final Leaderboard
Input: answers/, metrics.jsonl, elo_ratings.json, judgments.jsonl
Output: leaderboard.json, leaderboard.png, leaderboard.csv
"""

import argparse
import json
import csv
import logging
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_latency_data(answers_dir: str):
    """Load latency and token usage from answer files"""
    latency_data = {}
    answers_path = Path(answers_dir)

    for answer_file in answers_path.glob("*_answers.jsonl"):
        model_name = answer_file.stem.replace('_answers', '')
        latencies = []
        total_input_tokens = 0
        total_output_tokens = 0

        with open(answer_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                latencies.append(data.get('latency_ms', 0))
                usage = data.get('token_usage', {})
                total_input_tokens += usage.get('input_tokens', 0)
                total_output_tokens += usage.get('output_tokens', 0)

        latency_data[model_name] = {
            "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens
        }

    return latency_data


def load_metrics_data(metrics_file: str):
    """Load and aggregate metrics per model"""
    metrics_data = defaultdict(list)

    with open(metrics_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            model = data['model']
            metrics_data[model].append({
                "correctness": data.get('correctness', 0),
                "faithfulness": data.get('faithfulness', 0),
                "grounding": data.get('grounding', 0),
                "relevance": data.get('relevance', 0),
                "completeness": data.get('completeness', 0)
            })

    # Calculate averages
    avg_metrics = {}
    for model, metrics_list in metrics_data.items():
        if not metrics_list:
            continue

        avg = {
            "correctness": sum(m['correctness'] for m in metrics_list) / len(metrics_list),
            "faithfulness": sum(m['faithfulness'] for m in metrics_list) / len(metrics_list),
            "grounding": sum(m['grounding'] for m in metrics_list) / len(metrics_list),
            "relevance": sum(m['relevance'] for m in metrics_list) / len(metrics_list),
            "completeness": sum(m['completeness'] for m in metrics_list) / len(metrics_list)
        }
        avg["overall"] = sum(avg.values()) / len(avg)
        avg_metrics[model] = avg

    return avg_metrics


def load_elo_data(elo_file: str):
    """Load ELO ratings"""
    with open(elo_file, 'r') as f:
        return json.load(f)


def load_comparison_data(judgments_file: str):
    """Load pairwise comparison data"""
    comparisons = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0}))

    with open(judgments_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            model_x = data['model_x']
            model_y = data['model_y']
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


def generate_leaderboard(
    answers_dir: str,
    metrics_file: str,
    elo_file: str,
    judgments_file: str,
    output_dir: str
):
    """Generate final leaderboard with all data"""

    logger.info(f"\n{'='*60}")
    logger.info(f"Generating Final Leaderboard")
    logger.info(f"{'='*60}")

    # Load all data
    logger.info("Loading data...")
    latency_data = load_latency_data(answers_dir)
    metrics_data = load_metrics_data(metrics_file)
    elo_data = load_elo_data(elo_file)
    comparisons = load_comparison_data(judgments_file)

    elo_ratings = elo_data["ratings"]
    win_loss_tie = elo_data["win_loss_tie"]

    # Build leaderboard
    models = []
    for model_name in elo_ratings.keys():
        model_entry = {
            "name": model_name,
            "elo": elo_ratings[model_name],
            "metrics": {
                k: round(v, 2) for k, v in metrics_data.get(model_name, {}).items()
            },
            "performance": latency_data.get(model_name, {}),
            "win_loss_tie": win_loss_tie.get(model_name, {}),
            "comparisons": {
                opponent: {
                    "wins": comparisons[model_name][opponent]["wins"],
                    "losses": comparisons[model_name][opponent]["losses"],
                    "ties": comparisons[model_name][opponent]["ties"]
                }
                for opponent in comparisons[model_name]
            }
        }
        models.append(model_entry)

    # Sort by ELO
    models.sort(key=lambda x: x["elo"], reverse=True)

    # Add ranks
    for rank, model in enumerate(models, 1):
        model["rank"] = rank

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON
    leaderboard_json = output_path / "leaderboard.json"
    with open(leaderboard_json, 'w') as f:
        json.dump({"models": models}, f, indent=2)
    logger.info(f"✅ Saved: {leaderboard_json}")

    # Generate PNG
    logger.info("Generating visualization...")
    leaderboard_png = output_path / "leaderboard.png"

    model_names = [m["name"] for m in models]
    elos = [m["elo"] for m in models]

    plt.figure(figsize=(10, max(6, len(models) * 0.5)))
    bars = plt.barh(range(len(models)), elos, color='#4C78A8')
    plt.yticks(range(len(models)), model_names)
    plt.xlabel('ELO Rating', fontweight='bold')
    plt.title('LLM Leaderboard - ELO Rankings', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

    # Add value labels
    for i, (bar, elo) in enumerate(zip(bars, elos)):
        width = bar.get_width()
        plt.text(width + 10, bar.get_y() + bar.get_height()/2,
                f'{elo:.0f}', ha='left', va='center', fontweight='bold')

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(leaderboard_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {leaderboard_png}")

    # Generate CSV
    logger.info("Generating CSV...")
    leaderboard_csv = output_path / "leaderboard.csv"

    with open(leaderboard_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'model', 'elo', 'wins', 'losses', 'ties', 'win_rate',
            'correctness', 'faithfulness', 'grounding', 'relevance', 'completeness', 'overall',
            'avg_latency_ms', 'total_tokens'
        ])

        for model in models:
            wlt = model['win_loss_tie']
            total = wlt['wins'] + wlt['losses'] + wlt['ties']
            win_rate = wlt['wins'] / total if total > 0 else 0.0

            metrics = model['metrics']
            perf = model['performance']

            writer.writerow([
                model['rank'],
                model['name'],
                f"{model['elo']:.2f}",
                wlt['wins'],
                wlt['losses'],
                wlt['ties'],
                f"{win_rate:.4f}",
                metrics.get('correctness', 0),
                metrics.get('faithfulness', 0),
                metrics.get('grounding', 0),
                metrics.get('relevance', 0),
                metrics.get('completeness', 0),
                metrics.get('overall', 0),
                perf.get('avg_latency_ms', 0),
                perf.get('total_tokens', 0)
            ])

    logger.info(f"✅ Saved: {leaderboard_csv}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Final Rankings")
    logger.info(f"{'='*60}")

    for model in models:
        logger.info(f"{model['rank']}. {model['name']}: {model['elo']:.1f} ELO")

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Leaderboard Generated!")
    logger.info(f"{'='*60}")
    logger.info(f"Output directory: {output_dir}")

    return models


def main():
    parser = argparse.ArgumentParser(description='Generate final leaderboard')
    parser.add_argument('--answers-dir', required=True, help='Directory with model answers')
    parser.add_argument('--metrics', required=True, help='Path to metrics.jsonl')
    parser.add_argument('--elo', required=True, help='Path to elo_ratings.json')
    parser.add_argument('--judgments', required=True, help='Path to judgments.jsonl')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    args = parser.parse_args()

    generate_leaderboard(
        answers_dir=args.answers_dir,
        metrics_file=args.metrics,
        elo_file=args.elo,
        judgments_file=args.judgments,
        output_dir=args.output_dir
    )

    return 0


if __name__ == "__main__":
    exit(main())
