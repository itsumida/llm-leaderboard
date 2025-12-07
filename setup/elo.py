#!/usr/bin/env python3
"""
Step 5: Calculate ELO Ratings from Judgments
Input: judgments.jsonl
Output: elo_ratings.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def update_elo(elo_a: float, elo_b: float, score_a: float, k_factor: int = 32) -> Tuple[float, float]:
    """
    Update ELO ratings after a match

    Args:
        elo_a: Current ELO of player A
        elo_b: Current ELO of player B
        score_a: Actual score of A (1.0 for win, 0.5 for tie, 0.0 for loss)
        k_factor: K-factor (higher = more volatile ratings)

    Returns:
        Tuple of (new_elo_a, new_elo_b)
    """
    # Expected scores
    expected_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    expected_b = 1.0 - expected_a

    # Update ELO
    new_elo_a = elo_a + k_factor * (score_a - expected_a)
    new_elo_b = elo_b + k_factor * ((1.0 - score_a) - expected_b)

    return new_elo_a, new_elo_b


def calculate_elo_ratings(
    judgments_file: str,
    output_file: str,
    initial_rating: int = 1500,
    k_factor: int = 32
):
    """
    Calculate ELO ratings from judgments

    Args:
        judgments_file: Path to judgments JSONL
        output_file: Path to save ELO ratings JSON
        initial_rating: Starting ELO for all models
        k_factor: ELO K-factor
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Calculating ELO Ratings")
    logger.info(f"{'='*60}")
    logger.info(f"Initial rating: {initial_rating}")
    logger.info(f"K-factor: {k_factor}")

    # Load judgments
    judgments = []
    with open(judgments_file, 'r') as f:
        for line in f:
            judgments.append(json.loads(line))

    logger.info(f"Loaded {len(judgments)} judgments")

    # Initialize ELO ratings
    models = set()
    for j in judgments:
        models.add(j['model_x'])
        models.add(j['model_y'])

    elo_ratings = {model: float(initial_rating) for model in models}
    win_loss_tie = {model: {'wins': 0, 'losses': 0, 'ties': 0} for model in models}

    logger.info(f"Models: {', '.join(sorted(models))}")

    # Process judgments in order (ELO is order-dependent)
    for idx, judgment in enumerate(judgments, 1):
        model_x = judgment['model_x']
        model_y = judgment['model_y']
        result = judgment['judgment']

        # Determine score for model_x (A in the judgment)
        if result == 'A':
            score_x = 1.0
            win_loss_tie[model_x]['wins'] += 1
            win_loss_tie[model_y]['losses'] += 1
        elif result == 'B':
            score_x = 0.0
            win_loss_tie[model_x]['losses'] += 1
            win_loss_tie[model_y]['wins'] += 1
        else:  # TIE
            score_x = 0.5
            win_loss_tie[model_x]['ties'] += 1
            win_loss_tie[model_y]['ties'] += 1

        # Update ELO
        new_elo_x, new_elo_y = update_elo(
            elo_ratings[model_x],
            elo_ratings[model_y],
            score_x,
            k_factor
        )

        elo_ratings[model_x] = new_elo_x
        elo_ratings[model_y] = new_elo_y

        if idx % 100 == 0:
            logger.info(f"  Processed {idx}/{len(judgments)} judgments")

    # Sort by ELO
    sorted_models = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Final ELO Rankings")
    logger.info(f"{'='*60}")

    for rank, (model, elo) in enumerate(sorted_models, 1):
        wlt = win_loss_tie[model]
        total = wlt['wins'] + wlt['losses'] + wlt['ties']
        win_rate = wlt['wins'] / total if total > 0 else 0.0
        logger.info(f"{rank}. {model}: {elo:.1f} ELO (W:{wlt['wins']} L:{wlt['losses']} T:{wlt['ties']}, WR:{win_rate:.1%})")

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "ratings": {model: round(elo, 2) for model, elo in elo_ratings.items()},
        "win_loss_tie": win_loss_tie,
        "config": {
            "initial_rating": initial_rating,
            "k_factor": k_factor,
            "num_judgments": len(judgments)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"\n✅ Saved: {output_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Calculate ELO ratings from judgments')
    parser.add_argument('--judgments', required=True, help='Path to judgments.jsonl')
    parser.add_argument('--output', default='elo_output/elo_ratings.json', help='Output file')
    parser.add_argument('--initial-rating', type=int, default=1500, help='Initial ELO rating')
    parser.add_argument('--k-factor', type=int, default=32, help='ELO K-factor')
    args = parser.parse_args()

    calculate_elo_ratings(
        judgments_file=args.judgments,
        output_file=args.output,
        initial_rating=args.initial_rating,
        k_factor=args.k_factor
    )

    return 0


if __name__ == "__main__":
    exit(main())
