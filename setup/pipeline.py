#!/usr/bin/env python3
"""
Master Pipeline Script - Run All Steps
Orchestrates the complete LLM leaderboard pipeline
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd: list, step_name: str):
    """Run a command and handle errors"""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        logger.info(f"✅ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run complete LLM leaderboard pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Steps 3-6 (assuming you have top15.jsonl from steps 1-2)
  python pipeline.py --input rerank_output/top15.jsonl

  # Run specific steps only
  python pipeline.py --input rerank_output/top15.jsonl --steps 3,4,5,6

  # Use different judge model
  python pipeline.py --input rerank_output/top15.jsonl --judge-model anthropic/claude-3.5-sonnet

  # Add new model to existing leaderboard
  python pipeline.py --input rerank_output/top15.jsonl --models deepseek/deepseek-chat --existing-judgments judge_output/judgments.jsonl
        """
    )

    parser.add_argument('--input', required=True, help='Input file with top-15 docs (from step 2)')
    parser.add_argument('--models', required=True, help='Comma-separated OpenRouter model IDs')
    parser.add_argument('--num-queries', type=int, help='Limit to first N queries (optional, for testing)')
    parser.add_argument('--steps', default='3,4,5,6', help='Steps to run (default: 3,4,5,6)')
    parser.add_argument('--existing-judgments', help='Path to existing judgments (for adding new models)')
    parser.add_argument('--max-workers', type=int, default=10, help='Parallel workers for judging')
    parser.add_argument('--elo-initial', type=int, default=1500, help='Initial ELO rating')
    parser.add_argument('--elo-k-factor', type=int, default=32, help='ELO K-factor')

    args = parser.parse_args()

    # If num-queries is specified, create a temporary filtered input file
    if args.num_queries:
        import json
        from pathlib import Path

        logger.info(f"Limiting to first {args.num_queries} queries")
        temp_input = Path("temp_limited_queries.jsonl")

        with open(args.input, 'r') as f_in, open(temp_input, 'w') as f_out:
            for i, line in enumerate(f_in):
                if i >= args.num_queries:
                    break
                f_out.write(line)

        # Use the temp file as input
        original_input = args.input
        args.input = str(temp_input)
        logger.info(f"Created temporary input file: {temp_input}")

    # Parse steps to run
    steps_to_run = [int(s.strip()) for s in args.steps.split(',')]

    logger.info(f"\n{'='*60}")
    logger.info(f"LLM LEADERBOARD PIPELINE")
    logger.info(f"{'='*60}")
    logger.info(f"Steps to run: {', '.join([f'Step {s}' for s in steps_to_run])}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Judge: Azure GPT-5 (from .env)")

    # Define output directories
    answers_dir = "rag_output/answers"
    judge_output = "judge_output"
    elo_output = "elo_output"
    results_output = "results"

    success = True

    # Step 3: Generate LLM answers
    if 3 in steps_to_run:
        cmd = [
            sys.executable, "llm_rag.py",
            "--input", args.input,
            "--output-dir", answers_dir,
            "--models", args.models
        ]
        if not run_command(cmd, "Step 3: Generate LLM Answers"):
            success = False

    # Step 4: Judge answers
    if 4 in steps_to_run and success:
        cmd = [
            sys.executable, "judge.py",
            "--answers-dir", answers_dir,
            "--output-dir", judge_output,
            "--max-workers", str(args.max_workers)
        ]
        if args.existing_judgments:
            cmd.extend(["--existing-judgments", args.existing_judgments])
        if not run_command(cmd, "Step 4: Judge Answers"):
            success = False

    # Step 5: Calculate ELO
    if 5 in steps_to_run and success:
        cmd = [
            sys.executable, "elo.py",
            "--judgments", f"{judge_output}/judgments.jsonl",
            "--output", f"{elo_output}/elo_ratings.json",
            "--initial-rating", str(args.elo_initial),
            "--k-factor", str(args.elo_k_factor)
        ]
        if not run_command(cmd, "Step 5: Calculate ELO"):
            success = False

    # Step 6: Generate results
    if 6 in steps_to_run and success:
        cmd = [
            sys.executable, "results.py",
            "--answers-dir", answers_dir,
            "--metrics", f"{judge_output}/metrics.jsonl",
            "--elo", f"{elo_output}/elo_ratings.json",
            "--judgments", f"{judge_output}/judgments.jsonl",
            "--output-dir", results_output
        ]
        if not run_command(cmd, "Step 6: Generate Results"):
            success = False

    # Final summary
    logger.info(f"\n{'='*60}")
    if success:
        logger.info(f"✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"{'='*60}")
        logger.info(f"\nOutput locations:")
        logger.info(f"  - Answers: {answers_dir}/")
        logger.info(f"  - Judgments: {judge_output}/judgments.jsonl")
        logger.info(f"  - Metrics: {judge_output}/metrics.jsonl")
        logger.info(f"  - ELO: {elo_output}/elo_ratings.json")
        logger.info(f"  - Leaderboard: {results_output}/leaderboard.json")
        logger.info(f"  - Visualization: {results_output}/leaderboard.png")
        logger.info(f"  - CSV: {results_output}/leaderboard.csv")
        return 0
    else:
        logger.error(f"❌ PIPELINE FAILED")
        logger.error(f"{'='*60}")
        return 1


if __name__ == "__main__":
    exit(main())
