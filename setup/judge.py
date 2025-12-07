#!/usr/bin/env python3
"""
Step 4: Judge LLM Answers via OpenRouter
Input: answers/ directory with model answers
Output: judgments.jsonl and metrics.jsonl
"""

import argparse
import json
import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_answers(answers_dir: str):
    """Load all model answers from directory"""
    answers_data = {}
    model_names = []

    answers_path = Path(answers_dir)
    for answer_file in answers_path.glob("*_answers.jsonl"):
        # Extract model name from filename
        model_name = answer_file.stem.replace('_answers', '')
        model_names.append(model_name)
        answers_data[model_name] = {}

        with open(answer_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                query_id = data['query_id']
                answers_data[model_name][query_id] = data

        logger.info(f"Loaded {len(answers_data[model_name])} answers for {model_name}")

    return answers_data, model_names


def load_existing_judgments(judgments_file: str):
    """Load existing judgments to reuse"""
    existing = {}
    judgments_path = Path(judgments_file)

    if not judgments_path.exists():
        logger.info("No existing judgments file found")
        return existing

    with open(judgments_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Create order-independent key
            key = (data['query_id'], frozenset([data['model_x'], data['model_y']]))
            existing[key] = data

    logger.info(f"Loaded {len(existing)} existing judgments")
    return existing


def score_answer_azure(answer: str, query: str, context_docs: str, client: AzureOpenAI, deployment_id: str):
    """Score answer using Azure GPT-5"""
    system_content = f"""You are a STRICT and PRECISE evaluator. You must use the full 1–5 scale correctly, not generously.

IMPORTANT:
• A score of 5 is allowed, but ONLY when the answer is truly exceptional.
• You must justify every high score internally: if any flaw exists, the score must be lower.
• Do not inflate scores. Start from 1 and move upward only if the evidence clearly supports it.

Context Documents:
{context_docs}

SCORING RULES:

Correctness:
1 = mostly incorrect
2 = several factual issues
3 = mostly correct with some minor issues
4 = correct with only very small imperfections
5 = completely correct with no identifiable flaws

Faithfulness (supported by context):
1 = many unsupported claims or hallucinations
2 = several unsupported claims
3 = mostly supported but with gaps
4 = well supported with only minor exceptions
5 = every claim fully supported, no exceptions

Grounding (using the provided documents appropriately):
1 = grounding absent or incorrect
2 = weak grounding
3 = adequate grounding
4 = strong, consistent grounding
5 = impeccable grounding with no weaknesses

Relevance:
1 = mostly irrelevant
2 = partially relevant
3 = adequately relevant
4 = strongly relevant
5 = fully and precisely relevant to all parts of the query

Completeness:
1 = very incomplete
2 = major gaps
3 = adequately complete
4 = minor gaps
5 = fully complete with no missing elements

GUIDANCE:
• If you can identify ANY flaw in any category, reduce the score accordingly.
• Do not default to 4 or 5 unless the answer truly earns it.
• Use the full scale naturally: weak answers get 1–2, decent ones get 3, great ones get 4, rare ones get 5.

Return ONLY valid JSON:
{{"correctness": X, "faithfulness": X, "grounding": X, "relevance": X, "completeness": X}}"""

    prompt = f"""Query: {query}

Answer: {answer}

Rate 1-5 on each dimension:"""

    try:
        response = client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100,
            response_format={"type": "json_object"}
        )

        scores = json.loads(response.choices[0].message.content)

        # Validate scores are 1-5
        for key in ["correctness", "faithfulness", "grounding", "relevance", "completeness"]:
            if key not in scores:
                scores[key] = 3
            scores[key] = max(1, min(5, int(scores[key])))

        return scores

    except Exception as e:
        logger.warning(f"Failed to parse scores: {e}")
        return {"correctness": 3, "faithfulness": 3, "grounding": 3, "relevance": 3, "completeness": 3}


def judge_pairwise_azure(answer_a: str, answer_b: str, query: str, context_docs: str, model_a_name: str, model_b_name: str, client: AzureOpenAI, deployment_id: str):
    """Judge pairwise using Azure GPT-5"""
    system_content = f"""You are evaluating answers to queries based on provided context documents.

Context Documents:
{context_docs}

Evaluation Criteria - Select the answer that is:
1. More factually correct
2. Better supported by the provided context
3. Less speculative or fabricated (lower hallucination risk)
4. More comprehensive and thorough
5. More directly responsive to the specific question

Disregard style, formatting, and verbosity.

Reply strictly with one of: A, B, or TIE."""

    prompt = f"""Query: {query}

Answer A ({model_a_name}):
{answer_a}

Answer B ({model_b_name}):
{answer_b}

Reply: A, B, or TIE"""

    try:
        response = client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=3
        )

        judgment = response.choices[0].message.content.strip().upper()
        if judgment in ["A", "B", "TIE"]:
            return judgment
        return "TIE"

    except Exception as e:
        logger.error(f"Error in judgment: {e}")
        return "TIE"


def judge_single_comparison(args):
    """Judge a single comparison with score caching"""
    query_id, model_x, model_y, query_text, answers_data, azure_client, azure_deployment, score_cache = args

    # Get context docs (use from model_x, should be same for all)
    docs_used = answers_data[model_x][query_id]["documents_used"]
    context_docs = "\n\n".join([
        f"[{doc['rank']}] {doc['doc_id']}: {doc.get('text', '')}"
        for doc in docs_used
    ])

    answer_x = answers_data[model_x][query_id]['answer']
    answer_y = answers_data[model_y][query_id]['answer']

    # Score each answer (with caching)
    cache_key_x = (model_x, query_id)
    cache_key_y = (model_y, query_id)

    if cache_key_x not in score_cache:
        score_cache[cache_key_x] = score_answer_azure(
            answer=answer_x,
            query=query_text,
            context_docs=context_docs,
            client=azure_client,
            deployment_id=azure_deployment
        )

    if cache_key_y not in score_cache:
        score_cache[cache_key_y] = score_answer_azure(
            answer=answer_y,
            query=query_text,
            context_docs=context_docs,
            client=azure_client,
            deployment_id=azure_deployment
        )

    scores_x = score_cache[cache_key_x]
    scores_y = score_cache[cache_key_y]

    # Pairwise judgment (model_x is always A, model_y is always B)
    judgment = judge_pairwise_azure(
        answer_a=answer_x,
        answer_b=answer_y,
        query=query_text,
        context_docs=context_docs,
        model_a_name=model_x,
        model_b_name=model_y,
        client=azure_client,
        deployment_id=azure_deployment
    )

    return {
        'query_id': query_id,
        'model_x': model_x,
        'model_y': model_y,
        'judgment': judgment,
        'scores_x': scores_x,
        'scores_y': scores_y
    }


def main():
    parser = argparse.ArgumentParser(description='Judge LLM answers via Azure GPT-5')
    parser.add_argument('--answers-dir', required=True, help='Directory with model answers')
    parser.add_argument('--output-dir', default='judge_output', help='Output directory')
    parser.add_argument('--max-workers', type=int, default=10, help='Parallel workers')
    parser.add_argument('--existing-judgments', help='Path to existing judgments to reuse')
    args = parser.parse_args()

    # Check Azure API key
    azure_api_key = os.getenv('AZURE_API_KEY')
    azure_resource_name = os.getenv('AZURE_RESOURCE_NAME')
    azure_deployment = os.getenv('AZURE_DEPLOYMENT_ID', 'gpt-5')

    if not azure_api_key or not azure_resource_name:
        print("❌ Error: AZURE_API_KEY and AZURE_RESOURCE_NAME must be set in .env!")
        return 1

    # Initialize Azure OpenAI client
    azure_client = AzureOpenAI(
        api_key=azure_api_key,
        api_version="2024-02-15-preview",
        azure_endpoint=f"https://{azure_resource_name}.openai.azure.com"
    )

    logger.info(f"Using Azure GPT-5: {azure_deployment}")

    # Load answers
    answers_data, model_names = load_answers(args.answers_dir)

    if len(model_names) < 2:
        print("❌ Error: Need at least 2 models to compare")
        return 1

    # Load existing judgments if provided
    existing_judgments = {}
    if args.existing_judgments:
        existing_judgments = load_existing_judgments(args.existing_judgments)

    # Find common query IDs
    common_query_ids = set(answers_data[model_names[0]].keys())
    for model_name in model_names[1:]:
        common_query_ids &= set(answers_data[model_name].keys())
    common_query_ids = sorted(list(common_query_ids))

    logger.info(f"\n{'='*60}")
    logger.info(f"Judging Configuration")
    logger.info(f"{'='*60}")
    logger.info(f"Judge model: Azure GPT-5 ({azure_deployment})")
    logger.info(f"Models to compare: {', '.join(model_names)}")
    logger.info(f"Common queries: {len(common_query_ids)}")

    # Generate all model pairs
    model_names_sorted = sorted(model_names)
    pairs = list(itertools.combinations(model_names_sorted, 2))
    logger.info(f"Model pairs: {len(pairs)}")
    logger.info(f"Total comparisons: {len(common_query_ids) * len(pairs)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare output files
    judgments_file = output_dir / "judgments.jsonl"
    metrics_file = output_dir / "metrics.jsonl"

    all_judgments = []
    reused_count = 0
    new_count = 0

    # Score cache (critical optimization)
    score_cache = {}

    logger.info(f"\n{'='*60}")
    logger.info(f"Generating Judgments...")
    logger.info(f"{'='*60}")

    # Prepare ALL tasks across ALL queries (fully parallelized)
    all_tasks = []
    reused_judgments = []

    for query_id in common_query_ids:
        query_text = answers_data[model_names[0]][query_id]['question']

        for model_x, model_y in pairs:
            # Check if judgment exists
            key = (query_id, frozenset([model_x, model_y]))

            if key in existing_judgments:
                reused_judgments.append(existing_judgments[key])
                reused_count += 1
            else:
                all_tasks.append((query_id, model_x, model_y, query_text, answers_data, azure_client, azure_deployment, score_cache))
                new_count += 1

    logger.info(f"Total comparisons: {len(all_tasks) + reused_count}")
    logger.info(f"Reusing: {reused_count}, New: {new_count}")

    # Execute ALL new comparisons in parallel
    new_judgments = []
    if all_tasks:
        logger.info(f"Running {new_count} judgments in parallel with {args.max_workers} workers...")

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(judge_single_comparison, task) for task in all_tasks]

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                new_judgments.append(result)
                completed += 1

                # Progress update every 10 completions
                if completed % 10 == 0 or completed == len(all_tasks):
                    progress = completed * 100 / len(all_tasks)
                    logger.info(f"  [{completed}/{len(all_tasks)}] {progress:.1f}% - cached scores: {len(score_cache)}")

    # Combine reused and new judgments
    all_judgments = reused_judgments + new_judgments

    # Sort by query_id, then model pair for consistency
    all_judgments.sort(key=lambda x: (x['query_id'], x['model_x'], x['model_y']))

    # Save judgments
    logger.info(f"\n{'='*60}")
    logger.info(f"Saving Results...")
    logger.info(f"{'='*60}")

    with open(judgments_file, 'w') as f:
        for judgment in all_judgments:
            f.write(json.dumps(judgment) + '\n')
    logger.info(f"✅ Saved: {judgments_file}")

    # Save metrics (one entry per model per query)
    with open(metrics_file, 'w') as f:
        for model in model_names:
            for query_id in common_query_ids:
                cache_key = (model, query_id)
                if cache_key in score_cache:
                    scores = score_cache[cache_key]
                    metric_entry = {
                        "model": model,
                        "query_id": query_id,
                        **scores
                    }
                    f.write(json.dumps(metric_entry) + '\n')
    logger.info(f"✅ Saved: {metrics_file}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total judgments: {len(all_judgments)}")
    logger.info(f"New comparisons: {new_count}")
    logger.info(f"Reused judgments: {reused_count}")
    logger.info(f"Unique scores computed: {len(score_cache)}")

    total_api_calls = new_count + len(score_cache)
    without_caching = new_count * 3
    logger.info(f"\nAPI calls:")
    logger.info(f"  With caching: {total_api_calls}")
    logger.info(f"  Without caching: {without_caching}")
    logger.info(f"  Savings: {without_caching - total_api_calls} calls ({(without_caching - total_api_calls) / without_caching * 100:.1f}%)")

    return 0


if __name__ == "__main__":
    exit(main())
