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
import random
from dotenv import load_dotenv
from openai import OpenAI

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


def score_answer_azure(answer: str, query: str, context_docs: str, client: OpenAI, deployment_id: str):
    """Score answer using Azure GPT-5"""
    system_content = f"""You are a highly critical evaluator. 

CRITICAL MINDSET:
• Default assumption: No answer is perfect. Look for flaws first, not strengths.
• Score of 5 = Truly exceptional, publishable quality (reserve for <5% of answers)
• Score of 4 = Very good but has at least 1-2 minor issues
• Score of 3 = Acceptable/average with multiple issues
• Score of 2 = Below average with significant problems
• Score of 1 = Poor quality, major flaws

Context Documents:
{context_docs}

EVALUATION CRITERIA:

Correctness (factual accuracy):
5 = Perfect. Zero errors. Every fact is verifiable and precise.
4 = Very good. At most 1 minor imprecision or missing nuance.
3 = Acceptable. 2-3 minor issues OR 1 moderate issue.
2 = Below standard. Multiple errors or 1 major error.
1 = Poor. Mostly incorrect or seriously flawed.

Faithfulness (grounded in context):
5 = Every single claim has direct support. Nothing added from outside knowledge.
4 = Almost entirely supported. Maybe 1 very minor inference.
3 = Mostly supported but 2-3 claims lack clear evidence.
2 = Several claims unsupported or speculative.
1 = Many hallucinations or fabrications.

Grounding (evidence usage):
5 = Exceptional use of context. Could cite specific passages for every point.
4 = Strong grounding with very minor gaps in citation quality.
3 = Adequate but doesn't leverage context optimally.
2 = Weak connection to documents.
1 = Barely uses provided context.

Relevance (addresses the query):
5 = Perfectly targeted. Answers exactly what was asked, nothing more/less.
4 = Very relevant but slightly over/under-addresses one aspect.
3 = Generally on-topic but misses nuances or includes irrelevant info.
2 = Partially off-topic or misunderstands query.
1 = Mostly irrelevant.

Completeness (coverage):
5 = Comprehensive. All aspects covered with appropriate depth.
4 = Nearly complete. One minor gap or aspect could be deeper.
3 = Adequate coverage but 2+ notable gaps.
2 = Significant omissions.
1 = Major parts of query unanswered.

SCORING APPROACH:
1. Read answer critically. List flaws mentally.
2. Start from 3 (average) and adjust based on flaws/strengths.
3. Only use 5 if you cannot find ANY flaw in that dimension.
4. Use 4 sparingly - most "good" answers have issues.

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


def judge_pairwise_azure(answer_a: str, answer_b: str, query: str, context_docs: str, model_a_name: str, model_b_name: str, client: OpenAI, deployment_id: str):
    """Judge pairwise using Azure GPT-5"""
    system_content = f"""You are a rigorous, unbiased evaluator comparing two answers to the same query.

Context Documents:
{context_docs}

CRITICAL EVALUATION PROTOCOL:

1. CONTENT QUALITY ONLY - Ignore these factors completely:
   - Length (shorter or longer is neither better nor worse)
   - Formatting style (bullets vs paragraphs)
   - Writing tone or voice
   - Answer structure or organization

2. JUDGE STRICTLY ON:
   A. Factual Correctness - Which answer has more accurate information?
   B. Context Grounding - Which answer is better supported by the provided documents?
   C. Completeness - Which answer addresses more aspects of the query?
   D. Hallucination Risk - Which answer makes fewer unsupported claims?
   E. Relevance - Which answer more directly addresses what was asked?

3. DECISION RULES:
   - If both answers have identical content quality across all 5 dimensions → TIE
   - If one answer is clearly better on 2+ dimensions and not worse on others → Choose it
   - If one answer is marginally better on most dimensions → Choose it
   - If answers are roughly equal with trade-offs → TIE
   - Default to TIE when genuinely uncertain

4. FORBIDDEN BIASES:
   - DO NOT prefer longer answers simply because they're longer
   - DO NOT prefer shorter answers simply because they're concise
   - DO NOT prefer answers with more bullet points or better formatting
   - DO NOT assume more detail = higher quality (only if the details are correct and relevant)

RESPOND WITH ONLY ONE WORD: A, B, or TIE

Your judgment must be based purely on factual accuracy, grounding, completeness, and relevance."""

    prompt = f"""Query: {query}

Answer A:
{answer_a}

Answer B:
{answer_b}

Based strictly on factual accuracy, grounding in context, completeness, and relevance (ignoring length and style):
Reply with ONE WORD ONLY: A, B, or TIE"""

    try:
        response = client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5
        )

        judgment = response.choices[0].message.content.strip().upper()
        if judgment in ["A", "B", "TIE"]:
            return judgment
        # Extract first word if judge responded with explanation
        first_word = judgment.split()[0]
        if first_word in ["A", "B", "TIE"]:
            return first_word
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
    azure_endpoint = os.getenv('AZURE_ENDPOINT')
    azure_deployment = os.getenv('AZURE_DEPLOYMENT_ID', 'gpt-5-chat')

    if not azure_api_key or not azure_endpoint:
        print("❌ Error: AZURE_API_KEY and AZURE_ENDPOINT must be set in .env!")
        return 1

    # Initialize Azure OpenAI client (new format)
    azure_client = OpenAI(
        base_url=azure_endpoint,
        api_key=azure_api_key
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

        for pair in pairs:
            # Randomize A/B position for each query to avoid position bias
            model_x, model_y = (pair[0], pair[1]) if random.random() < 0.5 else (pair[1], pair[0])

            # Check if judgment exists
            key = (query_id, frozenset([model_x, model_y]))

            if key in existing_judgments:
                existing_j = existing_judgments[key]
                reused_judgments.append(existing_j)
                reused_count += 1

                # Add scores from existing judgment to cache for metrics generation
                cache_key_x = (model_x, query_id)
                cache_key_y = (model_y, query_id)
                if 'scores_x' in existing_j:
                    score_cache[cache_key_x] = existing_j['scores_x']
                if 'scores_y' in existing_j:
                    score_cache[cache_key_y] = existing_j['scores_y']
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
    if without_caching > 0:
        logger.info(f"  Savings: {without_caching - total_api_calls} calls ({(without_caching - total_api_calls) / without_caching * 100:.1f}%)")

    return 0


if __name__ == "__main__":
    exit(main())
