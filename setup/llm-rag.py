#!/usr/bin/env python3
"""
Step 3: Generate LLM Answers via OpenRouter
Input: top15.jsonl (reranked documents)
Output: answers/ directory with one JSONL per model
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE importing openrouter_client
load_dotenv()

from openrouter_client import call_openrouter, generate_answer_with_challenge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_answer_for_query(query_id: str, query: str, top_docs: list, model_id: str, enable_challenge: bool = False) -> dict:
    """Generate answer for a single query using a specific model"""

    # Build context from top documents
    context_docs = "\n\n".join([
        f"[{i+1}] {doc.get('doc_id', f'doc_{i}')}: {doc.get('text', '')}"
        for i, doc in enumerate(top_docs)
    ])

    # System prompt with context
    system_prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the information provided in the retrieved documents.


Context Documents:
{context_docs}

CRITICAL INSTRUCTIONS:

1. Use only information explicitly stated in the context.
Do not infer, assume, or add anything from general knowledge.

2. If the context does not contain enough information to fully answer the query, respond clearly:
"I cannot fully answer this question based on the provided documents."
Briefly mention which parts lack support.

3. Do not cite, quote, or reference document IDs.
Just give a clean, direct answer.

4. Match the language of the user's query.

5. NEVER mention that you are using documents, context, or any source material.
NEVER use phrases like:
- "Based on the provided documents"
- "According to the context"
- "The documents show"
- "From the information provided"
Simply answer as if the information is naturally known to you.

6. Keep the answer concise, factual, and grounded strictly in the provided text."""

    # Measure latency
    start_time = time.time()

    # Many models need more tokens to produce output
    # Start with higher baseline to avoid empty responses
    max_tokens = 3000

    # Reasoning models need even more tokens
    reasoning_models = ["gemini-3", "gemini-2.5", "gpt-oss", "gpt-5", "qwen3-30b-a3b-thinking", "grok-code", "glm-4", "deepseek", "z-ai", "claude-sonnet-4.6"]
    if any(x in model_id.lower() for x in reasoning_models):
        max_tokens = 5000

    # These specific models have shown empty response issues
    if any(x in model_id.lower() for x in ["gpt-5-nano", "glm-4", "deepseek-r1"]):
        max_tokens = 6000

    try:
        # Use multi-turn reasoning if enabled
        if enable_challenge:
            result = generate_answer_with_challenge(
                model=model_id,
                query=query,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=max_tokens,
                challenge_prompt="Are you sure? Think carefully."
            )
            answer = result["answer"]  # Final answer after challenge
            initial_answer = result.get("initial_answer", "")
        else:
            result = call_openrouter(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=max_tokens
            )
            answer = result["content"]
            initial_answer = None

        latency_ms = int((time.time() - start_time) * 1000)

        output = {
            "query_id": query_id,
            "question": query,
            "answer": answer,
            "documents_used": [
                {
                    "doc_id": doc.get("doc_id", f"doc_{i}"),
                    "rank": i + 1,
                    "text": doc.get("text", "")
                }
                for i, doc in enumerate(top_docs)
            ],
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "latency_ms": latency_ms,
            "token_usage": {
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"]
            }
        }

        # Include initial answer if challenge mode was used
        if enable_challenge and initial_answer:
            output["initial_answer"] = initial_answer
            output["reasoning_mode"] = "multi-turn-challenge"

        return output

    except Exception as e:
        logger.error(f"Error generating answer for {query_id} with {model_id}: {e}")
        return {
            "query_id": query_id,
            "question": query,
            "answer": f"Error: {str(e)}",
            "documents_used": [],
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "latency_ms": 0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0}
        }


def main():
    parser = argparse.ArgumentParser(description='Generate LLM answers via OpenRouter')
    parser.add_argument('--input', required=True, help='Input file with top-15 docs per query')
    parser.add_argument('--output-dir', default='rag_output/answers', help='Output directory')
    parser.add_argument('--models', required=True, help='Comma-separated list of OpenRouter model IDs')
    parser.add_argument('--enable-challenge', action='store_true', help='Enable multi-turn reasoning with challenge prompt')
    args = parser.parse_args()

    # Parse models
    model_ids = [m.strip() for m in args.models.split(',')]

    # Check API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set!")
        return 1

    # Load queries with top-15 docs
    queries = []
    with open(args.input, 'r') as f:
        for line in f:
            queries.append(json.loads(line))

    logger.info(f"Loaded {len(queries)} queries")
    logger.info(f"Models: {', '.join(model_ids)}")
    if args.enable_challenge:
        logger.info("🧠 Multi-turn reasoning with challenge enabled")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate answers for each model (parallelized)
    for model_id in model_ids:
        model_name = model_id.replace('/', '-').replace('_', '-')
        output_file = output_dir / f"{model_name}_answers.jsonl"

        logger.info(f"\n{'='*60}")
        logger.info(f"Generating answers with: {model_id}")
        logger.info(f"{'='*60}")

        # Prepare tasks for parallel execution
        tasks = []
        for query_data in queries:
            query_id = query_data['query_id']
            query_text = query_data['query']
            top_docs = query_data.get('top15_docs', query_data.get('top_docs', []))
            tasks.append((query_id, query_text, top_docs, model_id, args.enable_challenge))

        # Execute in parallel
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(generate_answer_for_query, *task) for task in tasks]

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                logger.info(f"  [{i}/{len(tasks)}] {result['query_id']} completed")

        # Sort by query_id to maintain order
        results.sort(key=lambda x: x['query_id'])

        # Write all results
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

        logger.info(f"✅ Saved: {output_file}")

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ All answers generated!")
    logger.info(f"{'='*60}")
    logger.info(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
