#!/usr/bin/env python3
"""
Enrich reranked results with actual document text from corpus
Input: reranked_results.jsonl (doc_ids only)
Output: reranked_results_with_text.jsonl (includes text)
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Enrich reranked results with document text')
    parser.add_argument('--reranked', required=True, help='Reranked results file (with doc_ids)')
    parser.add_argument('--corpus', required=True, help='Corpus file (with doc texts)')
    parser.add_argument('--output', required=True, help='Output file')
    args = parser.parse_args()

    # Load corpus into memory (doc_id -> text mapping)
    logger.info(f"Loading corpus from {args.corpus}")
    corpus = {}
    with open(args.corpus, 'r') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc['_id']
            text = doc.get('text', '')
            title = doc.get('title', '')
            # Combine title and text
            full_text = f"{title}\n{text}".strip() if title else text
            corpus[doc_id] = full_text

    logger.info(f"Loaded {len(corpus)} documents from corpus")

    # Process reranked results
    logger.info(f"Processing reranked results from {args.reranked}")
    enriched_queries = []
    missing_docs = 0

    with open(args.reranked, 'r') as f:
        for line in f:
            query_data = json.loads(line)

            # Convert results to top_docs with text
            top_docs = []
            for result in query_data.get('results', []):
                doc_id = result['doc_id']
                text = corpus.get(doc_id, '')

                if not text:
                    missing_docs += 1
                    logger.warning(f"Missing text for doc_id: {doc_id}")

                top_docs.append({
                    'doc_id': doc_id,
                    'rank': result['rank'],
                    'score': result['score'],
                    'text': text
                })

            # Create enriched query data
            enriched_query = {
                'query_id': query_data['query_id'],
                'query': query_data['query'],
                'top_docs': top_docs  # Use 'top_docs' field name expected by llm-rag.py
            }
            enriched_queries.append(enriched_query)

    # Write enriched results
    logger.info(f"Writing enriched results to {args.output}")
    with open(args.output, 'w') as f:
        for query_data in enriched_queries:
            f.write(json.dumps(query_data) + '\n')

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Enrichment Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Queries processed: {len(enriched_queries)}")
    logger.info(f"Missing documents: {missing_docs}")
    logger.info(f"Output: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
