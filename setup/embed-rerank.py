#!/usr/bin/env python3
"""
Unified Embed + Rerank Pipeline
Combines embedding generation, FAISS search, ZeRank reranking, and text enrichment in one script

Input: corpus.jsonl, queries.jsonl
Output: top15.jsonl (with full document text, ready for RAG pipeline)
"""

import argparse
import json
import numpy as np
import faiss
import requests
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
ZERANK_API_KEY = os.getenv("ZERANK_API_KEY")


def load_corpus(corpus_path: str) -> Dict[str, Dict]:
    """Load corpus from JSONL file"""
    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = doc
    logger.info(f"Loaded {len(corpus)} documents from corpus")
    return corpus


def load_queries(queries_path: str) -> Dict[str, str]:
    """Load queries from JSONL file"""
    queries = {}
    with open(queries_path) as f:
        for line in f:
            query = json.loads(line)
            queries[query['_id']] = query['text']
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def generate_query_embedding(query_text: str) -> np.ndarray:
    """Generate embedding for query using OpenAI API"""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query_text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def create_embeddings_and_index(corpus: Dict[str, Dict], cache_dir: Path) -> tuple:
    """Create embeddings and FAISS index from corpus"""
    embeddings_path = cache_dir / "embeddings.npy"
    index_path = cache_dir / "faiss_index.bin"

    # Check if cached files exist
    if embeddings_path.exists() and index_path.exists():
        logger.info("Loading cached embeddings and FAISS index...")
        embeddings = np.load(embeddings_path)
        index = faiss.read_index(str(index_path))
        corpus_ids = list(corpus.keys())
        logger.info(f"Loaded {len(embeddings)} cached embeddings")
        return embeddings, index, corpus_ids

    # Generate embeddings for all documents
    logger.info("Generating embeddings for corpus (this may take a while)...")
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    corpus_ids = list(corpus.keys())
    texts = [corpus[doc_id].get('text', '') for doc_id in corpus_ids]

    # Batch process embeddings
    embeddings_list = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts
        )

        for item in response.data:
            embeddings_list.append(item.embedding)

        time.sleep(0.5)  # Rate limiting

    embeddings = np.array(embeddings_list, dtype=np.float32)

    # Create FAISS index
    logger.info("Creating FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved embeddings and index to {cache_dir}")

    return embeddings, index, corpus_ids


def search_faiss(index: faiss.Index, query_embedding: np.ndarray, k: int) -> tuple:
    """Search FAISS index for top-k similar documents"""
    query_vec = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k)
    return scores[0], indices[0]


def rerank_with_zerank1(query: str, documents: List[str], top_n: int = 15) -> List[Dict]:
    """Rerank documents using ZeRank1 API"""
    if not ZERANK_API_KEY:
        logger.warning("ZeRank API key not found. Skipping reranking.")
        return []

    payload = {
        "model": "zerank-1",
        "query": query,
        "documents": documents,
        "top_n": top_n
    }

    headers = {
        "Authorization": f"Bearer {ZERANK_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://api.zeroentropy.dev/v1/models/rerank",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])

    except requests.exceptions.Timeout:
        logger.error("ZeRank API request timed out")
        return []
    except Exception as e:
        logger.error(f"ZeRank API error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Unified Embed + Rerank Pipeline')
    parser.add_argument('--corpus', required=True, help='Corpus JSONL file')
    parser.add_argument('--queries', required=True, help='Queries JSONL file')
    parser.add_argument('--output', required=True, help='Output file (top15.jsonl)')
    parser.add_argument('--top-k', type=int, default=15, help='Number of results to return per query')
    parser.add_argument('--retrieve-k', type=int, default=100, help='Number of candidates to retrieve before reranking')
    parser.add_argument('--cache-dir', default='.embed_cache', help='Directory for caching embeddings and index')
    args = parser.parse_args()

    # Check for required API keys
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("❌ OPENAI_API_KEY not found in environment variables")
        return 1

    if not ZERANK_API_KEY:
        logger.warning("⚠️  ZERANK_API_KEY not found. Will use FAISS scores only (no reranking)")

    # Load data
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)

    # Create or load embeddings and FAISS index
    cache_dir = Path(args.cache_dir)
    embeddings, index, corpus_ids = create_embeddings_and_index(corpus, cache_dir)

    # Process each query
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"🔍 Unified Embed + Rerank Pipeline")
    logger.info(f"{'='*70}")
    logger.info(f"📊 Corpus: {len(corpus)} documents")
    logger.info(f"❓ Queries: {len(queries)}")
    logger.info(f"🔝 Retrieve: Top {args.retrieve_k} candidates")
    logger.info(f"⭐ Rerank: Top {args.top_k} results")
    logger.info(f"💾 Output: {args.output}")
    logger.info(f"{'='*70}\n")

    with open(output_path, 'w') as out_f:
        for idx, (query_id, query_text) in enumerate(queries.items(), 1):
            logger.info(f"[{idx}/{len(queries)}] Query: {query_id}")
            logger.info(f"   Q: {query_text[:80]}...")

            # Generate query embedding
            start = time.time()
            query_embedding = generate_query_embedding(query_text)
            embed_time = (time.time() - start) * 1000
            logger.info(f"   ⏱️  Embedding: {embed_time:.0f}ms")

            # Search FAISS for candidates
            start = time.time()
            scores, indices = search_faiss(index, query_embedding, args.retrieve_k)
            search_time = (time.time() - start) * 1000
            logger.info(f"   ⏱️  FAISS search: {search_time:.0f}ms")

            # Prepare candidates for reranking
            candidates = []
            for score, idx_pos in zip(scores, indices):
                if idx_pos < len(corpus_ids):
                    doc_id = corpus_ids[idx_pos]
                    doc = corpus[doc_id]
                    text = doc.get('text', '')
                    title = doc.get('title', '')
                    full_text = f"{title}\n{text}".strip() if title else text

                    candidates.append({
                        'doc_id': doc_id,
                        'score': float(score),
                        'text': full_text
                    })

            # Rerank with ZeRank1 (if API key available)
            if ZERANK_API_KEY:
                start = time.time()
                documents = [c['text'] for c in candidates]
                reranked = rerank_with_zerank1(query_text, documents, top_n=args.top_k)
                rerank_time = (time.time() - start) * 1000
                logger.info(f"   ⏱️  ZeRank1: {rerank_time:.0f}ms")

                # Build final results with text included
                top_docs = []
                for rank, item in enumerate(reranked, 1):
                    original_idx = item['index']
                    top_docs.append({
                        'doc_id': candidates[original_idx]['doc_id'],
                        'rank': rank,
                        'score': item['relevance_score'],
                        'text': candidates[original_idx]['text']
                    })
            else:
                # No reranking - just take top K from FAISS with text
                top_docs = [
                    {
                        'doc_id': c['doc_id'],
                        'rank': rank,
                        'score': c['score'],
                        'text': c['text']
                    }
                    for rank, c in enumerate(candidates[:args.top_k], 1)
                ]

            # Save result (compatible with llm-rag.py)
            result_record = {
                'query_id': query_id,
                'query': query_text,
                'top15_docs': top_docs  # Use 'top15_docs' field name
            }

            out_f.write(json.dumps(result_record) + '\n')
            out_f.flush()

            logger.info(f"   ✓ Saved {len(top_docs)} results with text\n")

    logger.info(f"\n{'='*70}")
    logger.info(f"✅ Pipeline Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"📝 Output: {args.output}")
    logger.info(f"📊 Total queries processed: {len(queries)}")
    logger.info(f"🎯 Results per query: {args.top_k}")
    logger.info(f"\n💡 Next step: Generate LLM answers:")
    logger.info(f"   python3 llm-rag.py --input {args.output} --models <model_ids>")
    logger.info(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    exit(main())
