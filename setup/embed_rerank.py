#!/usr/bin/env python3
"""
Generate ZeRank2 reranked results for MS MARCO dataset
Creates reranked_results.jsonl file compatible with RAG Q&A pipeline
"""

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

# Configuration
CORPUS_PATH = Path("msmarco/corpus.jsonl")
QUERIES_PATH = Path("msmarco/queries.jsonl")
EMBEDDINGS_PATH = Path("embed-msmarco/all_embeddings.npy")
FAISS_INDEX_PATH = Path("embed-msmarco/faiss_index.bin")
OUTPUT_PATH = Path("msmarco/reranked_results.jsonl")

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
ZERANK_API_KEY = os.getenv("ZERANK_API_KEY")

# How many candidates to retrieve before reranking
RETRIEVE_TOP_K = 100
# How many to keep after reranking
RERANK_TOP_K = 15


def load_corpus() -> Dict[str, Dict]:
    """Load MS MARCO corpus"""
    corpus = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = doc
    logger.info(f"Loaded {len(corpus)} documents from corpus")
    return corpus


def load_queries() -> Dict[str, str]:
    """Load MS MARCO queries"""
    queries = {}
    with open(QUERIES_PATH) as f:
        for line in f:
            query = json.loads(line)
            queries[query['_id']] = query['text']
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def load_or_create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Load existing FAISS index or create new one"""
    if FAISS_INDEX_PATH.exists():
        logger.info("Loading existing FAISS index...")
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
    else:
        logger.info("Creating new FAISS index...")
        index = faiss.IndexFlatIP(EMBEDDING_DIM)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        # Save index
        FAISS_INDEX_PATH.parent.mkdir(exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        logger.info(f"Created and saved FAISS index with {index.ntotal} vectors")

    return index


def generate_query_embedding(query_text: str) -> np.ndarray:
    """Generate embedding for query using OpenAI API"""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query_text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


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


def generate_reranked_results():
    """Generate reranked results for all queries"""

    # Check for required API keys
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    if not ZERANK_API_KEY:
        logger.warning("⚠️  ZERANK_API_KEY not found. Reranking will be skipped.")
        logger.warning("    Add ZERANK_API_KEY to .env file to enable reranking")

    # Load data
    corpus = load_corpus()
    queries = load_queries()

    # Load embeddings
    logger.info("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")

    # Create corpus ID mapping (embeddings order to doc IDs)
    corpus_ids = list(corpus.keys())

    # Load or create FAISS index
    index = load_or_create_faiss_index(embeddings)

    # Process each query
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    # Clear output file if it exists
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
        logger.info(f"Removed old output file: {OUTPUT_PATH}")

    print(f"\n{'='*70}")
    print(f"🔍 Generating Reranked Results for MS MARCO")
    print(f"{'='*70}")
    print(f"📊 Corpus: {len(corpus)} documents")
    print(f"❓ Queries: {len(queries)}")
    print(f"🔝 Retrieve: Top {RETRIEVE_TOP_K} candidates")
    print(f"⭐ Rerank: Top {RERANK_TOP_K} results (ZeRank1)")
    print(f"💾 Output: {OUTPUT_PATH}")
    print(f"{'='*70}\n")

    with open(OUTPUT_PATH, 'w') as out_f:
        for idx, (query_id, query_text) in enumerate(queries.items(), 1):
            print(f"[{idx}/{len(queries)}] Query: {query_id}")
            print(f"   Q: {query_text[:80]}...")

            # Generate query embedding
            start = time.time()
            query_embedding = generate_query_embedding(query_text)
            embed_time = (time.time() - start) * 1000
            print(f"   ⏱️  Embedding: {embed_time:.0f}ms")

            # Search FAISS for candidates
            start = time.time()
            scores, indices = search_faiss(index, query_embedding, RETRIEVE_TOP_K)
            search_time = (time.time() - start) * 1000
            print(f"   ⏱️  FAISS search: {search_time:.0f}ms")

            # Prepare candidates for reranking
            candidates = []
            for score, idx_pos in zip(scores, indices):
                if idx_pos < len(corpus_ids):
                    doc_id = corpus_ids[idx_pos]
                    candidates.append({
                        'doc_id': doc_id,
                        'score': float(score),
                        'text': corpus[doc_id]['text']
                    })

            # Rerank with ZeRank1
            if ZERANK_API_KEY:
                start = time.time()
                documents = [c['text'] for c in candidates]
                reranked = rerank_with_zerank1(query_text, documents, top_n=RERANK_TOP_K)
                rerank_time = (time.time() - start) * 1000
                print(f"   ⏱️  ZeRank1: {rerank_time:.0f}ms")

                # Build final results
                final_results = []
                for rank, item in enumerate(reranked, 1):
                    original_idx = item['index']
                    final_results.append({
                        'rank': rank,
                        'doc_id': candidates[original_idx]['doc_id'],
                        'score': item['relevance_score']
                    })
            else:
                # No reranking - just take top K from FAISS
                final_results = [
                    {
                        'rank': rank,
                        'doc_id': c['doc_id'],
                        'score': c['score']
                    }
                    for rank, c in enumerate(candidates[:RERANK_TOP_K], 1)
                ]

            # Save result
            result_record = {
                'query_id': query_id,
                'query': query_text,
                'results': final_results
            }

            out_f.write(json.dumps(result_record) + '\n')
            out_f.flush()

            print(f"   ✓ Saved {len(final_results)} reranked results\n")

    print(f"\n{'='*70}")
    print(f"✅ Reranked Results Generation Complete!")
    print(f"{'='*70}")
    print(f"📝 Output: {OUTPUT_PATH}")
    print(f"📊 Total queries processed: {len(queries)}")
    print(f"🎯 Results per query: {RERANK_TOP_K}")
    print(f"\n💡 Next step: Run RAG Q&A pipeline with these results:")
    print(f"   python3 rag-qa.py --dataset-config configs/msmarco_dataset.yaml")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    generate_reranked_results()
