# LLM RAG Leaderboard Pipeline Guide

Complete guide for running each step of the evaluation pipeline individually.

---

## Pipeline Overview

```
1. embed      → Generate embeddings for queries
2. rerank     → Rerank documents using embeddings
3. llm-rag    → Generate model answers via OpenRouter
4. judge      → Judge answers using Azure GPT-5
5. elo        → Calculate Elo ratings
6. results    → Generate final leaderboard
```

---

## Directory Structure

```
datasets/
├── msmarco/
│   ├── queries.jsonl         # Queries for MS MARCO
│   ├── corpus.jsonl          # Document corpus
│   └── qrels.jsonl           # Query relevance judgments
└── your-dataset/
    ├── queries.jsonl
    ├── corpus.jsonl
    └── qrels.jsonl

embed-rerank/
├── msmarco/
│   ├── queries_embedded.jsonl    # Step 1 output
│   └── top15.jsonl               # Step 2 output
└── your-dataset/
    ├── queries_embedded.jsonl
    └── top15.jsonl

setup/
├── rag_output/
│   └── answers/                  # Step 3 output
├── judge_output/
│   ├── judgments.jsonl           # Step 4 output
│   └── metrics.jsonl
├── elo_output/
│   └── elo_ratings.json          # Step 5 output
└── results_output/
    ├── leaderboard.html          # Step 6 output
    └── leaderboard.csv
```

---

## Prerequisites

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```bash
# OpenRouter API (for model answers)
OPENROUTER_API_KEY=your_openrouter_key

# Azure OpenAI API (for judge)
AZURE_API_KEY=your_azure_key
AZURE_RESOURCE_NAME=your_resource_name
AZURE_DEPLOYMENT_ID=gpt-5
```

---

## Step 1: Embed Queries

Generate embeddings for your queries.

### MS MARCO Dataset

```bash
python setup/embed-queries.py \
  --input datasets/msmarco/queries.jsonl \
  --output embed-rerank/msmarco/queries_embedded.jsonl
```

### Your Custom Dataset

```bash
python setup/embed-queries.py \
  --input datasets/your-dataset/queries.jsonl \
  --output embed-rerank/your-dataset/queries_embedded.jsonl
```

### Subset of Queries

```bash
# Create a subset file first
head -n 10 datasets/msmarco/queries.jsonl > datasets/msmarco/queries_subset.jsonl

# Then embed
python setup/embed-queries.py \
  --input datasets/msmarco/queries_subset.jsonl \
  --output embed-rerank/msmarco/queries_subset_embedded.jsonl
```

**Output**: `queries_embedded.jsonl` with vector embeddings in `embed-rerank/{dataset}/`

---

## Step 2: Rerank Documents

Rerank documents based on query embeddings and corpus.

### MS MARCO Dataset

```bash
python setup/rerank.py \
  --input embed-rerank/msmarco/queries_embedded.jsonl \
  --corpus datasets/msmarco/corpus.jsonl \
  --output embed-rerank/msmarco/top15.jsonl \
  --top-k 15
```

### Your Custom Dataset

```bash
python setup/rerank.py \
  --input embed-rerank/your-dataset/queries_embedded.jsonl \
  --corpus datasets/your-dataset/corpus.jsonl \
  --output embed-rerank/your-dataset/top15.jsonl \
  --top-k 15
```

### Subset

```bash
python setup/rerank.py \
  --input embed-rerank/msmarco/queries_subset_embedded.jsonl \
  --corpus datasets/msmarco/corpus.jsonl \
  --output embed-rerank/msmarco/top15_subset.jsonl \
  --top-k 15
```

**Parameters**:
- `--corpus`: Path to document corpus
- `--top-k`: Number of top documents to retrieve (default: 15)

**Output**: `top15.jsonl` with top-15 documents per query in `embed-rerank/{dataset}/`

---

## Step 3: Generate Model Answers

Generate answers from models via OpenRouter using reranked documents.

### Single Model (MS MARCO)

```bash
python setup/llm-rag.py \
  --input embed-rerank/msmarco/top15.jsonl \
  --output-dir setup/rag_output/answers \
  --models "anthropic/claude-opus-4.5"
```

### Multiple Models

```bash
python setup/llm-rag.py \
  --input embed-rerank/msmarco/top15.jsonl \
  --output-dir setup/rag_output/answers \
  --models "anthropic/claude-opus-4.5,google/gemini-3-pro-preview"
```

### Add New Model (Incremental)

```bash
# Just add the new model - existing answers are preserved
python setup/llm-rag.py \
  --input embed-rerank/msmarco/top15.jsonl \
  --output-dir setup/rag_output/answers \
  --models "openai/gpt-4o"
```

### Subset of Queries

```bash
python setup/llm-rag.py \
  --input embed-rerank/msmarco/top15_subset.jsonl \
  --output-dir setup/rag_output/answers_subset \
  --models "anthropic/claude-opus-4.5,google/gemini-3-pro-preview"
```

### Custom Dataset

```bash
python setup/llm-rag.py \
  --input embed-rerank/your-dataset/top15.jsonl \
  --output-dir setup/rag_output/answers_your_dataset \
  --models "anthropic/claude-opus-4.5,google/gemini-3-pro-preview"
```

**Available Models** (via OpenRouter):
- `anthropic/claude-opus-4.5`
- `google/gemini-3-pro-preview`
- `openai/gpt-4o`
- `openai/gpt-4-turbo`
- `meta-llama/llama-3.3-70b-instruct`

**Output**: One JSONL file per model in `setup/rag_output/answers/`
- Example: `anthropic-claude-opus-4.5_answers.jsonl`

---

## Step 4: Judge Answers

Evaluate answers using Azure GPT-5.

### First Run

```bash
python setup/judge.py \
  --answers-dir setup/rag_output/answers \
  --output-dir setup/judge_output \
  --max-workers 10
```

### Incremental (Reuse Existing Judgments)

```bash
# After adding a new model, reuse previous judgments
python setup/judge.py \
  --answers-dir setup/rag_output/answers \
  --output-dir setup/judge_output \
  --existing-judgments setup/judge_output/judgments.jsonl \
  --max-workers 10
```

### Subset

```bash
python setup/judge.py \
  --answers-dir setup/rag_output/answers_subset \
  --output-dir setup/judge_output_subset \
  --max-workers 5
```

**Parameters**:
- `--max-workers`: Parallel API calls (default: 10)
- `--existing-judgments`: Path to previous judgments to reuse

**Output**:
- `judgments.jsonl`: Pairwise comparisons (A/B/TIE)
- `metrics.jsonl`: Dimensional scores (1-5 on 5 dimensions)

**Cost Optimization**:
- Score caching: Same model+query is only scored once
- Judgment reuse: Use `--existing-judgments` when adding new models
- Example: Adding 3rd model only judges new pairs, reuses existing

---

## Step 5: Calculate Elo Ratings

Compute Elo ratings from pairwise judgments.

### Full Dataset

```bash
python setup/elo.py \
  --judgments setup/judge_output/judgments.jsonl \
  --output setup/elo_output/elo_ratings.json
```

### Subset

```bash
python setup/elo.py \
  --judgments setup/judge_output_subset/judgments.jsonl \
  --output setup/elo_output/elo_ratings_subset.json
```

**Parameters**:
- `--k-factor`: Elo sensitivity (default: 32)
- `--initial-rating`: Starting Elo (default: 1500)

**Output**: `elo_ratings.json` with model rankings

---

## Step 6: Generate Results

Create final leaderboard visualization.

### Full Dataset

```bash
python setup/results.py \
  --elo setup/elo_output/elo_ratings.json \
  --metrics setup/judge_output/metrics.jsonl \
  --output-dir setup/results_output
```

### Subset

```bash
python setup/results.py \
  --elo setup/elo_output/elo_ratings_subset.json \
  --metrics setup/judge_output_subset/metrics.jsonl \
  --output-dir setup/results_output_subset
```

**Output**:
- `leaderboard.html`: Interactive web visualization
- `leaderboard.csv`: Raw data for analysis

---

## Common Workflows

### Test with 5 Queries (MS MARCO)

```bash
# 1. Create subset
head -n 5 datasets/msmarco/queries.jsonl > datasets/msmarco/queries_test.jsonl

# 2. Embed
python setup/embed-queries.py \
  --input datasets/msmarco/queries_test.jsonl \
  --output embed-rerank/msmarco/queries_test_embedded.jsonl

# 3. Rerank
python setup/rerank.py \
  --input embed-rerank/msmarco/queries_test_embedded.jsonl \
  --corpus datasets/msmarco/corpus.jsonl \
  --output embed-rerank/msmarco/top15_test.jsonl \
  --top-k 15

# 4. Generate answers (2 models)
python setup/llm-rag.py \
  --input embed-rerank/msmarco/top15_test.jsonl \
  --output-dir setup/rag_output/answers_test \
  --models "anthropic/claude-opus-4.5,google/gemini-3-pro-preview"

# 5. Judge
python setup/judge.py \
  --answers-dir setup/rag_output/answers_test \
  --output-dir setup/judge_output_test \
  --max-workers 5

# 6. Elo
python setup/elo.py \
  --judgments setup/judge_output_test/judgments.jsonl \
  --output setup/elo_output/elo_test.json

# 7. Results
python setup/results.py \
  --elo setup/elo_output/elo_test.json \
  --metrics setup/judge_output_test/metrics.jsonl \
  --output-dir setup/results_output_test
```

### Add a New Model

```bash
# 1. Generate answers for new model only (reuses existing top15.jsonl)
python setup/llm-rag.py \
  --input embed-rerank/msmarco/top15.jsonl \
  --output-dir setup/rag_output/answers \
  --models "openai/gpt-4o"

# 2. Re-judge with existing judgments (saves API calls)
python setup/judge.py \
  --answers-dir setup/rag_output/answers \
  --output-dir setup/judge_output \
  --existing-judgments setup/judge_output/judgments.jsonl \
  --max-workers 10

# 3. Recalculate Elo
python setup/elo.py \
  --judgments setup/judge_output/judgments.jsonl \
  --output setup/elo_output/elo_ratings.json

# 4. Regenerate leaderboard
python setup/results.py \
  --elo setup/elo_output/elo_ratings.json \
  --metrics setup/judge_output/metrics.jsonl \
  --output-dir setup/results_output
```

### Add a New Dataset

```bash
# 1. Prepare your dataset
mkdir -p datasets/my-dataset
# Add queries.jsonl, corpus.jsonl, qrels.jsonl to datasets/my-dataset/

# 2. Create output directory
mkdir -p embed-rerank/my-dataset

# 3. Embed queries
python setup/embed-queries.py \
  --input datasets/my-dataset/queries.jsonl \
  --output embed-rerank/my-dataset/queries_embedded.jsonl

# 4. Rerank documents
python setup/rerank.py \
  --input embed-rerank/my-dataset/queries_embedded.jsonl \
  --corpus datasets/my-dataset/corpus.jsonl \
  --output embed-rerank/my-dataset/top15.jsonl \
  --top-k 15

# 5. Continue with llm-rag, judge, elo, results steps
python setup/llm-rag.py \
  --input embed-rerank/my-dataset/top15.jsonl \
  --output-dir setup/rag_output/answers_my_dataset \
  --models "anthropic/claude-opus-4.5,google/gemini-3-pro-preview"
```

### Compare Different Query Subsets

```bash
# Subset 1: First 100 queries
head -n 100 datasets/msmarco/queries.jsonl > datasets/msmarco/queries_100.jsonl

# Subset 2: Random 50 queries
shuf -n 50 datasets/msmarco/queries.jsonl > datasets/msmarco/queries_random50.jsonl

# Run full pipeline for each subset with different output directories
```

---

## Monitoring Progress

### Check Dataset Files

```bash
# Count queries
wc -l datasets/msmarco/queries.jsonl

# Count corpus documents
wc -l datasets/msmarco/corpus.jsonl

# Check embedded queries
wc -l embed-rerank/msmarco/queries_embedded.jsonl

# Check reranked results
wc -l embed-rerank/msmarco/top15.jsonl
```

### Check Answer Files

```bash
# Count answers per model
wc -l setup/rag_output/answers/*.jsonl

# Check specific model
cat setup/rag_output/answers/anthropic-claude-opus-4.5_answers.jsonl | jq '.query_id'
```

### Check Judgment Progress

```bash
# Total judgments
wc -l setup/judge_output/judgments.jsonl

# Total scores
wc -l setup/judge_output/metrics.jsonl

# Check specific comparison
grep "104861" setup/judge_output/judgments.jsonl | jq .
```

### Check Elo Ratings

```bash
# View current standings
cat setup/elo_output/elo_ratings.json | jq '.rankings'
```

---

## Troubleshooting

### Issue: API Rate Limits

**Solution**: Reduce `--max-workers`

```bash
python setup/judge.py \
  --answers-dir setup/rag_output/answers \
  --output-dir setup/judge_output \
  --max-workers 3  # Lower parallelism
```

### Issue: Gemini 3 Empty Answers

**Solution**: Already fixed in `llm-rag.py` (uses 2000 max_tokens + reasoning mode)

### Issue: Judge Giving All 5s

**Solution**: Already fixed in `judge.py` (stricter scoring prompt)

### Issue: Need to Re-judge with Different Prompt

```bash
# 1. Delete old judgments
rm setup/judge_output/judgments.jsonl
rm setup/judge_output/metrics.jsonl

# 2. Re-run judge
python setup/judge.py \
  --answers-dir setup/rag_output/answers \
  --output-dir setup/judge_output \
  --max-workers 10
```

### Issue: Missing Corpus for Reranking

**Solution**: Ensure corpus path is correct

```bash
# Check corpus exists
ls -lh datasets/msmarco/corpus.jsonl

# Verify corpus format (should be JSONL with doc_id and text)
head -n 1 datasets/msmarco/corpus.jsonl | jq .
```

---

## Tips

1. **Start Small**: Test with 5-10 queries before running full dataset
2. **Reuse Judgments**: Always use `--existing-judgments` when adding models
3. **Monitor Costs**: Each judgment = 3 API calls (2 scores + 1 pairwise)
4. **Score Caching**: Judge caches scores, so adding models is cheap
5. **Parallel Workers**: Adjust `--max-workers` based on API rate limits
6. **Check Outputs**: Verify each step's output before proceeding
7. **Dataset Isolation**: Keep each dataset's embed/rerank outputs in separate folders
8. **Reuse Embeddings**: Steps 1-2 (embed + rerank) only need to run once per dataset

---

## Cost Estimates

### 5 Test Queries (2 Models)

- **Answers**: 10 calls (5 × 2)
- **Judgments**: 15 calls (10 scores + 5 pairwise)
- **Total**: ~25 API calls (~$0.75)

### 100 Queries (3 Models)

- **Answers**: 300 calls (100 × 3)
- **Judgments**: 600 calls (300 scores + 300 pairwise)
- **Total**: ~900 API calls (~$27)

### Adding 4th Model to Existing 3

- **Answers**: 100 new calls
- **Judgments**: 400 new calls (100 scores + 300 new pairwise)
- **Reused**: 300 scores + 300 pairwise = 600 saved
- **Total**: ~500 API calls (~$15)

---

## Questions?

Check the README or individual script help:

```bash
python setup/judge.py --help
python setup/llm-rag.py --help
python setup/elo.py --help
```
