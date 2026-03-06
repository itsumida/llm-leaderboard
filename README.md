# LLM RAG Leaderboard

Benchmark framework comparing LLM performance on RAG tasks across 3 datasets.

*Full results in `benchmarks.json`*

## Datasets

- **MS MARCO** - Web search Q&A
- **Paul Graham** - Essay comprehension
- **SciFact** - Scientific claim verification

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

## Add a New Model

See [ADD-MODEL.md](ADD-MODEL.md)

## Files

| File | Description |
|------|-------------|
| `benchmarks.json` | Full leaderboard with comparisons |
| `model-info.json` | Model metadata (cost, context, etc.) |
| `setup/` | Evaluation pipeline scripts |

## Evaluation Method

1. **Retrieve** - Top 15 docs via embedding + reranking
2. **Generate** - Each model answers using retrieved context
3. **Judge** - GPT-5 evaluates answers pairwise (A vs B)
4. **Score** - ELO ratings from pairwise outcomes

**Metrics:** Correctness, Faithfulness, Grounding, Relevance, Completeness (1-5 scale)

**Fairness notes:**
- All models answer the same 30 queries per dataset (90 total)
- Each model pair has equal comparisons (30 per dataset)
- Position bias exists (~40% A wins vs ~60% B wins) but affects all models equally since positions are balanced

## API Keys Required

```
OPENROUTER_API_KEY=xxx  # For LLM calls
AZURE_API_KEY=xxx       # For judge model
```
