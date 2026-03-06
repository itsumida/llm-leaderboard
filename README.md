# LLM RAG Leaderboard

Benchmark framework comparing LLM performance on RAG tasks across 3 datasets.

## Leaderboard

| Rank | Model | ELO | Win Rate |
|------|-------|-----|----------|
| 1 | claude-opus-4-6 | 1916 | 83.0% |
| 2 | claude-sonnet-4.6 | 1738 | 67.6% |
| 3 | gpt-5.1 | 1689 | 65.5% |
| 4 | claude-opus-4-5-20251101 | 1666 | 57.9% |
| 5 | claude-sonnet-4.5 | 1613 | 50.8% |
| 6 | gemini-3-flash-preview | 1571 | 60.6% |
| 7 | gemini-3-pro-preview | 1509 | 44.2% |
| 8 | grok-4-fast | 1492 | 44.9% |
| 9 | gpt-5.2 | 1491 | 40.1% |
| 10 | gpt-5.4 | 1418 | 31.9% |

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
