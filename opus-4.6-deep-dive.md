# Claude Opus 4.6: Testing Anthropic's Latest Against Our RAG Benchmark

**Umida Muratbekova** | [agentset.ai](https://agentset.ai) | February 6, 2026

Anthropic released Claude Opus 4.6 yesterday (February 5, 2026), claiming it's designed for "longer-running, agentic work" with a 1M token context window. We ran it through our standard RAG evaluation pipeline to see how it performs on real question-answering tasks.

## What We Tested

We evaluated Opus 4.6 on three datasets:
- **MSMARCO**: 30 factual queries (information retrieval)
- **Paul Graham Essays**: 30 queries requiring synthesis and long-form reasoning
- **SciFact**: 30 scientific claim verification tasks

Each model gets the same top-15 retrieved documents as context and generates answers. We then use pairwise comparison (GPT-5 as judge) to evaluate answers against 11 other frontier models, calculating ELO ratings similar to chess rankings.

The competing models include GPT-5.1, GPT-5.2, Gemini 3 Flash, Gemini 3 Pro, DeepSeek R1, Grok 4 Fast, and Opus 4.5 among others.

## Overall Results

**Opus 4.6 achieved the highest overall ELO: 1780.28**

```
Model Performance (Top 5):
1. Claude Opus 4.6    1780.28  (74.8% win rate)
2. GPT-5.1            1716.03  (65.3% win rate)
3. Grok 4 Fast        1616.31  (54.3% win rate)
4. Gemini 3 Flash     1569.71  (56.8% win rate)
5. GPT-5.2            1558.61  (41.8% win rate)
```

Out of 990 pairwise judgments, Opus 4.6 won 740 (74.8%), lost only 93, and tied 157. That's a decisive victory overall.

## Dataset Breakdown: Where It Excels

### MSMARCO: Dominant Performance

Opus 4.6 was nearly unbeatable on factual retrieval:

```
ELO: 1783.19 (#1)
Record: 268-16-46 (81.2% win rate)
Quality Metrics: Perfect 5.0/5.0 across all dimensions
Latency: 3.7s - 12.5s (avg 7.7s)
```

**Head-to-head highlights:**
- vs GPT-5.2: 28-1 (96.7% win rate in decisive judgments)
- vs Gemini 3 Flash: 19-4 (82.6%)
- vs DeepSeek R1: 25-1 (96.2%)
- vs GPT-OSS-120B: 30-0 (perfect sweep)

When you need precise, grounded answers to factual questions, Opus 4.6 is the best model we've tested. Zero hallucinations detected in manual review.

### Paul Graham Essays: Strong But Not #1

This is where things get interesting:

```
ELO: 1806.56 (#2)
Record: 255-52-23 (77.3% win rate)
Quality Metrics: Perfect 5.0/5.0
Latency: 11.2s - 26.0s (avg 16.8s)
```

Opus 4.6 ranked second on this dataset. **GPT-5.1 took first place with an ELO of 1869.98** - the highest single-dataset score we've ever recorded.

Head-to-head, GPT-5.1 beat Opus 4.6 20-8-2. Despite perfect quality scores, judges consistently preferred GPT-5.1's answers for these synthesis-heavy queries.

The latency tells a story: Opus 4.6 averaged 16.8 seconds per query on Paul Graham essays vs. 7.7 seconds on MSMARCO. It's thinking longer for complex reasoning tasks.

### SciFact: The Completeness Problem

```
ELO: 1751.09 (#1)
Record: 217-25-88 (65.8% win rate)
Quality Metrics: Overall 4.64/5.0
  - Completeness: 4.36/5.0 ← Notable gap
  - Correctness: 4.55/5.0
  - Faithfulness: 4.64/5.0
Latency: 4.7s - 19.1s (avg 10.2s)
```

Opus 4.6 ranked first on SciFact but with caveats:
1. **88 ties out of 330 judgments (26.7%)** - many matchups were too close to call
2. **Completeness score of 4.36** - the lowest quality dimension we measured

For comparison, GPT-5.1 scored perfect 5.0 on completeness. Opus 4.6's answers were typically shorter (avg 87 tokens vs GPT-5.1's 142 tokens) and sometimes missed supporting evidence.

**Takeaway**: If you need comprehensive coverage for scientific or technical writing, Opus 4.6 might be too concise.

## Opus 4.6 vs Opus 4.5: Major Upgrade

The improvement over the previous generation is substantial:

```
Overall:      4.6 wins 54-5 (91.5% decisive win rate)

By Dataset:
MSMARCO:     4.6 wins 15-3   (+95 ELO)
PG Essays:   4.6 wins 26-0   (+387 ELO) ← Massive jump
SciFact:     4.6 wins 13-2   (+213 ELO)
```

The Paul Graham essays improvement is remarkable. Opus 4.5 had an embarrassing 1419 ELO on that dataset (7th place), losing to nearly every competitor. Anthropic clearly identified and fixed a critical weakness in long-context reasoning.

## Latency & Token Usage

**Latency varies significantly by task complexity:**

```
Dataset       Avg      Min      Max
MSMARCO      7.7s     3.7s     12.5s
PG Essays    16.8s    11.2s    26.0s
SciFact      10.2s    4.7s     19.1s
```

Opus 4.6 appears to adaptively allocate compute based on query difficulty. Simple factual lookups are fast, complex synthesis tasks trigger extended reasoning.

**Token consumption (per 30 queries):**

```
Dataset       Input Tokens    Output Tokens
MSMARCO       52,536          10,460
PG Essays     159,188         18,896
SciFact       193,498         9,367
```

SciFact's high input token usage (193K) reflects the 1M context window handling longer scientific documents. However, lower output tokens (9,367) explain the completeness gap - it's being more concise than competitors.

**Cost at $5/M input, $25/M output:**
- MSMARCO: $0.52 per 30 queries
- PG Essays: $1.27
- SciFact: $1.20
- **Total: $2.99 for 90-query evaluation**

## The GPT-5.1 Paradox

Here's something counterintuitive: **Opus 4.6 has higher overall ELO (1780) but loses head-to-head to GPT-5.1** (38-27-25 record, 42.2% win rate).

How? ELO systems aggregate performance across all matchups, not just head-to-head. Opus 4.6 dominated weaker models while GPT-5.1 struggled against them but excelled on difficult tasks.

**Practical implication**: Overall leaderboard rankings don't guarantee superiority in direct comparisons. Task-specific evaluation matters more.

## What About the Competition?

**Grok 4 Fast (3rd place)** - Achieved 1616 ELO with the fastest latency: 3.9s on MSMARCO, 5.9s average overall. Best speed/quality trade-off if you need sub-10s responses.

**Gemini 3 Flash (4th place)** - Outperformed Gemini 3 Pro (8th place) by 92 ELO points despite being marketed as smaller/faster. Google's model lineup is confusing.

**DeepSeek R1 (last place)** - Disappointing 1273 ELO with 17.6% win rate. Whatever it's optimized for, it's not RAG question-answering.

## Recommendations

**Use Opus 4.6 when:**
- You need maximum accuracy on factual queries
- Grounding in source documents is critical
- You can tolerate 10-20s latency for quality
- Budget allows $5/$25 per M tokens

**Consider alternatives when:**
- Long-form synthesis is required → Use GPT-5.1 (superior on Paul Graham essays)
- Comprehensive coverage matters → Use GPT-5.1 (perfect completeness scores)
- Speed is critical → Use Grok 4 Fast (3x faster, only ~10% quality drop)
- Budget constrained → Use Gemini 3 Flash (1/10th the cost, competitive quality)

## Testing Notes

**Methodology**: Pairwise comparison using GPT-5 as judge across 990 total judgments. We reused 1,650 existing judgments between the 11 baseline models and generated 990 new judgments involving Opus 4.6.

**Limitations**:
- Single judge (GPT-5) introduces potential bias
- 30 queries per dataset provides adequate statistical power for rankings but individual matchups have ±10% confidence intervals
- RAG question-answering may not reflect performance on other tasks like coding or creative writing
- Release day testing (Feb 6) may not represent stable production performance

**Models tested**: Opus 4.6, Opus 4.5, GPT-5.1, GPT-5.2, GPT-OSS-120B, DeepSeek R1, Gemini 2.5 Pro, Gemini 3 Flash, Gemini 3 Pro, Grok 4 Fast, GLM 4.6, Qwen3-30B-A3B-Thinking

## Final Thoughts

Opus 4.6 is the strongest all-around RAG model we've tested, but it's not universally best:

- **Best at**: Factual retrieval, document grounding, precision
- **Good at**: Long-form reasoning (but GPT-5.1 is better)
- **Weak at**: Comprehensive coverage for technical content
- **Trade-off**: Premium pricing, moderate latency

If your RAG pipeline emphasizes accuracy and grounding over speed and comprehensiveness, Opus 4.6 is worth the cost. For long-form synthesis or when exhaustive coverage matters, GPT-5.1 remains the better choice.

The 387-point ELO improvement over Opus 4.5 on complex reasoning tasks shows Anthropic is moving fast. We'll continue tracking these models as they evolve.

---

**Benchmark code and data**: [github.com/agentset-ai/llm-leaderboard](https://github.com/yourusername/llm-leaderboard)

**Contact**: [hello@agentset.ai](mailto:hello@agentset.ai)

Last updated: February 6, 2026
