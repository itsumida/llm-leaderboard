# I Tested Claude Opus 4.6 Against 11 Top Models on RAG Tasks. Here's What I Found.

On February 5, 2026, Anthropic released Claude Opus 4.6 with [impressive claims](https://www.anthropic.com/news/claude-opus-4-6): extended context to 1M tokens, improved agentic capabilities, and state-of-the-art coding performance. I wanted to see how it actually performs on real RAG (Retrieval Augmented Generation) tasks compared to the competition.

I ran Opus 4.6 through my LLM RAG benchmark across three datasets: MSMARCO (information retrieval), Paul Graham essays (long-form reasoning), and SciFact (scientific fact-checking). The model competed against 11 other frontier models including GPT-5.1, GPT-5.2, DeepSeek R1, Gemini 3, and the previous Opus 4.5.

**TL;DR: Opus 4.6 ranked #1 overall with a 74.8% win rate, but GPT-5.1 beat it on long-form reasoning tasks, and it struggled with scientific completeness.**

## The Setup

I evaluate models using a pairwise comparison methodology inspired by Chatbot Arena:

1. **Answer Generation**: Each model generates answers to 30 queries per dataset (90 total), using top-15 retrieved documents as context
2. **Pairwise Judging**: GPT-5 (via Azure) compares every pair of answers for each query
3. **ELO Calculation**: Standard ELO rating system (K=32, starting at 1500)
4. **Quality Metrics**: 5-point scale evaluation (correctness, faithfulness, grounding, relevance, completeness)

The datasets:
- **MSMARCO**: 30 factual queries ("What is the population of Tokyo?")
- **Paul Graham (PG)**: 30 queries about startup/tech philosophy requiring synthesis
- **SciFact**: 30 scientific claims requiring verification

All answers were generated fresh on February 6, 2026. The competition included:
- Claude Opus 4.5, 4.6 (Anthropic)
- GPT-5.1, 5.2, GPT-OSS-120B (OpenAI)
- DeepSeek R1
- Gemini 2.5 Pro, 3 Flash, 3 Pro (Google)
- Grok 4 Fast (xAI)
- GLM 4.6 (Zhipu AI)
- Qwen3-30B-A3B Thinking (Alibaba)

## Overall Results: Opus 4.6 Takes the Crown

**Final ELO Rankings (Overall)**
```
Rank  Model                   ELO      Win Rate  Avg Latency
1.    Claude Opus 4.6        1780.28   74.8%     11.5s
2.    GPT-5.1                1716.03   65.3%     16.2s
3.    Grok 4 Fast            1616.31   54.3%      5.9s
4.    Gemini 3 Flash         1569.71   56.8%      7.8s
5.    GPT-5.2                1558.61   41.8%      5.4s
6.    Claude Opus 4.5        1548.65   48.5%      8.3s
7.    GLM 4.6                1487.40   37.0%     33.1s
8.    Gemini 3 Pro           1477.17   40.3%     17.9s
9.    Gemini 2.5 Pro         1376.01   29.6%     15.2s
10.   Qwen3-30B Thinking     1321.58   27.6%     12.3s
11.   GPT-OSS-120B           1274.39   16.6%     11.2s
12.   DeepSeek R1            1273.87   17.6%     18.3s
```

Claude Opus 4.6 achieved a dominant **740 wins** out of 990 total pairwise judgments (74.8% win rate), with only 93 losses and 157 ties. The 64-point ELO lead over second-place GPT-5.1 is substantial.

**But here's the surprise: GPT-5.1 actually beat Opus 4.6 head-to-head.**

## The GPT-5.1 Paradox

Looking at the direct comparison between Opus 4.6 and GPT-5.1:

```python
# Overall head-to-head (90 comparisons)
opus_4_6_vs_gpt_5_1 = {
    "wins": 38,
    "losses": 27,
    "ties": 25,
    "win_rate": 0.422  # 42.2% - LOSING!
}
```

Wait, what? Opus 4.6 has a higher overall ELO but *loses* to GPT-5.1? This is the ELO paradox in action—overall rankings don't guarantee head-to-head dominance.

Breaking it down by dataset reveals the story:

**MSMARCO (Factual Queries)**
- Opus 4.6 wins: 20
- GPT-5.1 wins: 3
- Ties: 7

**Paul Graham Essays (Long-form Reasoning)**
- Opus 4.6 wins: 8
- GPT-5.1 wins: 20 ← **GPT-5.1 dominates here**
- Ties: 2

**SciFact (Scientific Verification)**
- Opus 4.6 wins: 10
- GPT-5.1 wins: 4
- Ties: 16

On the Paul Graham dataset, GPT-5.1 achieved an ELO of **1869.98** (highest across all models/datasets), while Opus 4.6 scored 1806.56. This suggests **GPT-5.1 has superior long-form reasoning and synthesis abilities** when dealing with nuanced, philosophical content.

Meanwhile, Opus 4.6 crushed GPT-5.1 on MSMARCO's factual queries, achieving the highest MSMARCO ELO overall (1783.19) with an 81.2% win rate.

## Opus 4.6 vs. Opus 4.5: A Clear Upgrade

Against its predecessor, Opus 4.6 showed significant improvement:

```
Head-to-head: 54-5 (60% win rate, 31 ties)

Dataset Performance:
MSMARCO:  4.6 wins 15-3  (ELO: 1783 vs 1688)
PG:       4.6 wins 26-0  (ELO: 1806 vs 1419) ← Massive improvement
SciFact:  4.6 wins 13-2  (ELO: 1751 vs 1537)
```

The biggest improvement came on Paul Graham essays, where 4.5 badly underperformed (1419 ELO, 7th place). Opus 4.6 nearly jumped 400 ELO points on this dataset, suggesting Anthropic specifically improved long-context reasoning and synthesis.

Interestingly, while 4.6 improved, GPT-5.1 still maintained its edge on PG essays with an even higher ELO.

## Dataset-Specific Performance: Where Opus 4.6 Excels and Struggles

### MSMARCO: Opus 4.6 Dominates Factual Retrieval

On MSMARCO, Opus 4.6 was nearly unbeatable:

```
ELO: 1783.19 (1st place)
Win-Loss-Tie: 268-16-46 (81.2% win rate)
Quality Metrics: Perfect 5.0 across all dimensions
Latency: 7.7s average (3.7s min, 12.5s max)
```

Head-to-head highlights:
- vs GPT-5.2: **28-1** (one of the most lopsided matchups)
- vs Gemini 3 Flash: **19-4**
- vs DeepSeek R1: **25-1**
- vs GPT-OSS-120B: **30-0** (perfect sweep)

Only GPT-5.1 put up a real fight (3-20-7), and even that was decisive.

**Why MSMARCO?** This dataset rewards precision and factual accuracy with clear right/wrong answers. Opus 4.6's training clearly optimized for grounding answers strictly in provided context without hallucination.

### Paul Graham Essays: GPT-5.1 Reigns Supreme

This is where things get interesting:

```
GPT-5.1:       ELO 1869.98 (1st) - 282 wins
Opus 4.6:      ELO 1806.56 (2nd) - 255 wins
Grok 4 Fast:   ELO 1763.14 (3rd) - 238 wins
Gemini 3 Flash: ELO 1702.44 (4th) - 197 wins
```

Opus 4.6 had **52 losses** on PG essays (vs. 16 on MSMARCO, 25 on SciFact), with most coming from GPT-5.1 and other top models. The quality metrics were still perfect 5.0, but the answers apparently lacked something judges preferred in GPT-5.1's responses.

**Latency was also highest here**: Opus 4.6 averaged **16.8 seconds** per query on PG (vs. 7.7s on MSMARCO), ranging from 11.2s to 26s. The longer, more complex queries likely triggered deeper reasoning.

GPT-5.1 was even slower (29s average, max 43.9s), suggesting both models engaged extended thinking for these queries.

### SciFact: The Completeness Problem

SciFact revealed Opus 4.6's first quality metric weakness:

```
ELO: 1751.09 (1st place, but closer competition)
Win-Loss-Tie: 217-25-88 (65.8% win rate, lowest of 3 datasets)
Quality Metrics:
  - Correctness: 4.55 ← First non-perfect score
  - Faithfulness: 4.64
  - Grounding: 4.64
  - Relevance: 5.0
  - Completeness: 4.36 ← Weakest dimension
  - Overall: 4.64
```

Despite ranking 1st on SciFact, Opus 4.6 had **88 ties**—nearly 27% of comparisons. This suggests many head-to-heads were too close to call, indicating less decisive superiority.

The completeness score of 4.36 is notable. Scientific verification requires comprehensive coverage of evidence, nuanced hedging, and addressing counterarguments. Opus 4.6 may have been overly concise or missed supporting details.

Compare to GPT-5.1 on SciFact: perfect 5.0 across all metrics despite ranking 2nd (ELO 1630.98). GPT-5.1's answers were apparently more complete even when judges didn't prefer them overall.

## The Competition: Winners and Losers

### Grok 4 Fast: The Speed Demon (3rd Place)

Grok 4 Fast surprised me with 3rd overall (ELO 1616.31) and the **fastest latency**: 3.9s on MSMARCO, 9.1s on PG, 4.5s on SciFact (average 5.9s).

It beat Opus 4.6 head-to-head 13 times (out of 90), mostly on PG essays and SciFact. For latency-sensitive applications, Grok 4 Fast offers 50% faster responses than Opus 4.6 with only a modest quality drop.

### DeepSeek R1: The Disappointing Overhype

DeepSeek R1 placed **last** (ELO 1273.87, 17.6% win rate) despite massive hype:

```
Overall: 174 wins, 690 losses, 126 ties
vs Opus 4.6: 4-73-13 (4.4% win rate)
```

It got demolished across all datasets:
- MSMARCO: ELO 1131.77 (last place by 140 points)
- PG: ELO 1268.04 (10th place)
- SciFact: ELO 1421.80 (9th place)

The model averaged **18.3s latency** (slow) and scored only 4.79-4.98 on quality metrics. Whatever DeepSeek R1 is good at, it's not RAG-style question answering with short context windows.

### The Gemini 3 Surprise

Gemini 3 Flash (4th place, ELO 1569.71) significantly outperformed Gemini 3 Pro (8th place, ELO 1477.17) despite being marketed as the smaller, faster model.

Head-to-head: Gemini 3 Flash beat 3 Pro **44-27** with nearly identical latency (7.8s vs 17.9s—wait, Pro is *slower*?).

Gemini 2.5 Pro (9th place) performed even worse, suggesting Google's newer models regressed on RAG tasks. This is concerning.

### The Claude Opus 4.5 Collapse on PG Essays

One of the strangest findings: Opus 4.5 had a **1419.65 ELO on Paul Graham essays**—7th place, worse than models ranked far below it overall.

```
PG Dataset Results for Opus 4.5:
- Wins: 97
- Losses: 217 ← Destroyed by most models
- Ties: 16
- Win rate: 29.4%
```

It lost head-to-head matchups to:
- Opus 4.6: 0-26 (perfect sweep)
- GPT-5.1: 2-27
- Grok 4 Fast: 5-25
- Gemini 3 Flash: 2-26
- Even Qwen3-30B: 5-25

Something about Opus 4.5's training made it terrible at long-form synthesis tasks, which Anthropic clearly fixed in 4.6.

## Latency Analysis: Speed vs. Quality Trade-offs

**Fastest Models:**
1. GPT-5.2: 5.4s average (but 5th place overall)
2. Grok 4 Fast: 5.9s (3rd place) ← Best speed/quality ratio
3. Gemini 3 Flash: 7.8s (4th place)

**Slowest Models:**
1. GLM 4.6: 33.1s (7th place—slow AND mediocre)
2. DeepSeek R1: 18.3s (last place—slow AND worst)
3. Gemini 3 Pro: 17.9s (8th place)

Opus 4.6's 11.5s average puts it mid-pack, but with huge variance:
- MSMARCO: 7.7s (factual retrieval is fast)
- SciFact: 10.2s (moderate complexity)
- PG: 16.8s (extended reasoning kicks in)

The latency-ELO plot shows **no correlation** between speed and quality. Grok 4 Fast and GPT-5.2 are fast but diverge in ELO (1616 vs 1558). GLM 4.6 is slow but mediocre. Opus 4.6 is moderate speed with top performance.

**Insight**: Query complexity matters more than base model speed. Both Opus 4.6 and GPT-5.1 adaptively slow down on harder queries (PG essays), suggesting internal reasoning/verification loops.

## Quality Metrics: The Devil in the Details

Most models scored perfect or near-perfect 5.0 on MSMARCO and PG essays. SciFact revealed differences:

**SciFact Completeness Scores:**
```
GPT-5.1:           5.00 (most complete)
GPT-5.2:           4.82
Gemini 3 Flash:    4.91
DeepSeek R1:       4.91
Opus 4.6:          4.36 ← Notable weakness
Opus 4.5:          4.55
GPT-OSS-120B:      4.55 (lowest overall: 4.62)
```

This suggests Opus 4.6 optimizes for conciseness over exhaustiveness. For scientific or technical content requiring comprehensive coverage, GPT-5.1 may be preferable despite lower overall ELO.

## Token Usage: Opus 4.6 is Verbose

Looking at token consumption across datasets:

**MSMARCO (30 queries):**
```
Opus 4.6:   Input: 52,536  Output: 10,460  Total: 62,996
GPT-5.1:    Input: 47,219  Output: 10,771  Total: 57,990
GPT-5.2:    Input: 47,219  Output:  3,588  Total: 50,807 ← Most concise
```

**Paul Graham (30 queries):**
```
Opus 4.6:   Input: 159,188  Output: 18,896  Total: 178,084
GPT-5.1:    Input: 147,104  Output: 46,620  Total: 193,724 ← Most verbose
Grok 4:     Input: 152,864  Output: 20,853  Total: 173,717
```

**SciFact (30 queries):**
```
Opus 4.6:   Input: 193,498  Output:  9,367  Total: 202,865
GPT-5.1:    Input: 167,845  Output: 14,255  Total: 182,100
```

Opus 4.6 had the highest input token count on SciFact (193,498), likely from its extended context processing, but lower output tokens (9,367) than GPT-5.1 (14,255)—explaining the completeness score gap.

**Cost implications at Opus 4.6 pricing ($5/M input, $25/M output):**
- MSMARCO: $0.52
- PG: $1.27
- SciFact: $1.20
- **Total for 90 queries: $2.99**

GPT-5.1 (assuming similar pricing) would cost slightly more due to higher output tokens on PG essays.

## Methodology Notes and Limitations

### Judge Model Bias

I used **GPT-5 (via Azure)** as the judge. This could introduce bias favoring OpenAI models, but the results don't support this—Opus 4.6 dominated overall despite losing to GPT-5.1 on PG essays.

Using a single judge is standard practice (Chatbot Arena uses Claude, LMSYS uses various judges), but multi-judge ensembles would be more robust.

### Dataset Size

30 queries per dataset (90 total) is small but produces 990 pairwise judgments. Statistical significance is reasonable for overall rankings but individual matchup details have higher variance.

### Judgment Caching

I reused all existing judgments between the 11 other models (1,650 judgments) and only generated 990 new judgments involving Opus 4.6. This saved ~$15 in API costs and ensures consistency, but means judgments were made across multiple days with potentially different judge model versions.

### Opus 4.6 Model ID

I used `anthropic/claude-opus-4-6` via OpenRouter. Anthropic's API may have a different version or tuning. Release day performance can also vary due to infrastructure scaling.

### The "Error Response" Incident

Initial runs had 30/90 queries fail with 402 Payment Required errors (OpenRouter credit exhaustion). After rerunning, all queries succeeded. I filtered out the 330 judgments involving error responses and regenerated them with correct answers.

This highlights a real-world challenge: evaluating frontier models on release day often means dealing with API instability, rate limits, and billing issues.

## Key Takeaways

1. **Opus 4.6 is the strongest all-around RAG model** with a 74.8% win rate and 1780 ELO, but it's not perfect.

2. **GPT-5.1 beats Opus 4.6 on long-form reasoning tasks** (Paul Graham essays), achieving the highest single-dataset ELO ever recorded (1869.98). If your use case involves synthesis, analysis, or philosophical reasoning, GPT-5.1 is superior.

3. **Opus 4.6 dominates factual retrieval** (MSMARCO) with 81.2% win rate and perfect quality metrics. For grounded QA over documents, it's the best choice.

4. **Completeness is Opus 4.6's weakness**—it scored only 4.36 on SciFact completeness vs. 5.0 for GPT-5.1. When comprehensive coverage matters (technical docs, scientific writing), consider alternatives.

5. **Opus 4.6 is a huge upgrade over 4.5**, especially on long-context reasoning (+387 ELO on PG essays).

6. **Grok 4 Fast offers the best speed/quality trade-off** at 5.9s average latency with 3rd place overall. If latency matters, it's a strong alternative to Opus 4.6.

7. **DeepSeek R1 is overhyped for RAG tasks**—it placed last overall with a 17.6% win rate and poor performance across all datasets.

8. **Gemini 3's model lineup is confusing**—the "Flash" model beats "Pro" despite being marketed as smaller/faster. Gemini 2.5 Pro regressed further.

9. **Latency doesn't predict quality**—models adaptively adjust inference time based on query complexity. Opus 4.6 ranges from 3.7s to 26s depending on dataset.

10. **The RAG leaderboard differs from general LLM rankings**—models optimized for chat or coding may underperform on retrieval-grounded QA tasks. Always evaluate on your specific use case.

## Reproducing This Analysis

All code and data is available in my [llm-leaderboard](https://github.com/yourusername/llm-leaderboard) repository:

```bash
# Generate answers for a new model
python3 setup/llm-rag.py \
  --input msmarco-outputs/embed-rerank/msmarco/top15.jsonl \
  --output-dir msmarco-outputs/llm-rag \
  --models "anthropic/claude-opus-4-6"

# Generate judgments (reusing existing comparisons)
python3 setup/judge.py \
  --answers-dir msmarco-outputs/llm-rag \
  --output-dir msmarco-outputs/llm-judge \
  --existing-judgments msmarco-outputs/llm-judge/judgments.jsonl \
  --max-workers 10

# Calculate ELO ratings
python3 setup/elo.py \
  --judgments msmarco-outputs/llm-judge/judgments.jsonl \
  --output msmarco-outputs/llm-elo/elo_ratings.json

# Aggregate across datasets
python3 setup/aggregate_datasets.py
```

The pipeline uses:
- **OpenRouter API** for model inference (supports 100+ models)
- **Azure GPT-5** for pairwise judging
- **Standard ELO rating** (K=32, base 1500)

I spent about $25 total evaluating Opus 4.6:
- $5 for answer generation (90 queries across 3 datasets)
- $20 for judgment generation (990 new pairwise comparisons)

## What's Next?

I'm planning to add:
- **GPT-5.3-Codex** (announced Feb 5, not yet available via API)
- **Multi-judge ensembles** (Claude, Gemini, GPT-5) to reduce bias
- **Human evaluation** on a subset to validate judge agreement
- **Cost-normalized rankings** (ELO per dollar)

If you're interested in adding your model to the leaderboard, open an issue or PR!

---

*Have questions about the methodology or want to discuss the results? Find me on Twitter [@yourusername](https://twitter.com/yourusername) or comment below.*

*Updated: February 6, 2026*
