# LLM RAG Leaderboard

Evaluation framework for comparing LLM performance on RAG (Retrieval Augmented Generation) tasks.

## Current Results

**Top 3 Models (Aggregated across 3 datasets):**
1. **OpenAI GPT-5.1** - ELO: 1725.65
2. **X.AI Grok-4-Fast** - ELO: 1652.62
3. **Anthropic Claude Opus 4.5** - ELO: 1620.53

Evaluated on: MS MARCO, Paul Graham, SciFact

## Quick Start

### Adding a New Dataset

1. **Prepare dataset files:**
   ```
   datasets/YOUR_DATASET/
   ├── corpus.jsonl          # Documents
   └── queries.jsonl         # Queries to evaluate
   ```

2. **Run evaluation pipeline:**
   ```bash
   cd setup

   # Step 1: Embed & Rerank
   python3 embed-rerank.py \
     --corpus ../datasets/YOUR_DATASET/corpus.jsonl \
     --queries ../datasets/YOUR_DATASET/queries.jsonl \
     --output-dir ../YOUR_DATASET-outputs/embed-rerank

   # Step 2: Generate LLM answers
   python3 llm-rag.py \
     --input ../YOUR_DATASET-outputs/embed-rerank/top15.jsonl \
     --output-dir ../YOUR_DATASET-outputs/llm-rag \
     --models "anthropic/claude-opus-4.5,openai/gpt-5.1,x-ai/grok-4-fast"

   # Step 3: Judge answers
   python3 judge.py \
     --answers-dir ../YOUR_DATASET-outputs/llm-rag \
     --output-dir ../YOUR_DATASET-outputs/llm-judge

   # Step 4: Calculate ELO
   python3 elo.py \
     --judgments ../YOUR_DATASET-outputs/llm-judge/judgments.jsonl \
     --output ../YOUR_DATASET-outputs/llm-elo/elo_ratings.json

   # Step 5: Generate leaderboard
   python3 results.py \
     --answers-dir ../YOUR_DATASET-outputs/llm-rag \
     --metrics ../YOUR_DATASET-outputs/llm-judge/metrics.jsonl \
     --elo ../YOUR_DATASET-outputs/llm-elo/elo_ratings.json \
     --judgments ../YOUR_DATASET-outputs/llm-judge/judgments.jsonl \
     --output-dir ../YOUR_DATASET-outputs/llm-elo

   # Step 6: Visualize
   python3 visualize_leaderboard.py \
     --input ../YOUR_DATASET-outputs/llm-elo/leaderboard.json \
     --output ../YOUR_DATASET-outputs/llm-elo/leaderboard.png
   ```

### Adding a New LLM Model

1. **Add model to the list:**

   Edit `setup/models_9.json`:
   ```json
   {
     "models": [
       {
         "id": "provider/model-name",
         "display_name": "provider-model-name"
       }
     ]
   }
   ```

2. **Or specify directly in command:**
   ```bash
   python3 llm-rag.py \
     --input ../datasets/DATASET/top15.jsonl \
     --output-dir ../outputs/llm-rag \
     --models "anthropic/claude-opus-4.5,YOUR-NEW-MODEL"
   ```

3. **Ensure OpenRouter API key is set:**
   ```bash
   # In .env file
   OPENROUTER_API_KEY=your_key_here
   ```

4. **Rerun evaluation pipeline** (steps 2-6 from dataset section)

## Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys in .env
OPENROUTER_API_KEY=xxx
AZURE_API_KEY=xxx
AZURE_RESOURCE_NAME=xxx
AZURE_DEPLOYMENT_ID=gpt-5
```

## Output Structure

```
DATASET-outputs/
├── embed-rerank/        # Retrieval results
├── llm-rag/            # Model answers
├── llm-judge/          # Judgments & metrics
└── llm-elo/            # ELO ratings & leaderboard
```

## Aggregating Results

To combine multiple datasets:

```bash
cd setup
python3 aggregate_datasets.py
python3 visualize_leaderboard.py \
  --input ../aggregated_leaderboard_wrapped.json \
  --output ../aggregated_leaderboard.png
```

## Judge Configuration

- **Judge Model:** Azure GPT-5
- **Evaluation Dimensions:** Correctness, Faithfulness, Grounding, Relevance, Completeness
- **Comparison Method:** Pairwise judgments for ELO calculation
