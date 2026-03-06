# Add a New Model

## Step 1: Add to model-info.json

```json
{
  "name": "your-model-name",
  "display_name": "Your Model Name",
  "provider": "Provider",
  "license": "Proprietary",
  "cost_per_1m_input_tokens": 1.00,
  "cost_per_1m_output_tokens": 5.00,
  "release_date": "2026-03-01",
  "context_window": 128000,
  "about_model": "Brief description."
}
```

**Name format:** Use simple names without provider prefix (e.g., `gpt-5.1` not `openai-gpt-5.1`)

## Step 2: Generate Answers

```bash
cd setup

# Run for all 3 datasets
for dataset in msmarco pg scifact; do
  python3 llm-rag.py \
    --input ../${dataset}-outputs/embed-rerank/top15.jsonl \
    --output-dir ../${dataset}-outputs/llm-rag \
    --models "provider/your-model-name"
done
```

**OpenRouter model ID format:** `provider/model-name` (e.g., `openai/gpt-5.1`)

## Step 3: Run Judge

```bash
for dataset in msmarco pg scifact; do
  python3 judge.py \
    --answers-dir ../${dataset}-outputs/llm-rag \
    --output-dir ../${dataset}-outputs/llm-judge
done
```

## Step 4: Calculate ELO

```bash
for dataset in msmarco pg scifact; do
  python3 elo.py \
    --judgments ../${dataset}-outputs/llm-judge/judgments.jsonl \
    --output ../${dataset}-outputs/llm-elo/elo_ratings.json
done
```

## Step 5: Regenerate Leaderboard

```bash
python3 aggregate_datasets.py
python3 add_comparisons.py
```

## Step 6: Update model_names.py

Add mapping if your model has a provider prefix in answers:

```python
# In setup/model_names.py
NAME_MAPPING = {
    ...
    "provider-your-model-name": "your-model-name",
}
```

## Checklist

- [ ] Added to `model-info.json`
- [ ] Generated answers for all 3 datasets
- [ ] Ran judge on all datasets
- [ ] Calculated ELO ratings
- [ ] Regenerated `benchmarks.json`
- [ ] Added name mapping if needed

## Tips

- Use `--models "model1,model2,model3"` to run multiple models at once
- Check `*_answers.jsonl` files to verify responses aren't empty
- Judge compares new model against ALL existing models
