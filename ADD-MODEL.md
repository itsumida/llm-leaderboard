# Add a New Model

## 1. Add to model-info.json

```json
{
  "name": "your-model-name",
  "display_name": "Your Model Name",
  "provider": "Provider",
  "cost_per_1m_input_tokens": 1.00,
  "cost_per_1m_output_tokens": 5.00,
  "context_window": 128000
}
```

## 2. Generate Answers (all datasets)

```bash
cd setup

for dataset in msmarco pg scifact; do
  python3 llm-rag.py \
    --input ../${dataset}-outputs/embed-rerank/top15.jsonl \
    --output-dir ../${dataset}-outputs/llm-rag \
    --models "provider/your-model-name"
done
```

## 3. Run Judge

```bash
for dataset in msmarco pg scifact; do
  python3 judge.py \
    --answers-dir ../${dataset}-outputs/llm-rag \
    --output-dir ../${dataset}-outputs/llm-judge
done
```

## 4. Calculate ELO

```bash
for dataset in msmarco pg scifact; do
  python3 elo.py \
    --judgments ../${dataset}-outputs/llm-judge/judgments.jsonl \
    --output ../${dataset}-outputs/llm-elo/elo_ratings.json
done
```

## 5. Generate Benchmarks

```bash
python3 aggregate_datasets.py
```

This produces `benchmarks.json` with normalized names and comparisons.

## 6. Add Name Mapping (if needed)

If your model has a provider prefix in answers (e.g., `openai-gpt-6`), add mapping in `aggregate_datasets.py`:

```python
NAME_MAPPING = {
    ...
    "openai-gpt-6": "gpt-6",
}
```
