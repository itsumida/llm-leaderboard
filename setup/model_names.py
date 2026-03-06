#!/usr/bin/env python3
"""
Model name normalization module.
Ensures all benchmark entries use exact names from model-info2.json.
"""

import json
from pathlib import Path

# Canonical model names from model-info2.json
# These are the ONLY valid names for benchmark entries
CANONICAL_NAMES = [
    "claude-opus-4-5-20251101",
    "deepseek-r1",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-oss-120b",
    "qwen3-30b-a3b-thinking-2507",
    "grok-4-fast",
    "glm-4.6",
    "claude-opus-4-6",
    "claude-sonnet-4.6",
    "claude-sonnet-4.5",
    "gpt-5.4",
    "gpt-5.4-pro",
]

# Mapping from various formats to canonical names
# Keys are patterns that might appear in benchmark data
NAME_MAPPING = {
    # Anthropic models
    "anthropic-claude-opus-4-6": "claude-opus-4-6",
    "anthropic-claude-opus-4.5": "claude-opus-4-5-20251101",
    "anthropic-claude-opus-4-5-20251101": "claude-opus-4-5-20251101",
    "anthropic-claude-sonnet-4.6": "claude-sonnet-4.6",
    "anthropic-claude-sonnet-4.5": "claude-sonnet-4.5",

    # Google models
    "google-gemini-2.5-pro": "gemini-2.5-pro",
    "google-gemini-3-flash-preview": "gemini-3-flash-preview",
    "google-gemini-3-pro-preview": "gemini-3-pro-preview",

    # OpenAI models
    "openai-gpt-5.1": "gpt-5.1",
    "openai-gpt-5.2": "gpt-5.2",
    "openai-gpt-oss-120b": "gpt-oss-120b",

    # xAI models
    "x-ai-grok-4-fast": "grok-4-fast",

    # Zhipu AI models
    "z-ai-glm-4.6": "glm-4.6",

    # DeepSeek models
    "deepseek-deepseek-r1": "deepseek-r1",

    # Qwen models
    "qwen-qwen3-30b-a3b-thinking-2507": "qwen3-30b-a3b-thinking-2507",

    # Already canonical (pass through)
    "gpt-5.4": "gpt-5.4",
    "gpt-5.4-pro": "gpt-5.4-pro",
}


def normalize_model_name(name: str) -> str:
    """
    Convert any model name format to canonical name from model-info2.json.

    Args:
        name: Model name in any format (e.g., "anthropic-claude-opus-4-6")

    Returns:
        Canonical name (e.g., "claude-opus-4-6")
    """
    # Check direct mapping first
    if name in NAME_MAPPING:
        return NAME_MAPPING[name]

    # Check if already canonical
    if name in CANONICAL_NAMES:
        return name

    # Try to infer canonical name by removing common prefixes
    prefixes = ["anthropic-", "google-", "openai-", "x-ai-", "z-ai-", "deepseek-", "qwen-"]
    for prefix in prefixes:
        if name.startswith(prefix):
            stripped = name[len(prefix):]
            if stripped in CANONICAL_NAMES:
                return stripped
            # Check if stripped version is in mapping
            if stripped in NAME_MAPPING:
                return NAME_MAPPING[stripped]

    # Return original if no mapping found (will be flagged as unknown)
    print(f"WARNING: Unknown model name '{name}' - not in canonical list")
    return name


def normalize_comparisons(comparisons: dict) -> dict:
    """
    Normalize all comparison keys to canonical names.

    Args:
        comparisons: Dict with model names as keys

    Returns:
        Dict with canonical names as keys
    """
    normalized = {}
    for opponent, stats in comparisons.items():
        canonical_name = normalize_model_name(opponent)
        normalized[canonical_name] = stats
    return normalized


def load_canonical_names_from_file(model_info_path: str = None) -> list:
    """
    Load canonical names from model-info.json file.
    Falls back to hardcoded list if file not found.
    """
    if model_info_path is None:
        # Try to find model-info.json relative to this file
        model_info_path = Path(__file__).parent.parent / "model-info.json"

    try:
        with open(model_info_path) as f:
            models = json.load(f)
        return [m["name"] for m in models]
    except FileNotFoundError:
        return CANONICAL_NAMES


if __name__ == "__main__":
    # Test the normalization
    test_names = [
        "anthropic-claude-opus-4-6",
        "google-gemini-3-flash-preview",
        "openai-gpt-5.1",
        "gpt-5.4",
        "x-ai-grok-4-fast",
        "deepseek-deepseek-r1",
    ]

    print("Model name normalization test:")
    for name in test_names:
        canonical = normalize_model_name(name)
        print(f"  {name:40} -> {canonical}")
