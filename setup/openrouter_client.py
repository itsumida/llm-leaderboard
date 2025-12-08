"""
Simple OpenRouter client using requests library
Single unified client for all LLM interactions
"""

import os
import requests
import logging
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv

# Load .env from parent directory (repo root)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 500,
    site_url: str = "https://github.com/yourusername/llm-leaderboard",
    app_name: str = "llm-leaderboard"
) -> Dict:
    """
    Call OpenRouter API with any model

    Args:
        model: Model ID (e.g., 'openai/gpt-5', 'anthropic/claude-opus-4.5')
        messages: List of message dicts with 'role' and 'content'
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        site_url: Optional site URL for OpenRouter rankings
        app_name: Optional app name for OpenRouter rankings

    Returns:
        Dict with 'content', 'input_tokens', 'output_tokens'
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Optional headers for OpenRouter rankings
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Enable reasoning mode for models that support it
    if any(x in model.lower() for x in ["gemini-3", "gemini-2.5", "gpt-oss", "gpt-5", "qwen3-30b-a3b-thinking", "grok-code", "glm-4"]):
        payload["reasoning"] = {"enabled": True}

    try:
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        data = response.json()

        # Extract content and usage
        message = data["choices"][0]["message"]
        content = message.get("content", "")
        usage = data.get("usage", {})

        # For Gemini 3 with reasoning mode, content should contain the final answer
        # reasoning_details contains the thinking process (but we don't need it for single-turn QA)

        return {
            "content": content,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0)
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter API error for {model}: {e}")
        raise


def generate_answer(model: str, query: str, system_prompt: str = "", temperature: float = 0.0, max_tokens: int = 500) -> Dict:
    """
    Generate answer for RAG QA

    Args:
        model: Model ID
        query: User query
        system_prompt: Optional system prompt with context
        temperature: Temperature
        max_tokens: Max tokens

    Returns:
        Dict with 'answer', 'input_tokens', 'output_tokens'
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": query})

    result = call_openrouter(model, messages, temperature=temperature, max_tokens=max_tokens)

    return {
        "answer": result["content"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"]
    }


def judge_pairwise(
    answer_a: str,
    answer_b: str,
    query: str,
    context_docs: str,
    model_a_name: str,
    model_b_name: str,
    judge_model: str = "openai/gpt-5"
) -> str:
    """
    Judge which answer is better (A, B, or TIE)

    Args:
        answer_a: First answer
        answer_b: Second answer
        query: Original query
        context_docs: Context documents
        model_a_name: Name of model A
        model_b_name: Name of model B
        judge_model: Model to use for judging

    Returns:
        'A', 'B', or 'TIE'
    """
    system_content = f"""You are evaluating answers to queries based on provided context documents.

Context Documents:
{context_docs}

Evaluation Criteria - Select the answer that is:
1. More factually correct
2. Better supported by the provided context
3. Less speculative or fabricated (lower hallucination risk)
4. More comprehensive and thorough
5. More directly responsive to the specific question

Disregard style, formatting, and verbosity.

Reply strictly with one of: A, B, or TIE."""

    prompt = f"""Query: {query}

Answer A ({model_a_name}):
{answer_a}

Answer B ({model_b_name}):
{answer_b}

Reply: A, B, or TIE"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]

    result = call_openrouter(judge_model, messages, temperature=0, max_tokens=3)
    judgment = result["content"].strip().upper()

    if judgment in ["A", "B", "TIE"]:
        return judgment
    return "TIE"


def score_answer(
    answer: str,
    query: str,
    context_docs: str,
    judge_model: str = "openai/gpt-5"
) -> Dict[str, int]:
    """
    Score a single answer on 5 dimensions (1-5 scale)

    Args:
        answer: Answer to score
        query: Original query
        context_docs: Context documents
        judge_model: Model to use for scoring

    Returns:
        Dict with scores: correctness, faithfulness, grounding, relevance, completeness
    """
    system_content = f"""You are a STRICT evaluator. Use the full 1-5 scale. Be critical - reserve 5 for exceptional answers.

Context Documents:
{context_docs}

Rate on 1-5 scale:
- Correctness: Factual accuracy
- Faithfulness: Claims supported by context (no hallucinations)
- Grounding: Citations and evidence quality
- Relevance: Addresses the query
- Completeness: Coverage and thoroughness

Reply ONLY with valid JSON: {{"correctness": X, "faithfulness": X, "grounding": X, "relevance": X, "completeness": X}}"""

    prompt = f"""Query: {query}

Answer: {answer}

Rate 1-5 on each dimension:"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]

    result = call_openrouter(judge_model, messages, temperature=0, max_tokens=100)

    import json
    try:
        scores = json.loads(result["content"])

        # Validate scores are 1-5
        for key in ["correctness", "faithfulness", "grounding", "relevance", "completeness"]:
            if key not in scores:
                scores[key] = 3
            scores[key] = max(1, min(5, int(scores[key])))

        return scores

    except json.JSONDecodeError:
        logger.warning(f"Failed to parse scores, using defaults")
        return {
            "correctness": 3,
            "faithfulness": 3,
            "grounding": 3,
            "relevance": 3,
            "completeness": 3
        }
