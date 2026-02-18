#!/usr/bin/env python3
"""Quick test of Claude Sonnet 4.6 with reasoning mode via OpenRouter"""

import sys
sys.path.insert(0, 'setup')

from openrouter_client import call_openrouter

# Test with simple query
messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"}
]

print("Testing Claude Sonnet 4.6 with reasoning mode...")
print("Model: anthropic/claude-sonnet-4.6\n")

try:
    result = call_openrouter(
        model="anthropic/claude-sonnet-4.6",
        messages=messages,
        temperature=0.0,
        max_tokens=500
    )

    print("✅ Success!")
    print(f"\nResponse: {result['content']}")
    print(f"\nToken usage:")
    print(f"  Input tokens: {result['input_tokens']}")
    print(f"  Output tokens: {result['output_tokens']}")

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
