import anthropic
import os
from typing import AsyncIterator

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["content"] for c in chunks)
    return f"""You are a precise knowledge base assistant. Answer the user's question using ONLY the context provided below. If the answer is not in the context, say so clearly.

<context>
{context}
</context>

<question>
{query}
</question>

Answer:"""

async def stream_answer(query: str, chunks: list[dict]) -> AsyncIterator[str]:
    """Stream the LLM response using Anthropic's streaming API."""
    prompt = build_prompt(query, chunks)
    
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text_chunk in stream.text_stream:
            yield text_chunk