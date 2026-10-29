import os
from functools import lru_cache
from typing import AsyncIterator
from google import genai


@lru_cache
def get_client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in backend/.env")
    return genai.Client(api_key=key)


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
    stream = await get_client().aio.models.generate_content_stream(
        model="gemini-3.8-flash",
        contents=build_prompt(query, chunks),
        config=genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
        ),
    )
    async for chunk in stream:
        if chunk.text:
            yield chunk.text