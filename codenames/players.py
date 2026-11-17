"""Spymasters and guessers. LLM players use any OpenAI-compatible API, so the
same code works with Ollama (local), Inworld, OpenRouter, OpenAI, vLLM, ..."""
from __future__ import annotations

import os
import random
import threading

from .board import Board
from .parsing import parse_guesses
from .prompts import guesser_messages, spymaster_messages

# provider -> (default base_url, env var holding the API key, auth scheme)
PROVIDERS = {
    "ollama": ("http://localhost:11434/v1", None, "bearer"),
    "inworld": ("https://api.inworld.ai/v1", "INWORLD_API_KEY", "basic"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "bearer"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "bearer"),
}


class LLMClient:
    def __init__(self, model: str, provider: str = "ollama", base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.7, max_tokens: int = 200):
        from openai import OpenAI  # lazy import: not needed for human/random play

        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}, choose from {list(PROVIDERS)}")
        default_url, key_env, auth = PROVIDERS[provider]
        key = api_key or (os.getenv(key_env) if key_env else None) or "none"
        if key_env and key == "none":
            raise SystemExit(f"Set the environment variable {key_env} with your {provider} API key.")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            base_url=base_url or default_url,
            api_key=key,
            # Inworld expects 'Authorization: Basic <key>' instead of 'Bearer <key>'
            default_headers={"Authorization": f"Basic {key}"} if auth == "basic" else None,
            max_retries=5,  # back off and retry on rate limits / transient errors
        )

    def chat(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


# ---------------- spymasters: return raw text ----------------

class LLMSpymaster:
    def __init__(self, client: LLMClient):
        self.client = client

    def give_clue(self, board: Board) -> str:
        return self.client.chat(spymaster_messages(board))


class HumanSpymaster:
    def give_clue(self, board: Board) -> str:
        print("\n=== You are the spymaster ===")
        for role in ("team", "opponent", "neutral", "assassin"):
            print(f"{role:>9}: {', '.join(board.hidden_by_role(role))}")
        clue = input("Clue: ").strip()
        number = input("Number: ").strip()
        return f"CLUE: {clue}\nNUMBER: {number}"


# ---------------- guessers: return list of board words ----------------
# Every guesser sets self.last_raw (its unparsed answer) so it can be logged.

class LLMGuesser:
    """Thread-safe: last_raw is per thread, so it can be called from a thread pool."""

    def __init__(self, client: LLMClient):
        self.client = client
        self._local = threading.local()

    @property
    def last_raw(self) -> str | None:
        return getattr(self._local, "last_raw", None)

    def guess(self, board: Board, clue: str, number: int) -> list[str]:
        raw = self.client.chat(guesser_messages(board, clue, number))
        self._local.last_raw = raw
        return parse_guesses(raw, board, max_guesses=number)


class HumanGuesser:
    last_raw: str | None = None

    def guess(self, board: Board, clue: str, number: int) -> list[str]:
        print("\n=== You are the guesser ===")
        print("Board:", ", ".join(board.hidden_words))
        print(f"Clue: {clue}  ({number})")
        self.last_raw = input("Your guesses, comma-separated, most confident first: ")
        return parse_guesses(self.last_raw, board, max_guesses=number)


class RandomGuesser:
    """Sanity-check baseline: any decent spymaster should beat this easily."""
    last_raw: str | None = None

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def guess(self, board: Board, clue: str, number: int) -> list[str]:
        out = self.rng.sample(board.hidden_words, number)
        self.last_raw = ", ".join(out)
        return out
