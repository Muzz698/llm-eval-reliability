"""Minimal client for a local Ollama server (https://ollama.com): free, offline, reproducible."""
from __future__ import annotations

import requests


class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: float = 120):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, temperature: float = 0.0, seed: int = 0,
                 max_tokens: int = 8) -> str:
        r = requests.post(f"{self.host}/api/chat", timeout=self.timeout, json={
            "model": self.model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "seed": int(seed), "num_predict": max_tokens},
        })
        if r.status_code == 404:
            raise RuntimeError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
        r.raise_for_status()
        return r.json()["message"]["content"]
