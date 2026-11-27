#!/usr/bin/env python3
"""
Decoupled LLM Memory Framework v2.1 — Local Chat Runner
=======================================================
Zero-extra-dependency Terminal chat client connecting a local OpenAI-compatible
`llama-cpp-python` server (default: http://localhost:8000/v1/chat/completions,
Qwen2.5-1.5B-Instruct-GGUF) to the v2.1 FrontRoomController.

Uses only Python's standard library (`urllib.request`, `json`) plus `numpy`
already required by `FrontRoomController`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.layers.front_room_controller import FrontRoomController

DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Use the provided recent conversation "
    "and retrieved consolidated memory to answer accurately."
)


def make_llm_engine(
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> Callable[[str], str]:
    """Builds a callable `llm_engine(context) -> str` backed by localhost llama-cpp-python."""

    def _call_local_llm(context: str) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            api_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM server HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach local LLM server at {api_url} ({exc.reason}). "
                "Start llama_cpp.server first."
            ) from exc

        data = json.loads(raw)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Unexpected response from LLM server: {data}")
        message = choices[0].get("message") or {}
        return str(message.get("content", "")).strip()

    return _call_local_llm


def build_controller(max_context_tokens: int = 1764) -> FrontRoomController:
    """Initializes FrontRoomController with v2.1 gateway, hybrid retriever, and compactor."""
    return FrontRoomController(
        active_window_size=2,
        drift_threshold=0.65,
        max_retrieval_bundles=3,
        max_context_tokens=max_context_tokens,
        use_local_embeddings=True,
    )


def handle_command(cmd: str, controller: FrontRoomController) -> Optional[str]:
    """
    Handles Terminal slash commands (`/compact`, `/telemetry`, `/quit`, `/help`).
    Returns a string to print, or None if the REPL should exit.
    """
    normalized = cmd.strip().lower()
    if normalized in {"/quit", "/exit"}:
        return None
    if normalized == "/compact":
        report = controller.compact_memory()
        return (
            "[Compaction] "
            f"purged_rules={report.purged_distorted_rules}, "
            f"merged_rules={report.merged_rule_count}, "
            f"created_elephants={report.created_elephants}, "
            f"remaining_active={report.remaining_active_nodes}"
        )
    if normalized == "/telemetry":
        stats = controller.get_telemetry()
        return f"[Telemetry] {json.dumps(stats)}"
    if normalized == "/help":
        return "Commands: /compact | /telemetry | /quit"
    return f"Unknown command '{cmd}'. Try /compact, /telemetry, or /quit."


def run_chat_loop(
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    max_context_tokens: int = 1764,
) -> None:
    controller = build_controller(max_context_tokens=max_context_tokens)
    llm_engine = make_llm_engine(api_url=api_url, model=model)

    print("Decoupled LLM Memory Framework v2.1 — Local Chat")
    print(f"Server: {api_url} | Model: {model} | Token ceiling: {max_context_tokens}")
    print("Commands: /compact, /telemetry, /quit\n")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            out = handle_command(user_input, controller)
            if out is None:
                print("Goodbye.")
                break
            print(out)
            continue

        try:
            reply = controller.process_turn(user_input, llm_engine)
            print(f"\nAssistant> {reply}\n")
        except Exception as exc:
            print(f"\n[Error] {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local terminal chat with v2.1 FrontRoomController and llama-cpp-python."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="OpenAI-compatible chat completions URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name passed in payload")
    parser.add_argument("--max-tokens", type=int, default=1764, help="FrontRoom hard token ceiling")
    args = parser.parse_args()
    run_chat_loop(api_url=args.api_url, model=args.model, max_context_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
