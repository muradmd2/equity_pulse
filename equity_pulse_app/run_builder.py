"""CLI wrapper for calling the financial research graph directly."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from app.graph.builder import invoke_graph


def main() -> None:
    """Invoke the LangGraph builder from command-line tools such as OpenClaw."""

    load_dotenv()
    args = _parse_args()
    result = invoke_graph(
        args.query,
        session_id=args.session_id,
        model_name=args.model_name,
        openai_api_key=args.openai_api_key,
        checkpoint_db_path=args.checkpoint_db_path,
    )
    print(result.get("final_summary", ""))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the financial research LangGraph workflow directly."
    )
    parser.add_argument("query", help="Financial research query to run.")
    parser.add_argument("--session-id", help="Optional session/checkpoint thread ID.")
    parser.add_argument("--model-name", help="Optional OpenAI model override.")
    parser.add_argument("--openai-api-key", help="Optional per-request OpenAI API key.")
    parser.add_argument(
        "--checkpoint-db-path",
        help="Optional SQLite checkpoint database path override.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
