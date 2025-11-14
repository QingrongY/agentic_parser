"""CLI wiring for the agentic parser."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from interface.interaction_service import InteractionService
from llm.client import AIMLLLMClient
from pipeline.orchestrator import LogParsingOrchestrator

DEFAULTS = {
    "log_path": ROOT_DIR / "datasets/BGL/BGL_2k.log",
    "state_dir": ROOT_DIR / "state_v2",
    "config": ROOT_DIR / "config.json",
    "model": "gemini-2.0-flash",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic log parser prototype")
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=DEFAULTS["log_path"],
        help=f"Path to the log file (default: {DEFAULTS['log_path']})",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULTS["state_dir"],
        help=f"Directory for template libraries and outputs (default: {DEFAULTS['state_dir']})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULTS["config"],
        help=f"Path to AIML API config (default: {DEFAULTS['config']})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULTS["model"],
        help=f"AIML model identifier (default: {DEFAULTS['model']})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    llm_client = AIMLLLMClient(
        config_path=args.config,
        model=args.model,
    )
    orchestrator = LogParsingOrchestrator(
        llm_client,
        state_dir=args.state_dir,
        interaction_service=InteractionService(),
    )
    report = orchestrator.process(args.log_path)
    print("== PARSE REPORT ==")
    print(f"Routing bucket: {report.routing.source_id}")
    print(f"Processed lines: {report.processed_lines}")
    print(f"Matched lines: {report.matched_lines}")
    print(f"Learned templates: {report.learned_templates}")
    print("Artifacts:")
    print(f"  TSV: {report.artifacts.structured_output}")
    print(f"  Templates: {report.artifacts.templates_snapshot}")
    print(f"  Metrics: {report.artifacts.metrics_path}")


if __name__ == "__main__":
    main()
