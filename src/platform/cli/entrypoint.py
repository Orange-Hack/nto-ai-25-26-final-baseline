"""Canonical CLI entrypoint for platform pipeline operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.platform.cli.config_loader import load_config
from src.platform.core.logging import configure_logging
from src.platform.pipeline import STAGES, PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for run and validate commands."""
    parser = argparse.ArgumentParser(prog="python -m src.platform.cli.entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run whole pipeline or selected stage"
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to YAML config",
    )
    run_parser.add_argument(
        "--stage",
        type=str,
        choices=STAGES,
        default=None,
        help="Run target stage and dependencies only",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Run local pseudo-incident validation"
    )
    validate_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to YAML config",
    )
    return parser


def main() -> None:
    """Execute requested CLI command and report errors consistently."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        logs_dir = Path(config.get("logs", {}).get("dir", "./logs")).resolve()
        logger, log_path = configure_logging(logs_dir)
        logger.info("Log file: %s", log_path)
        runner = PipelineRunner(config=config, logger=logger)

        if args.command == "run":
            runner.run(stage=args.stage)
        elif args.command == "validate":
            result = runner.run_local_validation()
            print(json.dumps(result, ensure_ascii=False))
        else:
            raise RuntimeError(f"Unsupported command {args.command}")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"Pipeline failed: {exc}") from exc


if __name__ == "__main__":
    main()
