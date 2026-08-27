from __future__ import annotations

import argparse
from collections.abc import Sequence

_GATEWAY_DEPENDENCIES = frozenset(
    {"fastapi", "httpx", "openai", "pydantic", "starlette", "uvicorn"}
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="branchpilot-gateway",
        description="Run the text-only BranchPilot OpenAI-compatible gateway.",
    )
    parser.add_argument("--config", required=True, help="operator JSON configuration path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug"),
        default="info",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    try:
        import uvicorn

        from branchpilot.gateway.app import create_app
        from branchpilot.gateway.config import ConfigError, load_gateway_config
    except ModuleNotFoundError as exc:
        if exc.name is None or exc.name.partition(".")[0] not in _GATEWAY_DEPENDENCIES:
            raise
        raise SystemExit(
            "The gateway requires optional dependencies. "
            "Install BranchPilot with the 'gateway' extra."
        ) from exc
    try:
        config = load_gateway_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))
    app = create_app(config)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=1,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
