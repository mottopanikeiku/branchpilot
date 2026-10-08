from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from branchpilot.answers import extract_answer
from branchpilot.gateway.config import GatewayConfig, InboundAPIKeys, ModelRoute
from branchpilot.gateway.schemas import ChatCompletionRequest
from branchpilot.gateway.upstream import GatewayError, OpenAIUpstream, UpstreamSample
from branchpilot.runtime import PilotSession

_LOGGER = logging.getLogger("branchpilot.gateway")
_MAX_BODY_BYTES = 1_048_576


def _error_body(message: str, error_type: str, code: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "param": None, "code": code}}


class _BodyTooLarge(Exception):
    pass


class _BodyReadTimedOut(Exception):
    pass


class _BoundaryMiddleware:
    """Authenticate, then bound the complete request body before routing."""

    def __init__(
        self,
        app: ASGIApp,
        api_keys: InboundAPIKeys,
        body_read_timeout_s: float,
    ) -> None:
        self.app = app
        self.api_keys = api_keys
        self.body_read_timeout_s = body_read_timeout_s

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = "bp-" + secrets.token_urlsafe(18)
        scope.setdefault("state", {})["request_id"] = request_id
        path = scope.get("path", "")
        if not path.startswith("/v1/"):
            await self.app(scope, receive, send)
            return

        header_items = scope.get("headers", [])
        headers = {key.lower(): value for key, value in header_items}
        raw_auth = headers.get(b"authorization")
        token: str | None = None
        if raw_auth is not None:
            try:
                auth = raw_auth.decode("ascii")
            except UnicodeDecodeError:
                auth = ""
            parts = auth.split(" ")
            if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1]:
                token = parts[1]
        if token is None or not self.api_keys.matches(token):
            await self._send_error(
                send,
                401,
                "Invalid authentication credentials.",
                "authentication_error",
                "invalid_api_key",
                request_id,
                [(b"www-authenticate", b"Bearer")],
            )
            return

        content_lengths = [value for key, value in header_items if key.lower() == b"content-length"]
        if content_lengths:
            malformed = len(content_lengths) != 1 or not content_lengths[0].isdigit()
            if malformed:
                await self._send_error(
                    send,
                    400,
                    "The request is invalid.",
                    "invalid_request_error",
                    "invalid_request",
                    request_id,
                )
                return
            normalized_length = content_lengths[0].lstrip(b"0") or b"0"
            maximum_length = str(_MAX_BODY_BYTES).encode("ascii")
            too_large = len(normalized_length) > len(maximum_length) or (
                len(normalized_length) == len(maximum_length) and normalized_length > maximum_length
            )
            if too_large:
                await self._send_error(
                    send,
                    413,
                    "The request body is too large.",
                    "invalid_request_error",
                    "request_too_large",
                    request_id,
                )
                return

        consumed = 0
        deadline = asyncio.get_running_loop().time() + self.body_read_timeout_s

        async def bounded_receive() -> Message:
            nonlocal consumed
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise _BodyReadTimedOut
            try:
                message = await asyncio.wait_for(receive(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise _BodyReadTimedOut from exc
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > _MAX_BODY_BYTES:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except _BodyTooLarge:
            await self._send_error(
                send,
                413,
                "The request body is too large.",
                "invalid_request_error",
                "request_too_large",
                request_id,
            )
        except _BodyReadTimedOut:
            await self._send_error(
                send,
                408,
                "The request body was not received in time.",
                "invalid_request_error",
                "request_timeout",
                request_id,
            )

    @staticmethod
    async def _send_error(
        send: Send,
        status: int,
        message: str,
        error_type: str,
        code: str,
        request_id: str,
        extra_headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        import json

        body = json.dumps(_error_body(message, error_type, code), separators=(",", ":")).encode()
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"x-request-id", request_id.encode("ascii")),
        ]
        if extra_headers:
            headers.extend(extra_headers)
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})


def _exact_content(text: str) -> str | None:
    return text.strip() or None


def _numeric_strict(text: str) -> str | None:
    return extract_answer(text, strict=True)


def _error_response(error: GatewayError, request_id: str) -> JSONResponse:
    headers = {"x-request-id": request_id}
    if error.retry_after is not None:
        headers["retry-after"] = error.retry_after
    return JSONResponse(
        status_code=error.status_code,
        content=_error_body(error.message, error.error_type, error.code),
        headers=headers,
    )


def _invalid_request(message: str = "The request is invalid.") -> GatewayError:
    return GatewayError(400, "invalid_request_error", "invalid_request", message)


def _resolve_execution(
    request: ChatCompletionRequest,
    route: ModelRoute,
    strategies: Mapping[str, Any],
) -> tuple[Any, str, int]:
    strategy = route.strategy
    strategy_name = route.strategy_name
    horizon = min(route.max_samples, strategy.max_samples)
    overrides = request.branchpilot
    if overrides is None:
        return strategy, strategy_name, horizon
    supplied = overrides.model_dump(exclude_none=True)
    if not supplied:
        return strategy, strategy_name, horizon
    if not route.allow_client_overrides:
        raise _invalid_request("BranchPilot overrides are not enabled for this model.")
    if overrides.cost is not None:
        raise _invalid_request("The BranchPilot cost is fixed by the deployment plan.")
    if overrides.strategy is not None:
        if (
            overrides.strategy not in route.allowed_strategies
            or overrides.strategy not in strategies
        ):
            raise _invalid_request("The requested BranchPilot strategy is not allowed.")
        strategy = strategies[overrides.strategy]
        strategy_name = overrides.strategy
        maximum = getattr(strategy, "max_samples", None)
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise GatewayError(
                500, "api_error", "internal_error", "The gateway encountered an internal error."
            )
        horizon = min(route.max_samples, maximum)
    if overrides.max_samples is not None:
        if overrides.max_samples > horizon:
            raise _invalid_request("The requested max_samples exceeds the configured cap.")
        horizon = overrides.max_samples
    return strategy, strategy_name, horizon


def _request_body(request: ChatCompletionRequest, route: ModelRoute) -> dict[str, Any]:
    body = request.upstream_body()
    if "max_completion_tokens" in route.options:
        body.pop("max_tokens", None)
    elif "max_tokens" in route.options:
        body.pop("max_completion_tokens", None)
    body.update(route.options)
    body["upstream_model"] = route.upstream_model
    cap = route.max_completion_tokens
    requested = body.get("max_completion_tokens", body.get("max_tokens"))
    if requested is not None and (
        not isinstance(requested, int) or isinstance(requested, bool) or requested > cap
    ):
        raise _invalid_request("The requested completion token limit exceeds the configured cap.")
    if requested is None:
        body["max_completion_tokens"] = cap
    return body


async def _adaptive_request(
    request: ChatCompletionRequest,
    route: ModelRoute,
    upstream: OpenAIUpstream,
    extractor: Callable[[str], str | None],
    strategies: Mapping[str, Any],
    request_id: str,
    session_semaphore: asyncio.Semaphore,
    queue_timeout_s: float,
) -> tuple[list[UpstreamSample], str, str, str | None]:
    acquired = False
    try:
        try:
            await asyncio.wait_for(session_semaphore.acquire(), timeout=queue_timeout_s)
            acquired = True
        except asyncio.TimeoutError as exc:
            raise GatewayError(
                429,
                "rate_limit_error",
                "gateway_overloaded",
                "The gateway is temporarily overloaded.",
            ) from exc
        strategy, strategy_name, horizon = _resolve_execution(request, route, strategies)
        body = _request_body(request, route)
        first = await upstream.sample(
            body,
            extractor=extractor,
            public_request_id=request_id,
            sample_index=1,
        )
        samples = [first]
        session = PilotSession(
            strategy,
            request.question(),
            route.cost,
            prompt_tokens=first.usage.prompt_tokens,
            max_samples=horizon,
        )
        session.observe(first.sample)
        while session.should_continue:
            observed = await upstream.sample(
                body,
                extractor=extractor,
                public_request_id=request_id,
                sample_index=len(samples) + 1,
            )
            samples.append(observed)
            session.observe(observed.sample)
        answer = session.result().answer
        if answer is None:
            selection = "unparsed-fallback"
        else:
            selection = "majority"
            if not any(item.sample.answer == answer for item in samples):
                raise GatewayError(
                    500,
                    "api_error",
                    "internal_error",
                    "The gateway encountered an internal error.",
                )
        return samples, strategy_name, selection, answer
    except GatewayError:
        raise
    except Exception as exc:
        raise GatewayError(
            500, "api_error", "internal_error", "The gateway encountered an internal error."
        ) from exc
    finally:
        if acquired:
            session_semaphore.release()


async def _run_guarded(
    request: Request,
    operation: Any,
    timeout_s: float,
) -> Any:
    adaptive = asyncio.create_task(operation)

    async def disconnected() -> None:
        while not adaptive.done():
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.025)

    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait(
            {adaptive, watcher}, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
        )
        if adaptive in done:
            return await adaptive
        adaptive.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await adaptive
        if watcher in done:
            raise asyncio.CancelledError
        raise GatewayError(504, "api_error", "upstream_timeout", "The adaptive request timed out.")
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher


def _sum_details(samples: Sequence[UpstreamSample], attribute: str) -> dict[str, int] | None:
    detail_values = [getattr(item.usage, attribute) for item in samples]
    if any(value is None for value in detail_values):
        return None
    assert all(value is not None for value in detail_values)
    common = set(detail_values[0])
    for value in detail_values[1:]:
        common.intersection_update(value)
    if not common:
        return None
    return {name: sum(value[name] for value in detail_values) for name in sorted(common)}


def _success_response(
    route: ModelRoute,
    samples: Sequence[UpstreamSample],
    strategy_name: str,
    selection: str,
    final_answer: str | None,
    request_id: str,
) -> JSONResponse:
    selected = (
        samples[0]
        if final_answer is None
        else next(
            (item for item in samples if item.sample.answer == final_answer),
            samples[0],
        )
    )
    usage: dict[str, Any] = {
        "prompt_tokens": sum(item.usage.prompt_tokens for item in samples),
        "completion_tokens": sum(item.usage.completion_tokens for item in samples),
        "total_tokens": sum(item.usage.total_tokens for item in samples),
    }
    prompt_details = _sum_details(samples, "prompt_tokens_details")
    completion_details = _sum_details(samples, "completion_tokens_details")
    if prompt_details is not None:
        usage["prompt_tokens_details"] = prompt_details
    if completion_details is not None:
        usage["completion_tokens_details"] = completion_details
    content = {
        "id": "chatcmpl-bp-" + secrets.token_urlsafe(18),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": route.alias,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": selected.choice.content,
                    "refusal": selected.choice.refusal,
                },
                "logprobs": selected.choice.logprobs,
                "finish_reason": selected.choice.finish_reason,
            }
        ],
        "usage": usage,
    }
    return JSONResponse(
        content=content,
        headers={
            "x-request-id": request_id,
            "x-branchpilot-samples": str(len(samples)),
            "x-branchpilot-strategy": strategy_name,
            "x-branchpilot-selection": selection,
        },
    )


def create_app(
    config: GatewayConfig,
    *,
    strategies: Mapping[str, Any] | None = None,
    extractors: Mapping[str, Callable[[str], str | None]] | None = None,
    upstreams: Mapping[str, OpenAIUpstream] | None = None,
) -> FastAPI:
    strategy_registry = dict(strategies or {})
    extractor_registry: dict[str, Callable[[str], str | None]] = {
        "exact-content": _exact_content,
        "numeric-strict": _numeric_strict,
    }
    if extractors:
        extractor_registry.update(extractors)
    for route in config.models.values():
        if route.extractor not in extractor_registry:
            raise ValueError(f"unknown extractor {route.extractor!r}")
        unknown = route.allowed_strategies.difference(strategy_registry)
        if unknown:
            raise ValueError(f"unknown allowed strategy {sorted(unknown)[0]!r}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        clients = (
            dict(upstreams)
            if upstreams is not None
            else {
                name: OpenAIUpstream(item, config.queue_timeout_s)
                for name, item in config.upstreams.items()
            }
        )
        missing = set(config.upstreams).difference(clients)
        if missing:
            raise RuntimeError(f"missing upstream client {sorted(missing)[0]!r}")
        app.state.upstreams = clients
        app.state.session_semaphore = asyncio.Semaphore(config.max_concurrent_sessions)
        try:
            yield
        finally:
            await asyncio.gather(*(client.close() for client in clients.values()))

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        _BoundaryMiddleware,
        api_keys=config.inbound_api_keys,
        body_read_timeout_s=config.body_read_timeout_s,
    )

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        return _error_response(exc, request.state.request_id)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del exc
        return _error_response(_invalid_request(), request.state.request_id)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            error = GatewayError(
                404, "invalid_request_error", "not_found", "The route was not found."
            )
        else:
            error = GatewayError(
                400, "invalid_request_error", "invalid_request", "The request is invalid."
            )
        return _error_response(error, request.state.request_id)

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        _LOGGER.error(
            "gateway_request_failed",
            extra={"request_id": request.state.request_id, "exception_class": type(exc).__name__},
        )
        error = GatewayError(
            500, "api_error", "internal_error", "The gateway encountered an internal error."
        )
        return _error_response(error, request.state.request_id)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request) -> JSONResponse:
        request_id = request.state.request_id
        route = config.models.get(payload.model)
        if route is None:
            raise GatewayError(
                404,
                "invalid_request_error",
                "model_not_found",
                "The requested model does not exist.",
            )
        upstream = request.app.state.upstreams[route.upstream]
        samples, strategy_name, selection, final_answer = await _run_guarded(
            request,
            _adaptive_request(
                payload,
                route,
                upstream,
                extractor_registry[route.extractor],
                strategy_registry,
                request_id,
                request.app.state.session_semaphore,
                config.queue_timeout_s,
            ),
            config.request_timeout_s,
        )
        response = _success_response(
            route, samples, strategy_name, selection, final_answer, request_id
        )
        _LOGGER.info(
            "gateway_request_completed",
            extra={
                "request_id": request_id,
                "model": route.alias,
                "strategy": strategy_name,
                "samples": len(samples),
                "status": 200,
            },
        )
        return response

    return app
