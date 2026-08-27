from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

MAX_COMPLETION_TOKENS = 32_768


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TextMessage(StrictModel):
    role: Literal["developer", "system", "user", "assistant"]
    content: StrictStr
    name: StrictStr | None = Field(default=None, min_length=1, max_length=64, pattern=_NAME.pattern)


class TextResponseFormat(StrictModel):
    type: Literal["text"]


class JSONObjectResponseFormat(StrictModel):
    type: Literal["json_object"]


class JSONSchemaDefinition(StrictModel):
    name: StrictStr = Field(min_length=1, max_length=64, pattern=_NAME.pattern)
    description: StrictStr | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: StrictBool | None = None


class JSONSchemaResponseFormat(StrictModel):
    type: Literal["json_schema"]
    json_schema: JSONSchemaDefinition


ResponseFormat = Annotated[
    TextResponseFormat | JSONObjectResponseFormat | JSONSchemaResponseFormat,
    Field(discriminator="type"),
]


class BranchPilotOptions(StrictModel):
    strategy: StrictStr | None = Field(default=None, min_length=1, max_length=128)
    cost: StrictFloat | StrictInt | None = Field(default=None, ge=0)
    max_samples: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def finite_cost(self) -> BranchPilotOptions:
        if self.cost is not None and not math.isfinite(float(self.cost)):
            raise ValueError("cost must be finite")
        return self


class ChatCompletionRequest(StrictModel):
    model: StrictStr = Field(min_length=1, max_length=256)
    messages: list[TextMessage] = Field(min_length=1, max_length=256)
    frequency_penalty: StrictFloat | StrictInt | None = Field(default=None, ge=-2, le=2)
    presence_penalty: StrictFloat | StrictInt | None = Field(default=None, ge=-2, le=2)
    temperature: StrictFloat | StrictInt | None = Field(default=None, ge=0, le=2)
    top_p: StrictFloat | StrictInt | None = Field(default=None, ge=0, le=1)
    max_completion_tokens: StrictInt | None = Field(default=None, ge=1, le=MAX_COMPLETION_TOKENS)
    max_tokens: StrictInt | None = Field(default=None, ge=1, le=MAX_COMPLETION_TOKENS)
    stop: StrictStr | list[StrictStr] | None = None
    seed: StrictInt | None = None
    logit_bias: dict[StrictStr, Annotated[StrictInt, Field(ge=-100, le=100)]] | None = None
    logprobs: StrictBool | None = None
    top_logprobs: StrictInt | None = Field(default=None, ge=0, le=20)
    response_format: ResponseFormat | None = None
    user: StrictStr | None = Field(default=None, max_length=64)
    n: Literal[1] = 1
    stream: Literal[False] = False
    branchpilot: BranchPilotOptions | None = None

    @model_validator(mode="after")
    def validate_combinations(self) -> ChatCompletionRequest:
        if self.max_completion_tokens is not None and self.max_tokens is not None:
            raise ValueError("max_completion_tokens and max_tokens are mutually exclusive")
        if self.top_logprobs is not None and self.logprobs is not True:
            raise ValueError("top_logprobs requires logprobs=true")
        if isinstance(self.stop, list) and len(self.stop) > 4:
            raise ValueError("stop may contain at most four strings")
        if not any(message.role == "user" and message.content.strip() for message in self.messages):
            raise ValueError("messages must contain a non-empty user message")
        for field in ("frequency_penalty", "presence_penalty", "temperature", "top_p"):
            value = getattr(self, field)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{field} must be finite")
        return self

    def question(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user" and message.content.strip():
                return message.content
        raise RuntimeError("validated request has no user message")

    def upstream_body(self) -> dict[str, Any]:
        body = self.model_dump(
            by_alias=True,
            exclude={"model", "branchpilot"},
            exclude_none=True,
        )
        body["n"] = 1
        body["stream"] = False
        return body


class ErrorObject(StrictModel):
    message: StrictStr
    type: StrictStr
    param: StrictStr | None = None
    code: StrictStr


class ErrorEnvelope(StrictModel):
    error: ErrorObject
