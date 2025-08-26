"""
title: OpenAI Responses API Manifold — GPT-5 Enhanced
id: response_api_for_gpt5
author: Justin Kropp (original), frondesce (community mod)
contributors: GPT-5 Thinking (AI assistance)
source: https://github.com/jrkropp/open-webui-developer-toolkit
license: MIT
version: 0.8.23
description: Adds GPT-5 Responses API support (text.verbosity, reasoning.effort), streaming reasoning summary with throttling, “Thinking → 🧠”, and SSE fallback. Unofficial; credits retained.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────
import textwrap
from typing import Tuple
import asyncio
import datetime
import inspect
import json
import logging
import os
import re
import sys
import secrets
import time
from collections import defaultdict, deque
from contextvars import ContextVar
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)
from urllib.parse import urlparse

import aiohttp
from fastapi import Request
from pydantic import BaseModel, Field, model_validator

from open_webui.models.chats import Chats
from open_webui.models.models import ModelForm, Models

# ─────────────────────────────────────────────────────────────────────────────
# 2. Constants & Global Configuration
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_SUPPORT = {
    "web_search_tool": {
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
        "o3-pro",
        "o4-mini",
        "o3-deep-research",
        "o4-mini-deep-research",
        # >>> GPT-5
        "gpt-5",
        "gpt-5-mini",
    },
    "image_gen_tool": {
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1-nano",
        "o3",
        # >>> GPT-5
        "gpt-5",
        "gpt-5-mini",
    },
    "function_calling": {
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1-nano",
        "o3",
        "o4-mini",
        "o3-mini",
        "o3-pro",
        "o3-deep-research",
        "o4-mini-deep-research",
        # >>> GPT-5
        "gpt-5",
        "gpt-5-mini",
    },
    "reasoning": {
        "o3",
        "o4-mini",
        "o3-mini",
        "o3-pro",
        "o3-deep-research",
        "o4-mini-deep-research",
        # >>> GPT-5：Used to display the "Thinking..." status (whether there is a summary depends on the valve and upstream)
        "gpt-5",
        "gpt-5-mini",
    },
    "reasoning_summary": {
        "o3",
        "o4-mini",
        "o4-mini-high",
        "o3-mini",
        "o3-mini-high",
        "o3-pro",
        "o3-deep-research",
        "o4-mini-deep-research",
        # >>> GPT-5：If your upstream/account supports summary, add it
        "gpt-5",
        "gpt-5-mini",
    },
    "deep_research": {
        "o3-deep-research",
        "o4-mini-deep-research",
    },
}

DETAILS_RE = re.compile(
    r"<details\b[^>]*>.*?</details>|!\[.*?]\(.*?\)",
    re.S | re.I,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Models
# ─────────────────────────────────────────────────────────────────────────────
class CompletionsBody(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False

    class Config:
        extra = "allow"

    @model_validator(mode="after")
    def normalize_model(self) -> "CompletionsBody":
        # Compatible with various possible prefix writing methods to avoid emitting pipeline prefixes as upstream model names
        prefixes = [
            "openai_responses.",
            "openai_response.",
            "openai.",
            "responses.",
        ]
        for p in prefixes:
            if self.model.startswith(p):
                self.model = self.model.removeprefix(p)
                break
        if self.model in {"o3-mini-high", "o4-mini-high"}:
            self.model = self.model.removesuffix("-high")
            self.reasoning_effort = "high"
        return self


class ResponsesBody(BaseModel):
    model: str
    input: Union[str, List[Dict[str, Any]]]

    instructions: Optional[str] = ""
    stream: bool = False
    store: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    truncation: Optional[Literal["auto", "disabled"]] = None
    reasoning: Optional[Dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = True
    user: Optional[str] = None
    tool_choice: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    include: Optional[List[str]] = None

    class Config:
        extra = "allow"

    @staticmethod
    def transform_tools(
        tools: dict | list | None = None,
        *,
        strict: bool = False,
    ) -> list[dict]:
        if not tools:
            return []
        iterable = tools.values() if isinstance(tools, dict) else tools
        native, converted = [], []
        for item in iterable:
            if not isinstance(item, dict):
                continue
            if "spec" in item:
                spec = item["spec"]
                if isinstance(spec, dict):
                    converted.append(
                        {
                            "type": "function",
                            "name": spec.get("name", ""),
                            "description": spec.get("description", ""),
                            "parameters": spec.get("parameters", {}),
                        }
                    )
                continue
            if item.get("type") == "function" and "function" in item:
                fn = item["function"]
                if isinstance(fn, dict):
                    converted.append(
                        {
                            "type": "function",
                            "name": fn.get("name", ""),
                            "description": fn.get("description", ""),
                            "parameters": fn.get("parameters", {}),
                        }
                    )
                continue
            native.append(dict(item))

        if strict:
            for tool in converted:
                params = tool.setdefault("parameters", {})
                props = params.setdefault("properties", {})
                params["required"] = list(props)
                params["additionalProperties"] = False
                for schema in props.values():
                    t = schema.get("type")
                    schema["type"] = (
                        [t, "null"]
                        if isinstance(t, str)
                        else (
                            t + ["null"]
                            if isinstance(t, list) and "null" not in t
                            else t
                        )
                    )
                tool["strict"] = True

        canonical: dict[str, dict] = {}
        for t in native + converted:
            key = t["name"] if t.get("type") == "function" else t["type"]
            canonical[key] = t
        return list(canonical.values())

    @staticmethod
    def _build_mcp_tools(mcp_json: str) -> list[dict]:
        if not mcp_json or not mcp_json.strip():
            return []
        try:
            data = json.loads(mcp_json)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "REMOTE_MCP_SERVERS_JSON could not be parsed (%s); ignoring.", exc
            )
            return []
        items = data if isinstance(data, list) else [data]
        valid_tools: list[dict] = []
        for idx, obj in enumerate(items, start=1):
            if not isinstance(obj, dict):
                logging.getLogger(__name__).warning(
                    "REMOTE_MCP_SERVERS_JSON item %d ignored: not an object.", idx
                )
                continue
            label = obj.get("server_label")
            url = obj.get("server_url")
            if not (label and url):
                logging.getLogger(__name__).warning(
                    "REMOTE_MCP_SERVERS_JSON item %d ignored: "
                    "'server_label' and 'server_url' are required.",
                    idx,
                )
                continue
            allowed = {
                "server_label",
                "server_url",
                "require_approval",
                "allowed_tools",
                "headers",
            }
            tool = {"type": "mcp"}
            tool.update({k: v for k, v in obj.items() if k in allowed})
            valid_tools.append(tool)
        return valid_tools

    @staticmethod
    def transform_messages_to_input(
        messages: List[Dict[str, Any]],
        chat_id: Optional[str] = None,
        openwebui_model_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        required_item_ids: set[str] = set()
        if chat_id and openwebui_model_id:
            for m in messages:
                if (
                    m.get("role") == "assistant"
                    and m.get("content")
                    and contains_marker(m["content"])
                ):
                    for mk in extract_markers(m["content"], parsed=True):
                        required_item_ids.add(mk["ulid"])
        items_lookup: dict[str, dict] = {}
        if chat_id and openwebui_model_id and required_item_ids:
            items_lookup = fetch_openai_response_items(
                chat_id, list(required_item_ids), openwebui_model_id=openwebui_model_id
            )

        openai_input: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            raw_content = msg.get("content", "")
            if role == "system":
                continue
            if role == "user":
                content_blocks = msg.get("content") or []
                if isinstance(content_blocks, str):
                    content_blocks = [{"type": "text", "text": content_blocks}]
                block_transform = {
                    "text": lambda b: {"type": "input_text", "text": b.get("text", "")},
                    "image_url": lambda b: {
                        "type": "input_image",
                        "image_url": b.get("image_url", {}).get("url"),
                    },
                    "input_file": lambda b: {
                        "type": "input_file",
                        "file_id": b.get("file_id"),
                    },
                }
                openai_input.append(
                    {
                        "role": "user",
                        "content": [
                            block_transform.get(block.get("type"), lambda b: b)(block)
                            for block in content_blocks
                            if block
                        ],
                    }
                )
                continue
            if role == "developer":
                openai_input.append({"role": "developer", "content": raw_content})
                continue
            if "<details" in raw_content or "![" in raw_content:
                content = DETAILS_RE.sub("", raw_content).strip()
            else:
                content = raw_content
            if contains_marker(content):
                for segment in split_text_by_markers(content):
                    if segment["type"] == "marker":
                        mk = parse_marker(segment["marker"])
                        item = items_lookup.get(mk["ulid"])
                        if item:
                            openai_input.append(item)
                        else:
                            logging.warning(
                                f"Missing persisted item for ID: {mk['ulid']}"
                            )
                    elif segment["type"] == "text" and segment["text"].strip():
                        openai_input.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": segment["text"].strip(),
                                    }
                                ],
                            }
                        )
            else:
                if content:
                    openai_input.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    )
        return openai_input

    @classmethod
    def from_completions(
        ResponsesBody,
        completions_body: "CompletionsBody",
        chat_id: Optional[str] = None,
        openwebui_model_id: Optional[str] = None,
        **extra_params,
    ) -> "ResponsesBody":
        completions_dict = completions_body.model_dump(exclude_none=True)
        unsupported_fields = {
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "n",
            "stop",
            "response_format",
            "suffix",
            "stream_options",
            "audio",
            "function_call",
            "functions",
            "reasoning_effort",
            "max_tokens",
        }
        sanitized_params = {}
        for key, value in completions_dict.items():
            if key in unsupported_fields:
                logging.warning(f"Dropping unsupported parameter: '{key}'")
            else:
                sanitized_params[key] = value
        if "max_tokens" in completions_dict:
            sanitized_params["max_output_tokens"] = completions_dict["max_tokens"]
        effort = completions_dict.get("reasoning_effort")
        if effort:
            reasoning = sanitized_params.get("reasoning", {})
            reasoning.setdefault("effort", effort)
            sanitized_params["reasoning"] = reasoning
        instructions = next(
            (
                msg["content"]
                for msg in reversed(completions_dict.get("messages", []))
                if msg["role"] == "system"
            ),
            None,
        )
        if instructions:
            sanitized_params["instructions"] = instructions
        if "messages" in completions_dict:
            sanitized_params.pop("messages", None)
            sanitized_params["input"] = ResponsesBody.transform_messages_to_input(
                completions_dict.get("messages", []),
                chat_id=chat_id,
                openwebui_model_id=openwebui_model_id,
            )
        return ResponsesBody(
            **sanitized_params,
            **extra_params,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Controller: Pipe
# ─────────────────────────────────────────────────────────────────────────────
class Pipe:
    class Valves(BaseModel):
        # —— Commonly used (put on top)
        BASE_URL: str = Field(
            default=(
                (os.getenv("OPENAI_API_BASE_URL") or "").strip()
                or "https://api.openai.com/v1"
            ),
            description="Base URL for OpenAI/LiteLLM/custom endpoints.",
        )
        API_KEY: str = Field(
            default=(os.getenv("OPENAI_API_KEY") or "").strip() or "sk-xxxxx",
            description="OpenAI API key.",
        )
        MODEL_ID: str = Field(
            default="gpt-4.1, gpt-4o",
            description="Comma separated model IDs shown in WebUI.",
        )
        # Reasoning and Output Control (Group 2)
        ENABLE_REASONING_SUMMARY: Literal["auto", "concise", "detailed", None] = Field(
            default=None,
            description="Reasoning summary for models in FEATURE_SUPPORT['reasoning_summary']",
        )
        REASON_EFFORT: Optional[Literal["minimal", "low", "medium", "high"]] = Field(
            default=None,
            description="Reasoning effort for o3/o4-mini/GPT-5. ('minimal' only on GPT-5; others downgrade to 'low')",
        )
        TEXT_VERBOSITY: Optional[Literal["low", "medium", "high"]] = Field(
            default=None,
            description="(GPT-5 only) Controls text verbosity: low | medium | high. When unset, the field is omitted.",
        )
        TRUNCATION: Literal["auto", "disabled"] = Field(
            default="auto",
            description="Truncation strategy.",
        )
        PARALLEL_TOOL_CALLS: bool = Field(
            default=True,
            description="Allow parallel tool calls.",
        )
        MAX_TOOL_CALLS: Optional[int] = Field(
            default=None,
            description="Max total tool/function calls per response.",
        )
        MAX_FUNCTION_CALL_LOOPS: int = Field(
            default=5,
            description="Max outer execution loops.",
        )

        # Tools/Search (Group 3)
        ENABLE_WEB_SEARCH_TOOL: bool = Field(
            default=False,
            description="Enable OpenAI web_search_preview tool when supported.",
        )
        WEB_SEARCH_CONTEXT_SIZE: Literal["low", "medium", "high", None] = Field(
            default="medium",
            description="Web search context size.",
        )
        WEB_SEARCH_USER_LOCATION: Optional[str] = Field(
            default=None,
            description='User location JSON for web search, e.g. {"type":"approximate","country":"US",...}.',
        )
        PERSIST_TOOL_RESULTS: bool = Field(
            default=True,
            description="Persist tool call results across turns.",
        )
        REMOTE_MCP_SERVERS_JSON: Optional[str] = Field(
            default=None,
            description="[EXPERIMENTAL] JSON list of remote MCP servers to attach.",
        )

        # ——Other (last)
        USER_ID_FIELD: Literal["id", "email"] = Field(
            default="id",
            description="Which user identifier to send as 'user'.",
        )
        LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
            default=os.getenv("GLOBAL_LOG_LEVEL", "INFO").upper(),
            description="Logging level.",
        )

    class UserValves(BaseModel):
        LOG_LEVEL: Literal[
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "INHERIT"
        ] = Field(
            default="INHERIT",
            description="Per-user log level (INHERIT uses global).",
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "openai_responses"
        self.valves = self.Valves()
        self.session: aiohttp.ClientSession | None = None
        self.logger = SessionLogger.get_logger(__name__)

    async def pipes(self):
        model_ids = [
            model_id.strip()
            for model_id in self.valves.MODEL_ID.split(",")
            if model_id.strip()
        ]
        return [
            {"id": model_id, "name": f"OpenAI: {model_id}"} for model_id in model_ids
        ]

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __metadata__: dict[str, Any],
        __tools__: list[dict[str, Any]] | dict[str, Any] | None,
        __task__: Optional[dict[str, Any]] = None,
        __task_body__: Optional[dict[str, Any]] = None,
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None = None,
    ) -> AsyncGenerator[str, None] | str | None:

        valves = self._merge_valves(
            self.valves, self.UserValves.model_validate(__user__.get("valves", {}))
        )
        openwebui_model_id = __metadata__.get("model", {}).get("id", "")
        user_identifier = __user__[valves.USER_ID_FIELD]
        features = __metadata__.get("features", {}).get("openai_responses", {})

        SessionLogger.session_id.set(__metadata__.get("session_id", None))
        SessionLogger.log_level.set(
            getattr(logging, valves.LOG_LEVEL.upper(), logging.INFO)
        )

        completions_body = CompletionsBody.model_validate(body)
        responses_body = ResponsesBody.from_completions(
            completions_body=completions_body,
            **(
                {"chat_id": __metadata__["chat_id"]}
                if __metadata__.get("chat_id")
                else {}
            ),
            **(
                {"openwebui_model_id": openwebui_model_id} if openwebui_model_id else {}
            ),
            truncation=valves.TRUNCATION,
            user=user_identifier,
            **(
                {"max_tool_calls": valves.MAX_TOOL_CALLS}
                if valves.MAX_TOOL_CALLS is not None
                else {}
            ),
        )

        # Normalized model family name: remove date; gpt-5-* → gpt-5 # >>> GPT-5
        model_family = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", responses_body.model)
        if responses_body.model.startswith("gpt-5"):
            model_family = "gpt-5"

        # —— GPT-5 specific: Inject text field only when TEXT_VERBOSITY is configured to avoid 400
        if model_family.startswith("gpt-5"):
            v = (valves.TEXT_VERBOSITY or "").strip().lower()
            if v in {"low", "medium", "high"}:
                text_obj: Dict[str, Any] = {"format": {"type": "text"}, "verbosity": v}
                try:
                    setattr(responses_body, "text", text_obj)
                except Exception:
                    responses_body.__dict__["text"] = text_obj

        # -- All "inference model" efforts (o3/o4-mini/gpt-5)
        effort = (valves.REASON_EFFORT or "").strip().lower()
        if effort and model_family in FEATURE_SUPPORT["reasoning"]:
            # o3/o4-mini does not support 'minimal' and automatically downgrades to 'low'
            if not model_family.startswith("gpt-5") and effort == "minimal":
                effort = "low"
            if effort in {"low", "medium", "high"} or (
                model_family.startswith("gpt-5")
                and effort in {"minimal", "low", "medium", "high"}
            ):
                responses_body.reasoning = responses_body.reasoning or {}
                # Does not override values explicitly set by the user in the body
                responses_body.reasoning.setdefault("effort", effort)

        if __task__:
            self.logger.info("Detected task model: %s", __task__)
            return await self._run_task_model_request(
                responses_body.model_dump(), valves
            )

        if inspect.isawaitable(__tools__):
            __tools__ = await __tools__

        if __tools__ and model_family in FEATURE_SUPPORT["function_calling"]:
            responses_body.tools = ResponsesBody.transform_tools(
                tools=__tools__,
                strict=True,
            )

        if model_family in FEATURE_SUPPORT["web_search_tool"] and (
            valves.ENABLE_WEB_SEARCH_TOOL or features.get("web_search", False)
        ):
            responses_body.tools = responses_body.tools or []
            responses_body.tools.append(
                {
                    "type": "web_search_preview",
                    "search_context_size": valves.WEB_SEARCH_CONTEXT_SIZE,
                    **(
                        {"user_location": json.loads(valves.WEB_SEARCH_USER_LOCATION)}
                        if valves.WEB_SEARCH_USER_LOCATION
                        else {}
                    ),
                }
            )

        if valves.REMOTE_MCP_SERVERS_JSON:
            mcp_tools = ResponsesBody._build_mcp_tools(valves.REMOTE_MCP_SERVERS_JSON)
            if mcp_tools:
                responses_body.tools = (responses_body.tools or []) + mcp_tools

        if __tools__ and __metadata__.get("function_calling") != "native":
            supports_function_calling = (
                model_family in FEATURE_SUPPORT["function_calling"]
            )
            if supports_function_calling:
                await self._emit_notification(
                    __event_emitter__,
                    content=f"Enabling native function calling for model: {responses_body.model}. Please re-run your query.",
                    level="info",
                )
                update_openwebui_model_param(
                    openwebui_model_id, "function_calling", "native"
                )
            else:
                await self._emit_error(
                    __event_emitter__,
                    f"The selected model '{responses_body.model}' does not support tools. "
                    f"Disable tools or choose a supported model (e.g., {', '.join(FEATURE_SUPPORT['function_calling'])}).",
                )
                return

        if (
            model_family in FEATURE_SUPPORT["reasoning_summary"]
            and valves.ENABLE_REASONING_SUMMARY
        ):
            responses_body.reasoning = responses_body.reasoning or {}
            responses_body.reasoning["summary"] = valves.ENABLE_REASONING_SUMMARY

        if (
            model_family in FEATURE_SUPPORT["reasoning"]
            and responses_body.store is False
        ):
            responses_body.include = responses_body.include or []
            responses_body.include.append("reasoning.encrypted_content")

        self.logger.debug(
            "Transformed ResponsesBody: %s",
            json.dumps(
                responses_body.model_dump(exclude_none=True),
                indent=2,
                ensure_ascii=False,
            ),
        )

        if responses_body.stream:
            return await self._run_streaming_loop(
                responses_body, valves, __event_emitter__, __metadata__, __tools__
            )
        else:
            return await self._run_nonstreaming_loop(
                responses_body, valves, __event_emitter__, __metadata__, __tools__
            )

    # ─────────────────────────────────────────────────────────────────────
    # 4.3 Core Multi-Turn Handlers
    # ─────────────────────────────────────────────────────────────────────
    async def _run_streaming_loop(
        self,
        body: ResponsesBody,
        valves: Pipe.Valves,
        event_emitter: Callable[[Dict[str, Any]], Awaitable[None]],
        metadata: dict[str, Any] = {},
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        tools = tools or {}
        openwebui_model = metadata.get("model", {}).get("id", "")
        assistant_message = ""
        total_usage: dict[str, Any] = {}
        ordinal_by_url: dict[str, int] = {}
        emitted_citations: list[dict] = []

        status_indicator = ExpandableStatusIndicator(event_emitter)
        status_indicator._done = False

        model_family = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", body.model)
        if body.model.startswith("gpt-5"):
            model_family = "gpt-5"

        # Initially a normal prompt, which will be "upgraded" to 🧠 after receiving a summary # >>> UPGRADE
        if model_family in FEATURE_SUPPORT["reasoning"]:
            assistant_message = await status_indicator.add(
                assistant_message,
                status_title="Thinking…",
                status_content="Reading the question and building a plan to answer it. This may take a moment.",
            )

        # summary Aggregation and throttling status  # >>> STREAM-REASONING
        summary_upgraded = False
        summary_buffers: Dict[int, str] = {}
        _last_summary_emit = 0.0
        _EMIT_INTERVAL = 0.35  # ≥350ms refresh once
        _EMIT_MIN_CHARS = 80  # Add ≥80 characters and refresh again
        self._last_rendered_summary = ""  # Record rendered merged

        try:
            for loop_idx in range(valves.MAX_FUNCTION_CALL_LOOPS):
                final_response: dict[str, Any] | None = None
                async for event in self.send_openai_responses_streaming_request(
                    body.model_dump(exclude_none=True),
                    api_key=valves.API_KEY,
                    base_url=valves.BASE_URL,
                    valves=valves,
                ):
                    etype = event.get("type")

                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug("Received event: %s", etype)
                        if not etype.endswith(".delta"):
                            self.logger.debug(
                                "Event data: %s",
                                json.dumps(event, indent=2, ensure_ascii=False),
                            )

                    # >>> STREAM-REASONING：Reasoning summary (aggregation + upgraded title)
                    if etype in (
                        "response.reasoning_summary_text.delta",
                        "response.reasoning_summary.delta",
                    ):
                        delta = event.get("delta") or event.get("text_delta") or ""
                        if not delta:
                            continue

                        # First time receiving a summary: Upgrade the last status from "Thinking..." to "🧠 Thinking..."
                        if not summary_upgraded:
                            assistant_message = (
                                await status_indicator.update_last_status(
                                    assistant_message,
                                    new_title="🧠 Thinking…",
                                    emit=False,
                                )
                            )
                            summary_upgraded = True

                        idx = event.get("summary_index", 0)
                        summary_buffers[idx] = summary_buffers.get(idx, "") + delta
                        merged = "\n\n".join(
                            summary_buffers[i] for i in sorted(summary_buffers)
                        ).strip()

                        now = time.perf_counter()
                        newly_added = max(
                            0, len(merged) - len(self._last_rendered_summary)
                        )
                        should_emit = (
                            (now - _last_summary_emit) >= _EMIT_INTERVAL
                            or newly_added >= _EMIT_MIN_CHARS
                            or any(
                                p in delta for p in (". ", "\n", "。", "！", "？", "; ")
                            )
                        )

                        if should_emit:
                            assistant_message = (
                                await status_indicator.update_last_status(
                                    assistant_message, new_content=merged, emit=True
                                )
                            )
                            self._last_rendered_summary = merged
                            _last_summary_emit = now
                        else:
                            # Only updates the memory and does not trigger UI refresh
                            assistant_message = (
                                await status_indicator.update_last_status(
                                    assistant_message, new_content=merged, emit=False
                                )
                            )
                        continue

                    if etype == "response.output_text.delta":
                        delta = event.get("delta", "")
                        if delta:
                            assistant_message += delta
                            await event_emitter(
                                {
                                    "type": "chat:message",
                                    "data": {"content": assistant_message},
                                }
                            )
                        continue

                    if etype == "response.output_text.annotation.added":
                        ann = event["annotation"]
                        url = ann.get("url", "").removesuffix("?utm_source=openai")
                        title = ann.get("title", "").strip()
                        domain = urlparse(url).netloc.lower().lstrip("www.")
                        already_cited = url in ordinal_by_url
                        citation_number = (
                            ordinal_by_url[url]
                            if already_cited
                            else len(ordinal_by_url) + 1
                        )
                        if not already_cited:
                            ordinal_by_url[url] = citation_number
                            citation_payload = {
                                "source": {"name": domain, "url": url},
                                "document": [title],
                                "metadata": [
                                    {
                                        "source": url,
                                        "date_accessed": datetime.date.today().isoformat(),
                                    }
                                ],
                            }
                            await event_emitter(
                                {"type": "source", "data": citation_payload}
                            )
                            emitted_citations.append(citation_payload)
                        assistant_message += f" [{citation_number}]"
                        assistant_message = re.sub(
                            rf"\(\s*\[\s*{re.escape(domain)}\s*\]\([^)]+\)\s*\)",
                            " ",
                            assistant_message,
                            count=1,
                        ).strip()
                        await event_emitter(
                            {
                                "type": "chat:message",
                                "data": {"content": assistant_message},
                            }
                        )
                        continue

                    if etype == "response.output_item.added":
                        item = event.get("item", {})
                        item_type = item.get("type", "")
                        item_status = item.get("status", "")
                        if (
                            item_type == "message"
                            and item_status == "in_progress"
                            and len(status_indicator._items) > 0
                        ):
                            assistant_message = await status_indicator.add(
                                assistant_message,
                                status_title="📝 Responding to the user…",
                                status_content="",
                            )
                            continue

                    if etype == "response.output_item.done":
                        item = event.get("item", {})
                        item_type = item.get("type", "")
                        item_name = item.get("name", "unnamed_tool")

                        # Reasoning is complete: Flush first, then “Done thinking!”  # >>> STREAM-REASONING
                        if item_type == "reasoning":
                            merged = (
                                "\n\n".join(
                                    summary_buffers[i] for i in sorted(summary_buffers)
                                ).strip()
                                if summary_buffers
                                else ""
                            )
                            if merged and self._last_rendered_summary != merged:
                                assistant_message = (
                                    await status_indicator.update_last_status(
                                        assistant_message, new_content=merged, emit=True
                                    )
                                )
                                self._last_rendered_summary = merged

                            if valves.PERSIST_TOOL_RESULTS:
                                hidden_uid_marker = persist_openai_response_items(
                                    metadata.get("chat_id"),
                                    metadata.get("message_id"),
                                    [item],
                                    openwebui_model,
                                )
                                if hidden_uid_marker:
                                    self.logger.debug(
                                        "Persisted item: %s", hidden_uid_marker
                                    )
                                    assistant_message += hidden_uid_marker
                                    await event_emitter(
                                        {
                                            "type": "chat:message",
                                            "data": {"content": assistant_message},
                                        }
                                    )

                            assistant_message = await status_indicator.add(
                                assistant_message,
                                status_title="🧠 Done thinking!",
                                status_content="",
                            )
                            continue

                        # Other types: Optional persistence and display status
                        if valves.PERSIST_TOOL_RESULTS and item_type != "message":
                            hidden_uid_marker = persist_openai_response_items(
                                metadata.get("chat_id"),
                                metadata.get("message_id"),
                                [item],
                                openwebui_model,
                            )
                            if hidden_uid_marker:
                                self.logger.debug(
                                    "Persisted item: %s", hidden_uid_marker
                                )
                                assistant_message += hidden_uid_marker
                                await event_emitter(
                                    {
                                        "type": "chat:message",
                                        "data": {"content": assistant_message},
                                    }
                                )

                        title = f"Running `{item_name}`"
                        content = ""
                        if item_type == "function_call":
                            title = f"🛠️ Running the {item_name} tool…"
                            arguments = json.loads(item.get("arguments") or "{}")
                            args_formatted = ", ".join(
                                f"{k}={json.dumps(v)}" for k, v in arguments.items()
                            )
                            content = f"```python\n{item_name}({args_formatted})\n```"
                        elif item_type == "web_search_call":
                            title = "🔍 Hmm, let me quickly check online…"
                            action = item.get("action", {})
                            if action.get("type") == "search":
                                query = action.get("query")
                                title = (
                                    f"🔍 Searching the web for: `{query}`"
                                    if query
                                    else "🔍 Searching the web"
                                )
                            elif action.get("type") == "open_page":
                                title = "🔍 Opening web page…"
                                url = item.get("url")
                                if url:
                                    content = f"URL: `{url}`"
                        elif item_type == "file_search_call":
                            title = "📂 Let me skim those files…"
                        elif item_type == "image_generation_call":
                            title = "🎨 Let me create that image…"
                        elif item_type == "local_shell_call":
                            title = "💻 Let me run that command…"
                        elif item_type == "mcp_call":
                            title = "🌐 Let me query the MCP server…"

                        if title:
                            assistant_message = await status_indicator.add(
                                assistant_message,
                                status_title=title,
                                status_content=content,
                            )
                        continue

                    if etype == "response.completed":
                        final_response = event.get("response", {})
                        body.input.extend(final_response.get("output", []))
                        break

                # Backstop: SSE was cut off by the agent in advance but some content is already available  # >>> STREAM-REASONING
                if final_response is None:
                    if assistant_message.strip():
                        await self._emit_notification(
                            event_emitter,
                            "Stream closed by upstream before completion.",
                            level="warning",
                        )
                        break
                    raise ValueError(
                        "No final response received from OpenAI Responses API."
                    )

                usage = final_response.get("usage", {})
                if usage:
                    usage["turn_count"] = 1
                    usage["function_call_count"] = sum(
                        1
                        for i in final_response["output"]
                        if i["type"] == "function_call"
                    )
                    total_usage = merge_usage_stats(total_usage, usage)
                    await self._emit_completion(
                        event_emitter, content="", usage=total_usage, done=False
                    )

                calls = [
                    i for i in final_response["output"] if i["type"] == "function_call"
                ]
                if calls:
                    function_outputs = await self._execute_function_calls(calls, tools)
                    if valves.PERSIST_TOOL_RESULTS:
                        hidden_uid_marker = persist_openai_response_items(
                            metadata.get("chat_id"),
                            metadata.get("message_id"),
                            function_outputs,
                            openwebui_model,
                        )
                        self.logger.debug("Persisted item: %s", hidden_uid_marker)
                        if hidden_uid_marker:
                            assistant_message += hidden_uid_marker
                            await event_emitter(
                                {
                                    "type": "chat:message",
                                    "data": {"content": assistant_message},
                                }
                            )
                    for output in function_outputs:
                        assistant_message = await status_indicator.add(
                            assistant_message,
                            status_title="🛠️ Received tool result",
                            status_content=f"```python\n{output.get('output', '')}\n```",
                        )
                    body.input.extend(function_outputs)
                else:
                    break

        except Exception as e:  # pragma: no cover
            await self._emit_error(
                event_emitter,
                f"Error: {str(e)}",
                show_error_message=True,
                show_error_log_citation=True,
                done=True,
            )

        finally:
            if not status_indicator._done and status_indicator._items:
                assistant_message = await status_indicator.finish(assistant_message)

            if valves.LOG_LEVEL != "INHERIT":
                if event_emitter:
                    session_id = SessionLogger.session_id.get()
                    logs = SessionLogger.logs.get(session_id, [])
                    if logs:
                        await self._emit_citation(
                            event_emitter, "\n".join(logs), "Logs"
                        )

            await self._emit_completion(
                event_emitter, content="", usage=total_usage, done=True
            )
            logs_by_msg_id.clear()
            SessionLogger.logs.pop(SessionLogger.session_id.get(), None)

            chat_id = metadata.get("chat_id")
            message_id = metadata.get("message_id")
            if chat_id and message_id and emitted_citations:
                Chats.upsert_message_to_chat_by_id_and_message_id(
                    chat_id, message_id, {"sources": emitted_citations}
                )
            return assistant_message

    async def _run_nonstreaming_loop(
        self,
        body: ResponsesBody,
        valves: Pipe.Valves,
        event_emitter: Callable[[Dict[str, Any]], Awaitable[None]],
        metadata: Dict[str, Any] = {},
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        openwebui_model_id = metadata.get("model", {}).get("id", "")
        tools = tools or {}
        assistant_message = ""
        total_usage: Dict[str, Any] = {}
        reasoning_map: dict[int, str] = {}

        status_indicator = ExpandableStatusIndicator(event_emitter)
        status_indicator._done = False

        model_family = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", body.model)
        if body.model.startswith("gpt-5"):
            model_family = "gpt-5"

        if model_family in FEATURE_SUPPORT["reasoning"]:
            assistant_message = await status_indicator.add(
                assistant_message,
                status_title="Thinking…",
                status_content=(
                    "Reading the question and building a plan to answer it. This may take a moment."
                ),
            )

        try:
            for loop_idx in range(valves.MAX_FUNCTION_CALL_LOOPS):
                response = await self.send_openai_responses_nonstreaming_request(
                    body.model_dump(exclude_none=True),
                    api_key=valves.API_KEY,
                    base_url=valves.BASE_URL,
                    valves=valves,
                )

                items = response.get("output", [])

                for item in items:
                    item_type = item.get("type")
                    if item_type == "message":
                        for content in item.get("content", []):
                            if content.get("type") == "output_text":
                                assistant_message += content.get("text", "")
                    elif item_type == "reasoning_summary_text":
                        idx = item.get("summary_index", 0)
                        text = item.get("text", "")
                        if text:
                            reasoning_map[idx] = reasoning_map.get(idx, "") + text
                            title_match = re.findall(r"\*\*(.+?)\*\*", text)
                            title = (
                                title_match[-1].strip() if title_match else "Thinking…"
                            )
                            content = re.sub(r"\*\*(.+?)\*\*", "", text).strip()
                            assistant_message = await status_indicator.add(
                                assistant_message,
                                status_title="🧠 " + title,
                                status_content=content,
                            )
                    elif item_type == "reasoning":
                        parts = "\n\n---".join(
                            reasoning_map[i] for i in sorted(reasoning_map)
                        )
                        snippet = (
                            f'<details type="{__name__}.reasoning" done="true">\n'
                            f"<summary>Done thinking!</summary>\n{parts}</details>"
                        )
                        assistant_message += snippet
                        reasoning_map.clear()
                        if valves.PERSIST_TOOL_RESULTS:
                            hidden_uid_marker = persist_openai_response_items(
                                metadata.get("chat_id"),
                                metadata.get("message_id"),
                                [item],
                                metadata.get("model", {}).get("id"),
                            )
                            self.logger.debug("Persisted item: %s", hidden_uid_marker)
                            assistant_message += hidden_uid_marker
                    else:
                        if valves.PERSIST_TOOL_RESULTS:
                            hidden_uid_marker = persist_openai_response_items(
                                metadata.get("chat_id"),
                                metadata.get("message_id"),
                                [item],
                                metadata.get("model", {}).get("id"),
                            )
                        else:
                            hidden_uid_marker = ""
                        self.logger.debug("Persisted item: %s", hidden_uid_marker)
                        assistant_message += hidden_uid_marker
                        title = f"Running `{item.get('name', 'unnamed_tool')}`"
                        content = ""
                        if item_type == "function_call":
                            title = f"🛠️ Running the {item.get('name', 'unnamed_tool')} tool…"
                            arguments = json.loads(item.get("arguments") or "{}")
                            args_formatted = ", ".join(
                                f"{k}={json.dumps(v)}" for k, v in arguments.items()
                            )
                            content = f"```python\n{item.get('name', 'unnamed_tool')}({args_formatted})\n```"
                        elif item_type == "web_search_call":
                            title = "🔍 Hmm, let me quickly check online…"
                            action = item.get("action", {})
                            if action.get("type") == "search":
                                query = action.get("query")
                                title = (
                                    f"🔍 Searching the web for: `{query}`"
                                    if query
                                    else "🔍 Searching the web"
                                )
                            elif action.get("type") == "open_page":
                                title = "🔍 Opening web page…"
                                url = item.get("url")
                                if url:
                                    content = f"URL: `{url}`"
                        elif item_type == "file_search_call":
                            title = "📂 Let me skim those files…"
                        elif item_type == "image_generation_call":
                            title = "🎨 Let me create that image…"
                        elif item_type == "local_shell_call":
                            title = "💻 Let me run that command…"
                        elif item_type == "mcp_call":
                            title = "🌐 Let me query the MCP server…"
                        if title:
                            assistant_message = await status_indicator.add(
                                assistant_message,
                                status_title=title,
                                status_content=content,
                            )

                usage = response.get("usage", {})
                if usage:
                    usage["turn_count"] = 1
                    usage["function_call_count"] = sum(
                        1 for i in items if i.get("type") == "function_call"
                    )
                    total_usage = merge_usage_stats(total_usage, usage)
                    await self._emit_completion(
                        event_emitter, content="", usage=total_usage, done=False
                    )

                body.input.extend(items)

                calls = [i for i in items if i.get("type") == "function_call"]
                if calls:
                    function_outputs = await self._execute_function_calls(calls, tools)
                    if valves.PERSIST_TOOL_RESULTS:
                        hidden_uid_marker = persist_openai_response_items(
                            metadata.get("chat_id"),
                            metadata.get("message_id"),
                            function_outputs,
                            openwebui_model_id,
                        )
                        self.logger.debug("Persisted item: %s", hidden_uid_marker)
                        assistant_message += hidden_uid_marker
                    for output in function_outputs:
                        assistant_message = await status_indicator.add(
                            assistant_message,
                            status_title="🛠️ Received tool result",
                            status_content=f"```python\n{output.get('output', '')}\n```",
                        )
                    body.input.extend(function_outputs)
                else:
                    break

            final_text = assistant_message.strip()
            if not status_indicator._done and status_indicator._items:
                final_text = await status_indicator.finish(final_text)
            return final_text

        except Exception as e:  # pragma: no cover
            await self._emit_error(
                event_emitter,
                e,
                show_error_message=True,
                show_error_log_citation=True,
                done=True,
            )
        finally:
            if not status_indicator._done and status_indicator._items:
                assistant_message = await status_indicator.finish(assistant_message)
            logs_by_msg_id.clear()
            SessionLogger.logs.pop(SessionLogger.session_id.get(), None)

    # 4.4 Task Model Handling
    async def _run_task_model_request(
        self, body: Dict[str, Any], valves: Pipe.Valves
    ) -> Dict[str, Any]:
        task_body = {
            "model": body.get("model"),
            "instructions": body.get("instructions", ""),
            "input": body.get("input", ""),
            "stream": False,
        }
        response = await self.send_openai_responses_nonstreaming_request(
            task_body, api_key=valves.API_KEY, base_url=valves.BASE_URL, valves=valves
        )
        text_parts: list[str] = []
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text_parts.append(content.get("text", ""))
        message = "".join(text_parts)
        return message

    # 4.5 HTTP helpers
    async def send_openai_responses_streaming_request(
        self,
        request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        valves = valves or self.valves
        self.session = await self._get_or_init_http_session()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        url = base_url.rstrip("/") + "/responses"
        # Normalization and fallback strategy: use the original (possibly including text/minimal) first, and fall back if 400
        prepared_variants = list(self._iter_request_variants(request_body, valves))
        last_error: Exception | None = None
        for payload in prepared_variants:
            buf = bytearray()
            try:
                async with self.session.post(
                    url, json=payload, headers=headers
                ) as resp:
                    if resp.status >= 400:
                        body_text = await resp.text()
                        self.logger.error(
                            "Responses(stream) HTTP %s: %s",
                            resp.status,
                            body_text[:800],
                        )
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=body_text,
                            headers=resp.headers,
                        )
                    async for chunk in resp.content.iter_chunked(4096):
                        buf.extend(chunk)
                        start_idx = 0
                        while True:
                            newline_idx = buf.find(b"\n", start_idx)
                            if newline_idx == -1:
                                break
                            line = buf[start_idx:newline_idx].strip()
                            start_idx = newline_idx + 1
                            if (
                                not line
                                or line.startswith(b":")
                                or not line.startswith(b"data:")
                            ):
                                continue
                            data_part = line[5:].strip()
                            if data_part == b"[DONE]":
                                return
                            yield json.loads(data_part.decode("utf-8"))
                        if start_idx > 0:
                            del buf[:start_idx]
                    return
            except aiohttp.ClientResponseError as cre:
                last_error = cre
                if cre.status == 400:
                    continue
                raise
        if last_error:
            raise last_error

    async def send_openai_responses_nonstreaming_request(
        self,
        request_params: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
    ) -> Dict[str, Any]:
        valves = valves or self.valves
        self.session = await self._get_or_init_http_session()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = base_url.rstrip("/") + "/responses"
        prepared_variants = list(self._iter_request_variants(request_params, valves))
        last_error: Exception | None = None
        for payload in prepared_variants:
            try:
                async with self.session.post(
                    url, json=payload, headers=headers
                ) as resp:
                    if resp.status >= 400:
                        body_text = await resp.text()
                        self.logger.error(
                            "Responses(non-stream) HTTP %s: %s",
                            resp.status,
                            body_text[:800],
                        )
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=body_text,
                            headers=resp.headers,
                        )
                    return await resp.json()
            except aiohttp.ClientResponseError as cre:
                last_error = cre
                if cre.status == 400:
                    continue
                raise
        if last_error:
            raise last_error

    async def _get_or_init_http_session(self) -> aiohttp.ClientSession:
        if self.session is not None and not self.session.closed:
            self.logger.debug("Reusing existing aiohttp.ClientSession")
            return self.session
        self.logger.debug("Creating new aiohttp.ClientSession")
        connector = aiohttp.TCPConnector(
            limit=50, limit_per_host=10, keepalive_timeout=75, ttl_dns_cache=300
        )
        timeout = aiohttp.ClientTimeout(connect=30, sock_connect=30, sock_read=3600)
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            json_serialize=json.dumps,
        )
        return session

    # 4.6.x Request Body Normalization and Fallback
    def _prepare_openai_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        prepared = json.loads(json.dumps(payload))
        tools = prepared.get("tools")
        if isinstance(tools, list):
            normalized_tools: list[dict] = []
            for t in tools:
                if (
                    isinstance(t, dict)
                    and t.get("type") == "function"
                    and "function" not in t
                ):
                    fn = {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    }
                    nt = {
                        k: v
                        for k, v in t.items()
                        if k not in {"name", "description", "parameters"}
                    }
                    nt["function"] = fn
                    normalized_tools.append(nt)
                else:
                    normalized_tools.append(t)
            prepared["tools"] = normalized_tools
        return prepared

    def _iter_request_variants(self, payload: dict[str, Any], valves: "Pipe.Valves"):
        base = self._prepare_openai_payload(payload)
        yielded = set()

        def _key(d):
            try:
                return json.dumps(d, sort_keys=True)
            except Exception:
                return str(id(d))

        # 1) Original (may contain text / minimal)
        k = _key(base)
        if k not in yielded:
            yielded.add(k)
            yield base

        # 2) Remove the text field (some accounts do not support Responses.text)
        if "text" in base:
            p2 = json.loads(json.dumps(base))
            p2.pop("text", None)
            k = _key(p2)
            if k not in yielded:
                yielded.add(k)
                yield p2

        # 3) Set reasoning.effort from minimal → low (if present)
        r = (
            (base.get("reasoning") or {})
            if isinstance(base.get("reasoning"), dict)
            else {}
        )
        if r.get("effort") == "minimal":
            p3 = json.loads(json.dumps(base))
            if isinstance(p3.get("reasoning"), dict):
                p3["reasoning"]["effort"] = "low"
            k = _key(p3)
            if k not in yielded:
                yielded.add(k)
                yield p3

    # 4.6 Tool Execution
    @staticmethod
    async def _execute_function_calls(
        calls: list[dict], tools: dict[str, dict[str, Any]]
    ) -> list[dict]:
        def _make_task(call):
            tool_cfg = tools.get(call["name"])
            if not tool_cfg:
                return asyncio.sleep(0, result="Tool not found")
            fn = tool_cfg["callable"]
            args = json.loads(call["arguments"])
            if inspect.iscoroutinefunction(fn):
                return fn(**args)
            else:
                return asyncio.to_thread(fn, **args)

        tasks = [_make_task(call) for call in calls]
        results = await asyncio.gather(*tasks)
        return [
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": str(result),
            }
            for call, result in zip(calls, results)
        ]

    # 4.7 Emitters
    async def _emit_error(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]],
        error_obj: Exception | str,
        *,
        show_error_message: bool = True,
        show_error_log_citation: bool = False,
        done: bool = False,
    ) -> None:
        error_message = str(error_obj)
        self.logger.error("Error: %s", error_message)
        if show_error_message and event_emitter:
            await event_emitter(
                {
                    "type": "chat:completion",
                    "data": {"error": {"message": error_message}, "done": done},
                }
            )
            if show_error_log_citation:
                session_id = SessionLogger.session_id.get()
                logs = SessionLogger.logs.get(session_id, [])
                if logs:
                    await self._emit_citation(
                        event_emitter, "\n".join(logs), "Error Logs"
                    )
                else:
                    self.logger.warning(
                        "No debug logs found for session_id %s", session_id
                    )

    async def _emit_citation(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        document: str | list[str],
        source_name: str,
    ) -> None:
        if event_emitter is None:
            return
        doc_text = "\n".join(document) if isinstance(document, list) else document
        await event_emitter(
            {
                "type": "citation",
                "data": {
                    "document": [doc_text],
                    "metadata": [
                        {
                            "date_accessed": datetime.datetime.now().isoformat(),
                            "source": source_name,
                        }
                    ],
                    "source": {"name": source_name},
                },
            }
        )

    async def _emit_completion(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        *,
        content: str | None = "",
        title: str | None = None,
        usage: dict[str, Any] | None = None,
        done: bool = True,
    ) -> None:
        if event_emitter is None:
            return
        await event_emitter(
            {
                "type": "chat:completion",
                "data": {
                    "done": done,
                    "content": content,
                    **({"title": title} if title is not None else {}),
                    **({"usage": usage} if usage is not None else {}),
                },
            }
        )

    async def _emit_status(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        description: str,
        *,
        done: bool = False,
        hidden: bool = False,
    ) -> None:
        if event_emitter is None:
            return
        await event_emitter(
            {
                "type": "status",
                "data": {"description": description, "done": done, "hidden": hidden},
            }
        )

    async def _emit_notification(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        content: str,
        *,
        level: Literal["info", "success", "warning", "error"] = "info",
    ) -> None:
        if event_emitter is None:
            return
        await event_emitter(
            {"type": "notification", "data": {"type": level, "content": content}}
        )

    def _merge_valves(self, global_valves, user_valves) -> "Pipe.Valves":
        if not user_valves:
            return global_valves
        update = {
            k: v
            for k, v in user_valves.model_dump().items()
            if v is not None and str(v).lower() != "inherit"
        }
        return global_valves.model_copy(update=update)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Utility Classes
# ─────────────────────────────────────────────────────────────────────────────
logs_by_msg_id: dict[str, list[str]] = defaultdict(list)
current_session_id: ContextVar[str | None] = ContextVar(
    "current_session_id", default=None
)


class SessionLogger:
    session_id = ContextVar("session_id", default=None)
    log_level = ContextVar("log_level", default=logging.INFO)
    logs = defaultdict(lambda: deque(maxlen=2000))

    @classmethod
    def get_logger(cls, name=__name__):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.filters.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        def filter(record):
            record.session_id = cls.session_id.get()
            return record.levelno >= cls.log_level.get()

        logger.addFilter(filter)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(
            logging.Formatter("[%(levelname)s] [%(session_id)s] %(message)s")
        )
        logger.addHandler(console)
        mem = logging.Handler()
        mem.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        mem.emit = lambda r: (
            cls.logs[r.session_id].append(mem.format(r)) if r.session_id else None
        )
        logger.addHandler(mem)
        return logger


class ExpandableStatusIndicator:
    _BLOCK_RE = re.compile(
        r"<details\s+type=\"status\".*?</details>", re.DOTALL | re.IGNORECASE
    )

    def __init__(
        self,
        event_emitter: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
    ) -> None:
        self._event_emitter = event_emitter
        self._items: List[Tuple[str, List[str]]] = []
        self._started = time.perf_counter()
        self._done: bool = False

    async def add(
        self,
        assistant_message: str,
        status_title: str,
        status_content: Optional[str] = None,
        *,
        emit: bool = True,
    ) -> str:
        self._assert_not_finished("add")
        if not self._items or self._items[-1][0] != status_title:
            self._items.append((status_title, []))
        if status_content:
            self._items[-1][1].append(status_content.strip())
        return await self._render(assistant_message, emit)

    async def update_last_status(
        self,
        assistant_message: str,
        *,
        new_title: Optional[str] = None,
        new_content: Optional[str] = None,
        emit: bool = True,
    ) -> str:
        self._assert_not_finished("update_last_status")
        if not self._items:
            return await self.add(
                assistant_message, new_title or "Status", new_content, emit=emit
            )
        title, subs = self._items[-1]
        if new_title:
            title = new_title
        if new_content is not None:
            subs = [new_content.strip()]
        self._items[-1] = (title, subs)
        return await self._render(assistant_message, emit)

    async def finish(self, assistant_message: str, *, emit: bool = True) -> str:
        if self._done:
            return assistant_message
        elapsed = time.perf_counter() - self._started
        self._items.append((f"Finished in {elapsed:.1f} s", []))
        self._done = True
        return await self._render(assistant_message, emit)

    def _assert_not_finished(self, method: str) -> None:
        if self._done:
            raise RuntimeError(
                f"Cannot call {method}(): status indicator is already finished."
            )

    async def _render(self, assistant_message: str, emit: bool) -> str:
        block = self._render_status_block()
        full_msg = (
            self._BLOCK_RE.sub(block, assistant_message, 1)
            if self._BLOCK_RE.search(assistant_message)
            else f"{block}{assistant_message}"
        )
        if emit and self._event_emitter:
            await self._event_emitter(
                {"type": "chat:message", "data": {"content": full_msg}}
            )
        return full_msg

    def _render_status_block(self) -> str:
        lines: List[str] = []
        for title, subs in self._items:
            lines.append(f"- **{title}**")
            for sub in subs:
                sub_lines = sub.splitlines()
                if sub_lines:
                    lines.append(f"  - {sub_lines[0]}")
                    if len(sub_lines) > 1:
                        lines.extend(
                            textwrap.indent(
                                "\n".join(sub_lines[1:]), "    "
                            ).splitlines()
                        )
        body_md = "\n".join(lines) if lines else "_No status yet._"
        summary = self._items[-1][0] if self._items else "Working…"
        return f'<details type="status" done="{str(self._done).lower()}">\n<summary>{summary}</summary>\n\n{body_md}\n\n---</details>'


# ─────────────────────────────────────────────────────────────────────────────
# 6. Framework Integration Helpers
# ─────────────────────────────────────────────────────────────────────────────
def persist_openai_response_items(
    chat_id: str, message_id: str, items: List[Dict[str, Any]], openwebui_model_id: str
) -> str:
    if not items:
        return ""
    chat_model = Chats.get_chat_by_id(chat_id)
    if not chat_model:
        return ""
    pipe_root = chat_model.chat.setdefault("openai_responses_pipe", {"__v": 3})
    items_store = pipe_root.setdefault("items", {})
    messages_index = pipe_root.setdefault("messages_index", {})
    message_bucket = messages_index.setdefault(
        message_id, {"role": "assistant", "done": True, "item_ids": []}
    )
    now = int(datetime.datetime.utcnow().timestamp())
    hidden_uid_markers: List[str] = []
    for payload in items:
        item_id = generate_item_id()
        items_store[item_id] = {
            "model": openwebui_model_id,
            "created_at": now,
            "payload": payload,
            "message_id": message_id,
        }
        message_bucket["item_ids"].append(item_id)
        hidden_uid_marker = wrap_marker(
            create_marker(payload.get("type", "unknown"), ulid=item_id)
        )
        hidden_uid_markers.append(hidden_uid_marker)
    Chats.update_chat_by_id(chat_id, chat_model.chat)
    return "".join(hidden_uid_markers)


# ─────────────────────────────────────────────────────────────────────────────
# 7. General-Purpose Utility Functions
# ─────────────────────────────────────────────────────────────────────────────
def merge_usage_stats(total, new):
    for k, v in new.items():
        if isinstance(v, dict):
            total[k] = merge_usage_stats(total.get(k, {}), v)
        elif isinstance(v, (int, float)):
            total[k] = total.get(k, 0) + v
        else:
            total[k] = v if v is not None else total.get(k, 0)
    return total


def update_openwebui_model_param(openwebui_model_id: str, field: str, value: Any):
    model = Models.get_model_by_id(openwebui_model_id)
    if not model:
        return
    form_data = model.model_dump()
    form_data["params"] = dict(model.params or {})
    if form_data["params"].get(field) == value:
        return
    form_data["params"][field] = value
    form = ModelForm(**form_data)
    Models.update_model_by_id(openwebui_model_id, form)


def remove_details_tags_by_type(text: str, removal_types: list[str]) -> str:
    pattern_types = "|".join(map(re.escape, removal_types))
    pattern = rf'<details\b[^>]*\btype=["\'](?:{pattern_types})["\'][^>]*>.*?</details>'
    return re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)


ULID_LENGTH = 16
CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_SENTINEL = "[openai_responses:v2:"
_RE = re.compile(
    rf"\[openai_responses:v2:(?P<kind>[a-z0-9_]{{2,30}}):(?P<ulid>[A-Z0-9]{{{ULID_LENGTH}}})(?:\?(?P<query>[^\]]+))?\]:\s*#",
    re.I,
)


def _qs(d: dict[str, str]) -> str:
    return "&".join(f"{k}={v}" for k, v in d.items()) if d else ""


def _parse_qs(q: str) -> dict[str, str]:
    return dict(p.split("=", 1) for p in q.split("&")) if q else {}


def generate_item_id() -> str:
    return "".join(secrets.choice(CROCKFORD_ALPHABET) for _ in range(ULID_LENGTH))


def create_marker(
    item_type: str,
    *,
    ulid: str | None = None,
    model_id: str | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    if not re.fullmatch(r"[a-z0-9_]{2,30}", item_type):
        raise ValueError("item_type must be 2-30 chars of [a-z0-9_]")
    meta = {**(metadata or {})}
    if model_id:
        meta["model"] = model_id
    base = f"openai_responses:v2:{item_type}:{ulid or generate_item_id()}"
    return f"{base}?{_qs(meta)}" if meta else base


def wrap_marker(marker: str) -> str:
    return f"\n[{marker}]: #\n"


def contains_marker(text: str) -> bool:
    return _SENTINEL in text


def parse_marker(marker: str) -> dict:
    if not marker.startswith("openai_responses:v2:"):
        raise ValueError("not a v2 marker")
    _, _, kind, rest = marker.split(":", 3)
    uid, _, q = rest.partition("?")
    return {"version": "v2", "item_type": kind, "ulid": uid, "metadata": _parse_qs(q)}


def extract_markers(text: str, *, parsed: bool = False) -> list:
    found = []
    for m in _RE.finditer(text):
        raw = f"openai_responses:v2:{m.group('kind')}:{m.group('ulid')}"
        if m.group("query"):
            raw += f"?{m.group('query')}"
        found.append(parse_marker(raw) if parsed else raw)
    return found


def split_text_by_markers(text: str) -> list[dict]:
    segments = []
    last = 0
    for m in _RE.finditer(text):
        if m.start() > last:
            segments.append({"type": "text", "text": text[last : m.start()]})
        raw = f"openai_responses:v2:{m.group('kind')}:{m.group('ulid')}"
        if m.group("query"):
            raw += f"?{m.group('query')}"
        segments.append({"type": "marker", "marker": raw})
        last = m.end()
    if last < len(text):
        segments.append({"type": "text", "text": text[last:]})
    return segments


def fetch_openai_response_items(
    chat_id: str, item_ids: List[str], *, openwebui_model_id: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    chat_model = Chats.get_chat_by_id(chat_id)
    if not chat_model:
        return {}
    items_store = chat_model.chat.get("openai_responses_pipe", {}).get("items", {})
    lookup: Dict[str, Dict[str, Any]] = {}
    for item_id in item_ids:
        item = items_store.get(item_id)
        if not item:
            continue
        if openwebui_model_id:
            if item.get("model", "") != openwebui_model_id:
                continue
        lookup[item_id] = item.get("payload", {})
    return lookup
