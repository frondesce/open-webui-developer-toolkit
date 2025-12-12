"""
title: Limit conversation context per user
id: user_map_context_clip_filter
author: gpt-5.1
version: 0.7.1
description: Limit context messages by admin-defined per-user map, with admin users unlimited.
"""

from typing import Optional, Any, Dict
from pydantic import BaseModel, Field
import json


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )

        default_max_messages: int = Field(
            default=2,
            description="默认最大保留消息数（未匹配到用户时使用，消息条数）。",
        )

        keep_system: bool = Field(
            default=True, description="是否始终保留第一条 system 提示词。"
        )

        user_limits_json: str = Field(
            default="{}",
            description=(
                "管理员配置：用户上限映射表（JSON）。key 可用 email 或 user id。\n"
                "示例：\n"
                '{"test@abc.com":16,"c162258a-e1d3-48e3-8bfb-e513b1f15e83":8}'
            ),
        )

        debug_logging: bool = Field(
            default=False, description="是否打印调试日志到后端日志（Docker logs）。"
        )

    def __init__(self):
        self.valves = self.Valves()

    # ---------- user 解析 ----------
    def _resolve_current_user(
        self,
        body: Dict[str, Any],
        __user__: Optional[dict],
        user: Optional[dict],
    ) -> Optional[dict]:
        if isinstance(__user__, dict):
            return __user__
        if isinstance(user, dict):
            return user

        meta = body.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("user"), dict):
            return meta["user"]

        if isinstance(body.get("user"), dict):
            return body["user"]

        return None

    # ---------- 解析管理员维护的映射表 ----------
    def _parse_user_limits(self) -> Dict[str, int]:
        try:
            data = json.loads(self.valves.user_limits_json or "{}")
            if not isinstance(data, dict):
                return {}
            out: Dict[str, int] = {}
            for k, v in data.items():
                try:
                    n = int(v)
                    if n > 0:
                        out[str(k)] = n
                except Exception:
                    continue
            return out
        except Exception:
            return {}

    # ---------- 核心：决定当前用户的 max_messages ----------
    def _get_limit_for_user(self, user_obj: Optional[dict]) -> Optional[int]:
        if not isinstance(user_obj, dict):
            return None

        # 1️⃣ 管理员直接无限制
        if user_obj.get("role") == "admin":
            return None  # None = 不裁剪

        user_limits = self._parse_user_limits()
        uid = user_obj.get("id")
        email = user_obj.get("email")

        # 2️⃣ 优先按 id，其次 email
        if uid is not None and str(uid) in user_limits:
            return user_limits[str(uid)]

        if email is not None and str(email) in user_limits:
            return user_limits[str(email)]

        # 3️⃣ 未命中，走默认
        return self.valves.default_max_messages

    def inlet(
        self, body: dict, __user__: Optional[dict] = None, user: Optional[dict] = None
    ) -> dict:
        messages = body.get("messages", [])
        if not messages:
            return body

        current_user = self._resolve_current_user(body, __user__, user)
        limit = self._get_limit_for_user(current_user)

        before = len(messages)

        # 找 system
        system_prompt = None
        if self.valves.keep_system:
            system_prompt = next(
                (m for m in messages if m.get("role") == "system"),
                None,
            )

        # ---------- 不限制（管理员） ----------
        if limit is None:
            new_messages = messages

        # ---------- 有限制 ----------
        else:
            if system_prompt:
                non_system = [
                    m
                    for m in messages
                    if m is not system_prompt and m.get("role") != "system"
                ]
                clipped = non_system[-limit:]
                new_messages = [system_prompt] + clipped
            else:
                new_messages = messages[-limit:]

        body["messages"] = new_messages
        after = len(new_messages)

        if self.valves.debug_logging:
            try:
                print("\n[UserMapContextClipFilter DEBUG v0.7.1]")
                if isinstance(current_user, dict):
                    print(
                        "  user:",
                        {
                            "id": current_user.get("id"),
                            "email": current_user.get("email"),
                            "role": current_user.get("role"),
                        },
                    )
                print(
                    "  selected limit:",
                    limit if limit is not None else "UNLIMITED (admin)",
                )
                print("  messages before:", before, "after:", after)
                print("[UserMapContextClipFilter DEBUG END]\n")
            except Exception:
                pass

        return body

