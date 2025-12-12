# Limit conversation context per user

Filter for Open WebUI that clips conversation context per user (by ID or email).

## Features

- Admins (`role=admin`) have unlimited context.
- Per-user limits via **user id or email**.
- Priority: **id → email**.
- Fallback to `default_max_messages`.
- Optionally keep the first `system` message.

## Key Settings (Valves)

- `default_max_messages`
- `user_limits_json`
- `keep_system`
- `debug_logging`

## Example

```json
{
  "test@abc.com": 16,
  "c162258a-e1d3-48e3-8bfb-e513b1f15e83": 8
}
```

Copy `user_map_context_clip_filter.py` to Open WebUI under **Admin ▸ Filters** to enable.

---

# 按用户限制上下文消息条数

这是一个 Open WebUI 的 Filter 插件，用于根据用户 **ID / Email** 对上下文消息进行裁剪。

## 功能

- 管理员（`role=admin`）无限制上下文。
- 支持按 **user id / email** 设置不同 `max_messages`。
- 匹配优先级：**id → email**。
- 未命中用户使用 `default_max_messages`。
- 可选始终保留第一条 `system` 消息。

## 核心配置（Valves）

- `default_max_messages`
- `user_limits_json`
- `keep_system`
- `debug_logging`

## 示例

```json
{
  "test@abc.com": 16,
  "c162258a-e1d3-48e3-8bfb-e513b1f15e83": 8
}
```

将 `user_map_context_clip_filter.py` 上传到 Open WebUI 的 **Admin ▸ Filters** 即可启用。

