# OpenAI Responses Manifold

Enables advanced OpenAI features (function calling, web search, visible reasoning summaries, and more) directly in [Open WebUI](https://github.com/open-webui/open-webui).

**Now supports OpenAI’s GPT-5 family in the API — [Learn more](#gpt-5-model-support).**

This project started as an internal tool (200+ hours of optimization and testing) and is now open-sourced as a way to give back to the Open WebUI community.

> ✨ Like the manifold? 
> Show your support by sharing it with others and providing feedback through [GitHub Discussions](https://github.com/jrkropp/open-webui-developer-toolkit/discussions).  
> 💡 Pull Requests are welcome!


## Contents

* [Setup](#setup)
* [Features](#features)
* [Advanced Features](#advanced-features)
* [Tested Models](#tested-models)
* [GPT‑5 Model Support](#gpt5-model-support)
* [How It Works (Design Notes)](#how-it-works-design-notes)
* [Troubleshooting / FAQ](#troubleshooting--faq)

## Setup
1. In **Open WebUI ▸ Admin Panel ▸ Functions**, click **Import from Link**.
   
   <img width="450" alt="image" src="https://github.com/user-attachments/assets/4a5a0355-e0af-4fb8-833e-7d3dfb7f10e3" />

2. Paste one of the following links, depending on which version you want:

   * **Main** (recommended) – Stable production build with regular, tested updates:

     ```
     https://github.com/jrkropp/open-webui-developer-toolkit/blob/main/functions/pipes/openai_responses_manifold/openai_responses_manifold.py
     ```

   * **Alpha Preview** – Pre-release build with early features, typically 2–4 weeks ahead of main:

     ```
     https://github.com/jrkropp/open-webui-developer-toolkit/blob/alpha-preview/functions/pipes/openai_responses_manifold/openai_responses_manifold.py
     ```

3. **⚠️ Important: Set the Function ID to `openai_responses`.**
   
   This value is currently hardcoded in the pipe and must match exactly. It will become configurable in a future release.
   
   <img width="800" alt="image" src="https://github.com/user-attachments/assets/ffd3dd72-cf39-43fa-be36-56c6ac41477d" />

4. Done! 🎉

## Features

| Feature                            | Status          | Last updated | Notes                                                                                                                                                                                                                                                                                                        |
| ---------------------------------- | --------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Native function calling**        | ✅ GA            | 2025-06-04   | Sends full JSON tool specs directly to OpenAI models that support function calling. This API-enforced method ensures reliable, validated tool calls and allows **multiple tool calls in a single response**. Much more robust than Open WebUI’s default single-call router.                                  |
| **Visible reasoning summaries**    | ✅ GA            | 2025-08-07   | Enables OpenAI’s *reasoning summaries*, which explain how the model arrives at its answers. Displayed in collapsible `<details>` blocks for transparency, trust, and easier debugging.                                                                                                                       |
| **Encrypted reasoning tokens**     | ✅ GA            | 2025-08-07   | Saves reasoning tokens across tool-calling “turns” (and optionally whole conversations). Prevents the model from having to **re-reason from scratch** after each tool call, making responses faster, cheaper, and more cache-friendly.                                                                       |
| **Optimized token caching**        | ✅ GA            | 2025-06-03   | Ensures all tokens—including hidden ones like tool calls and reasoning—are re-sent in the same order. This unlocks OpenAI’s token caching, cutting **input costs by 50–75%** and lowering latency.                                                                                                           |
| **Web search tool**                | ✅ GA            | 2025-06-03   | Injects OpenAI’s `web_search` tool into supported models. With the valve enabled, the model can **self-decide when to search**. Alternatively, filters can toggle it on/off (mirroring ChatGPT’s behavior).                                                                                                  |
| **Task model support**             | ✅ GA            | 2025-08-07   | Detects when a request is for an **External Task Model** and routes it separately. Makes manifold models usable for lightweight routing or background tasks (e.g., with `gpt-4.1-nano`).                                                                                                                     |
| **Streaming responses (SSE)**      | ✅ GA            | 2025-06-04   | Supports **real-time streaming**, so users can watch responses appear live as the model generates them.                                                                                                                                                                                                      |
| **Usage pass-through**             | ✅ GA            | 2025-06-04   | Forwards API usage stats (tokens, caching data, etc.) into Open WebUI, visible in the frontend (hover the ℹ️ icon). Gives users **transparent cost and performance insights**.                                                                                                                               |
| **Response item persistence**      | ✅ GA            | 2025-06-27   | Persists hidden items (reasoning, tool calls) by embedding unique IDs in Markdown. Stored separately in the DB and reattached in later turns, so the **conversation can be rebuilt exactly** without losing hidden events.                                                                                   |
| **Open WebUI Notes compatibility** | ✅ GA            | 2025-07-14   | Works seamlessly with Open WebUI’s new **Notes feature** (currently preview). Ensures manifold-based models work even when chats are ephemeral (no `chat_id`).                                                                                                                                               |
| **Expandable status output**       | ✅ GA            | 2025-07-01   | Uses `<details>` blocks to create collapsible panels that show reasoning, tool calls, or progress. A workaround until Open WebUI adds native status panels ([discussion #14974](https://github.com/open-webui/open-webui/discussions/14974)).                                                                |
| **Inline citation events**         | ✅ GA (basic)    | 2025-07-28   | Adds inline citations (e.g., `[1]`) for **web search results**. Basic implementation works but still being refined. Style is adjustable with the `CITATION_STYLE` valve.                                                                                                                                     |
| **Truncation control**             | ✅ GA            | 2025-06-10   | Defaults to `auto`, meaning if token limits are exceeded, older context is trimmed instead of failing. You can also set `max_tokens` via custom parameters. See OpenAI’s [Responses API docs on truncation](https://community.openai.com/t/introducing-the-responses-api/1140929/12?utm_source=chatgpt.com). |
| **Custom param pass-through**      | ✅ GA            | 2025-06-14   | Supports Open WebUI’s **Custom Parameters**. Any params set in the GUI are passed through to OpenAI (e.g., `max_tokens` → `max_output_tokens`). Lets users fine-tune behavior without editing code.                                                                                                          |
| **Regenerate → `text.verbosity`**  | ✅ GA            | 2025-08-11   | Open WebUI v0.6.19 added regenerate buttons for “More Concise” / “Add Details.” The manifold maps these to the `text.verbosity` parameter for GPT-5 models. Falls back to prompt injection if not supported.                                                                                                 |
| **Filter-injected tools**          | ✅ GA            | 2025-08-28   | Lets developers build **companion filters** that add tools under `body.extra_tools`. The manifold merges these into `body.tools` before sending to OpenAI, removing duplicates. Enables features like **web search toggles** without breaking native function calling.                                       |
| **Image input (vision)**           | 🔄 In progress  | 2025-06-03   | Supports basic image input (Open WebUI converts uploads to base64 and forwards them). Works but inefficient for large images. A future version will switch to OpenAI’s **file upload API** for better performance.                                                                                           |
| **Image generation tool**          | 🕒 Backlog      | 2025-06-03   | Planned support for **creating and editing images** with OpenAI. Will include **multi-turn editing**, but depends on efficient image handling via file uploads first.                                                                                                                                        |
| **File upload / file search**      | 🕒 Backlog      | 2025-06-03   | Planned support for uploading files and querying their contents (e.g., PDFs, spreadsheets) directly in chat.                                                                                                                                                                                                 |
| **Code interpreter**               | 🕒 Backlog      | 2025-06-03   | Planned support for OpenAI’s **Python/code interpreter** tool. Would allow running code, analyzing data, and generating charts inside Open WebUI.                                                                                                                                                            |
| **Computer use**                   | 🕒 Backlog      | 2025-06-03   | Placeholder for OpenAI’s **computer use** tool (models interacting with apps or browsers). Not yet supported in Open WebUI.                                                                                                                                                                                  |
| **Live voice (Talk)**              | 🕒 Backlog      | 2025-06-03   | Planned support for **real-time voice conversations** (like ChatGPT’s Talk mode). Requires backend audio streaming support.                                                                                                                                                                                  |
| **Dynamic chat titles**            | 🕒 Backlog      | 2025-06-03   | Planned support for **auto-updating chat titles** during long tasks. Not yet implemented.                                                                                                                                                                                                                    |
| **MCP tool support**               | 🔄 Experimental | 2025-06-23   | Allows attaching **remote MCP servers** via the `REMOTE_MCP_SERVERS_JSON` valve. Experimental: implementation works but is not optimized, so **not production-ready**. Behavior may change.                                                                                                                  |

## Advanced Features
### Pseudo-model aliases

Use shorthand model names that automatically map to official OpenAI models with predefined reasoning levels.
Examples:

* `gpt-5-thinking` → `gpt-5` (medium reasoning)
* `gpt-5-thinking-high` → `gpt-5` (high reasoning)
* `o4-mini-high` → `o4-mini` (high reasoning effort)

See table below for full list of aliases.

### Debug logging

Set `LOG_LEVEL=debug` (pipe-level or per-user valve) to embed inline debug logs in assistant messages.
This surfaces details like:

* API request/response structure
* Tool merging behavior
* Hidden response items

Helpful for troubleshooting and understanding exactly how the manifold processes requests.

### Remote MCP servers (experimental)

Attach external [Model Context Protocol (MCP)](https://platform.openai.com/docs/guides/tools-remote-mcp) servers using the `REMOTE_MCP_SERVERS_JSON` valve.

* Accepts JSON describing one or more servers
  ⚠️ Still experimental: works, but **not recommended for production** yet.

### Filter-injected tools

Lets developers build **companion filters** that add OpenAI-style tools dynamically.

* Filters inject tools into `body.extra_tools`
* The manifold merges them into `body.tools` before sending the request
* Duplicates are removed automatically

E.g.,

```python
body.setdefault("extra_tools", []).append({
    "type": "function",
    "name": "weather_lookup",
    "description": "Get current weather by city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
})
```

This makes features like **on-demand web search toggles** possible without breaking native function calling.

## Tested Models
The manifold ahould work with any model that supports the **OpenAI Responses API**.  
Below are the official model IDs that have been tested and confirmed.

### Official Model IDs

| Family            | Model ID              | Type / Modality                  | Status | Notes |
|-------------------|-----------------------|----------------------------------|:------:|-------|
| **GPT-5**         | `gpt-5`               | Reasoning                        | ✅ | Standard GPT-5 reasoning model. |
|                   | `gpt-5-mini`          | Reasoning                        | ✅ | Smaller, faster, lower cost than `gpt-5`. |
|                   | `gpt-5-nano`          | Reasoning                        | ✅ | Ultra-lightweight reasoning; lowest cost. |
|                   | `gpt-5-chat-latest`   | Chat-tuned (non-reasoning)       | ✅ | Best for polished conversation. No tool calling. ([OpenAI Platform][1]) |
| **GPT-4.1**       | `gpt-4.1`             | Non-reasoning                    | ✅ | High-speed GPT-4 series model. |
| **GPT-4o**        | `gpt-4o`              | Text + image → text              | ✅ | Multimodal reasoning model. |
|                   | `chatgpt-4o-latest`   | Alias to ChatGPT’s GPT-4o build  | ✅ | Matches ChatGPT’s current 4o snapshot. ([OpenAI Platform][2]) |
| **O-series**      | `o3`                  | Reasoning                        | ✅ | Standard O-series reasoning model. |
|                   | `o3-pro`              | Reasoning (higher compute)       | ✅ | API-only; heavier, deeper reasoning. ([OpenAI Platform][3]) |
|                   | `o3-mini`             | Reasoning                        | ✅ | Lightweight O-series reasoning. ([OpenAI Platform][11]) |
|                   | `o4-mini`             | Reasoning                        | ✅ | Cost-efficient O-series reasoning. ([OpenAI][12]) |
| **Deep Research** | `o3-deep-research`    | Agentic deep-research            | ❌ | Not yet supported. ([OpenAI Platform][4]) |
|                   | `o4-mini-deep-research` | Agentic deep-research          | ❌ | Not yet supported. ([OpenAI Platform][5]) |
| **Utility/Coding**| `codex-mini-latest`   | Lightweight coding / agent model | ✅ | For code + lightweight reasoning. ([OpenAI Platform][6]) |

---

### Pseudo-Model Aliases (Convenience IDs)

These **aliases** are supported via the `MODELS` valve. They resolve to official models but may also apply presets (e.g., `reasoning_effort`).  
Useful for **routing, shorthand, or quick quality/cost tuning**. *(Subject to change as OpenAI updates models.)*

| Alias                          | Resolves To   | Preset(s)                    | Suggested Use |
|--------------------------------|---------------|------------------------------|---------------|
| `gpt-5-auto`                   | Dynamic GPT-5 | —                            | Automatically routes between GPT-5 chat/mini/nano. |
| `gpt-5-thinking`               | `gpt-5`       | Medium reasoning             | General high-quality tasks. ([OpenAI][8]) |
| `gpt-5-thinking-minimal`       | `gpt-5`       | `reasoning_effort="minimal"` | Faster/cheaper reasoning. ([OpenAI][8]) |
| `gpt-5-thinking-high`          | `gpt-5`       | `reasoning_effort="high"`    | Hard problems; max quality. ([OpenAI][8]) |
| `gpt-5-thinking-mini`          | `gpt-5-mini`  | Medium reasoning             | Budget-tilted reasoning. ([OpenAI Platform][9]) |
| `gpt-5-thinking-mini-minimal`  | `gpt-5-mini`  | Minimal reasoning            | Hidden/task routing use. ([OpenAI Platform][9]) |
| `gpt-5-thinking-nano`          | `gpt-5-nano`  | Medium reasoning             | Ultra-low cost; triage/routing. ([OpenAI Platform][10]) |
| `gpt-5-thinking-nano-minimal`  | `gpt-5-nano`  | Minimal reasoning            | Cheapest reasoning option. ([OpenAI Platform][10]) |
| `o3-mini-high`                 | `o3-mini`     | High reasoning effort        | Small + fast, but deeper thinking. ([OpenAI Platform][11]) |
| `o4-mini-high`                 | `o4-mini`     | High reasoning effort        | Balanced cost-efficient + deeper thinking. ([OpenAI][12]) |
| *(reserved)* `gpt-5-main`, `gpt-5-main-mini`, `gpt-5-thinking-pro` | — | — | Reserved placeholders; no current API models. ([OpenAI][8]) |

---

[1]: https://platform.openai.com/docs/models/gpt-5-chat-latest?utm_source=chatgpt.com "Model - OpenAI API"  
[2]: https://platform.openai.com/docs/models/chatgpt-4o-latest?utm_source=chatgpt.com "Model - OpenAI API"  
[3]: https://platform.openai.com/docs/models/o3-pro?utm_source=chatgpt.com "Model - OpenAI API"  
[4]: https://platform.openai.com/docs/models/o3-deep-research?utm_source=chatgpt.com "o3-deep-research"  
[5]: https://platform.openai.com/docs/guides/deep-research?utm_source=chatgpt.com "Deep research - OpenAI API"  
[6]: https://platform.openai.com/docs/models/codex-mini-latest?utm_source=chatgpt.com "Codex mini"  
[7]: https://platform.openai.com/docs/models/gpt-5?utm_source=chatgpt.com "Model - OpenAI API"  
[8]: https://openai.com/index/introducing-gpt-5-for-developers/?utm_source=chatgpt.com "Introducing GPT-5 for developers"  
[9]: https://platform.openai.com/docs/models/gpt-5-mini?utm_source=chatgpt.com "GPT-5 mini"  
[10]: https://platform.openai.com/docs/models/gpt-5-nano?utm_source=chatgpt.com "Model - OpenAI API"  
[11]: https://platform.openai.com/docs/models/o3-mini?utm_source=chatgpt.com "Model - OpenAI API"  
[12]: https://openai.com/index/introducing-o3-and-o4-mini/?utm_source=chatgpt.com "Introducing OpenAI o3 and o4-mini"  


## GPT-5 Model Support

The Responses Manifold supports the full **GPT-5 family** of models currently exposed in the API:

- `gpt-5` *(reasoning model)*
- `gpt-5-mini` *(reasoning model)*
- `gpt-5-nano` *(reasoning model)*
- `gpt-5-chat-latest` *(non-reasoning, fine-tuned for chat, does NOT support function calling / tools)*

In the public **ChatGPT app**, choosing *GPT-5* doesn’t mean you’re using one fixed model. Behind the scenes, OpenAI runs a **router layer** that inspects your request and decides whether to send it to a reasoning, minimal-reasoning, or non-reasoning GPT-5 variant.

This router is **not available in the API**. When using the API, you must select the model explicitly (`gpt-5`, `gpt-5-mini`, `gpt-5-nano`, or `gpt-5-chat-latest`).

To bridge this gap, the manifold includes an experimental **`gpt-5-auto`** model.  
- It isn’t a real OpenAI model ID.  
- Instead, the request is first passed to a lightweight classifier (currently `gpt-5-nano` with minimal reasoning + a routing prompt).  
- The classifier chooses the most suitable GPT-5 endpoint (reasoning, minimal-reasoning, or chat) for your request.  

⚠️ This router is an **early proof of concept** and will be optimized further in future releases.

### What you need to know
1. **Reasoning vs. non-reasoning**  
   - `gpt-5`, `gpt-5-mini`, and `gpt-5-nano` are **reasoning models** by default.  
   - Setting `reasoning_effort="minimal"` makes them cheaper/faster but they still perform reasoning.  
   - For a true **non-reasoning chat model**, use `gpt-5-chat-latest`.

2. **Tool calling**  
   - Reasoning models support **native tool calling** (function calls, web search, etc.).  
   - `gpt-5-chat-latest` does **not** support tool calling — it’s tuned only for polished chat.  
   - A future `gpt-5-main` model may bring **non-reasoning + tool support**.

3. **Latency and performance**  
   - Even with `"minimal"` effort, reasoning models take time to “think.”  
   - For **ultra-low latency tasks**, use `gpt-4.1-nano` until a GPT-5 task model is released.

4. **Output style**  
   - `gpt-5-chat-latest` is tuned for smooth, conversational answers and usually works without a system prompt.  
   - Reasoning models work best with a short style prompt (e.g., “Concise Markdown with headings and lists”).  
   - Example prompts are provided in the `system_prompts` folder.


[1]: https://openai.com/index/introducing-gpt-5-for-developers/ "Introducing GPT-5 for developers | OpenAI"  
[2]: https://cdn.openai.com/pdf/8124a3ce-ab78-4f06-96eb-49ea29ffb52f/gpt5-system-card-aug7.pdf "GPT-5 System Card (Aug 7, 2025)"


## How It Works (Design Notes)

### Persisting non-message items (function calls, tool outputs, reasoning tokens, …)

The **OpenAI Responses API** doesn’t just return text. A single response can include:
- reasoning steps,
- function/tool calls,
- tool results,
- assistant messages.

By default, **Open WebUI only saves** the `role` (`user` / `assistant`) and the assistant’s **visible text content**.  
That means all the “hidden work” (reasoning, tools used, results returned) would normally disappear.  
If we don’t capture it:
- tool calls may run again on regenerate,
- reasoning tokens are wasted (higher cost, slower),
- the model’s exact sequence of actions is lost.

The manifold solves this by persisting **all items** in sequence, not just the visible text.

---

### The persistence challenge

Open WebUI’s conversation history is built around a simple `messages[]` array where each entry looks like:

```json
{ "role": "user" | "assistant", "content": "..." }
````

* `role` identifies who sent the message.
* `content` is always rendered in the chat UI as visible text.

Here’s the problem:

* **Upstream filters** (which inject context, rewrite prompts, toggle tools, etc.) only modify the `messages[]` array.
* If the manifold built its own parallel storage, it would ignore those filter changes — breaking compatibility and leaving the manifold out of sync.

So we must keep using the **same `messages[]` array** that Open WebUI and its filters rely on.
But since `content` is always displayed in the UI, we need a way to tuck hidden data into assistant messages **without showing it to the user**.

---

### The solution: invisible markers

The manifold injects **empty Markdown reference links** into the assistant’s response text.
These links are ignored by the Open WebUI frontend (they don’t render), but they carry stable IDs that point to the hidden items stored in the DB.

Example marker:

```
[openai_responses:v2:function_call:01HX4Y2VW5VR2Z2H]: #
```

**Marker format:**

```
[openai_responses:v2:<item_type>:<id>[?model=<model_id>&k=v...]]: #
```

* `<item_type>` = event type (e.g., `function_call`, `reasoning`)
* `<id>` = unique 16-character ID
* optional query params (e.g., `model`)

> **Why not embed JSON directly?**
> Markers keep assistant messages lightweight and clipboard-safe, while the full payloads remain in the DB.

---

### Example: function call flow

1. **User asks a question**

```json
{ "role": "user", "content": "Calculate 34234 multiplied by pi." }
```

2. **Model emits a tool call**

```json
{
  "type": "function_call",
  "name": "calculator",
  "arguments": "{\"expression\":\"34234*pi\"}",
  "status": "completed"
}
```

* Stored under ID `01HX4Y2VW5VR2Z2H`
* Marker injected into assistant output:

```
[openai_responses:v2:function_call:01HX4Y2VW5VR2Z2H]: #
```

3. **Tool result is persisted the same way**

```
[openai_responses:v2:function_call_output:01HX4Y2VW6B091XE]: #
```

4. **Assistant shows visible text**

```
34234 multiplied by π ≈ 107,549.28.
```

5. **Final stream = hidden markers + visible text**

```
[openai_responses:v2:function_call:01HX4Y2VW5VR2Z2H?model=openai_responses.gpt-4o]: #
[openai_responses:v2:function_call_output:01HX4Y2VW6B091XE?model=openai_responses.gpt-4o]: #
The result of \(34234 \times \pi\) is approximately 107,549.28.
```

---

### Why this matters

By combining hidden markers with DB persistence:

* **No duplicate work** → tool calls and reasoning aren’t re-run on regenerate
* **Lower cost & latency** → caching saves \~50–75% input tokens
* **Filter compatibility** → upstream filters can still modify `messages[]` normally
* **Full fidelity history** → the exact reasoning + tool sequence is preserved and replayable

---

> **Tip for debugging:** Open browser **DevTools → Network**, inspect the chat POST payload, and you’ll see the hidden markers alongside the visible messages.
