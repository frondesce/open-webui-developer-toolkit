import json
import re
from dataclasses import dataclass

import pytest

from functions.pipes.openai_responses_manifold import openai_responses_manifold as mod


@pytest.fixture()
def dummy_chats(monkeypatch):
    """Simple in-memory Chats stub."""
    storage: dict[str, dict] = {}

    @dataclass
    class DummyChatModel:
        chat: dict

    class DummyChats:
        @staticmethod
        def get_chat_by_id(cid):
            chat = storage.get(cid)
            if chat is None:
                return None
            return DummyChatModel(chat)

        @staticmethod
        def update_chat_by_id(cid, chat):
            storage[cid] = chat
            return DummyChatModel(chat)

    monkeypatch.setattr(mod, "Chats", DummyChats)
    return storage


def test_marker_roundtrip():
    marker = mod.create_marker("function_call", ulid="01HX4Y2VW5VR2Z2H", model_id="gpt-4o")
    wrapped = mod.wrap_marker(marker)
    assert mod.contains_marker(wrapped)

    parsed = mod.parse_marker(marker)
    assert parsed["metadata"]["model"] == "gpt-4o"

    text = f"pre {wrapped} post"
    assert mod.extract_markers(text) == [marker]

    segments = mod.split_text_by_markers(text)
    assert segments[1] == {"type": "marker", "marker": marker}
    assert segments[0]["text"].startswith("pre")
    assert segments[-1]["text"].strip().endswith("post")


def test_persistence_fetch_and_input(dummy_chats):
    dummy_chats["c1"] = {"history": {"messages": {}}}
    marker1 = mod.persist_openai_response_items(
        "c1",
        "m1",
        [{"type": "function_call", "name": "calc", "arguments": "{}"}],
        "openai_responses.gpt-4o",
    )
    marker2 = mod.persist_openai_response_items(
        "c1",
        "m2",
        [{"type": "function_call", "name": "other", "arguments": "{}"}],
        "openai_responses.gpt-3.5",
    )

    uid1 = mod.extract_markers(marker1, parsed=True)[0]["ulid"]
    uid2 = mod.extract_markers(marker2, parsed=True)[0]["ulid"]

    fetched = mod.fetch_openai_response_items(
        "c1", [uid1, uid2], openwebui_model_id="openai_responses.gpt-4o"
    )
    assert list(fetched) == [uid1]

    messages = [{"role": "assistant", "content": marker1 + "ok"}]
    output = mod.ResponsesBody.transform_messages_to_input(
        messages,
        chat_id="c1",
        openwebui_model_id="openai_responses.gpt-4o",
    )
    assert output[0]["type"] == "function_call"
    assert output[1]["content"][0]["text"] == "ok"


def test_tool_transforms_and_mcp():
    tools = [
        {"spec": {"name": "add", "description": "", "parameters": {}}},
        {"type": "function", "function": {"name": "add", "parameters": {}}},
        {"type": "web_search"},
    ]
    out = mod.ResponsesBody.transform_tools(tools, strict=True)
    names = {t.get("name", t.get("type")) for t in out}
    assert names == {"add", "web_search"}
    for t in out:
        if t.get("type") == "function":
            assert t["strict"] is True
            assert t["parameters"]["additionalProperties"] is False

    mcp_json = json.dumps({"server_label": "main", "server_url": "https://x.y"})
    assert mod.ResponsesBody._build_mcp_tools(mcp_json) == [
        {"type": "mcp", "server_label": "main", "server_url": "https://x.y"}
    ]


@pytest.mark.parametrize("item_type", ["", "a", "bad!", "x" * 31])
def test_create_marker_rejects_bad_types(item_type):
    """Ensure invalid item_type values raise."""
    with pytest.raises(ValueError):
        mod.create_marker(item_type)


def test_marker_no_markers():
    text = "no markers"
    assert not mod.contains_marker(text)
    assert mod.extract_markers(text) == []
    assert mod.split_text_by_markers(text) == [{"type": "text", "text": text}]


def test_multiple_markers_and_parsing():
    m1 = mod.create_marker("fc", ulid="A" * 16)
    m2 = mod.create_marker("tool", ulid="B" * 16)
    txt = f"pre {mod.wrap_marker(m1)} mid {mod.wrap_marker(m2)} end"
    assert mod.extract_markers(txt) == [m1, m2]
    segs = mod.split_text_by_markers(txt)
    assert [s["type"] for s in segs] == ["text", "marker", "text", "marker", "text"]


def test_parse_marker_invalid_version():
    with pytest.raises(ValueError):
        mod.parse_marker("openai_responses:v1:bad")


def test_persist_missing_and_empty(dummy_chats):
    assert (
        mod.persist_openai_response_items(
            "x", "m", [{"type": "t"}], "model"
        )
        == ""
    )
    dummy_chats["c1"] = {"history": {"messages": {}}}
    assert mod.persist_openai_response_items("c1", "m", [], "model") == ""


def test_fetch_nonexistent(dummy_chats):
    dummy_chats["c1"] = {"history": {"messages": {}}}
    assert mod.fetch_openai_response_items("c1", ["bad"]) == {}


def test_duplicate_persistence(dummy_chats, monkeypatch):
    dummy_chats["c1"] = {"history": {"messages": {}}}
    monkeypatch.setattr(mod, "generate_item_id", lambda: "A" * 16)
    mod.persist_openai_response_items("c1", "m1", [{"type": "ab"}], "model")
    mod.persist_openai_response_items("c1", "m1", [{"type": "bb"}], "model")
    store = dummy_chats["c1"]["openai_responses_pipe"]["items"]
    assert list(store) == ["A" * 16]
    assert store["A" * 16]["payload"]["type"] == "bb"
    ids = dummy_chats["c1"]["openai_responses_pipe"]["messages_index"]["m1"][
        "item_ids"
    ]
    assert ids == ["A" * 16, "A" * 16]


def test_transform_messages_various(monkeypatch):
    monkeypatch.setattr(mod, "fetch_openai_response_items", lambda *a, **k: {})
    msgs = [
        {"role": "system", "content": "skip"},
        {"role": "user", "content": "hi"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "t"},
                {"type": "image_url", "image_url": {"url": "u"}},
                {"type": "unknown", "value": 1},
            ],
        },
        {"role": "developer", "content": "dev"},
        {"role": "assistant", "content": "ok"},
        {"content": "ignored"},
    ]
    out = mod.ResponsesBody.transform_messages_to_input(msgs)
    assert [o["role"] for o in out] == [
        "user",
        "user",
        "developer",
        "assistant",
        "assistant",
    ]
    assert out[1]["content"][1]["image_url"] == "u"
    assert out[1]["content"][2] == {"type": "unknown", "value": 1}
    # chat_id without model_id should still transform without raising
    out2 = mod.ResponsesBody.transform_messages_to_input(msgs, chat_id="c1")
    assert len(out2) == len(out)
    # model_id without chat_id should also transform without raising
    out3 = mod.ResponsesBody.transform_messages_to_input(
        msgs, openwebui_model_id="model"
    )
    assert len(out3) == len(out)


def test_transform_messages_missing_item(monkeypatch, dummy_chats):
    dummy_chats["c1"] = {"history": {"messages": {}}}
    marker = mod.wrap_marker(mod.create_marker("fc", ulid="B" * 16))
    monkeypatch.setattr(mod, "fetch_openai_response_items", lambda *a, **k: {})
    out = mod.ResponsesBody.transform_messages_to_input(
        [{"role": "assistant", "content": marker}],
        chat_id="c1",
        openwebui_model_id="model",
    )
    assert out == []


@pytest.mark.parametrize(
    "tools,expected",
    [
        (None, []),
        ([1, "x"], []),
        ({"bad": 1}, []),
    ],
)
def test_transform_tools_invalid(tools, expected):
    assert mod.ResponsesBody.transform_tools(tools) == expected


def test_transform_tools_dedup_and_unknown():
    tools = [
        {"spec": {"name": "add", "parameters": {"a": {"type": "number"}}}},
        {"type": "function", "function": {"name": "add", "parameters": {"b": 1}}},
        {"type": "foo"},
    ]
    out = mod.ResponsesBody.transform_tools(tools)
    names = {t.get("name", t.get("type")) for t in out}
    assert names == {"add", "foo"}
    func = next(t for t in out if t.get("type") == "function")
    assert func["parameters"].get("b") == 1


@pytest.mark.parametrize(
    "payload",
    ["", "{", json.dumps([1, {}]), json.dumps({"server_label": "x"})],
)
def test_build_mcp_tools_invalid(payload):
    assert mod.ResponsesBody._build_mcp_tools(payload) == []


def test_citation_cleanup_handles_backslash_domain():
    domain = "foo\\bar.com"
    msg = f"( [{domain}](https://example.com) ) text"
    safe_domain = re.escape(domain).replace("\\", r"\\")
    pattern = re.compile(
        r"\(\s*\[\s*" + safe_domain + r"\s*\]\([^)]+\)\s*\)"
    )
    result = pattern.sub(" ", msg, count=1)
    assert isinstance(result, str)
