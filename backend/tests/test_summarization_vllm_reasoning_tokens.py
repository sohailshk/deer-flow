"""The summarization token trigger counts reasoning that VllmChatModel replays (#8074)."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult

from deerflow.agents.middlewares.summarization_middleware import DeerFlowSummarizationMiddleware, create_summarization_middleware
from deerflow.config.memory_config import MemoryConfig
from deerflow.config.summarization_config import ContextSize, SummarizationConfig
from deerflow.models.vllm_provider import VllmChatModel


class _StaticChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "static-test-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


_REASONING = "think " * 40_000  # 240k chars, about 60k tokens


def _vllm() -> VllmChatModel:
    return VllmChatModel(model="Qwen/QwQ-32B", api_key="dummy", base_url="http://localhost:8000/v1")


def _history(reasoning_key: str = "reasoning") -> list:
    return [
        HumanMessage(content="hi"),
        AIMessage(content="ok", additional_kwargs={reasoning_key: _REASONING}),
    ]


def _middleware(model) -> DeerFlowSummarizationMiddleware:
    return DeerFlowSummarizationMiddleware(model=model, trigger=("tokens", 50_000), keep=("messages", 1), extensions=[])


def test_vllm_reasoning_is_counted_and_fires_the_token_trigger() -> None:
    middleware = _middleware(_vllm())
    messages = _history()

    tokens = middleware.token_counter(messages)

    assert tokens >= 60_000
    assert middleware._should_summarize(messages, tokens)


def test_vllm_reasoning_content_fallback_field_is_counted() -> None:
    middleware = _middleware(_vllm())

    assert middleware.token_counter(_history("reasoning_content")) >= 60_000


def test_vllm_reasoning_is_counted_by_the_trimming_counter_too() -> None:
    middleware = _middleware(_vllm())

    assert middleware._partial_token_counter(_history()) >= 60_000


def test_counter_accepts_list_content_and_leaves_other_messages_alone() -> None:
    middleware = _middleware(_vllm())
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content=[{"type": "text", "text": "ok"}], additional_kwargs={"reasoning": _REASONING}),
        ToolMessage(content="result", tool_call_id="call-1"),
    ]
    baseline = [message.model_copy(update={"additional_kwargs": {}}) for message in messages]

    assert middleware.token_counter(messages) - middleware.token_counter(baseline) >= 60_000
    assert messages[1].content == [{"type": "text", "text": "ok"}]  # state is not mutated


def test_messages_without_reasoning_count_as_before() -> None:
    middleware = _middleware(_vllm())
    messages = [HumanMessage(content="hi"), AIMessage(content="ok")]

    assert middleware.token_counter(messages) == count_tokens_approximately(messages, use_usage_metadata_scaling=True)
    assert middleware._partial_token_counter(messages) == count_tokens_approximately(messages)


def test_non_vllm_model_keeps_the_stock_counter() -> None:
    middleware = _middleware(_StaticChatModel())

    assert middleware.token_counter(_history()) < 20


def test_factory_counts_reasoning_for_a_vllm_run_model(monkeypatch) -> None:
    monkeypatch.setattr("deerflow.agents.middlewares.summarization_middleware.create_chat_model", lambda **kwargs: _vllm().with_config(tags=["middleware:summarize"]))
    models = [SimpleNamespace(name="qwen")]
    config = SimpleNamespace(
        summarization=SummarizationConfig(enabled=True, trigger=ContextSize(type="tokens", value=50_000)),
        memory=MemoryConfig(enabled=False),
        models=models,
        get_model_config=lambda name: next((model for model in models if model.name == name), None),
    )

    middleware = create_summarization_middleware(app_config=config, run_model_name="qwen", keep=("messages", 1))

    assert middleware is not None
    messages = _history()
    assert middleware._should_summarize(messages, middleware.token_counter(messages))
