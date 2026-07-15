from typing import TYPE_CHECKING, Any, cast, override

import pytest
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs.chat_result import ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from langchain.agents.factory import create_agent
from langchain.agents.middleware import InputAgentState
from langchain.agents.middleware.token_limit import (
    TokenBudgetExceededError,
    TokenBudgetMiddleware,
    TokenBudgetState,
)
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


class FakeUsageToolCallingModel(FakeToolCallingModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

        message = cast("AIMessage", result.generations[0].message)
        message.usage_metadata = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

        return result


@tool
def simple_tool(value: str) -> str:
    """A simple tool."""
    return value


def test_middleware_unit_functionality() -> None:
    """Test that the middleware works as expected in isolation."""
    # Test with end behavior
    middleware = TokenBudgetMiddleware(thread_total_limit=2, run_total_limit=1)

    runtime = Runtime()

    # Test when limits are not exceeded
    state = TokenBudgetState(messages=[], thread_total_tokens=0, run_total_tokens=0)
    result = middleware.before_model(state, runtime)
    assert result is None

    # Test when thread limit is exceeded
    state = TokenBudgetState(messages=[], thread_total_tokens=2, run_total_tokens=0)
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert (
        "Token budget reached or exceeded: thread total tokens limit (2/2)"
        in result["messages"][0].content
    )

    # Test when run limit is exceeded
    state = TokenBudgetState(messages=[], thread_total_tokens=1, run_total_tokens=1)
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert (
        "Token budget reached or exceeded: run total tokens limit (1/1)"
        in result["messages"][0].content
    )

    # Test with error behavior
    middleware_exception = TokenBudgetMiddleware(
        thread_total_limit=2, run_total_limit=1, exit_behavior="error"
    )

    # Test exception when thread limit exceeded
    state = TokenBudgetState(messages=[], thread_total_tokens=2, run_total_tokens=0)
    with pytest.raises(TokenBudgetExceededError) as exc_info:
        middleware_exception.before_model(state, runtime)

    assert "Token budget reached or exceeded: thread total tokens limit (2/2)" in str(
        exc_info.value
    )

    # Test exception when run limit exceeded
    state = TokenBudgetState(messages=[], thread_total_tokens=1, run_total_tokens=1)
    with pytest.raises(TokenBudgetExceededError) as exc_info:
        middleware_exception.before_model(state, runtime)

    assert "Token budget reached or exceeded: run total tokens limit (1/1)" in str(exc_info.value)


def test_thread_total_limit_with_create_agent() -> None:
    """Test that thread limits work correctly with create_agent."""
    model = FakeUsageToolCallingModel(
        input_tokens=2,
        output_tokens=0,
        total_tokens=2,
    )

    # Set thread limit to 1 (should be exceeded after exeeding 1 token)
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(thread_total_limit=1)],
        checkpointer=InMemorySaver(),
    )

    # First invocation should work - 1 model call, within thread limit
    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]}, {"configurable": {"thread_id": "thread1"}}
    )
    # Should complete successfully with 1 model call
    assert "messages" in result
    assert len(result["messages"]) == 2  # Human + AI messages

    # Second invocation in same thread should hit thread limit
    # The agent should jump to end after detecting the limit
    result2 = agent.invoke(
        {"messages": [HumanMessage("Hello again")]}, {"configurable": {"thread_id": "thread1"}}
    )

    assert "messages" in result2
    # The agent should have detected the limit and jumped to end with a limit exceeded message
    # So we should have: previous messages + new human message + limit exceeded AI message
    assert len(result2["messages"]) == 4  # Previous Human + AI + New Human + Limit AI
    assert isinstance(result2["messages"][0], HumanMessage)  # First human
    assert isinstance(result2["messages"][1], AIMessage)  # First AI response
    assert isinstance(result2["messages"][2], HumanMessage)  # Second human
    assert isinstance(result2["messages"][3], AIMessage)  # Limit exceeded message
    assert (
        "Token budget reached or exceeded: thread total tokens limit (2/1)"
        in result2["messages"][3].content
    )


def test_thread_input_limit_with_create_agent() -> None:
    """Test that thread input token limit is enforced."""
    model = FakeUsageToolCallingModel(
        input_tokens=2,
        output_tokens=100,
        total_tokens=102,
    )

    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(thread_input_limit=1)],
        checkpointer=InMemorySaver(),
    )

    agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "thread1"}},
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello again")]},
        {"configurable": {"thread_id": "thread1"}},
    )

    assert "thread input tokens limit (2/1)" in result["messages"][-1].content


def test_thread_output_limit_with_create_agent() -> None:
    """Test that thread output token limit is enforced."""
    model = FakeUsageToolCallingModel(
        input_tokens=100,
        output_tokens=2,
        total_tokens=102,
    )

    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(thread_output_limit=1)],
        checkpointer=InMemorySaver(),
    )

    agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "thread1"}},
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello again")]},
        {"configurable": {"thread_id": "thread1"}},
    )

    assert "thread output tokens limit (2/1)" in result["messages"][-1].content


def test_run_total_limit_with_create_agent() -> None:
    """Test that run limits work correctly with create_agent."""
    # Create a model that will make 2 calls
    model = FakeUsageToolCallingModel(
        tool_calls=[
            [{"name": "simple_tool", "args": {"input": "test"}, "id": "1"}],
            [],  # No tool calls on second call
        ],
        input_tokens=1,
        output_tokens=0,
        total_tokens=1,
    )

    # Set run total limit to 1 (should be exceeded after 1 call)
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(run_total_limit=1)],
        checkpointer=InMemorySaver(),
    )

    # This should hit the run limit after the first model call
    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]}, {"configurable": {"thread_id": "thread1"}}
    )

    assert "messages" in result
    # The agent should have made 1 model call then jumped to end with limit exceeded message
    # So we should have: Human + AI + Tool + Limit exceeded AI message
    assert len(result["messages"]) == 4  # Human + AI + Tool + Limit AI
    assert isinstance(result["messages"][0], HumanMessage)
    assert isinstance(result["messages"][1], AIMessage)
    assert isinstance(result["messages"][2], ToolMessage)
    assert isinstance(result["messages"][3], AIMessage)  # Limit exceeded message
    assert (
        "Token budget reached or exceeded: run total tokens limit (1/1)"
        in result["messages"][3].content
    )


def test_run_input_limit_with_create_agent() -> None:
    """Test that run input token limit is enforced."""
    model = FakeUsageToolCallingModel(
        tool_calls=[
            [{"name": "simple_tool", "args": {"input": "test"}, "id": "1"}],
            [],
        ],
        input_tokens=1,
        output_tokens=100,
        total_tokens=101,
    )

    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(run_input_limit=1)],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "thread1"}},
    )

    assert "run input tokens limit (1/1)" in result["messages"][-1].content


def test_run_output_limit_with_create_agent() -> None:
    """Test that run output token limit is enforced."""
    model = FakeUsageToolCallingModel(
        tool_calls=[
            [{"name": "simple_tool", "args": {"input": "test"}, "id": "1"}],
            [],
        ],
        input_tokens=100,
        output_tokens=1,
        total_tokens=101,
    )

    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(run_output_limit=1)],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "thread1"}},
    )

    assert "run output tokens limit (1/1)" in result["messages"][-1].content


def test_middleware_initialization_validation() -> None:
    """Test that middleware initialization validates parameters correctly."""
    # Test that at least one limit must be specified
    with pytest.raises(ValueError, match="At least one limit must be specified"):
        TokenBudgetMiddleware()

    # Test invalid exit behavior
    with pytest.raises(ValueError, match="Invalid exit_behavior"):
        TokenBudgetMiddleware(thread_total_limit=5, exit_behavior="invalid")  # type: ignore[arg-type]

    # Test valid initialization
    middleware = TokenBudgetMiddleware(thread_total_limit=5, run_total_limit=3)
    assert middleware.thread_total_limit == 5
    assert middleware.run_total_limit == 3
    assert middleware.exit_behavior == "end"

    # Test with only thread limit
    middleware = TokenBudgetMiddleware(thread_total_limit=5)
    assert middleware.thread_total_limit == 5
    assert middleware.run_total_limit is None

    # Test with only run limit
    middleware = TokenBudgetMiddleware(run_total_limit=3)
    assert middleware.thread_total_limit is None
    assert middleware.run_total_limit == 3


def test_exception_error_message() -> None:
    """Test that the exception provides clear error messages."""
    middleware = TokenBudgetMiddleware(
        thread_total_limit=2, run_total_limit=1, exit_behavior="error"
    )

    # Test thread limit exceeded
    state = TokenBudgetState(messages=[], thread_total_tokens=2, run_total_tokens=0)
    with pytest.raises(TokenBudgetExceededError) as exc_info:
        middleware.before_model(state, Runtime())

    error_msg = str(exc_info.value)
    assert "Token budget reached or exceeded" in error_msg
    assert "thread total tokens limit (2/2)" in error_msg

    # Test run limit exceeded
    state = TokenBudgetState(messages=[], thread_total_tokens=0, run_total_tokens=1)
    with pytest.raises(TokenBudgetExceededError) as exc_info:
        middleware.before_model(state, Runtime())

    error_msg = str(exc_info.value)
    assert "Token budget reached or exceeded" in error_msg
    assert "run total tokens limit (1/1)" in error_msg

    # Test both limits exceeded
    state = TokenBudgetState(messages=[], thread_total_tokens=2, run_total_tokens=1)
    with pytest.raises(TokenBudgetExceededError) as exc_info:
        middleware.before_model(state, Runtime())

    error_msg = str(exc_info.value)
    assert "Token budget reached or exceeded" in error_msg
    assert "thread total tokens limit (2/2)" in error_msg
    assert "run total tokens limit (1/1)" in error_msg


def test_run_limit_resets_between_invocations() -> None:
    """Test run limit resets between invocations.

    Test that run_total_tokens resets between invocations, but
    thread_total_tokens accumulates.
    """
    # First: No tool calls per invocation, so model does not increment token usage internally
    middleware = TokenBudgetMiddleware(
        thread_total_limit=14, run_total_limit=5, exit_behavior="error"
    )
    model = FakeUsageToolCallingModel(
        tool_calls=[[], [], [], []],
        input_tokens=2,
        output_tokens=2,
        total_tokens=4,
    )  # No tool calls, so only model call per run

    agent = create_agent(model=model, middleware=[middleware], checkpointer=InMemorySaver())

    thread_config: RunnableConfig = {"configurable": {"thread_id": "test_thread"}}
    agent.invoke(InputAgentState(messages=[HumanMessage("Hello")]), thread_config)
    agent.invoke(InputAgentState(messages=[HumanMessage("Hello again")]), thread_config)
    agent.invoke(InputAgentState(messages=[HumanMessage("Hello third")]), thread_config)
    agent.invoke(InputAgentState(messages=[HumanMessage("Hello fourth")]), thread_config)

    # Fifth run: should raise, thread total tokens limit (16/14)
    with pytest.raises(TokenBudgetExceededError) as exc_info:
        agent.invoke(InputAgentState(messages=[HumanMessage("Hello fifth")]), thread_config)
    error_msg = str(exc_info.value)
    assert "Token budget reached or exceeded" in error_msg
    assert "thread total tokens limit (16/14)" in error_msg


def test_missing_usage_metadata_is_ignored() -> None:
    """Test that missing usage metadata does not accumulate tokens."""
    # FakeToolCallingModel does not return usage_metadata
    model = FakeToolCallingModel()

    # Set thread_total_limit to low enough that the middleware would have
    # blocked the second call if usage metadata had been present.
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[TokenBudgetMiddleware(thread_total_limit=1)],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]}, {"configurable": {"thread_id": "thread1"}}
    )

    assert "messages" in result
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][1], AIMessage)
    assert result["messages"][1].usage_metadata is None

    # This should not raise any error as there is no token usage tracked from usage_metadata(None)
    result = agent.invoke(
        {"messages": [HumanMessage("Hello again")]}, {"configurable": {"thread_id": "thread1"}}
    )

    assert "messages" in result
    assert len(result["messages"]) == 4
