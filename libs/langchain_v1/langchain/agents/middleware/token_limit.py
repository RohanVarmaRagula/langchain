"""Token limit middleware for models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired, override

from langchain_core.messages.ai import AIMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.runtime import Runtime

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
    ResponseT,
    hook_config,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class TokenBudgetState(AgentState[ResponseT]):
    """State schema for `TokenBudgetMiddleware`.

    Extends `AgentState` with token tracking fields.

    Type Parameters:
        ResponseT: The type of structured response. Defaults to `'Any'`.
    """

    thread_total_tokens: NotRequired[Annotated[int, PrivateStateAttr]]
    thread_input_tokens: NotRequired[Annotated[int, PrivateStateAttr]]
    thread_output_tokens: NotRequired[Annotated[int, PrivateStateAttr]]
    run_total_tokens: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]
    run_input_tokens: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]
    run_output_tokens: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


def _build_limit_exceeded_message(
    thread_total_tokens: int,
    thread_input_tokens: int,
    thread_output_tokens: int,
    run_total_tokens: int,
    run_input_tokens: int,
    run_output_tokens: int,
    thread_total_limit: int | None,
    thread_input_limit: int | None,
    thread_output_limit: int | None,
    run_total_limit: int | None,
    run_input_limit: int | None,
    run_output_limit: int | None,
) -> str:
    """Builds a message indicating which token limits were exceeded.

    Args:
        thread_total_tokens (int): Current thread total token usage.
        thread_input_tokens (int): Current thread input token usage.
        thread_output_tokens (int): Current thread output token usage.
        run_total_tokens (int): Current run total token usage.
        run_input_tokens (int): Current run input token usage.
        run_output_tokens (int): Current run output token usage.
        thread_total_limit (int | None): Thread total token usage limit (if set).
        thread_input_limit (int | None): Thread input token usage limit (if set).
        thread_output_limit (int | None): Thread output token usage limit (if set).
        run_total_limit (int | None): Run total token usage limit (if set).
        run_input_limit (int | None): Run input token usage limit (if set).
        run_output_limit (int | None): Run output token usage limit (if set).

    Returns:
        str: A formatted message describing which token limits were exceeded.
    """
    exceeded_limits = []
    checks = [
        (thread_total_limit, thread_total_tokens, "thread total"),
        (thread_input_limit, thread_input_tokens, "thread input"),
        (thread_output_limit, thread_output_tokens, "thread output"),
        (run_total_limit, run_total_tokens, "run total"),
        (run_input_limit, run_input_tokens, "run input"),
        (run_output_limit, run_output_tokens, "run output"),
    ]
    for limit, tokens, name in checks:
        if limit is not None and tokens >= limit:
            exceeded_limits.append(f"{name} tokens limit ({tokens}/{limit})")

    return f"Token budget reached or exceeded: {', '.join(exceeded_limits)}"


class TokenBudgetExceededError(Exception):
    """Exception raised when token limits are exceeded.

    This exception is raised when the configured exit behaviour is `'error'` and
    either the thread or run token limit has been exceeded.
    """

    def __init__(
        self,
        thread_total_tokens: int,
        thread_input_tokens: int,
        thread_output_tokens: int,
        run_total_tokens: int,
        run_input_tokens: int,
        run_output_tokens: int,
        thread_total_limit: int | None,
        thread_input_limit: int | None,
        thread_output_limit: int | None,
        run_total_limit: int | None,
        run_input_limit: int | None,
        run_output_limit: int | None,
    ) -> None:
        """Initialize the exception with token usage information.

        Args:
            thread_total_tokens (int): Current thread total token usage.
            thread_input_tokens (int): Current thread input token usage.
            thread_output_tokens (int): Current thread output token usage.
            run_total_tokens (int): Current run total token usage.
            run_input_tokens (int): Current run input token usage.
            run_output_tokens (int): Current run output token usage.
            thread_total_limit (int | None): Thread total token usage limit (if set).
            thread_input_limit (int | None): Thread input token usage limit (if set).
            thread_output_limit (int | None): Thread output token usage limit (if set).
            run_total_limit (int | None): Run total token usage limit (if set).
            run_input_limit (int | None): Run input token usage limit (if set).
            run_output_limit (int | None): Run output token usage limit (if set).
        """
        self.thread_total = thread_total_tokens
        self.thread_input = thread_input_tokens
        self.thread_output = thread_output_tokens
        self.run_total = run_total_tokens
        self.run_input = run_input_tokens
        self.run_output = run_output_tokens
        self.thread_total_limit = thread_total_limit
        self.thread_input_limit = thread_input_limit
        self.thread_output_limit = thread_output_limit
        self.run_total_limit = run_total_limit
        self.run_input_limit = run_input_limit
        self.run_output_limit = run_output_limit

        msg = _build_limit_exceeded_message(
            thread_total_tokens,
            thread_input_tokens,
            thread_output_tokens,
            run_total_tokens,
            run_input_tokens,
            run_output_tokens,
            thread_total_limit,
            thread_input_limit,
            thread_output_limit,
            run_total_limit,
            run_input_limit,
            run_output_limit,
        )
        super().__init__(msg)


class TokenBudgetMiddleware(AgentMiddleware[TokenBudgetState[ResponseT], ContextT, ResponseT]):
    """Tracks token usage and enforces limits.

    This middleware monitors the number of tokens used during agent execution
    and can terminate the agent when specified limits are reached. It supports
    both thread-level and run-level token usage with configurable exit behaviors.

    Thread-level: The middleware tracks the token usage and persists
    token usage across multiple runs (invocations) of the agent.

    Run-level: The middleware tracks the token usage made during a single
    run (invocation) of the agent.

    Example:
        ```python
        from langchain.agents.middleware import TokenBudgetMiddleware
        from langchain.agents import create_agent

        # Create middleware with limits
        token_tracker = TokenBudgetMiddleware(
            thread_total_limit=100000, run_total_limit=20000, exit_behavior="end"
        )

        agent = create_agent("openai:gpt-5.5", middleware=[token_tracker])

        # Agent will automatically jump to end when limits are exceeded
        result = await agent.invoke({"messages": [HumanMessage("Help me with a task")]})
        ```
    """

    state_schema = TokenBudgetState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        thread_total_limit: int | None = None,
        thread_input_limit: int | None = None,
        thread_output_limit: int | None = None,
        run_total_limit: int | None = None,
        run_input_limit: int | None = None,
        run_output_limit: int | None = None,
        exit_behavior: Literal["end", "error"] = "end",
    ) -> None:
        """Initialize the token usage tracking middleware.

        Args:
            thread_total_limit (int | None): Thread total token usage limit.
                Default `None`, i.e, no limit.
            thread_input_limit (int | None): Thread input token usage limit.
                Default `None`, i.e, no limit.
            thread_output_limit (int | None): Thread output token usage limit.
                Default `None`, i.e, no limit.
            run_total_limit (int | None): Run total token usage limit.
                Default `None`, i.e, no limit.
            run_input_limit (int | None): Run input token usage limit.
                Default `None`, i.e, no limit.
            run_output_limit (int | None): Run output token usage limit.
                Default `None`, i.e, no limit.

            exit_behavior (Literal["end", "error"], optional):
                What to do when limits are exceeded.
                Defaults to `end`.

                - `'end'`: Jump to the end of the agent execution and
                    inject an artificial AI message indicating that the limit was
                    exceeded.
                - `'error'`: Raise a `TokenBudgetExceededError`

        Raises:
            ValueError: If both limits are `None`.
            ValueError: If `exit_behavior` is invalid.
        """
        super().__init__()

        limits = (
            thread_total_limit,
            thread_input_limit,
            thread_output_limit,
            run_total_limit,
            run_input_limit,
            run_output_limit,
        )
        if all(limit is None for limit in limits):
            msg = "At least one limit must be specified."
            raise ValueError(msg)

        if exit_behavior not in {"end", "error"}:
            msg = f"Invalid exit_behavior: {exit_behavior}. Expected 'end' or 'error'"
            raise ValueError(msg)

        self.thread_total_limit = thread_total_limit
        self.thread_input_limit = thread_input_limit
        self.thread_output_limit = thread_output_limit
        self.run_total_limit = run_total_limit
        self.run_input_limit = run_input_limit
        self.run_output_limit = run_output_limit
        self.exit_behavior = exit_behavior

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self, state: TokenBudgetState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Checks token usage limits before making a model call.

        Args:
            state (TokenBudgetState[ResponseT]):
                The current agent state containing token usage.

            runtime (Runtime[ContextT]):
                The langgraph runtime.

        Raises:
            TokenBudgetExceededError:
                If limits are exceeded and `exit_behavior`
                is `'error'`.

        Returns:
            dict[str, Any] | None:
                If limits are exceeded and exit_behavior is `'end'`, returns
                a `Command` to jump to the end with a limit exceeded message. Otherwise
                returns `None`.

        """
        thread_total_tokens = state.get("thread_total_tokens", 0)
        thread_input_tokens = state.get("thread_input_tokens", 0)
        thread_output_tokens = state.get("thread_output_tokens", 0)
        run_total_tokens = state.get("run_total_tokens", 0)
        run_input_tokens = state.get("run_input_tokens", 0)
        run_output_tokens = state.get("run_output_tokens", 0)

        # Check if any limits are exceeded already
        exceeded = (
            self.thread_total_limit is not None and thread_total_tokens >= self.thread_total_limit,
            self.thread_input_limit is not None and thread_input_tokens >= self.thread_input_limit,
            self.thread_output_limit is not None
            and thread_output_tokens >= self.thread_output_limit,
            self.run_total_limit is not None and run_total_tokens >= self.run_total_limit,
            self.run_input_limit is not None and run_input_tokens >= self.run_input_limit,
            self.run_output_limit is not None and run_output_tokens >= self.run_output_limit,
        )

        if any(exceeded):
            if self.exit_behavior == "error":
                raise TokenBudgetExceededError(
                    thread_total_tokens=thread_total_tokens,
                    thread_input_tokens=thread_input_tokens,
                    thread_output_tokens=thread_output_tokens,
                    run_total_tokens=run_total_tokens,
                    run_input_tokens=run_input_tokens,
                    run_output_tokens=run_output_tokens,
                    thread_total_limit=self.thread_total_limit,
                    thread_input_limit=self.thread_input_limit,
                    thread_output_limit=self.thread_output_limit,
                    run_total_limit=self.run_total_limit,
                    run_input_limit=self.run_input_limit,
                    run_output_limit=self.run_output_limit,
                )

            if self.exit_behavior == "end":
                limit_message = _build_limit_exceeded_message(
                    thread_total_tokens=thread_total_tokens,
                    thread_input_tokens=thread_input_tokens,
                    thread_output_tokens=thread_output_tokens,
                    run_total_tokens=run_total_tokens,
                    run_input_tokens=run_input_tokens,
                    run_output_tokens=run_output_tokens,
                    thread_total_limit=self.thread_total_limit,
                    thread_input_limit=self.thread_input_limit,
                    thread_output_limit=self.thread_output_limit,
                    run_total_limit=self.run_total_limit,
                    run_input_limit=self.run_input_limit,
                    run_output_limit=self.run_output_limit,
                )
                limit_ai_message = AIMessage(content=limit_message)

                return {"jump_to": "end", "messages": [limit_ai_message]}

        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: TokenBudgetState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async check of token usage before making model call.

        Args:
            state (TokenBudgetState[ResponseT]):
                The current agent state containing token usage.

            runtime (Runtime[ContextT]):
                The langgraph runtime.

        Returns:
            dict[str, Any] | None:
                If limits are exceeded and exit_behavior is `'end'`, returns
                a `Command` to jump to the end with a limit exceeded message. Otherwise
                returns `None`.

        """
        return self.before_model(state=state, runtime=runtime)

    @override
    def after_model(
        self, state: TokenBudgetState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Increments token usage from `usage_metadata` after a model call.

        Args:
            state (TokenBudgetState[ResponseT]):
                The current agent state containing token usage.

            runtime (Runtime[ContextT]):
                The langgraph runtime.

        Returns:
            dict[str, Any] | None:
                Returns updated state with incremented token usage.

        """
        messages = state.get("messages")
        if (
            not messages
            or not isinstance(messages[-1], AIMessage)
            or messages[-1].usage_metadata is None
        ):
            return None

        usage_metadata = messages[-1].usage_metadata

        input_tokens = usage_metadata.get("input_tokens", 0)
        output_tokens = usage_metadata.get("output_tokens", 0)
        total_tokens = usage_metadata.get("total_tokens", 0)

        return {
            "thread_total_tokens": state.get("thread_total_tokens", 0) + total_tokens,
            "thread_input_tokens": state.get("thread_input_tokens", 0) + input_tokens,
            "thread_output_tokens": state.get("thread_output_tokens", 0) + output_tokens,
            "run_total_tokens": state.get("run_total_tokens", 0) + total_tokens,
            "run_input_tokens": state.get("run_input_tokens", 0) + input_tokens,
            "run_output_tokens": state.get("run_output_tokens", 0) + output_tokens,
        }

    async def aafter_model(
        self, state: TokenBudgetState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async increment of token usage from `usage_metadata` after a model call.

        Args:
            state (TokenBudgetState[ResponseT]):
                The current agent state containing token usage.

            runtime (Runtime[ContextT]):
                The langgraph runtime.

        Returns:
            dict[str, Any] | None:
                Returns updated state with incremented token usage.

        """
        return self.after_model(
            state=state,
            runtime=runtime,
        )
