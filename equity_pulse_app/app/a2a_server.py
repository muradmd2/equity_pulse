"""A2A protocol boundary for the LangGraph financial research workflow."""

import asyncio

from fastapi import FastAPI

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)

from app.config import get_settings
from app.constants import (
    A2A_AGENT_NAME,
    A2A_AGENT_VERSION,
    A2A_JSONRPC_PATH,
    A2A_REST_PATH,
)


def run_financial_graph(query: str, session_id: str) -> dict[str, object]:
    """Load and invoke the graph only when an A2A task needs it."""

    from app.graph.builder import invoke_graph

    return invoke_graph(query, session_id=session_id)


class FinancialResearchAgentExecutor(AgentExecutor):
    """Translate A2A tasks into executions of the existing LangGraph workflow."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id
        user_message = context.message
        if not task_id or not context_id or not user_message:
            return

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )
        query = context.get_user_input().strip()
        if not query:
            await updater.requires_input(
                message=updater.new_agent_message(
                    parts=[Part(text="Please provide a financial research question.")]
                )
            )
            return

        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[user_message],
            )
        )
        await updater.start_work(
            message=updater.new_agent_message(
                parts=[Part(text="Researching the requested financial topic...")]
            )
        )

        try:
            # The graph and its provider clients are synchronous, so keep them off
            # the ASGI event loop while A2A clients await task updates.
            result = await asyncio.to_thread(
                run_financial_graph,
                query,
                context_id,
            )
            final_summary = str(result.get("final_summary") or "")
            if not final_summary:
                raise RuntimeError("The research workflow returned no final response.")

            await updater.add_artifact(
                parts=[Part(text=final_summary)],
                name="financial-research-response",
                metadata={
                    "contentType": "text/markdown",
                    "educationalOnly": True,
                    "sessionId": context_id,
                },
                last_chunk=True,
            )
            await updater.complete()
        except Exception as exc:
            await updater.failed(
                message=updater.new_agent_message(
                    parts=[Part(text=f"Financial research failed: {exc}")]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Publish the protocol cancellation state for the requested task."""

        if not context.task_id or not context.context_id:
            return
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.cancel()


def build_agent_card(base_url: str | None = None) -> AgentCard:
    """Build the public A2A Agent Card for discovery clients."""

    resolved_base_url = (base_url or get_settings().a2a_base_url).rstrip("/")
    return AgentCard(
        name=A2A_AGENT_NAME,
        description=(
            "Source-cited educational financial research using a LangGraph workflow. "
            "Outputs are not personalized financial advice."
        ),
        version=A2A_AGENT_VERSION,
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text", "task-status"],
        skills=[
            AgentSkill(
                id="financial_research",
                name="Financial research",
                description=(
                    "Research companies, market data, news, filings, analyst ratings, "
                    "financials, and descriptive technical indicators with citations."
                ),
                tags=["finance", "research", "stocks", "market-data"],
                examples=[
                    "How is CRM stock doing today?",
                    "Summarize Microsoft's latest report and company news.",
                ],
                input_modes=["text"],
                output_modes=["text", "task-status"],
            )
        ],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=f"{resolved_base_url}{A2A_JSONRPC_PATH}",
            ),
            AgentInterface(
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
                url=f"{resolved_base_url}{A2A_REST_PATH}",
            ),
        ],
    )


def register_a2a_server(app: FastAPI) -> None:
    """Register Agent Card, JSON-RPC, and HTTP+JSON routes on FastAPI."""

    agent_card = build_agent_card()
    request_handler = DefaultRequestHandler(
        agent_executor=FinancialResearchAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card=agent_card),
        jsonrpc_routes=create_jsonrpc_routes(
            request_handler=request_handler,
            rpc_url=A2A_JSONRPC_PATH,
        ),
        rest_routes=create_rest_routes(
            request_handler=request_handler,
            path_prefix=A2A_REST_PATH,
        ),
    )
