"""Focused tests for the A2A discovery and task boundary."""

from google.protobuf import json_format
from fastapi import FastAPI
from fastapi.testclient import TestClient

from a2a.types import Message, Part, Role, SendMessageRequest, TaskState

from app import a2a_server


def create_test_app() -> FastAPI:
    app = FastAPI()
    a2a_server.register_a2a_server(app)
    return app


def test_agent_card_advertises_supported_a2a_interfaces():
    client = TestClient(create_test_app())

    response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    card = response.json()
    assert card["name"] == "Equity Pulse Financial Research Agent"
    assert card["capabilities"]["streaming"] is True
    assert {interface["protocolBinding"] for interface in card["supportedInterfaces"]} == {
        "JSONRPC",
        "HTTP+JSON",
    }
    assert card["skills"][0]["id"] == "financial_research"


def test_a2a_rest_request_runs_graph_and_returns_artifact(monkeypatch):
    calls = []

    def fake_invoke_graph(query, session_id=None):
        calls.append((query, session_id))
        return {
            "final_summary": (
                "CRM research summary.\n\n## Sources\n- Example source\n\n"
                "Educational research only; not personalized financial advice."
            )
        }

    monkeypatch.setattr(a2a_server, "run_financial_graph", fake_invoke_graph)
    request = SendMessageRequest(
        message=Message(
            message_id="message-1",
            context_id="a2a-session-1",
            role=Role.ROLE_USER,
            parts=[Part(text="How is CRM stock doing today?")],
        )
    )

    response = TestClient(create_test_app()).post(
        "/a2a/rest/message:send",
        headers={"A2A-Version": "1.0"},
        json=json_format.MessageToDict(request),
    )

    assert response.status_code == 200
    task = response.json()["task"]
    assert task["status"]["state"] == TaskState.Name(TaskState.TASK_STATE_COMPLETED)
    assert task["artifacts"][0]["name"] == "financial-research-response"
    assert "## Sources" in task["artifacts"][0]["parts"][0]["text"]
    assert calls == [("How is CRM stock doing today?", "a2a-session-1")]


def test_a2a_jsonrpc_transport_is_available(monkeypatch):
    monkeypatch.setattr(
        a2a_server,
        "run_financial_graph",
        lambda query, session_id: {"final_summary": f"Research response for {query}"},
    )
    request = SendMessageRequest(
        message=Message(
            message_id="message-2",
            context_id="a2a-session-2",
            role=Role.ROLE_USER,
            parts=[Part(text="Summarize MSFT news")],
        )
    )

    response = TestClient(create_test_app()).post(
        "/a2a/jsonrpc",
        headers={"A2A-Version": "1.0"},
        json={
            "jsonrpc": "2.0",
            "id": "request-1",
            "method": "SendMessage",
            "params": json_format.MessageToDict(request),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "request-1"
    assert body["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
