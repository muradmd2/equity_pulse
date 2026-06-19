"""FastAPI entry point for the educational financial research app."""
from app.config import get_settings
from app.constants import APP_DESCRIPTION, APP_NAME
from app.graph.builder import invoke_graph, resume_session, stream_graph_sse
from app.utils.tracing import configure_langsmith_environment

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

load_dotenv()



configure_langsmith_environment()

app = FastAPI(title=APP_NAME, description=APP_DESCRIPTION)
STATIC_DIR = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    """Incoming financial research query request."""

    query: str = Field(min_length=1)
    session_id: str | None = None
    model_name: str | None = None
    openai_api_key: str | None = None
    stream: bool = False


@app.get("/health")
def health() -> dict[str, object]:
    """Return application health and missing configuration warnings."""

    settings = get_settings()
    missing_config = settings.missing_required_config
    return {
        "status": "ok",
        "app_env": settings.app_env,
        "missing_config": missing_config,
        "configuration_warnings": [
            f"Missing required configuration: {name}" for name in missing_config
        ],
    }


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the lightweight browser frontend."""

    return FileResponse(STATIC_DIR / "index.html")


@app.post("/query")
def query(request: QueryRequest):
    """Run a financial research query with optional server-sent streaming."""

    if request.stream:
        return StreamingResponse(
            stream_graph_sse(
                request.query,
                session_id=request.session_id,
                model_name=request.model_name,
                openai_api_key=request.openai_api_key,
            ),
            media_type="text/event-stream",
        )

    result = invoke_graph(
        request.query,
        session_id=request.session_id,
        model_name=request.model_name,
        openai_api_key=request.openai_api_key,
    )
    return {
        "session_id": result.get("session_id", request.session_id),
        "final_summary": result.get("final_summary", ""),
        "state": result,
    }


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, object]:
    """Return the latest checkpointed state for a session."""

    state = resume_session(session_id)
    return {
        "session_id": session_id,
        "state": state,
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
