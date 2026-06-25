# Financial AI Research App

Educational AI financial research application for learning LangGraph, multi-node agent workflows, parallel category retrieval, source-cited summarization, streaming, review-and-regenerate loops, LangSmith tracing, and SQLite checkpointing.

This project is for educational research only and is not personalized investment advice.

## Architecture

```text
FastAPI
  POST /query
  GET /sessions/{session_id}
  GET /health
  GET /.well-known/agent-card.json
  POST /a2a/jsonrpc
  /a2a/rest/*
    |
    v
LangGraph StateGraph
  START
    -> classify_query_node
    -> build_subqueries_node
    -> dynamic parallel category nodes
       -> company_info_node              yfinance
       -> competitor_news_node           Tavily
       -> sector_news_node               Tavily
       -> financials_node                yfinance + SEC filing refs
       -> company_news_node              Tavily
       -> current_stock_price_node       yfinance
       -> analyst_ratings_node           Finnhub + Tavily fallback
       -> latest_report_release_node     SEC EDGAR
       -> dow_jones_index_node           yfinance
       -> technical_analysis_node        yfinance + local indicators
    -> aggregate_results_node
    -> summary_node
    -> review_node, first pass only
       -> rewrite_query_node, at most once, when major gaps are found
       -> classify_query_node
    -> final_response_node
  END

SQLite checkpointing stores graph state by thread_id/session_id.
```

## Setup

The commands below use PowerShell. Run them from the repository root.

### 1. Open the application directory

```powershell
Set-Location .\equity_pulse_app
```

The application reads `.env` and creates its SQLite checkpoint database relative
to this directory, so run the remaining commands here.

### 2. Create and activate a Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks the activation script, allow scripts for the current shell and
try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Python 3.11 or newer is required.

### 3. Install the dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Create `.env`


Add the following configuration and replace the placeholder values in .env file:

```dotenv
# Required for LLM classification, summarization, and review
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-5-nano

# Required for web/news retrieval and Finnhub market data
TAVILY_API_KEY=your_tavily_api_key
FINNHUB_API_KEY=your_finnhub_api_key

# SEC EDGAR requires a descriptive User-Agent with real contact information
SEC_USER_AGENT="EquityPulse/1.0 your-email@example.com"

# Optional: only provide LANGSMITH_API_KEY when tracing is enabled
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=financial-langgraph-learning-app
```

OpenAI, Tavily, and Finnhub require API keys for their provider-backed features.
LangSmith is optional: leave `LANGSMITH_TRACING=false` and
`LANGSMITH_API_KEY` empty if you do not use it. yfinance does not require an API
key. Never commit `.env` or paste real keys into source files.

### 5. Launch FastAPI and the UI

Start the application from `equity_pulse_app`:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Then open:

- Web UI: `http://127.0.0.1:8000/`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Configuration health check: `http://127.0.0.1:8000/health`

Keep this terminal running while using the app. Press `Ctrl+C` to stop it.

> **Streamlit status:** This repository does not currently contain a Streamlit
> entry point or the `streamlit` package. Its included UI is the HTML interface
> served directly by FastAPI at `/`; no second UI process is required. A command
> such as `streamlit run streamlit_app.py` will not work unless a Streamlit app is
> added to the project.

## Environment Variables

Required for full provider-backed behavior:

```text
OPENAI_API_KEY=
TAVILY_API_KEY=
FINNHUB_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=financial-langgraph-learning-app
SEC_USER_AGENT="FinancialLangGraphLearningApp/1.0 contact@email.com"
```

Optional local settings:

```text
APP_ENV=local
LOG_LEVEL=INFO
MAX_REVIEW_ATTEMPTS=1
DEFAULT_LOOKBACK_DAYS=30
TECHNICAL_ANALYSIS_LOOKBACK_DAYS=180
TAVILY_TIMEOUT_SECONDS=10
TAVILY_MAX_RESULTS=5
FINNHUB_TIMEOUT_SECONDS=10
FINNHUB_NEWS_LOOKBACK_DAYS=30
SEC_EDGAR_TIMEOUT_SECONDS=10
SEC_EDGAR_MIN_REQUEST_INTERVAL_SECONDS=0.1
SQLITE_CHECKPOINT_DB=checkpoints.sqlite
A2A_BASE_URL=http://127.0.0.1:8000
```

`GET /health` reports missing configuration clearly.

## A2A Protocol

The same FastAPI process exposes the app as an A2A 1.0 agent:

- Agent Card: `GET /.well-known/agent-card.json`
- JSON-RPC transport: `POST /a2a/jsonrpc`
- HTTP+JSON transport: `/a2a/rest/*`

Set `A2A_BASE_URL` to the externally reachable server origin in deployed
environments so discovery clients receive usable interface URLs. A2A context IDs
are passed to LangGraph as checkpoint session IDs, and completed research is
returned as a source-cited text artifact.

## Run FastAPI

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/health
```

## API Usage

Non-streamed query:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/query `
  -ContentType "application/json" `
  -Body '{"query":"How is CRM stock doing today?","session_id":"demo-session"}'
```

Streamed query:

```powershell
Invoke-WebRequest `
  -Method Post `
  -Uri http://127.0.0.1:8000/query `
  -ContentType "application/json" `
  -Body '{"query":"How is CRM stock doing today?","session_id":"demo-session","stream":true}'
```

Resume a checkpointed session:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/sessions/demo-session
```

## Run Tests

```powershell
pytest
```

Tests mock external providers where practical and do not require live API calls for normal unit coverage.

## Example Queries

```text
How is CRM stock doing today?
What are the financials for ALKT?
What is the latest report released by Microsoft?
Why is the Dow Jones moving today?
Is CRM a buy?
Compare CRM with competitors based on news and financials.
```

The app must not answer with personalized buy/sell advice. Analyst-related output is framed as available analyst data, not a recommendation.

## LangGraph Nodes

- `classify_query_node`: extracts tickers, company names, and one or more query categories.
- `build_subqueries_node`: creates concise provider-friendly subqueries for selected categories.
- Category nodes: retrieve normalized `RetrievalResult` objects with citations and errors.
- `aggregate_results_node`: separates successful, partial, and failed results; deduplicates citations and repeated facts.
- `summary_node`: builds concise source-grounded sections only for retrieved/requested categories.
- `review_node`: checks for missing categories, missing citations, failed important nodes, and missing ticker/company information.
- `rewrite_query_node`: creates one focused retry query for missing information.
- `final_response_node`: returns the final summary, combining first and regenerated summaries when applicable.

## Streaming

`POST /query` with `"stream": true` returns server-sent events.

Streamed events include:

```text
Analyzing query...
Detected categories: current stock price, company news, analyst ratings
Fetching stock quote...
Searching recent company news...
Retrieving analyst ratings...
Summarizing results...
Review failed. Regenerating the response.
Finalizing response...
```

Summary text is streamed as token-like chunks where possible.

## Review And Regenerate

The review step runs only after the first summary. If it finds a major gap and no retry has happened yet, it:

- records `Review failed. Regenerating the response.`
- creates a focused rewritten query for missing information only
- sets `review_attempted = true`
- sets `iteration = 2`
- preserves first-pass retrieval results and `first_summary`
- routes back to classification

The second pass skips review and routes directly to the final response. Infinite review loops are not allowed.

## LangSmith Tracing

LangSmith settings are read from environment variables:

```text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=financial-langgraph-learning-app
```

Tests are written so LangSmith tracing does not require a real external API call.

## SQLite Checkpointing

The graph uses LangGraph `SqliteSaver`.

- Checkpoints are stored in `SQLITE_CHECKPOINT_DB`, default `checkpoints.sqlite`.
- Each API request can provide `session_id`.
- The graph uses `session_id` as LangGraph `thread_id`.
- `GET /sessions/{session_id}` returns the latest checkpointed state.

Checkpointed state includes original query, active query, detected categories, retrieval results, summaries, review status, and final answer.

## Limitations

- This is a learning app, not a production investment research system.
- Provider responses can be delayed, incomplete, rate-limited, or unavailable.
- yfinance market data may be delayed.
- Tavily search results depend on web indexing and source availability.
- Finnhub access can be quota-limited.
- SEC EDGAR requires a valid User-Agent and should be used politely.
- Classification currently includes deterministic local rules and a structured-output helper, but production tuning may need more robust entity resolution.
- Technical analysis is descriptive only and not predictive.

## Disclaimer

This app is for educational research only and is not personalized financial advice. It must not tell users they should buy or sell a security. Every external fact in final answers should be traceable to a source citation.
