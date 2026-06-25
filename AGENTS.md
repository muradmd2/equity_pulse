# Codex Project Instructions

## Project Purpose

This project is an educational AI financial research application for learning LangGraph, agentic workflows, multi-node graph orchestration, parallel node execution, tool-based financial data retrieval, LLM-based routing, summarization, review, streaming responses, LangSmith observability, and SQLite checkpointing.

The app accepts a user's financial query, classifies it into one or more research categories, runs the relevant category nodes in parallel, summarizes the collected information, reviews the answer for missing pieces, optionally regenerates once, and returns a final source-cited response.

This app is for learning and research only. It must never present itself as personalized investment advice.

## Architecture Rules

- Use Python 3.11+ with LangGraph, LangChain / LangChain OpenAI integration, OpenAI LLMs, FastAPI, Pydantic, pytest, python-dotenv, SQLite checkpointing with `SqliteSaver`, LangSmith tracing, and provider integrations such as Tavily, yfinance, Finnhub, and SEC EDGAR APIs.
- Model the workflow as a LangGraph `StateGraph`.
- Keep graph nodes modular and category-specific.
- The classifier must support multi-category financial queries.
- Relevant category nodes should run in parallel after classification.
- Do not run unrelated category nodes unless classification selected them or the user specifically asked for them.
- Each node should return structured results and should not crash the graph.
- Node failures should be represented as structured failed or partial results with errors, not unhandled exceptions.
- Preserve raw retrieved data separately from LLM-generated summaries.
- Keep string literals, reusable prompts, and LLM instructions in separate files where practical.
- Use constants for repeated category names, provider names, statuses, and configuration values.
- Support the Agent2Agent (A2A) protocol for agent discovery and interoperable task exchange. Expose an A2A Agent Card and use A2A message, task, status, artifact, and error semantics at the external boundary while keeping the internal LangGraph workflow modular and provider-agnostic.
- Do not implement unrelated chunks of the app unless specifically asked.
- Prefer small incremental changes with focused tests over large broad rewrites.

## Coding Style

- Follow existing project structure and patterns before introducing new abstractions.
- Use Pydantic models or typed schemas for structured inputs and outputs.
- Prefer structured LLM outputs for classification, review, and other routing-critical logic.
- Use deterministic model settings for classification and review logic, such as temperature `0`.
- Keep retrieval services separate from graph node orchestration.
- Keep provider-specific normalization close to the service or node that retrieves the data.
- Add concise comments for LangGraph concepts where helpful because this is a learning app.
- Avoid speculative financial language and unsupported conclusions in code, prompts, and summaries.

## Testing Rules

- Use `pytest` for tests.
- Add or update focused tests with each behavioral change.
- Prefer small incremental changes and tests.
- Test classification, routing, aggregation, citation preservation, review-loop behavior, node failure handling, and final-response guardrails as those areas are implemented.
- Tests should not require live external APIs unless explicitly marked or isolated.
- Mock external providers such as yfinance, Tavily, Finnhub, SEC EDGAR, OpenAI, and LangSmith in normal unit tests.
- Confirm that failed or partial category nodes do not crash the graph.
- Confirm that the final answer includes the educational disclaimer once final response behavior exists.

## Financial-Data Guardrails

- Clearly state that outputs are educational research only and not personalized financial advice.
- Do not use phrases such as "you should buy" or "you should sell."
- Prefer language such as "analysts currently rate," "the data suggests," "risks include," and "this may be worth researching further."
- Clearly separate sourced facts from interpretation.
- Mention data freshness where relevant.
- Mention when market data may be delayed.
- Do not hallucinate unavailable financial metrics.
- Do not use unsourced claims.
- Do not compare stocks as a recommendation unless the user explicitly asks for a comparison, and even then keep the comparison educational and source-grounded.
- Technical analysis must be descriptive, not predictive.

## Citation Requirements

- Every external fact needs citations.
- Do not invent citations.
- If a fact cannot be sourced, omit it or label it as unavailable.
- Each citation should include the provider, source title, URL when available, published date when available, retrieved timestamp, and a short snippet or evidence text when available.
- Final responses should include a `Sources` section.
- Preserve source metadata through retrieval, aggregation, summary, and final response steps.

## Structured Node Results

Each category node should normalize its output into a shared structured result shape with:

- `category`
- `subquery`
- `status`, such as `success`, `partial`, or `failed`
- `data`
- `summary`
- `citations`
- `errors`

When a provider fails, the node should catch the exception, include the error in the structured result, return any partial data available, and allow the graph to continue.

## Scope Discipline

Future Codex sessions should stay tightly aligned with the user's current request. Do not build extra nodes, provider integrations, UI, README content, or broad refactors unless the user specifically asks for them. When in doubt, make the smallest useful change and verify it with an appropriately small test.
