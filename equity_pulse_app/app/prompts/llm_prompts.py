"""Prompt templates for OpenAI-backed graph nodes."""

CLASSIFICATION_PROMPT = """Classify this financial research query for a LangGraph financial research app.

Return structured output with:
- detected_tickers
- detected_companies
- categories
- reasoning

Allowed category values:
- company_information
- competitor_news
- sector_news
- financials
- company_news
- current_stock_price
- analyst_ratings
- latest_report_release
- dow_jones_index
- technical_analysis

Routing rules:
- "how is stock doing" should include current_stock_price, company_news, technical_analysis.
- "is it a buy" should include analyst_ratings, financials, company_news, technical_analysis.
- "latest report" should include latest_report_release.
- "Dow Jones" should include dow_jones_index.
- Financial queries should include financials.
- Company overview queries should include company_information.
- If a company name is present, include it in detected_companies.
- If a ticker is present, include it in detected_tickers.
- If a company name is ambiguous, leave detected_tickers empty and explain the ambiguity in reasoning.
- If the query asks for today's news on a company then only return today's published news.
- Company's quote query should only include current quote price and no financials.


Query:
{query}
"""

SUMMARY_PROMPT = """
Do not present more data in the summary than what is being asked for by original_query.
Strip all data from the summary except for what is being requested.

You are summarizing source-grounded financial research for an educational app.

Use only the retrieved data and citations provided in the JSON context.
Do not add unsupported external facts, assumptions, forecasts, or personalized investment advice.

Include only sections relevant to the original query and detected categories.
Keep the answer concise.
Every external fact must be tied to the provided source citations when citations are available.
Always include this disclaimer: "This is for educational research only and is not personalized financial advice."

JSON context:
{context}
"""

REVIEW_PROMPT = """Review whether the summary answers the user's financial research query.

Return structured output with:
- missing_info_found
- missing_info
- web_search_query
- reasoning

Review rules:
- Compare the query against the summary_response.
- Look only for missing information caused by lack of data in the retrieved resources.
- If provided data does not contains the information requested, then consider this as missing info.
- If the information requested is not released or available yet, even then consider this as missing info.
- Mark missing_info_found as true only when a missing part of the query would make the summary materially incomplete.
- Do not require extra categories or facts that the user did not ask for.
- Do not add investment advice or unsupported financial claims.
- If missing_info_found is true, provide one concise Tavily web_search_query for only the unanswered part.
- Keep web_search_query search-oriented, no more than 16 words, and not a long instruction.
- If missing_info_found is false, web_search_query must be null and missing_info must be empty.

JSON context:
{context}
"""
