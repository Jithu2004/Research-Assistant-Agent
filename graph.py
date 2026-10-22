
import json
import os
import re
import traceback
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from state import ResearchState

load_dotenv()

FASTCRW_URL = "https://api.fastcrw.com/v1/search"

DEFAULT_MAX_ROUNDS = 2
RESULTS_PER_QUERY = 3
SEARCH_TIMEOUT = 25
LLM_TIMEOUT = 60


# --------------------------------------------------
# 1. Gemini model
# --------------------------------------------------

def get_llm():
    """Create the Gemini model and validate configuration."""

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing from your .env file."
        )

    if not model_name:
        raise RuntimeError(
            "GEMINI_MODEL is missing from your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        timeout=LLM_TIMEOUT,
        max_retries=1,
    )


def response_text(response: Any) -> str:
    """Convert a Gemini response into plain text."""

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])

        return "\n".join(parts).strip()

    return str(content).strip()


def parse_json(text: str) -> dict:
    """Parse JSON returned by Gemini, including fenced JSON."""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if match:
        try:
            result = json.loads(match.group(0))
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            pass

    return {}


# --------------------------------------------------
# 2. Plan the research
# --------------------------------------------------

def plan_research(state: ResearchState) -> dict:
    """Ask Gemini to create focused search queries."""

    topic = state["topic"]

    try:
        prompt = f"""
You are the planning component of a web research agent.

Research goal:
{topic}

Create 3 distinct web search queries covering different
important aspects of the topic.

Return only valid JSON in this format:
{{"queries": ["query one", "query two", "query three"]}}
"""

        result = parse_json(response_text(get_llm().invoke(prompt)))

        queries = result.get("queries", [])

        if not isinstance(queries, list):
            queries = []

        queries = [
            q.strip()
            for q in queries
            if isinstance(q, str) and q.strip()
        ][:3]

        if not queries:
            queries = [topic]

        return {
            "search_queries": queries,
            "sources": state.get("sources", []),
            "errors": state.get("errors", []),
            "research_round": 0,
            "max_rounds": state.get(
                "max_rounds", DEFAULT_MAX_ROUNDS
            ),
            "research_complete": False,
        }

    except Exception as exc:
        traceback.print_exc()

        return {
            "search_queries": [topic],
            "sources": state.get("sources", []),
            "errors": state.get("errors", []) + [
                f"Planning failed: {type(exc).__name__}: {exc}"
            ],
            "research_round": 0,
            "max_rounds": state.get(
                "max_rounds", DEFAULT_MAX_ROUNDS
            ),
            "research_complete": False,
        }


# --------------------------------------------------
# 3. Parse Fastcrw search results
# --------------------------------------------------

def extract_results(payload: Any) -> list[dict]:
    """Handle common Fastcrw response formats."""

    if isinstance(payload, list):
        return [
            item for item in payload
            if isinstance(item, dict)
        ]

    if not isinstance(payload, dict):
        return []

    data = payload.get("data", payload)

    if isinstance(data, list):
        return [
            item for item in data
            if isinstance(item, dict)
        ]

    if isinstance(data, dict):
        for key in ("results", "web", "items"):
            value = data.get(key)

            if isinstance(value, list):
                return [
                    item for item in value
                    if isinstance(item, dict)
                ]

    return []


# --------------------------------------------------
# 4. Search the web
# --------------------------------------------------

def search_web(state: ResearchState) -> dict:
    """Search Fastcrw and collect unique sources."""

    api_key = os.getenv("CRW_API_KEY")

    sources = list(state.get("sources", []))
    errors = list(state.get("errors", []))
    round_number = state.get("research_round", 0) + 1

    if not api_key:
        errors.append("CRW_API_KEY is missing from .env.")

        return {
            "sources": sources,
            "errors": errors,
            "research_round": round_number,
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    known_urls = {
        source.get("url")
        for source in sources
        if source.get("url")
    }

    queries = state.get("search_queries", [])

    for query in queries:
        print(f"[ResearchMind] Searching: {query}")

        try:
            response = requests.post(
                FASTCRW_URL,
                headers=headers,
                json={
                    "query": query,
                    "limit": RESULTS_PER_QUERY,
                    "scrapeOptions": {
                        "formats": ["markdown"],
                        "onlyMainContent": True,
                    },
                },
                timeout=SEARCH_TIMEOUT,
            )

            response.raise_for_status()
            results = extract_results(response.json())

            if not results:
                print(
                    f"[ResearchMind] No results returned for: {query}"
                )

            for item in results:
                url = item.get("url") or item.get("link") or ""
                title = item.get("title") or "Untitled source"

                content = (
                    item.get("markdown")
                    or item.get("content")
                    or item.get("description")
                    or item.get("snippet")
                    or item.get("text")
                    or ""
                )

                if url and url in known_urls:
                    continue

                if not content and not url:
                    continue

                source = {
                    "title": str(title),
                    "url": str(url),
                    "content": str(content)[:8000],
                    "query": query,
                }

                sources.append(source)

                if url:
                    known_urls.add(url)

        except Exception as exc:
            traceback.print_exc()

            errors.append(
                f"Search failed for '{query}': "
                f"{type(exc).__name__}: {exc}"
            )

    print(
        f"[ResearchMind] Round {round_number}: "
        f"{len(sources)} unique sources collected."
    )

    return {
        "sources": sources,
        "errors": errors,
        "research_round": round_number,
    }


# --------------------------------------------------
# 5. Evaluate the research
# --------------------------------------------------

def evaluate_research(state: ResearchState) -> dict:
    """Determine whether more research is needed."""

    sources = state.get("sources", [])
    round_number = state.get("research_round", 0)
    max_rounds = state.get("max_rounds", DEFAULT_MAX_ROUNDS)

    if round_number >= max_rounds:
        return {
            "research_complete": True,
            "evaluation": "Maximum research rounds reached.",
            "search_queries": [],
        }

    if not sources:
        return {
            "research_complete": True,
            "evaluation": "No usable sources were retrieved.",
            "search_queries": [],
            "errors": state.get("errors", []) + [
                "No web sources were retrieved. Check Fastcrw "
                "credentials, connectivity, and API response."
            ],
        }

    source_summaries = []

    for index, source in enumerate(sources[:12], start=1):
        source_summaries.append(
            f"[S{index}] {source.get('title', '')}\n"
            f"URL: {source.get('url', '')}\n"
            f"Content: {source.get('content', '')[:1000]}"
        )

    try:
        prompt = f"""
You are the quality evaluator for a web research agent.

Research goal:
{state["topic"]}

Current round: {round_number} of {max_rounds}

Collected sources:
{chr(10).join(source_summaries)}

Assess whether the sources adequately cover the research goal.
If important gaps remain, suggest up to 2 new and specific
search queries. Do not repeat existing research.

Return only valid JSON:
{{
  "complete": true,
  "reason": "Brief explanation",
  "next_queries": []
}}

Set complete to false only if additional research is useful.
"""

        result = parse_json(
            response_text(get_llm().invoke(prompt))
        )

        complete = bool(result.get("complete", True))
        reason = str(
            result.get("reason", "Evaluation completed.")
        )

        queries = result.get("next_queries", [])

        if not isinstance(queries, list):
            queries = []

        queries = [
            q.strip()
            for q in queries
            if isinstance(q, str) and q.strip()
        ][:2]

        if not queries:
            complete = True

        return {
            "research_complete": complete,
            "evaluation": reason,
            "search_queries": [] if complete else queries,
        }

    except Exception as exc:
        traceback.print_exc()

        return {
            "research_complete": True,
            "evaluation": (
                "Evaluation failed; using the collected sources."
            ),
            "search_queries": [],
            "errors": state.get("errors", []) + [
                f"Evaluation failed: {type(exc).__name__}: {exc}"
            ],
        }


def route_after_evaluation(state: ResearchState) -> str:
    """Route to another search round or report generation."""

    if state.get("research_complete", True):
        return "write_report"

    if state.get("research_round", 0) >= state.get(
        "max_rounds", DEFAULT_MAX_ROUNDS
    ):
        return "write_report"

    if not state.get("search_queries"):
        return "write_report"

    return "search_web"


# --------------------------------------------------
# 6. Generate the final report
# --------------------------------------------------

def write_report(state: ResearchState) -> dict:
    """Generate an evidence-based Markdown research report."""

    topic = state["topic"]
    sources = state.get("sources", [])
    errors = list(state.get("errors", []))

    if not sources:
        return {
            "report": (
                f"# Research Report: {topic}\n\n"
                "No usable web sources were retrieved, so an "
                "evidence-backed report could not be generated. "
                "Check the Fastcrw API configuration and try again."
            ),
            "errors": errors,
        }

    source_text = "\n\n".join(
        (
            f"[S{index}] {source.get('title', 'Untitled source')}\n"
            f"URL: {source.get('url', '')}\n"
            f"Search query: {source.get('query', '')}\n"
            f"Content: {source.get('content', '')[:2500]}"
        )
        for index, source in enumerate(sources[:12], start=1)
    )

    prompt = f"""
You are a careful research analyst.

Write a clear, structured Markdown report about:
{topic}

Use only the supplied sources for source-backed factual claims.
Cite claims inline using [S1], [S2], etc.
Never invent citations, source URLs, quotations, or evidence.
If sources disagree, explain the disagreement.
Clearly state uncertainties and limitations.

Research evaluation:
{state.get("evaluation", "Not provided")}

Collected sources:
{source_text}

Use these sections:
1. Title
2. Executive Summary
3. Main Findings
4. Supporting Evidence
5. Limitations and Uncertainties
6. Conclusion
7. References

In References, list each source ID, title, and URL.
Return the report in Markdown.
"""

    try:
        print("[ResearchMind] Generating final report...")

        report = response_text(get_llm().invoke(prompt))

        if not report:
            raise RuntimeError(
                "Gemini returned an empty report."
            )

        print("[ResearchMind] Report generated successfully.")

        return {
            "report": report,
            "errors": errors,
        }

    except Exception as exc:
        # Print the actual error and full traceback in the terminal.
        traceback.print_exc()

        errors.append(
            f"Report generation failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "report": (
                f"# Research Report: {topic}\n\n"
                "## Report Generation Failed\n\n"
                "The agent collected web sources, but Gemini "
                "could not generate the final report. Check the "
                "backend terminal for the detailed error."
            ),
            "errors": errors,
        }


# --------------------------------------------------
# 7. Build the LangGraph agent
# --------------------------------------------------

def build_graph():
    """Create and compile the research workflow."""

    workflow = StateGraph(ResearchState)

    workflow.add_node("plan_research", plan_research)
    workflow.add_node("search_web", search_web)
    workflow.add_node("evaluate_research", evaluate_research)
    workflow.add_node("write_report", write_report)

    workflow.add_edge(START, "plan_research")
    workflow.add_edge("plan_research", "search_web")
    workflow.add_edge("search_web", "evaluate_research")

    workflow.add_conditional_edges(
        "evaluate_research",
        route_after_evaluation,
        {
            "search_web": "search_web",
            "write_report": "write_report",
        },
    )

    workflow.add_edge("write_report", END)

    return workflow.compile()


research_graph = build_graph()
