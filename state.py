
from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    topic: str
    search_queries: list[str]
    sources: list[dict[str, Any]]
    report: str
    errors: list[str]

    research_round: int
    max_rounds: int
    research_complete: bool
    evaluation: str
