
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from graph import research_graph


# --------------------------------------------------
# 1. Initialize FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="ResearchMind AI",
    description=(
        "An AI-powered research agent that plans searches, "
        "collects web sources, evaluates research gaps, "
        "and generates research reports."
    ),
    version="2.0.0",
)


# --------------------------------------------------
# 2. Define request model
# --------------------------------------------------

class ResearchRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The topic you want the research agent to investigate.",
    )

    max_rounds: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of web-search rounds.",
    )


# --------------------------------------------------
# 3. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "ResearchMind AI agent API is running.",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "research": "/research",
            "documentation": "/docs",
        },
    }


# --------------------------------------------------
# 4. Health-check endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ResearchMind AI",
    }


# --------------------------------------------------
# 5. Research-agent endpoint
# --------------------------------------------------

@app.post("/research")
def research(request: ResearchRequest):
    topic = request.topic.strip()

    if len(topic) < 3:
        raise HTTPException(
            status_code=422,
            detail="Please provide a research topic with at least 3 characters.",
        )

    try:
        # Run the LangGraph research agent.
        result = research_graph.invoke(
            {
                "topic": topic,
                "search_queries": [],
                "sources": [],
                "report": "",
                "errors": [],
                "research_round": 0,
                "max_rounds": request.max_rounds,
                "research_complete": False,
                "evaluation": "",
            }
        )

        # Return the final report and agent execution details.
        return {
            "success": bool(result.get("report")),
            "topic": topic,
            "report": result.get("report", ""),
            "sources": result.get("sources", []),
            "search_queries": result.get("search_queries", []),
            "research_round": result.get("research_round", 0),
            "max_rounds": request.max_rounds,
            "research_complete": result.get(
                "research_complete", False
            ),
            "evaluation": result.get("evaluation", ""),
            "errors": result.get("errors", []),
        }

    except Exception as exc:
        # Keep detailed diagnostics in the backend terminal.
        print(
            f"Research agent failed: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The research agent encountered an error. "
                "Check the backend terminal for details."
            ),
        ) from exc
