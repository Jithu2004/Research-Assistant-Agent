
# AI Research Assistant Agent

A multi-step AI research agent built with **LangGraph**, **Google Gemini**, and **Fastcrw API**.

## Features

- **Dynamic Planning**: Uses Gemini to generate relevant search queries for a research topic.
- **Smart Retrieval**: Fetches web information and source content using the Fastcrw API.
- **Self-Evaluation**: Evaluates collected research to determine whether additional information is needed.
- **Iterative Research Loop**: Uses conditional routing to perform additional research rounds, up to a configurable maximum.
- **State Management**: Uses LangGraph state to maintain the research topic, queries, sources, and evaluation results across workflow steps.
- **Report Generation**: Uses Gemini to synthesize collected information into a structured research report.
- **REST API**: Exposes the research workflow through FastAPI.
- **Interactive UI**: Provides a Streamlit interface for submitting research topics and viewing reports.

## Core Flow

`Plan` → `Search` → `Evaluate` → `Retry (if needed)` → `Generate Report`

## Technologies Used

- Python
- LangGraph
- Google Gemini
- LangChain Core
- Fastcrw API
- FastAPI
- Streamlit
- Requests
- Pydantic
- Uvicorn
- python-dotenv

## Setup

1. Create a `.env` file and configure the environment variables:

   - `GOOGLE_API_KEY`
   - `GEMINI_MODEL`
   - `CRW_API_KEY`

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI backend:

   ```bash
   python -m uvicorn api:app --reload
   ```

5. Run the Streamlit frontend in a separate terminal:

   ```bash
   python -m streamlit run app.py
   ```

6. Open the API documentation at `http://127.0.0.1:8000/docs`.

## Files

- `state.py`: Defines the shared research state using `TypedDict`.
- `graph.py`: Implements research planning, web search, evaluation, report generation, and LangGraph workflow orchestration.
- `api.py`: Exposes the agent through FastAPI endpoints.
- `app.py`: Provides the Streamlit user interface.
- `requirements.txt`: Lists the project dependencies.
- `.env`: Stores API keys and model configuration.

## Agent Architecture

The agent uses LangGraph's `StateGraph` to coordinate multiple research steps. After retrieving web sources, the evaluation node determines whether the workflow should continue researching or proceed to report generation. The workflow maintains state across steps and uses conditional routing to control the research loop.
