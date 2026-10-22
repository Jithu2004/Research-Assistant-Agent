
import os

import requests
import streamlit as st

API_URL = os.getenv("RESEARCH_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AI Research Assistant")
st.write(
    "Research a topic using Fastcrw web search and Google Gemini. "
    "Generate structured reports with source citations."
)
st.divider()

topic = st.text_area(
    "What would you like to research?",
    placeholder=(
        "Example: Compare the benefits, risks, and recent developments "
        "of AI in education."
    ),
    height=130,
)

if st.button("Start Research", type="primary", use_container_width=True):
    clean_topic = topic.strip()

    if len(clean_topic) < 3:
        st.warning("Please enter a research topic with at least 3 characters.")
    else:
        try:
            with st.spinner(
                "Planning research, searching the web, and writing your report..."
            ):
                response = requests.post(
                    f"{API_URL}/research",
                    json={"topic": clean_topic},
                    timeout=600,
                )

            if response.ok:
                data = response.json()

                st.session_state["report"] = data.get("report", "")
                st.session_state["sources"] = data.get("sources", [])
                st.session_state["errors"] = data.get("errors", [])
                st.session_state["search_queries"] = data.get(
                    "search_queries", []
                )
                st.session_state["topic"] = data.get("topic", clean_topic)

            else:
                st.error(f"The API returned HTTP {response.status_code}.")
                try:
                    st.json(response.json())
                except ValueError:
                    st.code(response.text)

        except requests.Timeout:
            st.error(
                "The research request timed out. Check the backend and "
                "try again."
            )
        except requests.ConnectionError:
            st.error(
                "Could not connect to FastAPI. Start the backend in another "
                "terminal using: python -m uvicorn api:app --reload"
            )
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")

if "report" in st.session_state:
    st.divider()
    st.header("📄 Research Report")
    st.caption(f"Research topic: {st.session_state['topic']}")

    if st.session_state["report"]:
        st.markdown(st.session_state["report"])

        st.download_button(
            label="Download Report as Markdown",
            data=st.session_state["report"],
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.warning(
            "No report was generated. Review the errors below and check "
            "your API keys and backend terminal."
        )

    queries = st.session_state.get("search_queries", [])
    if queries:
        with st.expander(f"🔎 Search queries ({len(queries)})"):
            for query in queries:
                st.markdown(f"- {query}")

    sources = st.session_state.get("sources", [])
    with st.expander(f"🔗 Retrieved Sources ({len(sources)})"):
        if sources:
            for index, source in enumerate(sources, start=1):
                title = source.get("title", "Untitled source")
                url = source.get("url", "")
                query = source.get("query", "")

                st.markdown(f"**[S{index}] {title}**")

                if url:
                    st.markdown(f"[Open source]({url})")

                if query:
                    st.caption(f"Search query: {query}")

                content = source.get("content", "")
                if content:
                    st.write(content[:1000])

                st.divider()
        else:
            st.info("No sources were returned.")

    for error in st.session_state.get("errors", []):
        st.warning(error)
