"""Streamlit front-end for the Agentic AI Travel Planner."""
import json
import os
from datetime import date, timedelta

import streamlit as st

from agent import CITIES, DEFAULT_MODELS, PLACE_TYPES, build_llm, run_agent, split_answer

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")


def secret(name: str) -> str:
    """Read a key from Streamlit secrets or environment variables (never hard-code keys)."""
    try:
        return st.secrets.get(name, "") or os.environ.get(name, "")
    except Exception:  # no secrets file present
        return os.environ.get(name, "")


# ----------------------------- Sidebar: LLM settings -----------------------------
with st.sidebar:
    st.header("🔑 LLM settings")
    provider = st.selectbox("Provider", list(DEFAULT_MODELS))
    key_name = "OPENAI_API_KEY" if provider == "OpenAI" else "GOOGLE_API_KEY"
    api_key = st.text_input(f"{key_name}", value=secret(key_name), type="password",
                            help="Stored only in this session. On Streamlit Cloud, put it in Secrets.")
    model = st.text_input("Model", value=DEFAULT_MODELS[provider])
    st.caption("Weather comes from Open-Meteo (free, no key needed).")

# ----------------------------- Main: trip form -----------------------------
st.title("✈️ Agentic AI Travel Planner")
st.caption("A LangChain agent that picks flights, hotels and attractions, checks the weather "
           "and builds a budgeted day-wise itinerary.")

col1, col2, col3 = st.columns(3)
origin = col1.selectbox("From", CITIES, index=CITIES.index("Delhi"))
destination = col2.selectbox("To", CITIES, index=CITIES.index("Goa"))
start_date = col3.date_input("Start date", value=date.today() + timedelta(days=14))

col4, col5, col6 = st.columns(3)
days = col4.slider("Trip length (days)", 3, 7, 3)
travelers = col5.number_input("Travelers", 1, 10, 1)
budget = col6.number_input("Total budget (₹, 0 = no limit)", 0, 1_000_000, 0, step=1000)

col7, col8 = st.columns([2, 1])
interests = col7.multiselect("Interests", PLACE_TYPES, default=[])
hotel_pref = col8.selectbox("Hotel preference", ["Best value", "Cheapest", "Highest rated"])
notes = st.text_area("Anything else? (optional)", placeholder="e.g. prefer morning flights")

if st.button("Plan my trip", type="primary", use_container_width=True):
    if not api_key:
        st.error(f"Please enter your {key_name} in the sidebar.")
    elif origin == destination:
        st.error("Origin and destination must be different.")
    else:
        end_date = start_date + timedelta(days=days - 1)
        query = (
            f"Plan a {days}-day trip from {origin} to {destination}, "
            f"{start_date} to {end_date}, for {travelers} traveler(s). "
            f"Hotel preference: {hotel_pref}. "
            f"Interests: {', '.join(interests) if interests else 'no preference'}. "
            f"Total budget: {'₹' + str(budget) if budget else 'no limit'}. {notes}"
        )
        try:
            llm = build_llm(provider, api_key, model)
            with st.status("Agent is planning your trip...", expanded=True) as status:
                def show_step(step):
                    st.write(f"🔧 **{step['tool']}** — {json.dumps(step['args'])[:140]}")

                result = run_agent(llm, query, on_step=show_step)
                status.update(label="Plan ready!", state="complete", expanded=False)

            text, data = split_answer(result["answer"])
            tab1, tab2, tab3 = st.tabs(["📋 Itinerary", "🧾 JSON", "🧠 Agent reasoning"])
            with tab1:
                st.markdown(text)
            with tab2:
                if data:
                    st.json(data)
                    st.download_button("Download JSON", json.dumps(data, indent=2),
                                       "itinerary.json", "application/json")
                else:
                    st.info("The agent did not return a JSON block this time.")
            with tab3:
                for i, step in enumerate(result["trace"], 1):
                    with st.expander(f"{i}. {step['tool']}"):
                        st.write("**Input**")
                        st.json(step["args"])
                        st.write("**Output**")
                        st.code(step["result"], language="json")
        except ImportError as exc:
            st.error(f"Missing package: {exc}. Run `pip install -r requirements.txt`.")
        except Exception as exc:  # noqa: BLE001 - show API/auth errors in the UI
            st.error(f"Something went wrong: {exc}")
