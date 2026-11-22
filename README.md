# Agentic AI Travel Planner (LangChain + Streamlit)

An agent that plans 3-7 day trips from three JSON datasets (flights, hotels, places) plus live
weather from Open-Meteo. The LLM chooses which tools to call, then writes a day-wise itinerary
with flight, hotel, weather, budget and reasoning (readable text + JSON).

## Structure
| File | Purpose |
|---|---|
| `travel_tools.py` | Flight / hotel / places / weather / budget logic (no LLM needed) |
| `agent.py` | LangChain tools + tool-calling agent loop (OpenAI or Gemini) |
| `app.py` | Streamlit UI |
| `data/` | flights.json, hotels.json, places.json |
| `travel_planner_colab.ipynb` | Same project as a Google Colab notebook |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Enter your API key in the sidebar (OpenAI or Google Gemini), or copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.

## Deploy to Streamlit Community Cloud
1. Create an empty GitHub repo, then run `./deploy.sh https://github.com/<you>/<repo>.git`
2. On https://share.streamlit.io choose the repo, main file `app.py`
3. Add your key under Settings -> Secrets, then Deploy

## Notes
- The sample flights are dated 2025 and cover limited routes; the agent says when dates differ.
- Weather: live forecast for the next ~16 days, otherwise the same dates last year (clearly labelled).
