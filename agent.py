"""LangChain tool-calling travel agent.

The LLM decides which tools to call (flights, hotels, places, weather, budget),
reads the results, and writes the final itinerary. Works with OpenAI or Google
Gemini - only the API key and model name change.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import travel_tools as tt

CITIES = tt.CITIES
PLACE_TYPES = tt.available_place_types()


# --------------------------------------------------------------------------
# Tools (thin wrappers - the docstrings are what the LLM reads)
# --------------------------------------------------------------------------
@tool
def flight_search_tool(origin: str, destination: str, travel_date: str = "",
                       sort_by: str = "cheapest") -> str:
    """Search flights from origin city to destination city. travel_date is YYYY-MM-DD
    (optional). sort_by is 'cheapest' or 'fastest'. Returns flights with price, times and
    duration. If no flight exists, returns the available routes so you can try a connection."""
    return json.dumps(tt.search_flights(origin, destination, travel_date, sort_by))


@tool
def hotel_recommendation_tool(city: str, min_stars: int = 1, max_price_per_night: float = 0,
                              sort_by: str = "best_value") -> str:
    """Recommend hotels in a city. min_stars 1-5. max_price_per_night in INR (0 = no cap).
    sort_by is 'best_value', 'cheapest' or 'highest_rated'."""
    return json.dumps(tt.recommend_hotels(city, min_stars, max_price_per_night,
                                          sort_by=sort_by))


@tool
def places_discovery_tool(city: str, place_types: Optional[list[str]] = None,
                          min_rating: float = 0, limit: int = 8) -> str:
    """Find attractions in a city, best rated first. place_types can include: beach, fort,
    lake, market, monument, museum, park, temple. Leave place_types empty for all types."""
    return json.dumps(tt.discover_places(city, place_types, min_rating, limit))


@tool
def weather_lookup_tool(city: str, start_date: str, end_date: str) -> str:
    """Get daily weather (condition, max/min temperature in C, rain in mm) for a city between
    start_date and end_date (YYYY-MM-DD). Uses the free Open-Meteo API."""
    return json.dumps(tt.get_weather(city, start_date, end_date))


@tool
def budget_estimation_tool(flight_price: float, hotel_price_per_night: float, nights: int,
                           travelers: int = 1, daily_local_expense: float = 1500,
                           return_flight_price: float = 0) -> str:
    """Estimate total trip cost in INR: flights + hotel + food/local travel.
    flight_price is per person one-way; add return_flight_price if a return flight is used.
    daily_local_expense is per person per day."""
    return json.dumps(tt.estimate_budget(flight_price, hotel_price_per_night, nights,
                                         travelers, daily_local_expense, return_flight_price))


TOOLS = [flight_search_tool, hotel_recommendation_tool, places_discovery_tool,
         weather_lookup_tool, budget_estimation_tool]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = f"""You are an expert travel planning agent. Plan trips using ONLY the tools
provided; never invent flights, hotels or places.

Cities in the datasets: {", ".join(CITIES)}.

Workflow:
1. Search flights origin -> destination (cheapest unless the user says otherwise). Also search the
   return flight destination -> origin. If no direct route exists, try a connection through
   another city and explain it. If still impossible, say so honestly.
2. Recommend hotels in the destination that respect the user's budget and preference.
3. Find attractions matching the user's interests (use all types if none are given).
4. Get weather for the trip dates (start date to start date + days - 1).
5. Estimate the budget with the budget tool (nights = days - 1).
6. Write the final answer.

Rules:
- Flight data is sample data and its dates may not match the requested dates. If the dates differ,
  say so clearly and present it as the closest available option.
- If the user gave a total budget and the plan exceeds it, say so and suggest cheaper choices.
- Spread places across days (2-3 per day, best rated first, no repeats). If there are fewer places
  than needed, add free-time/rest suggestions instead of making up places.
- Explain WHY you chose each flight and hotel.

Final answer format (exactly):
## Your <N>-Day Trip to <Destination> (<start> - <end>)
**Flight:** ... **Hotel:** ... (one short line each, with price)
### Weather  - one line per day
### Itinerary - Day 1, Day 2 ... each with places and a short note
### Budget - flight, hotel, food & local travel, total
### Why these choices - 2-4 bullets
Then a fenced ```json block with this structure:
{{"trip_summary": {{"destination": "", "start_date": "", "end_date": "", "days": 0, "travelers": 0}},
 "flight": {{}}, "return_flight": {{}}, "hotel": {{}},
 "weather": [{{"date": "", "condition": "", "max_temp_c": 0}}],
 "itinerary": [{{"day": 1, "date": "", "places": [], "notes": ""}}],
 "budget": {{"flight": 0, "hotel": 0, "food_and_local": 0, "total": 0}},
 "reasoning": ""}}
"""


# --------------------------------------------------------------------------
# LLM factory
# --------------------------------------------------------------------------

DEFAULT_MODELS = {"Google Gemini": "gemini-3.5-flash-lite", "OpenAI": "gpt-4o-mini"}


def build_llm(provider: str, api_key: str, model: str = "", temperature: float = 0.2):
    """Create a chat model for the chosen provider."""
    model = model or DEFAULT_MODELS[provider]
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)# type: ignore
    if provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key,
                                      temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider}")


def _text(content) -> str:
    """Message content can be a string or a list of parts (Gemini) - normalise to text."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        parts.append(part if isinstance(part, str) else part.get("text", ""))
    return "".join(parts)


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------
def run_agent(llm, user_query: str, max_steps: int = 12,
              on_step: Optional[Callable[[dict], None]] = None) -> dict:
    """Run the ReAct-style tool-calling loop.

    Returns {"answer": str, "trace": [ {tool, args, result} ... ]}.
    on_step is called after every tool call (used by Streamlit to show progress).
    """
    llm_with_tools = llm.bind_tools(TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_query)]
    trace: list[dict] = []

    for _ in range(max_steps):
        ai = llm_with_tools.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return {"answer": _text(ai.content), "trace": trace}

        for call in ai.tool_calls:
            try:
                result = TOOL_MAP[call["name"]].invoke(call["args"])
            except Exception as exc:  # noqa: BLE001 - let the LLM see the error and recover
                result = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            step = {"tool": call["name"], "args": call["args"], "result": result}
            trace.append(step)
            if on_step:
                on_step(step)
            messages.append(ToolMessage(content=result, tool_call_id=call["id"],
                                        name=call["name"]))

    return {"answer": "The agent reached its step limit before finishing. Try again or "
                      "simplify the request.", "trace": trace}


def split_answer(answer: str) -> tuple[str, Optional[dict]]:
    """Separate the human-readable itinerary from the trailing JSON block."""
    import re
    match = re.search(r"```json\s*(\{.*?\})\s*```", answer, re.DOTALL)
    if not match:
        return answer, None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return answer, None
    return answer[:match.start()].rstrip(), data
