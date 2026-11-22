"""Core travel-planning logic: flights, hotels, places, weather and budget.

These are plain Python functions that return JSON-serialisable data.
`agent.py` wraps them as LangChain tools. Keeping them separate makes them
easy to test without an LLM or an API key.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data"

# Latitude / longitude for every city present in the datasets (used for weather).
CITY_COORDS = {
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "goa": (15.2993, 74.1240),
    "bangalore": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "hyderabad": (17.3850, 78.4867),
    "kolkata": (22.5726, 88.3639),
    "jaipur": (26.9124, 75.7873),
}
CITIES = [c.title() for c in CITY_COORDS]

# WMO weather codes -> readable text (Open-Meteo returns these).
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Dense drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
    81: "Heavy rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm",
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _load(name: str) -> list[dict]:
    """Load one of the JSON datasets from the data folder."""
    path = DATA_DIR / f"{name}.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found: {path}") from exc


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def available_place_types() -> list[str]:
    """All attraction types that exist in places.json."""
    return sorted({p["type"] for p in _load("places")})


# --------------------------------------------------------------------------
# Flights
# --------------------------------------------------------------------------
def _duration_hours(flight: dict) -> float:
    dep = datetime.fromisoformat(flight["departure_time"])
    arr = datetime.fromisoformat(flight["arrival_time"])
    return round((arr - dep).total_seconds() / 3600, 1)


def search_flights(origin: str, destination: str, travel_date: str = "",
                   sort_by: str = "cheapest", limit: int = 5) -> dict:
    """Find flights for a route.

    sort_by: "cheapest" or "fastest". If travel_date (YYYY-MM-DD) is given,
    flights on that exact date are ranked first, then the closest dates.
    """
    try:
        matches = [
            f for f in _load("flights")
            if _norm(f["from"]) == _norm(origin) and _norm(f["to"]) == _norm(destination)
        ]
        if not matches:
            routes = sorted({f"{f['from']} -> {f['to']}" for f in _load("flights")})
            return {"found": False,
                    "message": f"No flight from {origin} to {destination} in the dataset.",
                    "hint": "Try a connecting route via another city.",
                    "available_routes": routes}

        target = datetime.strptime(travel_date, "%Y-%m-%d").date() if travel_date else None
        results = []
        for f in matches:
            dep = datetime.fromisoformat(f["departure_time"])
            # Sample flights are all from 2025, so compare by time of year (ignore the year)
            # and treat Dec/Jan as neighbours.
            gap = 0
            if target:
                diff = abs(dep.timetuple().tm_yday - target.timetuple().tm_yday)
                gap = min(diff, 365 - diff)
            results.append({
                **f,
                "duration_hours": _duration_hours(f),
                "exact_date_match": bool(target and dep.date() == target),
                "days_from_requested_date_ignoring_year": gap if target else None,
            })

        key = (lambda r: r["duration_hours"]) if sort_by == "fastest" else (lambda r: r["price"])
        # With a date, closeness is primary; then price/duration.
        results.sort(key=lambda r: (r["days_from_requested_date_ignoring_year"] or 0, key(r)))
        return {"found": True, "count": len(results), "flights": results[:limit]}
    except ValueError:
        return {"found": False, "message": "travel_date must be in YYYY-MM-DD format."}
    except Exception as exc:  # noqa: BLE001 - tool must never crash the agent
        return {"found": False, "message": f"Flight search failed: {exc}"}


# --------------------------------------------------------------------------
# Hotels
# --------------------------------------------------------------------------
def recommend_hotels(city: str, min_stars: int = 1, max_price_per_night: float = 0,
                     amenities: list[str] | None = None, sort_by: str = "best_value",
                     limit: int = 5) -> dict:
    """Recommend hotels in a city.

    sort_by: "best_value" (stars per rupee), "cheapest" or "highest_rated".
    max_price_per_night = 0 means no price cap.
    """
    try:
        wanted = {_norm(a) for a in (amenities or [])}
        hotels = [
            h for h in _load("hotels")
            if _norm(h["city"]) == _norm(city)
            and h["stars"] >= min_stars
            and (not max_price_per_night or h["price_per_night"] <= max_price_per_night)
            and wanted.issubset({_norm(a) for a in h["amenities"]})
        ]
        if not hotels:
            return {"found": False, "message": f"No hotels in {city} match those filters."}

        if sort_by == "cheapest":
            hotels.sort(key=lambda h: h["price_per_night"])
        elif sort_by == "highest_rated":
            hotels.sort(key=lambda h: (-h["stars"], h["price_per_night"]))
        else:
            hotels.sort(key=lambda h: -h["stars"] / h["price_per_night"])
        return {"found": True, "count": len(hotels), "hotels": hotels[:limit]}
    except Exception as exc:  # noqa: BLE001
        return {"found": False, "message": f"Hotel search failed: {exc}"}


# --------------------------------------------------------------------------
# Places
# --------------------------------------------------------------------------
def discover_places(city: str, place_types: list[str] | None = None,
                    min_rating: float = 0, limit: int = 8) -> dict:
    """Recommend attractions in a city, best rated first."""
    try:
        types = {_norm(t) for t in (place_types or [])}
        places = [
            p for p in _load("places")
            if _norm(p["city"]) == _norm(city)
            and p["rating"] >= min_rating
            and (not types or _norm(p["type"]) in types)
        ]
        if not places:
            return {"found": False,
                    "message": f"No places in {city} match those filters.",
                    "available_types": available_place_types()}
        places.sort(key=lambda p: -p["rating"])
        return {"found": True, "count": len(places), "places": places[:limit]}
    except Exception as exc:  # noqa: BLE001
        return {"found": False, "message": f"Place search failed: {exc}"}


# --------------------------------------------------------------------------
# Weather (Open-Meteo, free, no API key)
# --------------------------------------------------------------------------
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum"


def get_weather(city: str, start_date: str, end_date: str) -> dict:
    """Daily weather for a city between two dates (YYYY-MM-DD).

    Open-Meteo forecasts cover ~16 days ahead. For dates further out we return
    what the weather was on the same dates one year earlier and say so.
    """
    try:
        coords = CITY_COORDS.get(_norm(city))
        if not coords:
            return {"found": False, "message": f"Unknown city '{city}'. Known: {CITIES}"}

        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if end < start:
            return {"found": False, "message": "end_date is before start_date."}

        today = date.today()
        mode = "forecast"
        url = FORECAST_URL
        if start < today - timedelta(days=1) or end > today + timedelta(days=15):
            # Outside the forecast window -> use historical data for the same dates.
            url, mode = ARCHIVE_URL, "historical"
            while end >= today - timedelta(days=7):
                start -= timedelta(days=365)
                end -= timedelta(days=365)

        resp = requests.get(url, params={
            "latitude": coords[0], "longitude": coords[1], "daily": DAILY_VARS,
            "timezone": "auto", "start_date": start.isoformat(), "end_date": end.isoformat(),
        }, timeout=15)
        resp.raise_for_status()
        daily = resp.json()["daily"]

        days = []
        for i, d in enumerate(daily["time"]):
            days.append({
                "date": d,
                "condition": WEATHER_CODES.get(daily["weather_code"][i], "Unknown"),
                "max_temp_c": daily["temperature_2m_max"][i],
                "min_temp_c": daily["temperature_2m_min"][i],
                "rain_mm": daily["precipitation_sum"][i],
            })
        note = ("Live forecast." if mode == "forecast" else
                "Trip dates are outside the 16-day forecast window, so this shows the actual "
                f"weather on the same dates a year earlier ({days[0]['date']} onward) as a guide.")
        return {"found": True, "city": city, "source": mode, "note": note, "days": days}
    except requests.RequestException as exc:
        return {"found": False, "message": f"Weather API error: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"found": False, "message": f"Weather lookup failed: {exc}"}


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------
def estimate_budget(flight_price: float, hotel_price_per_night: float, nights: int,
                    travelers: int = 1, daily_local_expense: float = 1500,
                    return_flight_price: float = 0) -> dict:
    """Total trip cost = flights + hotel rooms + food/local travel.

    Rooms are assumed to sleep two people. daily_local_expense is per person.
    """
    try:
        rooms = math.ceil(travelers / 2)
        flights_total = (flight_price + return_flight_price) * travelers
        hotel_total = hotel_price_per_night * nights * rooms
        local_total = daily_local_expense * (nights + 1) * travelers
        return {
            "travelers": travelers, "rooms": rooms, "nights": nights,
            "flight_total": round(flights_total),
            "hotel_total": round(hotel_total),
            "food_and_local_travel_total": round(local_total),
            "grand_total": round(flights_total + hotel_total + local_total),
            "currency": "INR",
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Budget estimation failed: {exc}"}
