"""
Challenge 3: Nacenění aut - speciální vozidla (Vehicle Pricing >3.5t)

Input:  Vehicle identification details
Output: Estimated market value with sources

Priority indicators (most to least important):
  1. Tovární značka + Model (Make + Model)
  2. Typ nástavby (Body type / superstructure)
  3. Rok výroby (Year of manufacture)
  4. Najeté km (Mileage)
"""

import json
import os
import re
import statistics
import threading
import time

import google.generativeai as genai
from google import genai as genai_new
from google.genai import types as genai_types
import psycopg2
from psycopg2.extras import Json
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Challenge 3: Vehicle Pricing")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://hackathon:hackathon@localhost:5432/hackathon"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


class GeminiTracker:
    """Wrapper around Gemini that tracks token usage."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.enabled = bool(api_key)
        if self.enabled:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        self._lock = threading.Lock()

    def generate(self, prompt, **kwargs):
        if not self.enabled:
            raise RuntimeError("Gemini API key not configured")
        response = self.model.generate_content(prompt, **kwargs)
        with self._lock:
            self.request_count += 1
            meta = getattr(response, "usage_metadata", None)
            if meta:
                self.prompt_tokens += getattr(meta, "prompt_token_count", 0)
                self.completion_tokens += getattr(meta, "candidates_token_count", 0)
                self.total_tokens += getattr(meta, "total_token_count", 0)
        return response

    def get_metrics(self):
        with self._lock:
            return {
                "gemini_request_count": self.request_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }

    def reset(self):
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.request_count = 0


gemini = GeminiTracker(GEMINI_API_KEY)

# New SDK client for grounding support
genai_client = genai_new.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GROUNDING_MODEL = "gemini-2.5-flash"

EUR_TO_CZK = 25.0


def get_db():
    return psycopg2.connect(DATABASE_URL)


def get_cached(key: str):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT value FROM cache WHERE key = %s", (key,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return row[0]
    except Exception:
        pass
    return None


def set_cache(key: str, value: dict):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO cache (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, created_at = NOW()",
            (key, Json(value)),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


@app.on_event("startup")
def init_db():
    for _ in range(15):
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )"""
            )
            conn.commit()
            cur.close()
            conn.close()
            return
        except Exception:
            time.sleep(1)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return gemini.get_metrics()


@app.post("/metrics/reset")
def reset_metrics():
    gemini.reset()
    return {"status": "reset"}


@app.post("/solve")
def solve(payload: dict):
    make = payload.get("make", "")
    model_name = payload.get("model", "")
    year = payload.get("year")
    mileage_km = payload.get("mileage_km")
    body_type = payload.get("body_type", "")
    additional_info = payload.get("additional_info", "")

    # Check cache
    cache_key = f"vehicle:{make}:{model_name}:{year}:{mileage_km}:{body_type}"
    cached = get_cached(cache_key)
    if cached:
        return cached

    # Build prompt for Gemini with Google Search grounding
    prompt = f"""You are a heavy vehicle pricing expert. Search the web for real, current market listings for this vehicle:

Make: {make}
Model: {model_name}
Year: {year}
Mileage: {mileage_km} km
Body type: {body_type}
Additional info: {additional_info}

Search on truck marketplaces: truck1.eu, autoline.info, mobile.de, tipcars.com, mascus.com, and other truck/machinery sales sites.

Find at least 5 similar vehicles currently listed for sale. Matching priority:
1. Make and model (most important - must match)
2. Body type (should match or be very similar)
3. Year of manufacture (within ±3 years)
4. Similar mileage range

For each listing, provide the actual listing URL, title, price in original currency, and price converted to CZK.
Use conversion rate: 1 EUR = {EUR_TO_CZK} CZK.

IMPORTANT: Return ONLY a JSON object in this exact format, no other text:
{{
  "listings": [
    {{
      "url": "https://actual-listing-url...",
      "title": "Vehicle listing title",
      "price": 50000,
      "currency": "EUR",
      "price_czk": 1250000
    }}
  ]
}}

Only include real listings with actual URLs. Return at least 5 listings."""

    # Call Gemini with Google Search grounding (new SDK)
    grounding_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
    config = genai_types.GenerateContentConfig(tools=[grounding_tool])
    response = genai_client.models.generate_content(
        model=GROUNDING_MODEL,
        contents=prompt,
        config=config,
    )
    # Track usage
    with gemini._lock:
        gemini.request_count += 1
        meta = getattr(response, "usage_metadata", None)
        if meta:
            gemini.prompt_tokens += getattr(meta, "prompt_token_count", 0)
            gemini.completion_tokens += getattr(meta, "candidates_token_count", 0)
            gemini.total_tokens += getattr(meta, "total_token_count", 0)
    text = response.text

    # Extract JSON from response (may be wrapped in markdown code blocks)
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        json_str = json_match.group(0) if json_match else text

    try:
        data = json.loads(json_str)
        listings = data.get("listings", [])
    except (json.JSONDecodeError, AttributeError):
        listings = []

    # Also try to extract URLs from grounding metadata as fallback
    grounding_urls = {}
    try:
        metadata = response.candidates[0].grounding_metadata
        if metadata and metadata.grounding_chunks:
            for chunk in metadata.grounding_chunks:
                if hasattr(chunk, "web") and chunk.web:
                    grounding_urls[chunk.web.title] = chunk.web.uri
    except Exception:
        pass

    # Process listings into sources
    prices_czk = []
    sources = []
    for listing in listings:
        price = listing.get("price", 0)
        currency = listing.get("currency", "EUR")
        price_czk = listing.get("price_czk")

        if not price_czk or price_czk <= 0:
            if currency == "CZK":
                price_czk = int(price)
            else:
                price_czk = int(price * EUR_TO_CZK)

        price_czk = int(price_czk)

        if price_czk > 0:
            prices_czk.append(price_czk)
            sources.append({
                "url": listing.get("url", ""),
                "title": listing.get("title", ""),
                "price": price,
                "currency": currency,
                "price_czk": price_czk,
            })

    # If not enough listings from JSON, try a simpler follow-up call
    if len(sources) < 3:
        sources, prices_czk = _fallback_search(
            make, model_name, year, mileage_km, body_type, additional_info,
            sources, prices_czk,
        )

    # Calculate estimate
    if prices_czk:
        estimated_value = int(statistics.median(prices_czk))
        price_min = min(prices_czk)
        price_max = max(prices_czk)
    else:
        estimated_value = None
        price_min = None
        price_max = None

    # Determine source sites for methodology
    site_names = set()
    for s in sources:
        try:
            host = s["url"].split("/")[2]
            site_names.add(host.replace("www.", ""))
        except (IndexError, AttributeError):
            pass

    result = {
        "estimated_value_czk": estimated_value,
        "currency": "CZK",
        "price_range_czk": {
            "min": price_min,
            "max": price_max,
        },
        "sources": sources[:10],
        "methodology": f"Median of {len(prices_czk)} listings from {', '.join(site_names) or 'web search'}",
    }

    set_cache(cache_key, result)
    return result


def _fallback_search(make, model_name, year, mileage_km, body_type, additional_info, sources, prices_czk):
    """Second attempt with a simpler prompt if the first didn't yield enough listings."""
    prompt = f"""Search the web for "{make} {model_name}" trucks for sale.
Year around {year}, body type: {body_type}.

Return a JSON array of at least 5 real listings:
[
  {{"url": "https://...", "title": "...", "price": 50000, "currency": "EUR", "price_czk": 1250000}}
]

Use 1 EUR = {EUR_TO_CZK} CZK. Only real listings with actual URLs."""

    try:
        grounding_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
        config = genai_types.GenerateContentConfig(tools=[grounding_tool])
        response = genai_client.models.generate_content(
            model=GROUNDING_MODEL,
            contents=prompt,
            config=config,
        )
        with gemini._lock:
            gemini.request_count += 1
        text = response.text

        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                json_str = json_match.group(0) if json_match else text

        data = json.loads(json_str)
        if isinstance(data, dict):
            listings = data.get("listings", [])
        else:
            listings = data

        for listing in listings:
            price = listing.get("price", 0)
            currency = listing.get("currency", "EUR")
            price_czk = listing.get("price_czk")

            if not price_czk or price_czk <= 0:
                if currency == "CZK":
                    price_czk = int(price)
                else:
                    price_czk = int(price * EUR_TO_CZK)

            price_czk = int(price_czk)

            if price_czk > 0:
                prices_czk.append(price_czk)
                sources.append({
                    "url": listing.get("url", ""),
                    "title": listing.get("title", ""),
                    "price": price,
                    "currency": currency,
                    "price_czk": price_czk,
                })
    except Exception:
        pass

    return sources, prices_czk


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
