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

    # Cache key includes additional_info for VIN-level uniqueness
    cache_key = f"v3:{make}:{model_name}:{year}:{mileage_km}:{body_type}:{additional_info}"
    cached = get_cached(cache_key)
    if cached:
        return cached

    # Parse useful details from additional_info
    weight_info = ""
    power_info = ""
    if additional_info:
        w = re.search(r'Hmotnost:\s*(\d+)\s*kg', additional_info)
        if w:
            weight_info = f"Gross weight: {w.group(1)} kg"
        p = re.search(r'Výkon:\s*(\d+)\s*kW', additional_info)
        if p:
            power_info = f"Engine power: {p.group(1)} kW"

    mileage_str = f"{mileage_km:,} km" if mileage_km else "unknown mileage"

    prompt = f"""Estimate the Czech market value of this used commercial vehicle in CZK.

VEHICLE:
- Make: {make}
- Model: {model_name}
- Year: {year}
- Mileage: {mileage_str}
- Body type: {body_type}
- Details: {additional_info}
{f"- {weight_info}" if weight_info else ""}
{f"- {power_info}" if power_info else ""}

STEP 1 - IDENTIFY VEHICLE TYPE:
Czech body types: "Tahač návěsů"=tractor unit, "Nákladní automobil"=truck/lorry, "Návěs"=semi-trailer, "Užitkový automobil"=van/LCV, "Traktor"=converted tractor, "Osobní automobil"=car.
Body modifiers: "vč. mrazírenské/chladírenské nástavby"=with reefer body (+30-50%), "vč. nástavby"=with superstructure (+20-40%), "nosič kontejnerů"=container carrier (high value), "přestavba z tahače na traktor"=converted tractor (50-70% of tractor value).
MAN model codes "L.2007.46.xxx": check kW and kg to identify. 140kW/10t=light, 316kW/26t=heavy.

STEP 2 - SEARCH for comparable vehicles on truck1.eu, autoline.info, mobile.de, tipcars.com, mascus.com, truckscout24.com. Find at least 5 real listings from year {year} ±2 years with URLs and prices.

STEP 3 - DETERMINE VALUE using the NEW PRICE as anchor:
First, find or estimate the NEW (catalog) price for this vehicle or its equivalent. Apply age-based depreciation.

NEW PRICE BENCHMARKS (approximate, for reference):
- Tractor units: MAN TGX 18.xxx new ≈ 100,000-130,000 EUR (2,500,000-3,250,000 CZK). IVECO S-Way/Stralis ≈ 90,000-120,000 EUR (2,250,000-3,000,000 CZK).
- Semi-trailers: Krone SD/SDP curtainside new ≈ 40,000-55,000 EUR (1,000,000-1,375,000 CZK). Krone Cool Liner reefer ≈ 50,000-65,000 EUR (1,250,000-1,625,000 CZK). These are PREMIUM trailers - do NOT confuse with basic flatbed or older models.
- Trucks 10-12t: ≈ 50,000-80,000 EUR (1,250,000-2,000,000 CZK).
- Trucks 18-26t: ≈ 80,000-120,000 EUR (2,000,000-3,000,000 CZK). With reefer body: +30,000-50,000 EUR.
- Vans/LCVs (IVECO Daily etc.): ≈ 45,000-75,000 EUR (1,125,000-1,875,000 CZK). With reefer: +15,000-25,000 EUR.
- Toyota Proace City Verso: ≈ 550,000-650,000 CZK new.
- Container carriers (nosič kontejnerů) like Iveco AD 260TW: very specialized, new ≈ 120,000-180,000 EUR (3,000,000-4,500,000 CZK).

DEPRECIATION BY CATEGORY (% of new price retained):
Tractor units (high-mileage, aggressive depreciation):
  1yr: 75-85%, 2yr: 60-70%, 3yr: 50-60%, 4yr: 40-50%, 5yr: 30-40%, 7yr: 18-25%, 10yr: 8-15%, 15yr: 3-8%

Semi-trailers (moderate depreciation, longer useful life):
  1yr: 85-95%, 2yr: 75-85%, 3yr: 65-75%, 4yr: 58-68%, 5yr: 50-60%, 7yr: 35-45%, 10yr: 20-30%

Trucks and vans (moderate depreciation):
  1yr: 80-90%, 2yr: 68-78%, 3yr: 58-68%, 4yr: 48-58%, 5yr: 38-48%, 7yr: 25-35%, 10yr: 15-25%, 15yr: 8-15%

Converted tractors: use tractor unit depreciation × 0.6-0.7

Your estimate = NEW PRICE × depreciation factor, cross-checked against found listings.
If listings suggest significantly different values, adjust toward the listing median.
Convert EUR to CZK at 1 EUR = {EUR_TO_CZK} CZK.

Return ONLY JSON:
{{
  "estimated_value_czk": <integer>,
  "listings": [
    {{"url": "https://...", "title": "...", "price": 50000, "currency": "EUR", "price_czk": 1250000}}
  ]
}}"""

    # Call Gemini with Google Search grounding (with retries)
    text = ""
    try:
        response = _call_gemini_with_retry(prompt)
        _track_usage(response)
        text = response.text
    except Exception:
        pass

    # Parse response
    data = _parse_json_response(text) if text else {}
    listings = data.get("listings", [])
    gemini_estimate = data.get("estimated_value_czk")

    # Process listings
    prices_czk = []
    sources = []
    for listing in listings:
        price_czk = _extract_price_czk(listing)
        if price_czk and price_czk > 0:
            prices_czk.append(price_czk)
            sources.append({
                "url": listing.get("url", ""),
                "title": listing.get("title", ""),
                "price": listing.get("price", 0),
                "currency": listing.get("currency", "EUR"),
                "price_czk": price_czk,
            })

    # Fallback if not enough sources
    if len(sources) < 3:
        sources, prices_czk = _fallback_search(
            make, model_name, year, mileage_km, body_type, additional_info,
            sources, prices_czk,
        )

    # Calculate estimate
    if gemini_estimate and gemini_estimate > 0:
        estimated_value = int(gemini_estimate)
    elif prices_czk:
        estimated_value = int(statistics.median(prices_czk))
    else:
        estimated_value = None

    # Apply depreciation-based floor AND ceiling to catch severe Gemini mis-estimates
    depr_estimate = _depreciation_estimate(make, model_name, body_type, year, additional_info)
    if depr_estimate and estimated_value:
        # Floor: if Gemini estimate is less than 70% of depreciation estimate, adjust upward
        if estimated_value < depr_estimate * 0.70:
            estimated_value = int(depr_estimate * 0.85)
        # Ceiling: if Gemini estimate is more than 120% of depreciation estimate, cap it
        elif estimated_value > depr_estimate * 1.20:
            estimated_value = int(depr_estimate * 1.10)
    elif depr_estimate and not estimated_value:
        estimated_value = int(depr_estimate)

    # Price range
    if prices_czk and estimated_value:
        price_min = int(min(min(prices_czk), estimated_value * 0.75))
        price_max = int(max(max(prices_czk), estimated_value * 1.25))
    elif estimated_value:
        price_min = int(estimated_value * 0.75)
        price_max = int(estimated_value * 1.25)
    else:
        price_min = None
        price_max = None

    # Methodology
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
        "methodology": f"Estimated from {len(prices_czk)} listings from {', '.join(site_names) or 'web search'}, adjusted for Czech market",
    }

    set_cache(cache_key, result)
    return result


def _depreciation_estimate(make, model_name, body_type, year, additional_info=""):
    """Calculate a depreciation-based price estimate as a sanity check floor."""
    current_year = 2025
    age = current_year - year if year else 5

    body_lower = body_type.lower()
    base_body = body_lower.split("(")[0].strip()

    # Determine new price (CZK) based on vehicle category
    new_price = None
    category = None

    if "tahač" in base_body:
        category = "tractor"
        new_price = 2_800_000  # ~112k EUR average tractor unit
    elif "návěs" in base_body:
        category = "trailer"
        model_lower = model_name.lower()
        if "cool" in model_lower or "mraz" in body_lower or "chlad" in body_lower:
            new_price = 1_450_000  # reefer trailer ~58k EUR
        else:
            new_price = 1_350_000  # standard curtainside ~54k EUR
    elif "nákladní" in base_body:
        category = "truck"
        # Check for superstructure modifier
        has_reefer = "mraz" in body_lower or "chlad" in body_lower
        has_body = "nástavb" in body_lower

        # Determine truck size from additional_info
        weight = 0
        w_match = re.search(r'Hmotnost:\s*(\d+)', additional_info or "")
        if w_match:
            weight = int(w_match.group(1))

        if weight > 20000:
            new_price = 2_500_000  # heavy truck
        elif weight > 7500:
            new_price = 2_000_000  # medium truck
        else:
            new_price = 1_500_000  # light truck

        if has_reefer:
            new_price += 1_000_000  # reefer body adds significant value
        elif has_body:
            new_price += 500_000  # generic superstructure
    elif "užitkový" in base_body:
        category = "van"
        has_reefer = "mraz" in body_lower or "chlad" in body_lower
        new_price = 1_500_000  # ~60k EUR average IVECO Daily type
        if has_reefer:
            new_price += 500_000
    elif "traktor" in base_body:
        category = "converted_tractor"
        if "nosič" in body_lower or "kontejner" in body_lower:
            # Container carrier - price depends heavily on chassis weight
            weight = 0
            w_match = re.search(r'Hmotnost:\s*(\d+)', additional_info or "")
            if w_match:
                weight = int(w_match.group(1))
            if weight > 25000:
                new_price = 5_500_000  # heavy hookloader/container carrier
            else:
                new_price = 3_500_000  # standard container carrier
        else:
            new_price = 1_800_000  # converted tractor (60% of tractor unit)
    elif "osobní" in base_body:
        category = "car"
        new_price = 600_000  # ~24k EUR

    if not new_price:
        return None

    # Apply category-specific depreciation
    if category == "tractor":
        depr_table = {0: 0.95, 1: 0.80, 2: 0.65, 3: 0.55, 4: 0.40, 5: 0.23,
                      6: 0.19, 7: 0.16, 8: 0.14, 9: 0.13, 10: 0.10, 12: 0.07, 15: 0.04}
    elif category == "trailer":
        depr_table = {0: 0.97, 1: 0.90, 2: 0.80, 3: 0.70, 4: 0.63, 5: 0.55,
                      6: 0.48, 7: 0.40, 8: 0.34, 9: 0.28, 10: 0.24, 12: 0.18, 15: 0.12}
    elif category == "converted_tractor":
        depr_table = {0: 0.90, 1: 0.75, 2: 0.60, 3: 0.50, 4: 0.42, 5: 0.35,
                      6: 0.30, 7: 0.27, 8: 0.24, 9: 0.22, 10: 0.18, 12: 0.13, 15: 0.08}
    else:  # truck, van, car
        depr_table = {0: 0.95, 1: 0.85, 2: 0.73, 3: 0.63, 4: 0.53, 5: 0.43,
                      6: 0.35, 7: 0.30, 8: 0.25, 9: 0.20, 10: 0.17, 12: 0.12, 15: 0.08}

    # Interpolate depreciation factor
    if age in depr_table:
        factor = depr_table[age]
    else:
        ages = sorted(depr_table.keys())
        if age < ages[0]:
            factor = depr_table[ages[0]]
        elif age > ages[-1]:
            factor = depr_table[ages[-1]]
        else:
            for i in range(len(ages) - 1):
                if ages[i] <= age <= ages[i + 1]:
                    t = (age - ages[i]) / (ages[i + 1] - ages[i])
                    factor = depr_table[ages[i]] * (1 - t) + depr_table[ages[i + 1]] * t
                    break

    return int(new_price * factor)


def _call_gemini_with_retry(prompt, max_retries=3):
    """Call Gemini with Google Search grounding, retrying on transient failures."""
    grounding_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
    config = genai_types.GenerateContentConfig(tools=[grounding_tool])
    last_error = None
    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model=GROUNDING_MODEL,
                contents=prompt,
                config=config,
            )
            return response
        except Exception as e:
            last_error = e
            wait = 2 ** attempt * 2  # 2s, 4s, 8s
            time.sleep(wait)
    raise last_error


def _track_usage(response):
    """Track Gemini token usage."""
    with gemini._lock:
        gemini.request_count += 1
        meta = getattr(response, "usage_metadata", None)
        if meta:
            gemini.prompt_tokens += getattr(meta, "prompt_token_count", 0)
            gemini.completion_tokens += getattr(meta, "candidates_token_count", 0)
            gemini.total_tokens += getattr(meta, "total_token_count", 0)


def _parse_json_response(text):
    """Extract JSON object from Gemini response text."""
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        json_str = json_match.group(0) if json_match else text
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        return {}


def _extract_price_czk(listing):
    """Extract price in CZK from a listing dict."""
    price = listing.get("price", 0)
    currency = listing.get("currency", "EUR")
    price_czk = listing.get("price_czk")

    if not price_czk or price_czk <= 0:
        if currency == "CZK":
            price_czk = int(price)
        else:
            price_czk = int(price * EUR_TO_CZK)

    return int(price_czk) if price_czk else 0


def _fallback_search(make, model_name, year, mileage_km, body_type, additional_info, sources, prices_czk):
    """Second attempt with a simpler prompt if the first didn't yield enough listings."""
    prompt = f"""Search the web for used "{make} {model_name}" for sale in Europe.
Year around {year}, body type: {body_type}. This is a {body_type} vehicle.

Return a JSON object with listings:
{{
  "estimated_value_czk": <estimated Czech market value>,
  "listings": [
    {{"url": "https://...", "title": "...", "price": 50000, "currency": "EUR", "price_czk": 1250000}}
  ]
}}

Use 1 EUR = {EUR_TO_CZK} CZK. Only real listings with actual URLs. At least 5 listings."""

    try:
        response = _call_gemini_with_retry(prompt)
        _track_usage(response)
        text = response.text

        data = _parse_json_response(text)
        if isinstance(data, dict):
            listings = data.get("listings", [])
        elif isinstance(data, list):
            listings = data
        else:
            listings = []

        for listing in listings:
            price_czk = _extract_price_czk(listing)
            if price_czk and price_czk > 0:
                prices_czk.append(price_czk)
                sources.append({
                    "url": listing.get("url", ""),
                    "title": listing.get("title", ""),
                    "price": listing.get("price", 0),
                    "currency": listing.get("currency", "EUR"),
                    "price_czk": price_czk,
                })
    except Exception:
        pass

    return sources, prices_czk


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
