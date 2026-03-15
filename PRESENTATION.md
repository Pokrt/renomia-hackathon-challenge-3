# Challenge 3: Vehicle Pricing — Special Vehicles (>3.5t)

---

## Architecture

```
Client → FastAPI /solve → PostgreSQL Cache (hit?) → Return cached
                                    ↓ (miss)
                          Gemini 2.5 Flash + Google Search Grounding
                                    ↓
                          Parse structured JSON response
                                    ↓
                          Code-based Depreciation Guard (floor + ceiling)
                                    ↓
                          Cache & Return
```

## How It Works

1. **Gemini + Google Search Grounding** — LLM searches real marketplace listings (TruckScout24, Autoline, Mobile.de, Tipcars) live
2. **Depreciation Guard** — Code-based sanity check using category-specific new prices & depreciation curves (tractors, trailers, trucks, vans, converted tractors)
   - **Floor**: if Gemini < 70% of depreciation estimate → snap up to 85%
   - **Ceiling**: if Gemini > 120% of depreciation estimate → cap at 110%
3. **PostgreSQL caching** — identical queries return instantly

## Key Design Decisions

| Decision | Why |
|---|---|
| Gemini with Search Grounding (not Firecrawl) | Single API call searches multiple sites, handles Czech/German/EU markets |
| Category-specific depreciation curves | Tractors lose value fast (23% at 5yr), trailers hold (55% at 5yr) — insurance reality |
| Floor + Ceiling approach | LLM finds real listings but can hallucinate; code-based guard catches outliers |

## Results (60 training samples)

| Metric | Score |
|---|---|
| **Price Accuracy (50%)** | 0.803 |
| **Sources (25%)** | 1.000 |
| **Price Range (15%)** | 0.900 |
| **Currency (10%)** | 1.000 |
| **Weighted Total** | **0.886** |

- **21/60 perfect** (±10%), **43/60 good** (±20%), only **4/60 bad** (>±40%)
- Avg response time: **7.2s** per vehicle

## Tech Stack

`Python 3.12` · `FastAPI` · `Gemini 2.5 Flash` · `Google Search Grounding` · `PostgreSQL` · `Docker`
