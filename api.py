import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import data as d

app = FastAPI(title="Obesity Landscape API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- in-memory cache --------------------------------------------------------
_cache: dict = {}
_ready = False


def _refresh() -> None:
    global _ready
    try:
        print("[refresh] fetching stock prices...")
        prices = d.get_stock_prices()
        _cache["prices"] = prices

        print("[refresh] fetching balance sheet / dry powder...")
        _cache["drypowder"] = d.get_dry_powder()

        print("[refresh] fetching ClinicalTrials.gov pipeline...")
        pipeline = d.get_pipeline()
        _cache["pipeline"]  = pipeline
        _cache["catalysts"] = d.get_catalysts(pipeline)

        print("[refresh] computing ARIMA forecasts...")
        forecasts = {}
        for ticker, ticker_data in prices.items():
            fc = d.get_arima_forecast(ticker, ticker_data)
            if fc:
                forecasts[ticker] = fc
        _cache["forecasts"] = forecasts

        print("[refresh] fetching firepower data...")
        _cache["firepower"] = d.get_firepower()

        print("[refresh] fetching obesity biotech targets...")
        _cache["obesity_targets"] = d.get_obesity_targets()

        _cache["last_updated"] = time.time()
        _ready = True
        print("[refresh] complete.")
    except Exception as exc:
        print(f"[refresh] ERROR: {exc}")


@app.on_event("startup")
async def startup() -> None:
    threading.Thread(target=_refresh, daemon=True).start()


# ---- endpoints --------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":       "ready" if _ready else "loading",
        "tickers":      list(d.TICKERS.keys()),
        "last_updated": _cache.get("last_updated"),
    }


@app.get("/tickers")
def tickers():
    return d.TICKERS


@app.get("/stocks")
def stocks():
    return _cache.get("prices", {})


@app.get("/pipeline")
def pipeline():
    return _cache.get("pipeline", [])


@app.get("/catalysts")
def catalysts():
    return _cache.get("catalysts", {})


@app.get("/drypowder")
def drypowder():
    return _cache.get("drypowder", [])


@app.get("/forecast/{ticker}")
def forecast(ticker: str):
    forecasts = _cache.get("forecasts", {})
    return forecasts.get(ticker.upper(), {"error": "not ready yet", "ticker": ticker.upper()})


@app.get("/firepower")
def firepower():
    return _cache.get("firepower", {"companies": [], "last_updated": None})


@app.get("/obesity-targets")
def obesity_targets():
    return _cache.get("obesity_targets", [])
