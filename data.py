"""
Data fetching layer for the obesity competitive landscape API.
All functions are synchronous (run in background threads on startup).
"""
import time
import warnings
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ---------- ticker universe -----------------------------------------------

TICKERS: dict[str, dict] = {
    "NVO":   {"name": "Novo Nordisk",           "ct_sponsor": "Novo Nordisk"},
    "LLY":   {"name": "Eli Lilly",              "ct_sponsor": "Eli Lilly"},
    "AMGN":  {"name": "Amgen",                  "ct_sponsor": "Amgen"},
    "AZN":   {"name": "AstraZeneca",            "ct_sponsor": "AstraZeneca"},
    "RHHBY": {"name": "Roche",                  "ct_sponsor": "Hoffmann-La Roche"},
    "VKTX":  {"name": "Viking Therapeutics",    "ct_sponsor": "Viking Therapeutics"},
    "GPCR":  {"name": "Structure Therapeutics", "ct_sponsor": "Structure Therapeutics"},
    "ALT":   {"name": "Altimmune",              "ct_sponsor": "Altimmune"},
    "ZEAL":  {"name": "Zealand Pharma",         "ct_sponsor": "Zealand Pharma"},
}

CTA_BASE = "https://clinicaltrials.gov/api/v2/studies"

# ---------- stock prices + SMAs -------------------------------------------

def get_stock_prices() -> dict:
    """3-year daily OHLCV with pre-computed SMAs and index-normalised price."""
    result = {}
    for ticker, meta in TICKERS.items():
        try:
            hist: pd.DataFrame = yf.Ticker(ticker).history(period="3y")
            if hist.empty:
                continue
            hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
            hist["sma50"]  = hist["Close"].rolling(50).mean()
            hist["sma200"] = hist["Close"].rolling(200).mean()
            base = hist["Close"].iloc[0]
            hist["normalized"] = (hist["Close"] / base * 100).round(2)
            records = []
            for date, row in hist.iterrows():
                records.append({
                    "date":       date.strftime("%Y-%m-%d"),
                    "close":      round(float(row["Close"]), 2),
                    "volume":     int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                    "sma50":      round(float(row["sma50"]), 2)  if not pd.isna(row["sma50"])  else None,
                    "sma200":     round(float(row["sma200"]), 2) if not pd.isna(row["sma200"]) else None,
                    "normalized": float(row["normalized"]),
                })
            result[ticker] = {"meta": meta, "prices": records}
        except Exception as exc:
            print(f"[prices] {ticker}: {exc}")
    return result

# ---------- M&A dry-powder (balance sheet) --------------------------------

def get_dry_powder() -> list:
    """Cash, debt, market cap, FCF from yfinance.info — ranked by net cash."""
    result = []
    for ticker, meta in TICKERS.items():
        try:
            info = yf.Ticker(ticker).info
            total_cash = info.get("totalCash") or 0
            total_debt = info.get("totalDebt") or 0
            market_cap = info.get("marketCap") or 0
            fcf        = info.get("freeCashflow") or 0
            net_cash   = total_cash - total_debt
            result.append({
                "ticker":            ticker,
                "name":              meta["name"],
                "market_cap":        market_cap,
                "total_cash":        total_cash,
                "total_debt":        total_debt,
                "net_cash":          net_cash,
                "free_cashflow":     fcf,
                "net_cash_pct_mcap": round(net_cash / market_cap * 100, 1) if market_cap else 0,
            })
        except Exception as exc:
            print(f"[drypowder] {ticker}: {exc}")
    return sorted(result, key=lambda x: x["net_cash"], reverse=True)

# ---------- ClinicalTrials.gov pipeline -----------------------------------

_PHASE_ORDER = {
    "Phase 3": 0, "Phase 2/Phase 3": 1, "Phase 2": 2,
    "Phase 1/Phase 2": 3, "Phase 1": 4, "N/A": 5,
}

def _phase_str(phases: list) -> str:
    if not phases:
        return "N/A"
    # PHASE3 → Phase 3, PHASE1_PHASE2 → Phase 1/Phase 2
    raw = phases[0]
    return raw.replace("_", "/").replace("PHASE", "Phase ").replace("Phase ", "Phase ").strip()

def get_pipeline() -> list:
    """Fetch obesity-relevant trials from ClinicalTrials.gov for each sponsor."""
    all_studies = []
    for ticker, meta in TICKERS.items():
        try:
            params = {
                "query.cond":           "obesity",
                "query.spons":          meta["ct_sponsor"],
                "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,NOT_YET_RECRUITING,COMPLETED",
                "filter.studyType":     "INTERVENTIONAL",
                "pageSize":             50,
                "format":               "json",
            }
            r = requests.get(CTA_BASE, params=params, timeout=30)
            r.raise_for_status()
            studies = r.json().get("studies", [])
            for s in studies:
                proto   = s.get("protocolSection", {})
                id_mod  = proto.get("identificationModule", {})
                stat_mod = proto.get("statusModule", {})
                spon_mod = proto.get("sponsorCollaboratorsModule", {})
                des_mod  = proto.get("designModule", {})
                arms_mod = proto.get("armsInterventionsModule", {})

                drugs = [
                    i.get("name", "") for i in arms_mod.get("interventions", [])
                    if i.get("type", "") in {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"}
                ]

                all_studies.append({
                    "nct_id":            id_mod.get("nctId", ""),
                    "title":             id_mod.get("briefTitle", "")[:120],
                    "phase":             _phase_str(des_mod.get("phases", [])),
                    "status":            stat_mod.get("overallStatus", ""),
                    "sponsor":           spon_mod.get("leadSponsor", {}).get("name", meta["name"]),
                    "ticker":            ticker,
                    "company":           meta["name"],
                    "primary_completion": stat_mod.get("primaryCompletionDateStruct", {}).get("date", ""),
                    "start_date":        stat_mod.get("startDateStruct", {}).get("date", ""),
                    "drugs":             drugs[:3],
                })
            time.sleep(0.4)  # polite rate-limiting
        except Exception as exc:
            print(f"[pipeline] {ticker}: {exc}")

    all_studies.sort(key=lambda x: (x["ticker"], _PHASE_ORDER.get(x["phase"], 6)))
    return all_studies

# ---------- catalyst dates (for chart overlays) ---------------------------

def get_catalysts(pipeline: list) -> dict:
    """Extract upcoming primary completion dates as chart catalyst markers."""
    catalysts: dict[str, list] = {}
    for study in pipeline:
        ticker = study["ticker"]
        date   = study.get("primary_completion", "")
        if not date or study["status"] == "COMPLETED":
            continue
        drug_label = study["drugs"][0] if study["drugs"] else "study"
        catalysts.setdefault(ticker, []).append({
            "date":    date,
            "label":   f"{drug_label} · {study['phase']}",
            "nct_id":  study["nct_id"],
            "status":  study["status"],
        })
    for ticker in catalysts:
        catalysts[ticker].sort(key=lambda x: x["date"])
    return catalysts

# ---------- ARIMA 30-day forecast -----------------------------------------

def get_arima_forecast(ticker: str, price_data: dict) -> dict | None:
    """Fit ARIMA(5,1,0) on last 252 trading days, return 30-day forecast + 80% CI."""
    try:
        prices = [p["close"] for p in price_data["prices"]]
        if len(prices) < 120:
            return None
        series = pd.Series(prices[-252:])
        fit = ARIMA(series, order=(5, 1, 0)).fit()
        fc  = fit.get_forecast(steps=30)
        mean    = fc.predicted_mean
        ci      = fc.conf_int(alpha=0.2)  # 80% CI

        last_date  = pd.Timestamp(price_data["prices"][-1]["date"])
        fc_dates   = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)

        return {
            "ticker":   ticker,
            "forecast": [
                {
                    "date":  d.strftime("%Y-%m-%d"),
                    "value": round(float(v), 2),
                    "lower": round(float(lo), 2),
                    "upper": round(float(hi), 2),
                }
                for d, v, lo, hi in zip(fc_dates, mean, ci.iloc[:, 0], ci.iloc[:, 1])
            ],
        }
    except Exception as exc:
        print(f"[arima] {ticker}: {exc}")
        return None
