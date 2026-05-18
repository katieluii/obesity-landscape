"""
Data fetching layer for the obesity competitive landscape API.
All functions are synchronous (run in background threads on startup).
"""
from __future__ import annotations
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

_FX_CACHE: dict[str, float] = {}

def _to_usd(value: float, currency: str) -> float:
    """Convert a financial value from `currency` to USD using a live spot rate."""
    if currency == "USD" or not value:
        return value
    if currency not in _FX_CACHE:
        try:
            hist = yf.Ticker(f"{currency}USD=X").history(period="5d")
            _FX_CACHE[currency] = float(hist["Close"].dropna().iloc[-1])
        except Exception as exc:
            print(f"[fx] {currency}: {exc}")
            _FX_CACHE[currency] = 1.0
    return value * _FX_CACHE[currency]


def get_dry_powder() -> list:
    """Cash, debt, market cap, FCF from yfinance — all values normalised to USD."""
    result = []
    for ticker, meta in TICKERS.items():
        try:
            t    = yf.Ticker(ticker)
            info = t.info

            fin_currency = info.get("financialCurrency") or "USD"

            # Filing date from annual balance sheet (most recent column)
            try:
                bs = t.balance_sheet
                filing_date = bs.columns[0].strftime("%b %Y") if not bs.empty else "—"
            except Exception:
                filing_date = "—"

            # Raw financial figures (may be in fin_currency, not USD)
            total_cash = info.get("totalCash") or 0
            total_debt = info.get("totalDebt") or 0
            fcf        = info.get("freeCashflow") or 0
            # marketCap is shares × USD price — already in USD for US-listed tickers
            market_cap = info.get("marketCap") or 0

            # Convert to USD if needed
            if fin_currency != "USD":
                total_cash = _to_usd(total_cash, fin_currency)
                total_debt = _to_usd(total_debt, fin_currency)
                fcf        = _to_usd(fcf, fin_currency)

            net_cash = total_cash - total_debt
            result.append({
                "ticker":            ticker,
                "name":              meta["name"],
                "filing_date":       filing_date,
                "filing_currency":   fin_currency,
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


# ---------- firepower analysis -----------------------------------------------

FIREPOWER_TICKERS: dict[str, str] = {
    "JNJ":   "Johnson & Johnson",
    "RHHBY": "Roche",
    "MRK":   "Merck",
    "NVO":   "Novo Nordisk",
    "NVS":   "Novartis",
    "ABBV":  "AbbVie",
    "LLY":   "Eli Lilly",
    "PFE":   "Pfizer",
    "AZN":   "AstraZeneca",
    "AMGN":  "Amgen",
}

# Companies where Normalized EBITDA is more representative than reported EBITDA
# JNJ: $5.9B unusual items (talc reversal + investment gains) inflate FY2025 reported figure
# ABBV: ~$7.4B/yr Allergan intangible amortization depresses reported EBITDA
# PFE: restructuring charges and Paxlovid wind-down distort reported figure
_USE_NORMALIZED_EBITDA = {"JNJ", "ABBV", "PFE"}

# Companies in active manufacturing buildout — use OCF not FCF-after-capex
# LLY: $50B+ US manufacturing plan; PFE: TrumpRx commitments; NVO: Catalent integration
_CAPEX_HEAVY = {"LLY", "NVO", "PFE"}

_EBITDA_NOTES: dict[str, str] = {
    "JNJ":   "† Reported EBITDA $41.1B includes $5.9B unusual items (talc litigation reversal + investment gains). Normalized EBITDA $35.2B used.",
    "ABBV":  "† Normalized EBITDA used; reported EBITDA $17.6B includes ~$7.4B/yr Allergan acquisition intangible amortization.",
    "PFE":   "† Normalized EBITDA used; reported EBITDA $16.8B distorted by restructuring charges and Paxlovid wind-down.",
    "RHHBY": "‡ FY2025 EBITDA +31% YoY in CHF (cost reduction despite flat revenue). Converted at period-end CHF/USD 1.2631. Lower than yfinance .info due to TTM methodology differences.",
}


def _get_fx_at_date(currency: str, date_str: str) -> tuple[float, str | None]:
    """Get currency→USD spot rate at a specific historical date (tz-aware fix)."""
    if currency == "USD":
        return 1.0, None
    pair  = f"{currency}USD=X"
    d     = pd.Timestamp(date_str)
    start = (d - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end   = (d + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    try:
        hist = yf.Ticker(pair).history(start=start, end=end)
        if hist.empty:
            return 1.0, f"FX rate for {currency} unavailable"
        hist.index = hist.index.tz_localize(None)
        filtered = hist[hist.index <= d]
        if filtered.empty:
            filtered = hist
        return float(filtered["Close"].iloc[-1]), None
    except Exception as e:
        return 1.0, f"FX {currency}: {e}"


def _safe(df: pd.DataFrame, *rows) -> tuple[float | None, str | None]:
    """Return (value, row_name) for first non-NaN row in df (most recent col)."""
    for r in rows:
        if r in df.index:
            v = df.loc[r].iloc[0]
            if pd.notna(v):
                return float(v), r
    return None, None


def _b(val: float | None, fx: float) -> float | None:
    if val is None:
        return None
    return round(val * fx / 1e9, 1)


def get_firepower() -> dict:
    """
    M&A firepower for 10 big pharma companies from audited annual statements.
    All monetary values in USD billions. Sorted by stretch_b descending.
    """
    from datetime import date as _date
    today = _date.today().isoformat()
    companies = []

    for ticker, name in FIREPOWER_TICKERS.items():
        try:
            t    = yf.Ticker(ticker)
            info = t.info
            inc  = t.income_stmt
            bs   = t.balance_sheet
            cf   = t.cashflow

            warnings: list[str] = []
            fin_cur = info.get("financialCurrency", "USD")
            period  = str(bs.columns[0].date()) if not bs.empty else None
            mkt_cap = info.get("marketCap") or 0

            fx, fx_err = _get_fx_at_date(fin_cur, period) if period else (1.0, "no period")
            if fx_err:
                warnings.append(fx_err)

            days_old = (pd.Timestamp(today) - pd.Timestamp(period)).days if period else None
            if days_old and days_old > 180:
                warnings.append(f"Statement {days_old}d old — data may be stale")

            # EBITDA
            ebitda_rep,  _ = _safe(inc, "EBITDA")
            ebitda_norm, _ = _safe(inc, "Normalized EBITDA")
            if ticker in _USE_NORMALIZED_EBITDA and ebitda_norm:
                ebitda = ebitda_norm
                ebitda_src = "income_statement_normalized"
            elif ebitda_rep:
                ebitda = ebitda_rep
                ebitda_src = "income_statement"
            else:
                oi, _ = _safe(inc, "Operating Income", "EBIT")
                da, _ = _safe(cf,  "Depreciation And Amortization",
                                   "Depreciation Amortization Depletion")
                if oi and da:
                    ebitda = oi + abs(da)
                    ebitda_src = "computed_oi_da"
                else:
                    ebitda = info.get("ebitda")
                    ebitda_src = "info_fallback"
                    warnings.append("EBITDA from .info fallback")

            # Net Debt
            nd_val, nd_src = _safe(bs, "Net Debt")
            cash_raw, _    = _safe(bs, "Cash Cash Equivalents And Short Term Investments",
                                       "Cash And Cash Equivalents")
            debt_raw, _    = _safe(bs, "Total Debt")
            if nd_val is None:
                if debt_raw and cash_raw:
                    nd_val  = debt_raw - cash_raw
                    nd_src  = "computed"
                else:
                    nd_val  = (info.get("totalDebt") or 0) - (info.get("totalCash") or 0)
                    nd_src  = "info_fallback"
                    warnings.append("Net debt from .info fallback")

            # FCF
            fcf_rep, _ = _safe(cf, "Free Cash Flow")
            ocf,     _ = _safe(cf, "Operating Cash Flow",
                                   "Cash Flow From Continuing Operating Activities")
            capex,   _ = _safe(cf, "Capital Expenditure", "Purchase Of PPE")
            fcf_norm   = ocf if ticker in _CAPEX_HEAVY else fcf_rep
            fcf_src    = "operating_cashflow" if ticker in _CAPEX_HEAVY else "fcf_line"

            # USD billions
            ebitda_b   = _b(ebitda, fx)
            nd_b       = _b(nd_val, fx)
            cash_b     = _b(cash_raw, fx)
            debt_b     = _b(debt_raw, fx)
            fcf_rep_b  = _b(fcf_rep, fx)
            fcf_norm_b = _b(fcf_norm, fx)
            ocf_b      = _b(ocf, fx)
            capex_b    = _b(capex, fx)
            mkt_cap_b  = round(mkt_cap / 1e9, 1)

            nd_ebitda  = round(nd_b / ebitda_b, 2)         if (ebitda_b and ebitda_b > 0 and nd_b is not None) else None
            fcf_pct    = round(fcf_norm_b / ebitda_b * 100) if (fcf_norm_b and ebitda_b and ebitda_b > 0) else None
            c_fp = round(max(0.0, 3.0 * ebitda_b - nd_b), 1) if (ebitda_b and nd_b is not None) else None
            s_fp = round(max(0.0, 5.0 * ebitda_b - nd_b), 1) if (ebitda_b and nd_b is not None) else None

            companies.append({
                "ticker":               ticker,
                "name":                 name,
                "market_cap_b":         mkt_cap_b,
                "total_cash_b":         cash_b,
                "total_debt_b":         debt_b,
                "ebitda_b":             ebitda_b,
                "net_debt_b":           nd_b,
                "nd_ebitda":            nd_ebitda,
                "fcf_reported_b":       fcf_rep_b,
                "fcf_normalized_b":     fcf_norm_b,
                "ocf_b":                ocf_b,
                "capex_b":              capex_b,
                "fcf_ebitda_pct":       fcf_pct,
                "comfortable_b":        c_fp,
                "stretch_b":            s_fp,
                "statement_period_end": period,
                "filing_currency":      fin_cur,
                "fx_rate":              round(fx, 4),
                "data_quality": {
                    "ebitda_source":   ebitda_src,
                    "net_debt_source": nd_src,
                    "fcf_source":      fcf_src,
                    "warnings":        warnings,
                    "ebitda_note":     _EBITDA_NOTES.get(ticker),
                },
            })
        except Exception as exc:
            print(f"[firepower] {ticker}: {exc}")

    companies.sort(key=lambda x: (x.get("stretch_b") or 0), reverse=True)
    return {
        "companies": companies,
        "last_updated": time.time(),
        "methodology_note": (
            "Firepower calculated from audited FY2025 financial statements. "
            "EBITDA, net debt, and free cash flow taken from company-reported line items "
            "rather than TTM aggregations. For companies in active manufacturing expansion "
            "(LLY, NVO, PFE), FCF reflects operating cash flow excluding one-time strategic capex. "
            "Comfortable firepower = capacity to lever to 3× Net Debt/EBITDA; Stretch = 5×. "
            "All data from FY2025 annual filings (period end 2025-12-31). "
            "M&A completed or announced in 2026 is not yet reflected in balance sheet figures."
        ),
    }


# ---------- obesity biotech targets ------------------------------------------

OBESITY_BIOTECH_META: dict[str, dict] = {
    "VKTX": {
        "name":                   "Viking Therapeutics",
        "lead_asset":             "VK2735",
        "mechanism":              "Dual GLP-1/GIP receptor agonist",
        "ct_query":               "VK2735",
        "peak_sales_est_b":       8.0,
        "is_obesity_indication":  True,
        "is_differentiated":      True,
        "has_poc_data":           True,
        # No mgmt runway guidance found — compute from yfinance with 1.5× forward multiplier
        "runway_months_mgmt":     None,
        "forward_burn_multiplier": 1.5,
    },
    "GPCR": {
        "name":                   "Structure Therapeutics",
        "lead_asset":             "aleniglipron",          # was GSBR-209 (incorrect)
        "mechanism":              "Oral small-molecule GLP-1 receptor agonist (biased agonist)",
        "ct_query":               "aleniglipron",
        "peak_sales_est_b":       5.0,
        "is_obesity_indication":  True,
        "is_differentiated":      True,
        "has_poc_data":           True,  # ACCESS Ph2: 16.3% placebo-adj. weight loss
        # Q1 2026 earnings: "funded through end of 2028" — ~31 months from 2026-03-31
        "runway_months_mgmt":     31,
        "report_date":            "2026-03-31",
    },
    "ALT": {
        "name":                   "Altimmune",
        "lead_asset":             "pemvidutide",
        "mechanism":              "GLP-1/glucagon dual agonist",
        "ct_query":               "pemvidutide",
        "peak_sales_est_b":       2.5,
        "is_obesity_indication":  True,
        "is_differentiated":      True,
        "has_poc_data":           True,
        # Q1 2026: CEO — "through Phase 3 MASH 52-wk readout expected 2029" (pro forma $535M cash)
        "runway_months_mgmt":     36,
        "report_date":            "2026-03-31",
        "cash_override_m":        535.0,  # pro forma post-April 2026 $225M offering
    },
    "CRBP": {
        "name":                   "Corbus Pharmaceuticals",
        "lead_asset":             "CRB-913",
        "mechanism":              "Peripheral CB1 inverse agonist",
        "ct_query":               "CRB-913",
        "peak_sales_est_b":       1.5,
        "is_obesity_indication":  True,
        "is_differentiated":      True,
        "has_poc_data":           False,  # Phase 1b — no obesity PoC data yet
        # Q1 2026: "expected to fund operations into 2028" — ~22 months from 2026-03-31
        "runway_months_mgmt":     22,
        "report_date":            "2026-03-31",
        "cash_override_m":        138.2,
    },
}


def _get_asset_trial_info(asset_query: str) -> dict:
    try:
        params = {
            "query.intr":           asset_query,
            "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,NOT_YET_RECRUITING",
            "pageSize":             10,
            "format":               "json",
        }
        r = requests.get(CTA_BASE, params=params, timeout=20)
        r.raise_for_status()
        studies = r.json().get("studies", [])
        if not studies:
            return {"phase": "N/A", "next_catalyst": None}
        phases, catalysts = [], []
        for s in studies:
            proto   = s.get("protocolSection", {})
            des_mod = proto.get("designModule", {})
            stat_mod = proto.get("statusModule", {})
            phase   = _phase_str(des_mod.get("phases", []))
            date    = stat_mod.get("primaryCompletionDateStruct", {}).get("date", "")
            if phase != "N/A":
                phases.append(phase)
            if date:
                catalysts.append(date)
        best = min(phases, key=lambda p: _PHASE_ORDER.get(p, 6)) if phases else "N/A"
        return {"phase": best, "next_catalyst": sorted(catalysts)[0] if catalysts else None}
    except Exception as e:
        print(f"[targets/{asset_query}]: {e}")
        return {"phase": "N/A", "next_catalyst": None}


def get_obesity_targets() -> list:
    """
    Biotech target screening: financials + ClinicalTrials.gov phase/catalyst.
    Returns two scoring dimensions per target:
      - deal_pressure: forced-seller dynamics (runway, stage, valuation)
      - strategic_fit:  asset attractiveness to big pharma (indication, mechanism, de-risking)
    """
    result = []
    for ticker, meta in OBESITY_BIOTECH_META.items():
        try:
            t    = yf.Ticker(ticker)
            info = t.info
            cf   = t.cashflow
            bs   = t.balance_sheet

            mkt_cap_b = round((info.get("marketCap") or 0) / 1e9, 2)

            # Cash: use management override (post-offering) if available, else balance sheet
            cash_override = meta.get("cash_override_m")
            if cash_override:
                cash_m = float(cash_override)
            else:
                cash_raw, _ = _safe(bs, "Cash Cash Equivalents And Short Term Investments",
                                        "Cash And Cash Equivalents")
                cash_m = round((cash_raw or 0) / 1e6, 0)

            # Runway: use management guidance if available, else compute from FCF
            mgmt_runway = meta.get("runway_months_mgmt")
            if mgmt_runway is not None:
                runway_months    = int(mgmt_runway)
                multiplier       = meta.get("forward_burn_multiplier", 1.0)
                # Implied monthly burn from cash / runway
                monthly_burn_m   = round(cash_m / runway_months, 1) if runway_months else 0
                quarterly_burn_m = round(monthly_burn_m * 3, 0)
            else:
                fcf_raw, _ = _safe(cf, "Free Cash Flow")
                if fcf_raw is None:
                    fcf_raw = info.get("freeCashflow") or 0
                multiplier       = meta.get("forward_burn_multiplier", 1.5)
                ttm_annual_burn  = abs(fcf_raw) * multiplier if fcf_raw else 0
                quarterly_burn_m = round(ttm_annual_burn / 4 / 1e6, 0)
                monthly_burn_m   = round(quarterly_burn_m / 3, 1)
                runway_months    = round(cash_m / monthly_burn_m) if monthly_burn_m > 0 else None

            trial = _get_asset_trial_info(meta["ct_query"])
            time.sleep(0.3)

            # ── Deal pressure scoring ────────────────────────────────────────
            sig_runway   = runway_months is not None and runway_months < 24
            sig_stage    = trial["phase"] in ("Phase 2", "Phase 2/Phase 3", "Phase 3")
            sig_cheap    = mkt_cap_b < 2 * meta["peak_sales_est_b"]
            dp_score     = sum([sig_runway, sig_stage, sig_cheap])
            dp_badge     = "High" if dp_score == 3 else ("Med" if dp_score == 2 else "Low")

            # ── Strategic fit scoring ────────────────────────────────────────
            sig_obesity   = meta.get("is_obesity_indication", False)
            sig_diff      = meta.get("is_differentiated", False)
            sig_derisk    = (
                trial["phase"] in ("Phase 3", "Phase 2/Phase 3") or
                meta.get("has_poc_data", False)
            )
            sf_score      = sum([sig_obesity, sig_diff, sig_derisk])
            sf_badge      = "High" if sf_score == 3 else ("Med" if sf_score == 2 else "Low")

            result.append({
                "ticker":            ticker,
                "name":              meta["name"],
                "market_cap_b":      mkt_cap_b,
                "cash_m":            cash_m,
                "quarterly_burn_m":  quarterly_burn_m,
                "monthly_burn_m":    monthly_burn_m,
                "runway_months":     runway_months,
                "runway_source":     "management_guidance" if mgmt_runway else "computed",
                "runway_report_date": meta.get("report_date"),
                "lead_asset":        meta["lead_asset"],
                "mechanism":         meta["mechanism"],
                "peak_sales_est_b":  meta["peak_sales_est_b"],
                "phase":             trial["phase"],
                "next_catalyst":     trial["next_catalyst"],
                # Deal pressure
                "deal_pressure_score":   dp_score,
                "deal_pressure_badge":   dp_badge,
                "deal_pressure_signals": {
                    "short_runway":  sig_runway,
                    "late_stage":    sig_stage,
                    "cheap_vs_peak": sig_cheap,
                },
                # Strategic fit
                "strategic_fit_score":   sf_score,
                "strategic_fit_badge":   sf_badge,
                "strategic_fit_signals": {
                    "obesity_indication": sig_obesity,
                    "differentiated":     sig_diff,
                    "derisked":           sig_derisk,
                },
                # Legacy fields kept for backward compatibility
                "acq_score":  dp_score,
                "acq_badge":  dp_badge,
                "acq_signals": {
                    "short_runway":  sig_runway,
                    "late_stage":    sig_stage,
                    "cheap_vs_peak": sig_cheap,
                },
            })
        except Exception as exc:
            print(f"[targets] {ticker}: {exc}")

    return sorted(result, key=lambda x: x["strategic_fit_score"] * 10 + x["deal_pressure_score"], reverse=True)
