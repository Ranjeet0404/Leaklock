# packages/laas_mv_core/laas_mv_core_v4_1.py
"""
LaaS M&V Core — Version 4.1

What this does (pilot-ready, CFO-safe):
- IPMVP Option A style normalization (production + optional temperature).
- Negative months are visible (we NEVER hide losses).
- Capture factor (unload/idle) applied ONLY when theoretical savings > 0.
- TOU-aware ₹ calculation with conservative allocation and safe fallbacks.
- Optional demand (kW) and PF line savings (placeholders supported).
- Confidence grade "A"/"B" with explicit reasons.
- 32-char lineage hash of all inputs for audit trail.

Inputs/Outputs are plain dataclasses + dict to keep it lightweight for your pilot.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import json
import hashlib


# -------------------------- Helpers -------------------------- #

def _r(x: float, nd: int = 2) -> float:
    """Round helper."""
    return round(float(x), nd)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_div(n: float, d: float, default: float = 0.0) -> float:
    return n / d if d not in (0.0, 0) else default


# ---------------------- Data Contracts ----------------------- #

@dataclass(frozen=True)
class MVPeriod:
    site_id: str
    period_start_utc: str  # ISO string, e.g. "2025-10-01T00:00:00Z"
    period_end_utc: str    # ISO string


@dataclass(frozen=True)
class Tariff:
    # Option 1: Weighted average only (pilot default)
    weighted_avg_inr_per_kwh: Optional[float] = None
    # Option 2: Full TOU (fill in Phase 2)
    kwh_actual_by_slot: Optional[Dict[str, float]] = None
    rate_inr_per_kwh_by_slot: Optional[Dict[str, float]] = None
    # Policy for allocation when TOU is present (kept for future use)
    tou_allocation_policy: str = "actual_share"  # ["actual_share", "uniform_price", "peak_first"]


@dataclass(frozen=True)
class DemandLine:
    baseline_peak_kw: Optional[float] = None
    actual_peak_kw: Optional[float] = None
    demand_rate_inr_per_kw: Optional[float] = None


@dataclass(frozen=True)
class PFLine:
    # Optional PF (power factor) penalties/credits in ₹ (if present on bill)
    baseline_pf_rupees: Optional[float] = None
    actual_pf_rupees: Optional[float] = None


@dataclass(frozen=True)
class MVInputs:
    # Energy (kWh)
    baseline_kwh: float
    actual_kwh: float

    # Normalization
    baseline_units: Optional[float] = None    # production units in baseline period
    actual_units: Optional[float] = None      # production units in current period

    baseline_temp_c: Optional[float] = None
    actual_temp_c: Optional[float] = None
    beta_per_degC: float = 0.0                # default 0.0; set per-site later via regression
    temp_norm_not_required: bool = False      # True for controlled environments

    # Compressor behavior
    unload_pct: float = 0.25                  # fraction of power that is "uncaptured" when fixed-speed idles
    capture_override: Optional[float] = None  # [0..1] to force capture in special cases

    # Pricing / Tariff
    tariff: Tariff = Tariff()

    # Optional other bill lines
    demand: Optional[DemandLine] = None
    pfline: Optional[PFLine] = None

    # Data quality
    data_completeness_pct: float = 100.0      # uptime of meter/ingest, % (used for confidence)

    # Meta
    method_version: str = "v4.1"


# ---------------------- Core Calculations -------------------- #

def _production_multiplier(baseline_units: Optional[float],
                           actual_units: Optional[float]) -> Tuple[float, bool, str]:
    """
    Returns: (multiplier, clamped, reason_if_any)
    Multiplier is clamped to [0.5, 2.0] to avoid wild swings in thin data.
    """
    reason = ""
    if baseline_units is None or actual_units is None:
        return 1.0, False, "Production normalization disabled (missing data)."
    if actual_units <= 0:
        return 1.0, True, "Production actual units <= 0; multiplier set to 1.0."

    mult = baseline_units / actual_units
    mult_clamped = False
    mult_clamped_val = _clamp(mult, 0.5, 2.0)
    if mult_clamped_val != mult:
        mult_clamped = True
        reason = f"Production multiplier clamped from {_r(mult,3)} to {_r(mult_clamped_val,3)}."

    return mult_clamped_val, mult_clamped, reason or ""


def _temperature_multiplier(baseline_temp_c: Optional[float],
                            actual_temp_c: Optional[float],
                            beta_per_degC: float,
                            temp_norm_not_required: bool) -> Tuple[float, Optional[str]]:
    """
    Simple linear temp normalization:
    multiplier = 1 + beta * (baseline_temp - actual_temp)
    Returns multiplier and an optional reason note.
    """
    if temp_norm_not_required:
        return 1.0, "Temperature normalization not required (controlled environment)."

    if (baseline_temp_c is None or actual_temp_c is None or beta_per_degC == 0.0):
        return 1.0, "Temperature normalization skipped (β=0 or missing inputs)."

    mult = 1.0 + beta_per_degC * (baseline_temp_c - actual_temp_c)
    return mult, None


def _energy_rupees(billed_saved_kwh: float, t: Tariff, reasons: list) -> Tuple[float, Dict[str, float], str]:
    """
    Conservatively allocates kWh savings into TOU slots by ACTUAL usage share.
    If TOU data is incomplete, falls back to weighted-avg rate (or 9.50).
    Returns: (total_inr, by_slot_map, assumption_text)
    """
    slot_map: Dict[str, float] = {}
    # TOU path present?
    if t.kwh_actual_by_slot and t.rate_inr_per_kwh_by_slot:
        actual = t.kwh_actual_by_slot
        rates = t.rate_inr_per_kwh_by_slot
        total_actual = sum(max(0.0, float(v)) for v in actual.values())

        if total_actual <= 0.0:
            # fall back to weighted average
            rate = t.weighted_avg_inr_per_kwh if t.weighted_avg_inr_per_kwh else 9.50
            return _r(billed_saved_kwh * rate, 0), {}, f"TOU fallback: used weighted avg ₹{rate}/kWh (no actual slot totals)."

        # Allocate proportional to actual usage shares
        total_inr = 0.0
        missing_rate_slots = []
        for slot, kwh_act in actual.items():
            share = _safe_div(kwh_act, total_actual, 0.0)
            slot_kwh_saved = billed_saved_kwh * share

            # rate fallback if missing: use weighted average or 9.50
            rate = rates.get(slot)
            if rate is None or float(rate) <= 0.0:
                rate = t.weighted_avg_inr_per_kwh if t.weighted_avg_inr_per_kwh else 9.50
                missing_rate_slots.append(slot)

            slot_map[slot] = _r(slot_kwh_saved * float(rate), 0)
            total_inr += slot_map[slot]

        assumption = "TOU allocation by actual-usage share."
        if missing_rate_slots:
            reasons.append(f"Missing TOU rate for slots {missing_rate_slots}; used weighted average for those.")
            assumption += " Missing slot rates fell back to weighted average."
        return _r(total_inr, 0), slot_map, assumption

    # No TOU → weighted average
    rate = t.weighted_avg_inr_per_kwh if t.weighted_avg_inr_per_kwh else 9.50
    return _r(billed_saved_kwh * rate, 0), {}, f"No TOU provided; used weighted avg ₹{rate}/kWh."


def _demand_rupees(d: Optional[DemandLine], reasons: list) -> float:
    """
    Demand savings (kW line). If provided and valid, we compute:
    (baseline_peak_kw - actual_peak_kw) * demand_rate.
    Negative values are allowed (losses) and will be shown.
    """
    if not d:
        return 0.0
    if d.baseline_peak_kw is None or d.actual_peak_kw is None or d.demand_rate_inr_per_kw is None:
        reasons.append("Demand inputs incomplete; demand savings not computed.")
        return 0.0
    diff = float(d.baseline_peak_kw) - float(d.actual_peak_kw)
    return _r(diff * float(d.demand_rate_inr_per_kw), 0)


def _pf_rupees(p: Optional[PFLine], reasons: list) -> float:
    """
    PF line as plain ₹ difference if supplied.
    """
    if not p:
        return 0.0
    if p.baseline_pf_rupees is None or p.actual_pf_rupees is None:
        reasons.append("PF inputs incomplete; PF savings not computed.")
        return 0.0
    return _r(float(p.baseline_pf_rupees) - float(p.actual_pf_rupees), 0)


# --------------------------- Public -------------------------- #

def calculate_savings(period: MVPeriod, inp: MVInputs) -> Dict:
    """
    Main entry point.
    Returns a plain dict with kWh, ₹ components, confidence, reasons, and lineage hash (32 chars).
    """
    reasons: list[str] = []

    # Validate inputs lightly
    if not (0.0 <= inp.unload_pct <= 1.0):
        raise ValueError("unload_pct must be within [0, 1].")

    # --- Production normalization
    prod_mult, prod_clamped, prod_reason = _production_multiplier(inp.baseline_units, inp.actual_units)
    if prod_reason:
        reasons.append(prod_reason)

    # --- Temperature normalization
    temp_mult, temp_note = _temperature_multiplier(
        inp.baseline_temp_c, inp.actual_temp_c, inp.beta_per_degC, inp.temp_norm_not_required
    )
    if temp_note:
        reasons.append(temp_note)

    # --- Adjusted actual kWh after normalizations
    adj_actual_kwh = float(inp.actual_kwh) * prod_mult * temp_mult

    # --- Theoretical savings (can be negative)
    theoretical_saved_kwh = float(inp.baseline_kwh) - adj_actual_kwh

    # --- Capture (only if savings are positive)
    if theoretical_saved_kwh > 0:
        capture = inp.capture_override if inp.capture_override is not None else (1.0 - float(inp.unload_pct))
        capture = _clamp(capture, 0.0, 1.0)
    else:
        capture = 1.0  # we don't "capture" losses; we report them fully

    captured_saved_kwh = theoretical_saved_kwh * capture

    # --- Billing logic (we never bill on losses)
    performance_loss_kwh = 0.0
    if captured_saved_kwh < 0:
        performance_loss_kwh = -captured_saved_kwh  # positive number of lost kWh (for visibility)
    billed_saved_kwh = max(0.0, captured_saved_kwh)
    if billed_saved_kwh != captured_saved_kwh:
        reasons.append(f"Negative month: performance loss of {_r(performance_loss_kwh,2)} kWh recorded (no billing).")

    # --- Energy ₹ with TOU fallback
    energy_inr, energy_inr_by_slot, tou_assumption = _energy_rupees(billed_saved_kwh, inp.tariff, reasons)

    # --- Optional: demand and PF lines
    demand_inr = _demand_rupees(inp.demand, reasons)
    pf_inr = _pf_rupees(inp.pfline, reasons)

    # --- Totals
    total_saved_inr = _r(energy_inr + demand_inr + pf_inr, 0)
    performance_loss_inr = 0.0
    if performance_loss_kwh > 0:
        # Value the loss at the same effective energy rate used above.
        # If TOU present, approximate using average ₹/kWh from computed energy_inr.
        eff_rate = 0.0
        if billed_saved_kwh > 0:
            eff_rate = _safe_div(energy_inr, billed_saved_kwh, 0.0)
        if eff_rate <= 0.0:
            # fallback to weighted avg or 9.50
            eff_rate = inp.tariff.weighted_avg_inr_per_kwh if inp.tariff.weighted_avg_inr_per_kwh else 9.50
        performance_loss_inr = _r(-performance_loss_kwh * eff_rate, 0)  # negative ₹ for display

    # --- Confidence grading
    confidence = "A"
    if inp.data_completeness_pct < 98.0:
        confidence = "B"
        reasons.append(f"Data completeness {inp.data_completeness_pct:.1f}% (<98%).")

    # Temperature normalization: allow β=0 if site explicitly marks not required
    if inp.beta_per_degC == 0.0 and not inp.temp_norm_not_required:
        confidence = "B"
        reasons.append("β (temperature coefficient) is 0.0 (no regression yet).")

    # Production normalization missing or clamped → lower confidence
    if inp.baseline_units is None or inp.actual_units is None:
        confidence = "B"
        reasons.append("Production normalization data missing.")
    elif prod_clamped:
        confidence = "B"
        reasons.append("Production multiplier clamped (thin or abnormal production data).")

    # --- Lineage hash (store inputs as used)
    lineage_payload = {
        "period": asdict(period),
        "inputs": {
            **asdict(inp),
            # dataclasses nested objects are already JSON-able since we used only std types
        },
        "computed": {
            "prod_mult": prod_mult,
            "temp_mult": temp_mult,
            "capture": capture,
            "tou_assumption": tou_assumption
        }
    }
    lineage_str = json.dumps(lineage_payload, sort_keys=True)
    lineage_hash = hashlib.sha256(lineage_str.encode()).hexdigest()[:32]

    # --- Build response
    resp = {
        # kWh
        "baseline_kwh": _r(inp.baseline_kwh, 2),
        "actual_kwh": _r(inp.actual_kwh, 2),
        "adj_actual_kwh": _r(adj_actual_kwh, 2),
        "theoretical_saved_kwh": _r(theoretical_saved_kwh, 2),
        "captured_saved_kwh": _r(captured_saved_kwh, 2),
        "billed_saved_kwh": _r(billed_saved_kwh, 2),
        "performance_loss_kwh": _r(performance_loss_kwh, 2),

        # ₹ components
        "energy_savings_inr": int(energy_inr),
        "demand_savings_inr": int(demand_inr),
        "pf_savings_inr": int(pf_inr),
        "performance_loss_inr": int(performance_loss_inr),  # negative if loss month
        "total_saved_inr": int(total_saved_inr),

        # Tariff breakdown (if TOU)
        "energy_inr_by_slot": energy_inr_by_slot,
        "tou_assumption": tou_assumption,

        # Normalization/capture context
        "prod_multiplier": _r(prod_mult, 3),
        "temp_multiplier": _r(temp_mult, 3),
        "capture_factor": _r(capture, 3),

        # Meta
        "confidence": confidence,
        "confidence_reasons": reasons,
        "method_version": inp.method_version,
        "lineage_hash": lineage_hash,

        # Policy echoes (so PDF can print assumptions)
        "unload_pct": _r(inp.unload_pct, 3),
        "temp_norm_not_required": inp.temp_norm_not_required,
        "beta_per_degC": _r(inp.beta_per_degC, 5),
        "data_completeness_pct": _r(inp.data_completeness_pct, 1),
    }
    return resp


# ---------------------- Quick local test --------------------- #

if __name__ == "__main__":
    # Simple smoke run (no TOU; temp norm off)
    period = MVPeriod(site_id="S1",
                      period_start_utc="2025-10-01T00:00:00Z",
                      period_end_utc="2025-10-31T23:59:59Z")
    inp = MVInputs(
        baseline_kwh=2400,
        actual_kwh=1800,
        baseline_units=1000,
        actual_units=950,
        temp_norm_not_required=True,      # skip temp normalization for this smoke test
        unload_pct=0.25,
        tariff=Tariff(weighted_avg_inr_per_kwh=9.50),
        data_completeness_pct=99.5
    )
    out = calculate_savings(period, inp)
    print(json.dumps(out, indent=2))
