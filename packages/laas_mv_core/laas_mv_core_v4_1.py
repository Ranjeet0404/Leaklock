# laas_mv_core_v4_1.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Mapping, Tuple
import json, hashlib

@dataclass(frozen=True)
class MVPeriod:
    site_id: str
    period_start_utc: str
    period_end_utc: str

@dataclass(frozen=True)
class TariffTOU:
    kwh_baseline_by_slot: Optional[Mapping[str, float]] = None
    kwh_actual_by_slot: Optional[Mapping[str, float]] = None
    rate_inr_per_kwh_by_slot: Optional[Mapping[str, float]] = None
    weighted_avg_inr_per_kwh: Optional[float] = None

@dataclass(frozen=True)
class DemandLine:
    baseline_peak_kw: Optional[float] = None
    actual_peak_kw: Optional[float] = None
    demand_rate_inr_per_kw: Optional[float] = None

@dataclass(frozen=True)
class PFLine:
    baseline_pf_rupees: Optional[float] = None
    actual_pf_rupees: Optional[float] = None

@dataclass(frozen=True)
class MVInputs:
    # Energy
    baseline_kwh: float
    actual_kwh: float
    # Normalization
    baseline_units: Optional[float] = None
    actual_units: Optional[float] = None
    baseline_temp_c: Optional[float] = None
    actual_temp_c: Optional[float] = None
    beta_per_degC: float = 0.0
    temp_norm_not_required: bool = False

    # Compressor behavior
    unload_pct: float = 0.25                 # must be 0..1
    capture_override: Optional[float] = None # e.g., 0.95 after stop/start or VSD

    # Pricing
    tariff: TariffTOU = TariffTOU()
    demand: DemandLine = DemandLine()
    pfline: PFLine = PFLine()
    tou_allocation_policy: str = "actual_share"  # future: "uniform_price" | "peak_first"

    # Quality / governance
    data_completeness_pct: float = 100.0
    allow_prod_multiplier_min: float = 0.5
    allow_prod_multiplier_max: float = 2.0
    prod_clamp_reason: Optional[str] = None

    # Meta
    method_version: str = "IPMVP-A-v4.1"

def calculate_savings(period: MVPeriod, inp: MVInputs) -> Dict:
    # Validate core inputs once (removes need for downstream clamps)
    if not (0.0 <= inp.unload_pct <= 1.0):
        raise ValueError("unload_pct must be between 0 and 1.")
    if inp.baseline_kwh < 0 or inp.actual_kwh < 0:
        raise ValueError("kWh cannot be negative.")
    if inp.baseline_units is not None and inp.baseline_units <= 0:
        raise ValueError("baseline_units must be > 0 when provided.")

    reasons = []

    # Production normalization (linear) with clamp
    prod_multiplier_raw = 1.0
    if inp.baseline_units is not None and inp.actual_units is not None and inp.actual_units > 0:
        prod_multiplier_raw = inp.baseline_units / inp.actual_units
    prod_multiplier = _clamp(prod_multiplier_raw, inp.allow_prod_multiplier_min, inp.allow_prod_multiplier_max)
    prod_clamped = (prod_multiplier != prod_multiplier_raw)
    if prod_clamped:
        reasons.append(
            f"Production multiplier clamped ({prod_multiplier_raw:.3f}→{prod_multiplier:.3f})."
            + (f" Reason: {inp.prod_clamp_reason}" if inp.prod_clamp_reason else "")
        )
    if inp.baseline_units is None or inp.actual_units is None:
        reasons.append("Production normalization data missing.")

    # Temperature normalization per policy
    temp_multiplier = 1.0
    if not inp.temp_norm_not_required and inp.beta_per_degC != 0.0 and \
       inp.baseline_temp_c is not None and inp.actual_temp_c is not None:
        temp_multiplier = 1.0 + inp.beta_per_degC * (inp.baseline_temp_c - inp.actual_temp_c)
    else:
        if inp.temp_norm_not_required:
            reasons.append("No temperature normalization (client policy: none).")
        else:
            reasons.append("No temperature normalization (β not regressed yet or temps missing).")

    # Adjusted actual kWh
    adj_actual_kwh = inp.actual_kwh * prod_multiplier * temp_multiplier

    # Analytic truth (can be negative)
    theoretical_saved_kwh = inp.baseline_kwh - adj_actual_kwh

    # Capture factor: only for positive savings
    if theoretical_saved_kwh > 0:
        capture = inp.capture_override if inp.capture_override is not None else (1.0 - inp.unload_pct)
    else:
        capture = 1.0
    captured_saved_kwh = theoretical_saved_kwh * capture

    # Disclose losses (never billed)
    performance_loss_kwh = max(0.0, -captured_saved_kwh)

    # Billing truth (no fees on losses)
    billed_saved_kwh = max(0.0, captured_saved_kwh)
    if billed_saved_kwh == 0.0 and captured_saved_kwh < 0.0:
        reasons.append(f"Negative month: Performance loss of {_r(performance_loss_kwh,2)} kWh recorded; billed at ₹0.")

    # Energy ₹ (TOU-aware if slots exist; else weighted average). Marks fallback use.
    energy_inr, tou_breakdown, tou_assumption, used_rate_fallback = _energy_rupees(inp, billed_saved_kwh)
    if used_rate_fallback:
        reasons.append("One or more TOU slot rates missing: fell back to weighted average for those slots.")

    # Demand ₹ (optional)
    demand_inr = 0.0
    if all(v is not None for v in (inp.demand.baseline_peak_kw, inp.demand.actual_peak_kw, inp.demand.demand_rate_inr_per_kw)):
        demand_delta_kw = max(0.0, inp.demand.baseline_peak_kw - inp.demand.actual_peak_kw)
        demand_inr = demand_delta_kw * float(inp.demand.demand_rate_inr_per_kw)
    else:
        reasons.append("No demand savings claimed (insufficient kW inputs or compressor not setting peak).")

    # PF ₹ (optional)
    pf_inr = 0.0
    if inp.pfline.baseline_pf_rupees is not None and inp.pfline.actual_pf_rupees is not None:
        pf_inr = max(0.0, inp.pfline.baseline_pf_rupees - inp.pfline.actual_pf_rupees)
    else:
        reasons.append("No PF savings included (PF bill line not provided).")

    total_inr = round(energy_inr + demand_inr + pf_inr, 0)

    # Confidence grading
    confidence = "A"
    if inp.data_completeness_pct < 98.0:
        confidence = "B"; reasons.append(f"Data completeness {inp.data_completeness_pct:.1f}% (<98%).")
    if not inp.temp_norm_not_required and inp.beta_per_degC == 0.0:
        confidence = "B"; reasons.append("β expected but not provided (regress in Phase 2).")
    if prod_clamped or (inp.baseline_units is None or inp.actual_units is None):
        confidence = "B"

    # Lineage hash (32 hex)
    lineage_payload = {"period": asdict(period), "inputs": _inputs_for_hash(inp), "method_version": inp.method_version}
    lineage_str = json.dumps(lineage_payload, sort_keys=True)
    lineage_hash = hashlib.sha256(lineage_str.encode()).hexdigest()[:32]

    # Performance loss ₹ (for display; never billed)
    perf_loss_inr = 0.0
    if performance_loss_kwh > 0:
        eff_rate = float(inp.tariff.weighted_avg_inr_per_kwh or 9.50)
        # If we had slot rates, we could compute a blended rate; for transparency keep it simple here.
        if tou_breakdown is not None and billed_saved_kwh > 0:
            eff_rate = (energy_inr / billed_saved_kwh)
        perf_loss_inr = - performance_loss_kwh * eff_rate

    return {
        # kWh accounting
        "theoretical_saved_kwh": _r(theoretical_saved_kwh, 2),
        "capture_factor": _r(capture, 3),
        "captured_saved_kwh": _r(captured_saved_kwh, 2),
        "billed_saved_kwh": _r(billed_saved_kwh, 2),
        "performance_loss_kwh": _r(performance_loss_kwh, 2),
        "adj_actual_kwh": _r(adj_actual_kwh, 2),

        # ₹ lines
        "energy_savings_inr": round(energy_inr, 0),
        "energy_tou_breakdown": tou_breakdown,
        "demand_savings_inr": round(demand_inr, 0),
        "pf_savings_inr": round(pf_inr, 0),
        "total_saved_inr": int(total_inr),
        "performance_loss_inr": round(perf_loss_inr, 0),
        "tou_assumption": tou_assumption,  # "slot_rates" | "weighted_avg" | "slot_rates(fallback)"
        "tou_allocation_policy": inp.tou_allocation_policy,

        # normalization
        "prod_multiplier_raw": _r(prod_multiplier_raw, 4),
        "prod_multiplier_used": _r(prod_multiplier, 4),
        "prod_clamped": prod_clamped,
        "temp_multiplier": _r(temp_multiplier, 4),

        # meta
        "ipmvp": "Option A",
        "confidence": confidence,
        "confidence_reasons": reasons,
        "method_version": inp.method_version,
        "lineage_hash": lineage_hash,
    }

def _energy_rupees(inp: MVInputs, billed_saved_kwh: float) -> Tuple[float, Optional[Dict[str, float]], str, bool]:
    """Returns (₹, slot_breakdown|None, assumption, used_rate_fallback_flag)"""
    kbh = inp.tariff.kwh_baseline_by_slot
    kah = inp.tariff.kwh_actual_by_slot
    rates = inp.tariff.rate_inr_per_kwh_by_slot
    wav = float(inp.tariff.weighted_avg_inr_per_kwh or 9.50)

    if kbh and kah and rates and len(kbh) == len(kah) == len(rates) and sum(kah.values()) > 0:
        total_actual = float(sum(kah.values()))
        slot_rupees: Dict[str, float] = {}
        energy_inr = 0.0
        used_fallback = False
        for slot, kwh_actual in kah.items():
            share = max(0.0, kwh_actual / total_actual)
            slot_kwh_saved = billed_saved_kwh * share
            rate = rates.get(slot)
            if rate is None:
                rate = wav   # fallback to weighted average if a slot rate is missing
                used_fallback = True
            slot_inr = slot_kwh_saved * float(rate)
            slot_rupees[slot] = round(slot_inr, 0)
            energy_inr += slot_inr
        assumption = "slot_rates(fallback)" if used_fallback else "slot_rates"
        return energy_inr, slot_rupees, assumption, used_fallback
    else:
        return billed_saved_kwh * wav, None, "weighted_avg", False

def _inputs_for_hash(inp: MVInputs) -> Dict:
    return asdict(inp)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _r(x: float, nd: int) -> float:
    return float(round(x, nd))
