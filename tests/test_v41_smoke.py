from packages.laas_mv_core.laas_mv_core_v4_1 import MVPeriod, MVInputs, calculate_savings

def test_positive_month_basic():
    period = MVPeriod(site_id="S1",
                      period_start_utc="2025-10-01T00:00:00Z",
                      period_end_utc="2025-10-31T23:59:59Z")
    inp = MVInputs(
        baseline_kwh=2400, actual_kwh=1800,
        baseline_units=1000, actual_units=950,
        temp_norm_not_required=True,   # skip temp norm for pilot
        unload_pct=0.25,
        tariff={"weighted_avg_inr_per_kwh": 9.5}
    )
    out = calculate_savings(period, inp)
    assert out["billed_saved_kwh"] > 0
    assert out["total_saved_inr"] > 0
