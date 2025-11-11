from datetime import datetime

def price_for(ts: str, discom: str) -> float:
    # Placeholder – will be replaced when Gemini drops real tariffs
    hour = datetime.strptime(ts, "%Y-%m-%d %H:%M").hour
    if discom == "MSEDCL":
        if 18 <= hour <= 22 or 9 <= hour <= 12:
            return 10.50   # peak
        else:
            return 6.20    # off-peak
    return 8.50  # default

def test_peak_rate():
    assert price_for("2025-11-11 19:30", "MSEDCL") == 10.50

def test_offpeak_rate():
    assert price_for("2025-11-11 02:30", "MSEDCL") == 6.20
