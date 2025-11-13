from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from packages.laas_mv_core.laas_mv_core_v4_1 import (
    MVPeriod, MVInputs, Tariff, DemandLine, PFLine, calculate_savings
)

app = FastAPI(title="LaaS M&V API", version="0.4.1")

class CalcRequest(BaseModel):
    period: Dict[str, Any]
    inputs: Dict[str, Any]

@app.post("/mv/calc")
def mv_calc(req: CalcRequest):
    # Build MVPeriod
    period = MVPeriod(**req.period)

    # Build MVInputs and convert nested dicts to their dataclasses
    inputs_dict = dict(req.inputs)

    t = inputs_dict.get("tariff")
    if isinstance(t, dict):
        inputs_dict["tariff"] = Tariff(**t)

    d = inputs_dict.get("demand")
    if isinstance(d, dict):
        inputs_dict["demand"] = DemandLine(**d)

    p = inputs_dict.get("pfline")
    if isinstance(p, dict):
        inputs_dict["pfline"] = PFLine(**p)

    inputs = MVInputs(**inputs_dict)

    return calculate_savings(period, inputs)
