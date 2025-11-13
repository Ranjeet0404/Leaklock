from fastapi import FastAPI
from pydantic import BaseModel
from packages.laas_mv_core.laas_mv_core_v4_1 import MVPeriod, MVInputs, calculate_savings

app = FastAPI(title="LaaS M&V API", version="0.4.1")

class CalcRequest(BaseModel):
    period: dict
    inputs: dict

@app.post("/mv/calc")
def mv_calc(req: CalcRequest):
    period = MVPeriod(**req.period)
    inputs = MVInputs(**req.inputs)
    return calculate_savings(period, inputs)
