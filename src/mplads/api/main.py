"""FastAPI application entrypoint for the MPLADS ML system.

Currently wires up Module 1 (Cost Overrun Detection) only, per the
one-module-at-a-time build order.
"""
from fastapi import FastAPI

from mplads.api.routers import module1, module1b, module2, underspend_risk
from mplads.module7_pulse_score.router import router as module7_router
from mplads.module8_report_generation.router import router as module8_router
from mplads.module9_dispatch.router import router as module9_router

app = FastAPI(
    title="MPLADS ML System",
    description="Machine learning services for MPLADS project monitoring.",
    version="0.1.0",
)

app.include_router(module1.router)
app.include_router(module2.router)
app.include_router(module7_router)
app.include_router(module8_router)
app.include_router(module9_router)
app.include_router(underspend_risk.router)
app.include_router(module1b.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
