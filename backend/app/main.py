"""
Coal Quality Intelligence Platform — FastAPI application entry point.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import DB_PATH, PROJECT_ROOT
from .api.environmental_routes import router as environmental_router
from .api.analytics_routes import router as analytics_router
from .api.prediction_routes import router as prediction_router
from .api.explainability_routes import router as explainability_router
from .api.scenario_routes import router as scenario_router
from .api.report_routes import router as report_router

app = FastAPI(
    title="Coal Quality Intelligence Platform",
    description="Environmental Intelligence Engine — Coal India Limited",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(environmental_router)
app.include_router(analytics_router)
app.include_router(prediction_router)
app.include_router(explainability_router)
app.include_router(scenario_router)
app.include_router(report_router)

FRONTEND_PATH = PROJECT_ROOT / "frontend_integrated.html"

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    if not FRONTEND_PATH.exists():
        return {"error": f"Frontend not found at {FRONTEND_PATH}"}
    return FileResponse(str(FRONTEND_PATH))

@app.get("/health")
def health():
    return {
        "status": "operational",
        "db_path": DB_PATH,
        "db_exists": Path(DB_PATH).exists(),
        "frontend_path": str(FRONTEND_PATH),
        "frontend_exists": FRONTEND_PATH.exists(),
    }
