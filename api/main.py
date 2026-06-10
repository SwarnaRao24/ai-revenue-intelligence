"""
api/main.py
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.predict import router as predict_router
from api.services.model_service import model_service

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="AI-Powered Churn & Revenue Intelligence API",
    description=(
        "Production-grade ML API for customer churn prediction, "
        "revenue forecasting, risk assessment, and LLM-powered insights."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    model_service.load_all()


# ── Health ────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models": "loaded",
    }


# ── Root ──────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Churn & Revenue Intelligence API",
        "docs": "/docs",
        "health": "/health",
    }


# ── Include routers ───────────────────────────────────────────
app.include_router(predict_router)
