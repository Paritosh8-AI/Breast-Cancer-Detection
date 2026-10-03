"""
FastAPI Microservice for Breast Cancer Sentinel 2.0.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from cancer_ai.config import settings
from cancer_ai.api.routes import router
from cancer_ai.models.ensemble import ensemble_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cancer_ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Breast Cancer Sentinel 2.0 AI Engines...")
    if ensemble_model.model is not None:
        logger.info("Clinical Ensemble model loaded successfully.")
    else:
        logger.warning("Ensemble model not found, running in fallback mode.")
    yield
    logger.info("Breast Cancer Sentinel service shutdown.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="""
# 🔬 Breast Cancer Sentinel 2.0 – Clinical AI Diagnostic Support
### High-Precision Cytology Ensemble • Sub-Millisecond Inference • Transparent Explainable AI

Empowering pathologists and oncologists with probabilistic risk stratification and 
morphological biomarker explanations based on Fine Needle Aspirate (FNA) cell nuclei characteristics.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Root"])
def root():
    return {
        "system": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "OPERATIONAL",
        "documentation": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cancer_ai.api.app:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
