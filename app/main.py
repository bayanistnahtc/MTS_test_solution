import logging

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from haystack import Pipeline

from app.core.config import settings
from app.models import HealthCheck, QueryRequest
from app.rag_pipeline import create_rag_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    app.state.rate_limit_store = {}
    app.state.pipeline = create_rag_pipeline(
        classifier_path=settings.classifier_path,
    )
    yield

    # Cleanup resources
    await app.state.pipeline.aclose()


# Initialize application
app = FastAPI(
    title="RAG Service API",
    version="1.0.1",
    lifespan=lifespan
    )

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_pipeline() -> Pipeline:
    if not app.state.pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized"
        )
    return app.state.pipeline


# API Endpoints
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Service health check endpoint"""
    return {
        "status": "OK",
        "model_loaded": app.state.pipeline is not None
    }


@app.post(
        "/ask",
        tags=["RAG"],
        summary="Process Query",
        response_description="Generated Answer",
)
async def ask_question(
    request: QueryRequest,
    pipeline: Pipeline = Depends(get_pipeline),
):
    """Main endpoint for querying the RAG system"""
    try:
        logger.info(f"Processing query: {request.query[:50]}...")

        # Execute pipeline
        result = pipeline.run(
            {
                "router": {"text": request.query},
                "prompt_builder": {
                    "template_variables": {"question": request.query}
                },
                "generator": {
                    "generation_kwargs": {
                        "max_tokens": request.max_length,
                        "temperature": request.temperature
                    }
                }
            },
            include_outputs_from=["router", "retriever"]
        )

        return {
            "answer": result["generator"]["replies"][0].text,
            "retrieval_used": "LABEL_1" in result.get("router", {}),
            "documents_retrieved": result.get("retriever", {}).get("documents", [])
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info" if settings.debug else "warning"
    )
