import os
import sys

# Ensure the project root is in the path when running via uvicorn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.classifier import HybridClassifer
from src.recommender import RecommendationEngine

# ── Pydantic Schemas ──────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Item or merchant name to classify")

class BulkClassifyRequest(BaseModel):
    items: list[str] = Field(..., min_length=1, description="List of item/merchant names to classify")

class TransactionInput(BaseModel):
    name: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0, description="Transaction amount in ₹")
    category: str = Field(..., min_length=1)

class RecommendRequest(BaseModel):
    transactions: list[TransactionInput] = Field(..., min_length=1)

class ClassifyResponse(BaseModel):
    item: str
    category: str
    confidence: float
    method: str

class HealthResponse(BaseModel):
    status: str
    service: str

class RecommendResponse(BaseModel):
    tip: str

# ── App & Singletons ─────────────────────────────────────────────────

app = FastAPI(
    title="AI Recommendation System",
    description="HybridClassifier (Fuzzy + Zero-Shot NLP) & RecommendationEngine for money management.",
    version="1.0.0",
)

# Instantiated once at startup – model loading happens here
classifier = HybridClassifer()
recommender = RecommendationEngine()

# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Simple health-check endpoint."""
    return {"status": "ok", "service": "AI Recommendation System"}


@app.post("/classify", response_model=ClassifyResponse)
def classify_item(req: ClassifyRequest):
    """Classify a single item/merchant name into a spending category."""
    result = classifier.classify(req.name)
    return result


@app.post("/bulk-classify", response_model=list[ClassifyResponse])
def bulk_classify(req: BulkClassifyRequest):
    """
    Classify a list of item names in one request.
    Reduces network round-trips from the Next.js backend.
    """
    results = [classifier.classify(name) for name in req.items]
    return results


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """Generate a spending tip from a list of categorized transactions."""
    transactions = [tx.model_dump() for tx in req.transactions]
    tip = recommender.generate_tips(transactions)
    return {"tip": tip}
