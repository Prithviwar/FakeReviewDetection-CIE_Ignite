from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint
from typing import Optional
from utils import score_review

app = FastAPI(
    title="Fake Review Detection API",
    description="API for detecting fake or suspicious product reviews using ML and rule-based scoring.",
    version="1.0.0"
)

class ReviewRequest(BaseModel):
    review: str
    rating: conint(ge=1, le=5) # type: ignore
    client_id: Optional[str] = "default"

class ReviewResponse(BaseModel):
    flag: str
    score: int
    ml_fake_probability: float
    reasons: list[str]

class BatchReviewRequest(BaseModel):
    reviews: list[ReviewRequest]

class BatchReviewResponse(BaseModel):
    results: list[ReviewResponse]

@app.post("/analyze", response_model=ReviewResponse)
def analyze_review(req: ReviewRequest):
    if not req.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    
    # Process the review using the existing utils function
    result = score_review(req.review, req.rating, client_id=req.client_id)
    
    return ReviewResponse(**result)

@app.post("/analyze-batch", response_model=BatchReviewResponse)
def analyze_batch(req: BatchReviewRequest):
    if not req.reviews:
        raise HTTPException(status_code=400, detail="Batch list cannot be empty.")
    
    results = []
    # Temporary history list for isolated batch duplication checking
    batch_history = []
    
    for r in req.reviews:
        if not r.review.strip():
            # Skip empty reviews in batch
            continue
            
        res = score_review(r.review, r.rating, explicit_history=batch_history)
        results.append(ReviewResponse(**res))
        
        # Add to the temporary batch history so future reviews in THIS batch are checked against it
        batch_history.append(r.review)
        
    return BatchReviewResponse(results=results)
