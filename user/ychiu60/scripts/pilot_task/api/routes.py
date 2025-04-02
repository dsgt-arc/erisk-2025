from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
from src.depression_detector import DepressionDetector
from api.app import get_depression_detector

# Define your models
class MessageRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class MessageResponse(BaseModel):
    response: str
    user_id: str
    assessment: Optional[Dict[str, Any]] = None

class AssessmentResponse(BaseModel):
    severity: str
    indicators: List[str]
    recommendations: List[str]

# Create a router
router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Depression Detection Chatbot API"}

@router.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest, detector: DepressionDetector = Depends(get_depression_detector)):
    """Process a chat message."""
    try:
        response = detector.process_message(request.user_id, request.message)
        assessment = detector.get_assessment(request.user_id)
        return {
            "response": response,
            "user_id": request.user_id,
            "assessment": assessment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assessment/{user_id}", response_model=AssessmentResponse)
async def get_assessment(user_id: str, detector: DepressionDetector = Depends(get_depression_detector)):
    """Get assessment for a user."""
    try:
        assessment = detector.get_assessment(user_id)
        return assessment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/session/{user_id}")
async def clear_session(user_id: str, detector: DepressionDetector = Depends(get_depression_detector)):
    """Clear a user's session."""
    try:
        detector.session_manager.clear_session(user_id)
        return {"message": f"Session for user {user_id} cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))