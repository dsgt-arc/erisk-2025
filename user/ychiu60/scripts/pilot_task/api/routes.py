# from fastapi import APIRouter, Depends
# import app
# from src.depression_detector import DepressionDetector
# from api.app import AssessmentResponse, MessageResponse, MessageRequest, get_depression_detector, initialize_depression_detector

# # Create a router
# router = APIRouter()

# @router.get("/")
# async def root():
#     return {"message": "Depression Detection Chatbot API"}

# @router.post("/chat", response_model=MessageResponse)
# async def chat(request: MessageRequest, detector: DepressionDetector = Depends(get_depression_detector)):
#     result = detector.process_message(request.message, request.user_id)
#     return result

# @router.get("/assessment/{user_id}", response_model=AssessmentResponse)
# async def get_assessment(user_id: str, detector: DepressionDetector = Depends(get_depression_detector)):
#     assessment = detector.get_assessment(user_id)
#     return assessment

# @router.delete("/session/{user_id}")
# async def clear_session(user_id: str, detector: DepressionDetector = Depends(get_depression_detector)):
#     detector.session_manager.clear_session(user_id)
#     return {"message": f"Session for user {user_id} cleared"}

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from src.depression_detector import DepressionDetector

# Define your models
class MessageRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class MessageResponse(BaseModel):
    response: str
    user_id: str
    depression_assessment: Optional[Dict[str, Any]] = None

class AssessmentResponse(BaseModel):
    indicators: Dict[str, int]
    severity: str
    recommendation: str

# Create a router
router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Depression Detection Chatbot API"}

@router.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest, detector: DepressionDetector = Depends(get_depression_detector)):
    result = detector.process_message(request.message, request.user_id)
    return result

@router.get("/assessment/{user_id}", response_model=AssessmentResponse)
async def get_assessment(user_id: str, detector: DepressionDetector = Depends(get_depression_detector)):
    assessment = detector.get_assessment(user_id)
    return assessment

@router.delete("/session/{user_id}")
async def clear_session(user_id: str, detector: DepressionDetector = Depends(get_depression_detector)):
    detector.session_manager.clear_session(user_id)
    return {"message": f"Session for user {user_id} cleared"}

# Import this from app.py
from api.app import get_depression_detector

# # API routes
# @app.get("/")
# async def root():
#     return {"message": "Depression Detection Chatbot API"}

# @app.post("/chat", response_model=MessageResponse)
# async def chat(request: MessageRequest, detector: DepressionDetector = Depends(get_depression_detector)):
#     result = detector.process_message(request.message, request.user_id)
#     return result

# @app.get("/assessment/{user_id}", response_model=AssessmentResponse)
# async def get_assessment(user_id: str, detector: DepressionDetector = Depends(get_depression_detector)):
#     assessment = detector.get_assessment(user_id)
#     return assessment

# @app.delete("/session/{user_id}")
# async def clear_session(user_id: str, detector: DepressionDetector = Depends(get_depression_detector)):
#     detector.session_manager.clear_session(user_id)
#     return {"message": f"Session for user {user_id} cleared"}

# @app.on_event("startup")
# async def startup_event():
#     # Initialize the depression detector on startup
#     try:
#         initialize_depression_detector()
#     except Exception as e:
#         print(f"Error during startup: {str(e)}")
#         # Continue anyway, will try to initialize when endpoints are called