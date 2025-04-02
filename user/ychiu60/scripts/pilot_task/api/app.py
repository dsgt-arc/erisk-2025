from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data_loader import load_and_process_data
from src.vector_store import setup_vector_store
from src.rag_chain import create_rag_components, RAGComponents
from src.session_manager import UserSessionManager
from src.depression_detector import DepressionDetector

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Depression Detection Chatbot API",
    description="API for a depression detection chatbot using RAG",
    version="1.0.0"
)

# Global variables
depression_detector = None

# Initialize depression detector
def initialize_depression_detector():
    """Initialize the depression detector."""
    global depression_detector
    
    # Check if already initialized
    if depression_detector is not None:
        return depression_detector
    
    try:
        # Load data and create vector store
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed_posts.csv")
        data = load_and_process_data(data_path)
        vector_store = setup_vector_store(data)
        
        # Create RAG components
        rag_components = create_rag_components(
            model_name="microsoft/DialoGPT-medium",
            device="cpu",
            vectorstore=vector_store
        )
        
        # Create session manager
        session_manager = UserSessionManager()
        
        # Create depression detector
        depression_detector = DepressionDetector(rag_components, session_manager)
        
        return depression_detector
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize depression detector: {str(e)}"
        )

# Dependency to get the depression detector
async def get_depression_detector():
    if depression_detector is None:
        return initialize_depression_detector()
    return depression_detector

# Include router
from api.routes import router
app.include_router(router)