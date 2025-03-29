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
from src.rag_chain import create_rag_chain
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
    global depression_detector
    
    # Check if already initialized
    if depression_detector is not None:
        return depression_detector
    
    # Get paths from environment
    data_path = os.environ.get("DATA_PATH", "data/MentalChat16K/Synthetic_Data_10K.csv")
    vector_db_path = os.environ.get("VECTOR_DB_PATH", "./chroma_db")
    
    try:
        # Load vector store
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma
        
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=embeddings
        )
        
        # Create RAG chain
        rag_chain = create_rag_chain(vectorstore)
        
        # Initialize session manager and depression detector
        session_manager = UserSessionManager()
        depression_detector = DepressionDetector(rag_chain, session_manager)
        
        return depression_detector
    except Exception as e:
        print(f"Error initializing depression detector: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize depression detector: {str(e)}")

# Dependency to get the depression detector
def get_depression_detector():
    detector = initialize_depression_detector()
    if detector is None:
        raise HTTPException(status_code=500, detail="Depression detector not initialized")
    return detector

# Include router
from api.routes import router
app.include_router(router)