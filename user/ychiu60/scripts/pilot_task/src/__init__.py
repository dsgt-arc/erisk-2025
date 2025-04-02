"""
Depression Detection System

This package contains the core components for the depression detection chatbot.
"""

from .data_loader import load_and_process_data
from .vector_store import setup_vector_store
from .rag_chain import create_rag_components, Message, RAGChainError, RAGComponents
from .session_manager import UserSessionManager, MessageRole
from .depression_detector import DepressionDetector
from .categories import BDICategories

__version__ = "0.1.0"
