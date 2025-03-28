import os
import re
from datetime import datetime
from typing import Dict, Any, List

from pyspark.sql import SparkSession

def create_spark_session(app_name="Depression_Detection"):
    """
    Create and return a SparkSession.
    
    Args:
        app_name: Name of the Spark application
        
    Returns:
        SparkSession object
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def extract_assessment(response: str) -> Dict[str, Any]:
    """
    Extract depression assessment from response text.
    
    Args:
        response: Response text containing assessment
        
    Returns:
        Dictionary with assessment details
    """
    # Extract assessment section
    assessment_pattern = r'<assessment>(.*?)</assessment>'
    assessment_match = re.search(assessment_pattern, response, re.DOTALL)
    
    if not assessment_match:
        return {}
    
    assessment_text = assessment_match.group(1).strip()
    
    # Extract indicators
    indicators_pattern = r'Depression indicators:\s*\[(.*?)\]'
    indicators_match = re.search(indicators_pattern, assessment_text, re.DOTALL)
    indicators = []
    if indicators_match:
        indicators = [i.strip() for i in indicators_match.group(1).split(',') if i.strip()]
    
    # Extract severity
    severity_pattern = r'Severity estimate:\s*(\w+)'
    severity_match = re.search(severity_pattern, assessment_text, re.DOTALL)
    severity = severity_match.group(1).strip() if severity_match else "unknown"
    
    # Extract recommendation
    recommendation_pattern = r'Recommended action:\s*(.*?)(?:\n|$)'
    recommendation_match = re.search(recommendation_pattern, assessment_text, re.DOTALL)
    recommendation = recommendation_match.group(1).strip() if recommendation_match else ""
    
    return {
        "indicators": indicators,
        "severity": severity,
        "recommendation": recommendation,
        "timestamp": datetime.now().isoformat()
    }

def format_chat_history(messages: List) -> str:
    """
    Format chat history for display.
    
    Args:
        messages: List of chat messages
        
    Returns:
        Formatted chat history string
    """
    formatted = []
    for msg in messages:
        role = "You" if msg.type == "human" else "Bot"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent prompt injection.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    # Remove assessment tags if present
    text = re.sub(r'<assessment>.*?</assessment>', '', text, flags=re.DOTALL)
    
    # Remove system prompt formatting
    text = re.sub(r'system:', '', text, flags=re.IGNORECASE)
    
    return text.strip()