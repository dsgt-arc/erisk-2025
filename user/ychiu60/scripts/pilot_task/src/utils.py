import os
import re
from datetime import datetime
from typing import Dict, Any, List, Protocol

ASSESSMENT_PATTERN = r'<assessment>(.*?)</assessment>'
INDICATORS_PATTERN = r'Depression indicators:\s*\[(.*?)\]'
SEVERITY_PATTERN = r'Severity estimate:\s*(\w+)'
RECOMMENDATION_PATTERN = r'Recommended action:\s*(.*?)(?:\n|$)'

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
        
    Raises:
        ValueError: If response is empty or None
    """
    if not response:
        raise ValueError("Response text cannot be empty")
    
    # Extract assessment section
    assessment_pattern = ASSESSMENT_PATTERN
    assessment_match = re.search(assessment_pattern, response, re.DOTALL)
    
    if not assessment_match:
        return {}
    
    assessment_text = assessment_match.group(1).strip()
    
    # Extract indicators
    indicators_pattern = INDICATORS_PATTERN
    indicators_match = re.search(indicators_pattern, assessment_text, re.DOTALL)
    indicators = []
    if indicators_match:
        indicators = [i.strip() for i in indicators_match.group(1).split(',') if i.strip()]
    
    # Extract severity
    severity_pattern = SEVERITY_PATTERN
    severity_match = re.search(severity_pattern, assessment_text, re.DOTALL)
    severity = severity_match.group(1).strip() if severity_match else "unknown"
    
    # Extract recommendation
    recommendation_pattern = RECOMMENDATION_PATTERN
    recommendation_match = re.search(recommendation_pattern, assessment_text, re.DOTALL)
    recommendation = recommendation_match.group(1).strip() if recommendation_match else ""
    
    return {
        "indicators": indicators,
        "severity": severity,
        "recommendation": recommendation,
        "timestamp": datetime.now().isoformat()
    }

class Message(Protocol):
    type: str
    content: str

def format_chat_history(messages: List[Message]) -> str:
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