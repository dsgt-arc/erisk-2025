import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

class MessageRole(Enum):
    """Enum for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    def __str__(self):
        return self.value

class UserSessionManager:
    """Manages user sessions and conversation history."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the session manager."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.storage_path = storage_path or "session_data"
        os.makedirs(self.storage_path, exist_ok=True)
        
    def get_session(self, user_id: str) -> Dict[str, Any]:
        """Get or create a session for a user."""
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                "messages": [],
                "assessment": {
                    "severity": "minimal",
                    "indicators": {},
                    "recommendations": []
                },
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat()
            }
        return self.sessions[user_id]
        
    def add_message(self, user_id: str, message: Dict[str, str]) -> None:
        """Add a message to a user's session."""
        session = self.get_session(user_id)
        # Ensure message is JSON serializable by using string values
        if isinstance(message.get('role'), MessageRole):
            message['role'] = str(message['role'])
        session["messages"].append(message)
        session["last_active"] = datetime.now().isoformat()
        self._save_session(user_id)
        
    def get_messages(self, user_id: str) -> List[Dict[str, str]]:
        """Get all messages for a user."""
        session = self.get_session(user_id)
        return session["messages"]
        
    def clear_session(self, user_id: str) -> None:
        """Clear a user's session."""
        if user_id in self.sessions:
            del self.sessions[user_id]
        self._delete_session_file(user_id)
        
    def update_assessment(self, user_id: str, assessment: Dict[str, Any]) -> None:
        """Update the depression assessment for a user."""
        session = self.get_session(user_id)
        session["assessment"].update(assessment)
        self._save_session(user_id)
        
    def get_assessment(self, user_id: str) -> Dict[str, Any]:
        """Get the current depression assessment for a user."""
        session = self.get_session(user_id)
        return session["assessment"]
        
    def _save_session(self, user_id: str) -> None:
        """Save a session to disk."""
        if user_id in self.sessions:
            file_path = os.path.join(self.storage_path, f"{user_id}.json")
            with open(file_path, 'w') as f:
                json.dump(self.sessions[user_id], f)
                
    def _delete_session_file(self, user_id: str) -> None:
        """Delete a session file from disk."""
        file_path = os.path.join(self.storage_path, f"{user_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            
    def load_session(self, user_id: str) -> bool:
        """Load a session from disk."""
        file_path = os.path.join(self.storage_path, f"{user_id}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.sessions[user_id] = json.load(f)
            return True
        return False