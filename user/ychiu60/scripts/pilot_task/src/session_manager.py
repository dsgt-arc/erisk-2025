from langchain.memory import ConversationBufferMemory
from typing import Dict, List, Any

class UserSessionManager:
    def __init__(self):
        """Initialize the session manager."""
        self.sessions = {}
    
    def get_session(self, user_id: str) -> Dict[str, Any]:
        """
        Get or create a user session.
        
        Args:
            user_id: User ID
            
        Returns:
            User session
        """
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                "memory": ConversationBufferMemory(return_messages=True, memory_key="chat_history"),
                "depression_indicators": []
            }
        return self.sessions[user_id]
    
    def add_message(self, user_id: str, role: str, content: str) -> None:
        """
        Add a message to the user's conversation history.
        
        Args:
            user_id: User ID
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        session = self.get_session(user_id)
        if role == "user":
            session["memory"].chat_memory.add_user_message(content)
        else:
            session["memory"].chat_memory.add_ai_message(content)
    
    def get_chat_history(self, user_id: str) -> List:
        """
        Get the user's conversation history.
        
        Args:
            user_id: User ID
            
        Returns:
            List of conversation messages
        """
        session = self.get_session(user_id)
        return session["memory"].chat_memory.messages
    
    def update_depression_indicators(self, user_id: str, indicators: List[str]) -> None:
        """
        Update the user's depression indicators.
        
        Args:
            user_id: User ID
            indicators: List of depression indicators
        """
        session = self.get_session(user_id)
        session["depression_indicators"].extend(indicators)
    
    def get_depression_indicators(self, user_id: str) -> List[str]:
        """
        Get the user's depression indicators.
        
        Args:
            user_id: User ID
            
        Returns:
            List of depression indicators
        """
        session = self.get_session(user_id)
        return session["depression_indicators"]
    
    def clear_session(self, user_id: str) -> None:
        """
        Clear a user session.
        
        Args:
            user_id: User ID
        """
        if user_id in self.sessions:
            del self.sessions[user_id]