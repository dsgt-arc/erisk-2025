import re
import uuid
from collections import Counter
from typing import Dict, List, Any, Optional

class DepressionDetector:
    def __init__(self, rag_chain, session_manager):
        """
        Initialize the depression detector.
        
        Args:
            rag_chain: RAG chain components
            session_manager: Session manager for conversation history
        """
        self.rag_chain = rag_chain
        self.session_manager = session_manager
    
    def process_message(self, user_input: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user message and detect depression signs.
        
        Args:
            user_input: User message text
            user_id: User ID (generated if not provided)
            
        Returns:
            Dictionary with response and depression assessment
        """
        # Generate a user ID if not provided
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        # Get user session
        self.session_manager.get_session(user_id)
        chat_history = self.session_manager.get_chat_history(user_id)
        
        # Add user message to history
        self.session_manager.add_message(user_id, "user", user_input)
        
        # Generate response using RAG
        response = self._run_rag_chain(user_input, chat_history)
        
        # Extract assessment if present
        assessment_pattern = r'Depression indicators:\s*\[(.*?)\]\s*Severity estimate:\s*(\w+)\s*Recommended action:\s*(.*?)(?:\n|$)'
        assessment_match = re.search(assessment_pattern, response, re.DOTALL)
        
        visible_response = response
        depression_assessment = None
        
        if assessment_match:
            # Extract the components from the regex match
            indicators_text = assessment_match.group(1).strip()
            severity = assessment_match.group(2).strip()
            recommendation = assessment_match.group(3).strip()
            
            # Remove the assessment section from the visible response
            visible_response = re.sub(assessment_pattern, '', response, flags=re.DOTALL).strip()
            
            # Extract and update indicators
            indicators = [i.strip() for i in indicators_text.split(',') if i.strip()]
            self.session_manager.update_depression_indicators(user_id, indicators)
            
            # Create assessment dictionary
            depression_assessment = {
                "indicators": dict(Counter(indicators)),
                "severity": severity.lower(),
                "recommendation": recommendation
            }
        
        # Add AI response to history (only the visible part)
        self.session_manager.add_message(user_id, "assistant", visible_response)
        
        return {
            "response": visible_response,
            "user_id": user_id,
            "depression_assessment": depression_assessment
        }
    
    def _run_rag_chain(self, user_input: str, chat_history: List) -> str:
        """
        Run the RAG chain with proper error handling.
        
        Args:
            user_input: User input text
            chat_history: Chat history
            
        Returns:
            Response text
        """
        try:
            # Extract components from rag_chain
            llm = self.rag_chain["llm"]
            output_parser = self.rag_chain["output_parser"]
            query_transform_chain = self.rag_chain["query_transform_chain"]
            retriever = self.rag_chain["retriever"]
            conversation_prompt = self.rag_chain["conversation_prompt"]
            
            # Transform query based on conversation history
            transformed_query = query_transform_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Retrieve relevant documents
            docs = retriever.get_relevant_documents(transformed_query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate conversation messages
            messages = conversation_prompt.invoke({
                "input": user_input,
                "chat_history": chat_history,
                "context": context
            })
            
            # Run through LLM
            response = llm.invoke(messages)
            
            # Parse output
            return output_parser.invoke(response)
        except Exception as e:
            print(f"Error in RAG chain: {str(e)}")
            return f"I'm having trouble processing your message. Can you try again? (Error: {str(e)})"
    
    def get_assessment(self, user_id: str) -> Dict[str, Any]:
        """
        Get the depression assessment for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Depression assessment dictionary
        """
        indicators = self.session_manager.get_depression_indicators(user_id)
        
        if not indicators:
            return {
                "indicators": {},
                "severity": "none",
                "recommendation": "continue monitoring"
            }
        
        # Count occurrence of each indicator
        indicator_counts = Counter(indicators)
        
        # Determine severity
        if len(indicator_counts) >= 5 or any(count >= 3 for count in indicator_counts.values()):
            severity = "high"
            recommendation = "professional referral needed"
        elif len(indicator_counts) >= 3 or any(count >= 2 for count in indicator_counts.values()):
            severity = "moderate"
            recommendation = "professional referral needed"
        else:
            severity = "low"
            recommendation = "continue monitoring"
            
        return {
            "indicators": dict(indicator_counts),
            "severity": severity,
            "recommendation": recommendation
        }
    
    
# import re
# import uuid
# from collections import Counter
# from typing import Dict, List, Any, Optional
# from rag_chain import run_rag_chain

# class DepressionDetector:
#     def __init__(self, rag_chain, session_manager):
#         """
#         Initialize the depression detector.
        
#         Args:
#             rag_chain: RAG chain components
#             session_manager: Session manager for conversation history
#         """
#         self.rag_chain = rag_chain
#         self.session_manager = session_manager
    
#     def process_message(self, user_input: str, user_id: Optional[str] = None) -> Dict[str, Any]:
#         """
#         Process a user message and detect depression signs.
        
#         Args:
#             user_input: User message text
#             user_id: User ID (generated if not provided)
            
#         Returns:
#             Dictionary with response and depression assessment
#         """
#         # Generate a user ID if not provided
#         if user_id is None:
#             user_id = str(uuid.uuid4())
        
#         # Get user session
#         self.session_manager.get_session(user_id)
#         chat_history = self.session_manager.get_chat_history(user_id)
        
#         # Add user message to history
#         self.session_manager.add_message(user_id, "user", user_input)
        
#         # Generate response using RAG
        
#         response = run_rag_chain(self.rag_chain, user_input, chat_history)
        
#         # # Extract assessment if present
#         # assessment_pattern = r'<assessment>(.*?)</assessment>'
#         # Fix this regex pattern to properly extract the assessment
#         assessment_pattern = r'Depression indicators:\s*\[(.*?)\]\s*Severity estimate:\s*(\w+)\s*Recommended action:\s*(.*?)(?:\n|$)'
#         assessment_match = re.search(assessment_pattern, response, re.DOTALL)
        
#         visible_response = response
#         if assessment_match:
#             assessment_text = assessment_match.group(1).strip()
#             visible_response = re.sub(assessment_pattern, '', visible_response, flags=re.DOTALL).strip()
            
#             # Extract depression indicators from assessment
#             self._extract_and_update_indicators(user_id, assessment_text)
        
#         # Add AI response to history
#         self.session_manager.add_message(user_id, "assistant", visible_response)
        
#         return {
#             "response": visible_response,
#             "user_id": user_id,
#             "depression_assessment": self.get_assessment(user_id) if assessment_match else None
#         }
    
#     def _extract_and_update_indicators(self, user_id: str, assessment_text: str) -> None:
#         """
#         Extract depression indicators from assessment text and update user session.
        
#         Args:
#             user_id: User ID
#             assessment_text: Assessment text
#         """
#         indicators_pattern = r'Depression indicators:\s*\[(.*?)\]'
#         indicators_match = re.search(indicators_pattern, assessment_text, re.DOTALL)
        
#         if indicators_match:
#             indicators = [i.strip() for i in indicators_match.group(1).split(',') if i.strip()]
#             self.session_manager.update_depression_indicators(user_id, indicators)
    
#     def get_assessment(self, user_id: str) -> Dict[str, Any]:
#         """
#         Get the depression assessment for a user.
        
#         Args:
#             user_id: User ID
            
#         Returns:
#             Depression assessment dictionary
#         """
#         indicators = self.session_manager.get_depression_indicators(user_id)
        
#         if not indicators:
#             return {
#                 "indicators": {},
#                 "severity": "none",
#                 "recommendation": "continue monitoring"
#             }
        
#         # Count occurrence of each indicator
#         indicator_counts = Counter(indicators)
        
#         # Determine severity
#         if len(indicator_counts) >= 5 or any(count >= 3 for count in indicator_counts.values()):
#             severity = "high"
#         elif len(indicator_counts) >= 3 or any(count >= 2 for count in indicator_counts.values()):
#             severity = "moderate"
#         else:
#             severity = "low"
            
#         return {
#             "indicators": dict(indicator_counts),
#             "severity": severity,
#             "recommendation": "professional referral needed" if severity in ["moderate", "high"] else "continue monitoring"
#         }