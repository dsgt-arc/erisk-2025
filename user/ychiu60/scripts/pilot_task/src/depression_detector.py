import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from .rag_chain import RAGComponents, Message, RAGChainError
from .session_manager import UserSessionManager, MessageRole
from .categories import BDICategories

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepressionDetectorError(Exception):
    """Base exception for depression detector errors"""
    pass

class Config:
    MAX_CHAT_HISTORY = 4
    MAX_DOCS = 3
    SESSION_TIMEOUT_HOURS = 24

class BDICategories:
    # BDI-II categories and indicators with severity levels (0-3), adapted for chat
    INDICATORS = {
        "sadness": {
            0: ["happy", "good", "great", "awesome", "fine", "ok", "okay"],
            1: ["sad", "down", "blue", "unhappy", "not happy", "feeling down"],
            2: ["very sad", "really sad", "so sad", "always sad", "depressed"],
            3: ["extremely sad", "worst", "terrible", "can't take it", "miserable"]
        },
        "pessimism": {
            0: ["hopeful", "optimistic", "positive", "looking forward", "excited"],
            1: ["worried", "concerned", "unsure", "not sure", "uncertain"],
            2: ["hopeless", "no hope", "won't work", "nothing helps", "pointless"],
            3: ["no future", "never get better", "only worse", "completely hopeless"]
        },
        "past_failure": {
            0: ["success", "good job", "doing well", "achieved", "proud"],
            1: ["failed", "messed up", "mistake", "wrong", "not good enough"],
            2: ["failure", "keep failing", "always fail", "nothing right"],
            3: ["total failure", "complete failure", "worthless", "useless"]
        },
        "loss_of_pleasure": {
            0: ["enjoy", "fun", "happy", "love", "like", "great"],
            1: ["boring", "not fun", "meh", "whatever", "don't care"],
            2: ["no fun", "hate", "nothing fun", "joyless", "empty"],
            3: ["nothing matters", "no point", "meaningless", "dead inside"]
        },
        "guilty_feelings": {
            0: ["fine", "okay", "good", "alright", "no problem"],
            1: ["guilty", "bad", "sorry", "regret", "ashamed"],
            2: ["very guilty", "really bad", "horrible", "awful"],
            3: ["worst person", "unforgivable", "evil", "terrible person"]
        },
        "self_dislike": {
            0: ["like myself", "good about", "confident", "proud", "happy with"],
            1: ["not confident", "insecure", "unsure", "doubt"],
            2: ["hate myself", "dislike", "bad person", "awful person"],
            3: ["despise myself", "loathe", "disgusting", "worthless"]
        },
        "suicidal_thoughts": {
            0: ["good", "fine", "okay", "alright", "living"],
            1: ["don't want to be here", "rather not live", "tired of life"],
            2: ["want to die", "end it", "kill", "suicide", "suicidal"],
            3: ["going to end it", "planning suicide", "last goodbye", "final message"]
        },
        "crying": {
            0: ["happy", "good", "fine", "okay", "strong"],
            1: ["cry", "crying", "tears", "emotional", "upset"],
            2: ["crying a lot", "always crying", "can't stop crying"],
            3: ["breaking down", "constant tears", "sobbing", "devastated"]
        },
        "agitation": {
            0: ["calm", "relaxed", "peaceful", "chill", "fine"],
            1: ["anxious", "nervous", "restless", "uneasy", "worried"],
            2: ["very anxious", "really nervous", "can't relax", "stressed"],
            3: ["panic", "freaking out", "losing it", "can't handle"]
        },
        "loss_of_interest": {
            0: ["interested", "care", "engaged", "like", "love"],
            1: ["not interested", "don't care", "whatever", "meh"],
            2: ["no interest", "nothing matters", "who cares", "pointless"],
            3: ["given up", "dead inside", "abandoned everything", "nothing left"]
        },
        "indecisiveness": {
            0: ["sure", "certain", "know", "decided", "clear"],
            1: ["unsure", "maybe", "not sure", "confused", "idk"],
            2: ["can't decide", "don't know", "lost", "stuck"],
            3: ["paralyzed", "helpless", "completely lost", "no idea"]
        },
        "changes_in_sleep": {
            0: ["sleep well", "good sleep", "rested", "fine"],
            1: ["tired", "sleepy", "can't sleep", "sleeping too much"],
            2: ["exhausted", "insomnia", "always tired", "no rest"],
            3: ["no sleep", "completely exhausted", "never sleep"]
        },
        "changes_in_appetite": {
            0: ["eating well", "hungry", "good appetite", "normal"],
            1: ["not hungry", "eating less", "eating more", "no appetite"],
            2: ["can't eat", "eating too much", "lost appetite"],
            3: ["stopped eating", "never eat", "always eating", "binge"]
        }
    }

    # BDI-II scoring thresholds - adjusted for chat context
    SEVERITY_THRESHOLDS = {
        "minimal": (0, 5),     # Lowered from 0-13
        "mild": (6, 10),      # Lowered from 14-19
        "moderate": (11, 15),  # Lowered from 20-28
        "severe": (16, 63)     # Lowered from 29-63
    }
    
    # Adjust thresholds for chat context where messages are shorter
    CHAT_SEVERITY_MULTIPLIER = 1  # Removed multiplier since we adjusted thresholds
    
    # Special handling for critical indicators that should trigger immediate concern
    CRITICAL_INDICATORS = {
        "suicidal_thoughts": 2,  # Level 2 or higher triggers critical response
        "self_dislike": 3,      # Level 3 triggers critical response
        "loss_of_pleasure": 3    # Level 3 triggers critical response
    }

    RECOMMENDATIONS = {
        "minimal": ["Continue the conversation to help me understand how you're feeling."],
        "mild": ["Consider talking to a counselor or therapist.", "Monitor symptoms and consider professional support if they persist."],
        "moderate": ["Consider consulting a mental health professional.", "Monitor symptoms more frequently"],
        "severe": ["Strongly recommend professional mental health support.", "Contact emergency services or crisis hotline if symptoms worsen."],
        "critical": ["Immediate professional help is strongly recommended. Please contact a mental health crisis hotline or emergency services."]
    }

class DepressionDetector:
    def __init__(self, rag_components: RAGComponents, session_manager: UserSessionManager):
        """
        Initialize the depression detector.
        
        Args:
            rag_components: RAG chain components
            session_manager: Session manager instance
        """
        self.rag_components = rag_components
        self.session_manager = session_manager
        self._validate_rag_components()

    def _validate_rag_components(self) -> None:
        """Validate that all required RAG components are present."""
        required_components = ["llm", "output_parser", "query_transform_chain", 
                             "retriever", "conversation_chain", "conversation_prompt"]
        missing = [comp for comp in required_components if not hasattr(self.rag_components, comp)]
        if missing:
            raise ValueError(f"Missing required RAG components: {', '.join(missing)}")

    def _format_chat_history(self, messages: List[Message]) -> str:
        """Format chat history for the RAG chain."""
        formatted = []
        for msg in messages[-4:]:  # Get last 2 exchanges (4 messages)
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        return "\n".join(formatted)

    def _analyze_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Analyze a message for depression indicators.
        
        Args:
            user_id: User ID for session tracking
            message: Message to analyze
            
        Returns:
            Dict containing severity and indicators found
        """
        # Extract depression indicators from message
        indicators = self._extract_depression_indicators(message)
        
        # Assess severity based on indicators
        severity, recommendations, key_concerns, bdi_score = self._assess_severity(indicators)
        
        # Store assessment in session
        assessment = {
            "severity": severity,
            "indicators": indicators,
            "recommendations": recommendations,
            "key_concerns": key_concerns,
            "bdi_score": bdi_score,
            "timestamp": datetime.now().isoformat()
        }
        self.session_manager.update_assessment(user_id, assessment)
        
        return assessment

    def _run_rag_chain(self, query: str, chat_history: str, context: str) -> str:
        """Run the RAG chain to generate a response."""
        try:
            # Transform query
            search_query = self.rag_components.query_transform_chain.invoke({"input": query})
            
            # Get response from conversation chain
            response = self.rag_components.conversation_chain.invoke({
                "context": context,
                "chat_history": chat_history,
                "input": query
            })
            
            # Extract text response
            if isinstance(response, dict):
                return response.get("text", str(response))
            else:
                return str(response)
            
        except Exception as e:
            logger.error(f"Error running RAG chain: {str(e)}")
            raise RAGChainError(f"Failed to generate response: {str(e)}")

    def process_message(self, message: str, user_id: str = None) -> str:
        try:
            # Transform query using predict
            search_query = self.rag_components.query_transform_chain.predict(
                input=message
            )
            
            # Get relevant documents
            docs = self.rag_components.retriever.get_relevant_documents(search_query)
            
            # Format chat history if user_id is provided
            chat_history = ""
            if user_id:
                session = self.session_manager.get_session(user_id)
                if session and "messages" in session:
                    chat_history = self._format_chat_history(session["messages"])

            # Generate response using predict
            response = self.rag_components.conversation_chain.predict(
                input=message,
                context=docs[0].page_content if docs else "",
                chat_history=chat_history
            )

            # Update depression indicators if user_id is provided
            if user_id:
                indicators = self._extract_depression_indicators(message)
                if indicators:
                    self._analyze_message(user_id, message)
            
            return response.strip() if response else "I apologize, but I encountered an error. Please try again."
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I apologize, but I encountered an error. Please try again."

    def get_assessment(self, user_id: str) -> Dict[str, Any]:
        """Get the current depression assessment for a user."""
        try:
            session = self.session_manager.get_session(user_id)
            assessment = session.get("assessment", {
                "severity": "minimal",
                "indicators": {},
                "recommendations": BDICategories.RECOMMENDATIONS["minimal"]
            })
            
            # Ensure recommendations are included
            if "recommendations" not in assessment:
                severity = assessment.get("severity", "minimal")
                assessment["recommendations"] = BDICategories.RECOMMENDATIONS.get(severity, 
                    BDICategories.RECOMMENDATIONS["minimal"])
            
            severity = assessment["severity"]
            indicators = assessment["indicators"]
            
            # Get recommendations from BDICategories
            recommendations = BDICategories.RECOMMENDATIONS.get(severity, [])
            
            # For critical cases, add crisis hotline information
            if severity == "critical" or any(ind == "suicidal_thoughts" for ind in indicators):
                crisis_info = [
                    "IMMEDIATE ACTION REQUIRED: Please seek immediate help.",
                    "National Crisis Hotline (US): 988",
                    "Crisis Text Line: Text HOME to 741741"
                ]
                recommendations = crisis_info + recommendations
            elif not recommendations:
                recommendations = ["Please seek professional help if symptoms persist or worsen."]
            
            return {
                "severity": severity,
                "recommendations": recommendations,
                "indicators": list(indicators.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting assessment: {str(e)}")
            return {
                "severity": "unknown",
                "recommendations": ["Unable to assess at this time. If you're in crisis, please call 988."],
                "indicators": []
            }

    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        try:
            current_time = datetime.now()
            timeout = timedelta(hours=24)  # 24-hour timeout
            self.session_manager.clear_session(current_time - timeout)
        except Exception as e:
            logger.warning(f"Session cleanup failed: {str(e)}")

    def _extract_depression_indicators(self, text: str) -> Dict[str, int]:
        """
        Extract depression indicators from text using BDI-II categories.
        
        Args:
            text: User message text
            
        Returns:
            Dictionary of categories and their severity scores (0-3)
        """
        text = text.lower().strip()
        if not text:
            return {}
        
        # Initialize scores for all categories
        scores = {category: 0 for category in BDICategories.INDICATORS.keys()}
        
        # For each category
        for category, severity_levels in BDICategories.INDICATORS.items():
            # Check each severity level in reverse (highest first)
            for severity in reversed(range(4)):  # 3, 2, 1, 0
                # Look for any indicator words at this severity level
                indicators = severity_levels[severity]
                for indicator in indicators:
                    # Check for exact matches or word boundaries
                    if (
                        indicator in text or 
                        any(word.strip('.,!?') == indicator for word in text.split())
                    ):
                        scores[category] = max(scores[category], severity)
                        break
                        
            # Special handling for compound indicators
            if category == "suicidal_thoughts":
                # Check for phrases like "want to kill myself" or "thinking about suicide"
                suicide_phrases = [
                    "kill myself", "end my life", "take my life",
                    "commit suicide", "thinking about suicide",
                    "considering suicide", "plan to die", "want to die"
                ]
                if any(phrase in text for phrase in suicide_phrases):
                    scores[category] = 3  # Set to highest severity
                
                # Check for the word "kill" in context
                if "kill" in text:
                    scores[category] = max(scores[category], 2)
        
        # Filter out categories with zero scores
        return {k: v for k, v in scores.items() if v > 0}

    def _assess_severity(self, indicators: Dict[str, int]) -> Tuple[str, str, List[str]]:
        """
        Assess depression severity based on BDI-II scoring.
        
        Args:
            indicators: Dictionary of categories and their severity scores
            
        Returns:
            Tuple of (severity_level, recommendation, key_concerns)
        """
        # Calculate BDI-II score (0-63 scale)
        # Each indicator is scored 0-3, and we have multiple indicators
        # Scale the total to match BDI-II range (0-63)
        raw_score = sum(indicators.values())
        max_possible = len(BDICategories.INDICATORS) * 3  # Max score per indicator is 3
        bdi_score = int((raw_score / max_possible) * 63)  # Scale to 0-63 range
        
        # Check for critical indicators first
        for category, threshold in BDICategories.CRITICAL_INDICATORS.items():
            if indicators.get(category, 0) >= threshold:
                return (
                    "critical",
                    "Immediate professional help is strongly recommended. Please contact a mental health crisis hotline or emergency services.",
                    [f"Severe {category.replace('_', ' ')} detected", "Immediate intervention needed"]
                )
        
        # Identify key concerns (categories with high scores)
        key_concerns = [
            f"Severe {category.replace('_', ' ')}" 
            for category, score in indicators.items() 
            if score >= 2
        ]
        
        # Determine severity based on thresholds
        sorted_thresholds = sorted(
            BDICategories.SEVERITY_THRESHOLDS.items(),
            key=lambda x: x[1][1],  # Sort by max_score
            reverse=True  # Most severe first
        )
        
        for severity, (min_score, max_score) in sorted_thresholds:
            if bdi_score >= min_score:
                if severity == "severe":
                    return (
                        severity,
                        "Your symptoms indicate severe depression. Professional help is strongly recommended.",
                        key_concerns or ["Multiple severe depression symptoms"]
                    )
                elif severity == "moderate":
                    return (
                        severity,
                        "Your symptoms suggest moderate depression. Please consider speaking with a mental health professional.",
                        key_concerns or ["Multiple moderate depression symptoms"]
                    )
                elif severity == "mild":
                    return (
                        severity,
                        "You're showing signs of mild depression. Consider talking to a counselor or therapist.",
                        key_concerns or ["Some depression symptoms present"]
                    )
                else:  # minimal
                    return (
                        "minimal" if not key_concerns else "mild",  # Upgrade to mild if there are key concerns
                        "Your symptoms are mild. Continue monitoring your mental health." if key_concerns else "Your symptoms are minimal, but continue monitoring your mental health.",
                        key_concerns or ["Minimal depression indicators"],
                        bdi_score
                    )
        
        # Fallback (should not reach here)
        return (
            "minimal",
            "Continue monitoring and maintaining good mental health practices.",
            ["No significant depression indicators detected"],
            0  # Default BDI score
        )

    def _analyze_indicator_trends(self, user_id: str, timeframe_days: int = 7) -> Dict[str, Any]:
        """
        Analyze trends in BDI-II severity scores over time.
        
        Args:
            user_id: User ID
            timeframe_days: Number of days to analyze
            
        Returns:
            Dictionary containing trend analysis with changes in severity scores
        """
        chat_history = self.session_manager.get_chat_history(user_id)
        if not chat_history:
            return {
                "trend": "insufficient_data",
                "changes": {},
                "recommendations": ["Not enough data for trend analysis"]
            }
        
        # Group severity scores by day
        daily_indicators = {}
        cutoff_time = datetime.now() - timedelta(days=timeframe_days)
        
        for msg in chat_history:
            if msg["role"] == "user":
                msg_time = datetime.fromtimestamp(msg.get("timestamp", 0))
                if msg_time < cutoff_time:
                    continue
                    
                day_key = msg_time.date()
                if day_key not in daily_indicators:
                    daily_indicators[day_key] = {}
                
                # Extract severity scores from message
                msg_scores = self._extract_depression_indicators(msg["content"])
                for category, score in msg_scores.items():
                    # Keep highest score for each category per day
                    daily_indicators[day_key][category] = max(
                        score,
                        daily_indicators[day_key].get(category, 0)
                    )
        
        # Analyze trends
        if len(daily_indicators) < 2:
            return {
                "trend": "insufficient_data",
                "changes": {},
                "recommendations": ["Continue monitoring for more accurate assessment"]
            }
        
        # Calculate changes in severity scores
        changes = {}
        sorted_days = sorted(daily_indicators.keys())
        first_day = sorted_days[0]
        last_day = sorted_days[-1]
        
        for category in BDICategories.INDICATORS.keys():
            first_score = daily_indicators[first_day].get(category, 0)
            last_score = daily_indicators[last_day].get(category, 0)
            
            if first_score == 0 and last_score == 0:
                continue
                
            change = last_score - first_score
            if change != 0:
                changes[category] = {
                    "direction": "increased" if change > 0 else "decreased",
                    "magnitude": abs(change),
                    "current_score": last_score
                }
        
        # Determine overall trend
        if ("suicidal_thoughts" in changes and 
            changes["suicidal_thoughts"]["current_score"] >= BDICategories.CRITICAL_INDICATORS["suicidal_thoughts"]):
            trend = "critical_deterioration"
            recommendations = [
                "Immediate professional intervention recommended",
                "Contact emergency services or crisis hotline"
            ]
        elif sum(1 for c in changes.values() if c["direction"] == "increased" and c["magnitude"] >= 1) > len(changes) / 2:
            trend = "deteriorating"
            recommendations = [
                "Consider professional mental health support",
                "Monitor symptoms more frequently"
            ]
        elif sum(1 for c in changes.values() if c["direction"] == "decreased") > len(changes) / 2:
            trend = "improving"
            recommendations = [
                "Continue current support strategies",
                "Maintain monitoring and self-care practices"
            ]
        else:
            trend = "stable"
            recommendations = [
                "Continue current monitoring",
                "Maintain support systems and coping strategies"
            ]
        
        return {
            "trend": trend,
            "changes": changes,
            "recommendations": recommendations
        }

    def _retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query to search for
            
        Returns:
            List of relevant documents
        """
        try:
            # Use invoke instead of get_relevant_documents
            return self.rag_components.retriever.invoke(query)
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    def _get_recommendations(self, severity: str, indicators: Dict[str, int]) -> List[str]:
        """Get recommendations based on severity and indicators."""
        try:
            # Get base recommendations for severity level
            recommendations = list(BDICategories.RECOMMENDATIONS.get(severity, 
                BDICategories.RECOMMENDATIONS["minimal"]))
            
            # Add critical recommendations if needed
            if severity == "critical" or any(
                indicators.get(indicator, 0) >= threshold 
                for indicator, threshold in BDICategories.CRITICAL_INDICATORS.items()
            ):
                recommendations.extend(BDICategories.RECOMMENDATIONS["critical"])
            
            # Add specific recommendations based on indicators
            if "suicide" in indicators or "self_harm" in indicators:
                recommendations.insert(0, "IMPORTANT: Please seek immediate professional help.")
                recommendations.append("Contact the National Suicide Prevention Lifeline at 988 (US).")
            
            if "loss_of_pleasure" in indicators:
                recommendations.append("Try to engage in activities you used to enjoy, even if you don't feel like it at first.")
            
            if "guilt" in indicators or "self_dislike" in indicators:
                recommendations.append("Remember that your thoughts are not facts. Be kind to yourself.")
            
            return recommendations[:5]  # Return top 5 most relevant recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return ["Continue the conversation to help me understand how you're feeling."]

    def _get_response_template(self, severity: str, indicators: Dict[str, List[str]]) -> Dict[str, str]:
        """Get appropriate response template based on severity."""
        try:
            templates = {
                "minimal": {
                    "acknowledgment": "I hear you. ",
                    "support": "I'm here to listen and support you. ",
                    "question": "Would you like to tell me more about what's on your mind?"
                },
                "mild": {
                    "acknowledgment": "That sounds challenging. ",
                    "support": "It's good that you're sharing these feelings. ",
                    "question": "How long have you been feeling this way?"
                },
                "moderate": {
                    "acknowledgment": "I can hear how difficult this is for you. ",
                    "support": "You're not alone in this. ",
                    "question": "Have you considered talking to a counselor about these feelings?"
                },
                "severe": {
                    "acknowledgment": "I'm very concerned about what you're sharing. ",
                    "support": "These feelings are serious, but there is help available. ",
                    "question": "Would you be willing to speak with a mental health professional?"
                },
                "critical": {
                    "acknowledgment": "I'm very worried about your safety right now. ",
                    "support": "Your life has value and there are people who want to help. ",
                    "action": "Please call 988 (US) or your local emergency services immediately."
                }
            }
            
            return templates.get(severity, templates["minimal"])
            
        except Exception as e:
            logger.error(f"Error getting response template: {str(e)}")
            return templates["minimal"]