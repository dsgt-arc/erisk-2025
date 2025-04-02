# Standard library imports
from datetime import datetime
import os
import gc
import logging
import json
import subprocess
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

# Third-party imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Local imports
from .session_manager import MessageRole

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Disable TF-XLA warnings
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add debug logger
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('debug.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
debug_logger.addHandler(handler)

# Force CPU mode for stable execution
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_cache')

# Update imports
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline  # Use community package

class MessageRole(str, Enum):
    """Enum for message roles that inherits from str for JSON serialization"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
    
    @classmethod
    def _missing_(cls, value):
        """Handle missing values by returning the value itself if it's a valid role"""
        if value in ["user", "assistant", "system"]:
            return cls(value)
        return None

@dataclass
class Message:
    """Message class with JSON serialization support"""
    role: str
    content: str
    timestamp: str

    def to_dict(self) -> Dict[str, str]:
        """Convert message to dictionary"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }

@dataclass
class RAGComponents:
    """Container for RAG chain components."""
    llm: Any
    tokenizer: Any
    retriever: Any
    query_transform_chain: Any
    conversation_prompt: Any
    conversation_chain: Any
    output_parser: Any

class RAGChainError(Exception):
    """Custom exception for RAG chain errors"""
    pass

# Add BDI-II criteria mapping
BDI_CRITERIA = {
    "sadness": {
        "keywords": ["sad", "unhappy", "miserable", "depressed"],
        "severity_scale": {
            0: "I do not feel sad",
            1: "I feel sad much of the time",
            2: "I am sad all the time",
            3: "I am so sad or unhappy that I can't stand it"
        }
    },
    "pessimism": {
        "keywords": ["hopeless", "future", "worse", "improve", "better"],
        "severity_scale": {
            0: "I am not discouraged about my future",
            1: "I feel more discouraged about my future than I used to be",
            2: "I do not expect things to work out for me",
            3: "I feel my future is hopeless and will only get worse"
        }
    },
    "past_failure": {
        "keywords": ["fail", "failure", "unsuccessful", "mistake", "wrong"],
        "severity_scale": {
            0: "I do not feel like a failure",
            1: "I have failed more than I should have",
            2: "As I look back, I see a lot of failures",
            3: "I feel I am a total failure as a person"
        }
    },
    "loss_of_pleasure": {
        "keywords": ["enjoy", "pleasure", "satisfaction", "happy", "fun"],
        "severity_scale": {
            0: "I get as much pleasure as I ever did from things",
            1: "I don't enjoy things as much as I used to",
            2: "I get very little pleasure from things",
            3: "I can't get any pleasure from things"
        }
    }
    # Add more BDI-II criteria as needed
}

class ConversationTracker:
    def __init__(self, storage_path="conversation_logs"):
        self.storage_path = storage_path
        self.current_session = []
        self.depression_indicators = {
            "bdi_scores": {},  # Track BDI-II scores for each criterion
            "severity_history": [],  # Track overall severity over time
            "criteria_matches": {}  # Track matched criteria and context
        }
        os.makedirs(storage_path, exist_ok=True)
    
    def analyze_message(self, message: str) -> Dict[str, Any]:
        """Analyze a message for BDI-II criteria matches"""
        matches = {}
        message_lower = message.lower()
        
        for criterion, data in BDI_CRITERIA.items():
            for keyword in data["keywords"]:
                if keyword in message_lower:
                    context = self._get_keyword_context(message_lower, keyword)
                    if criterion not in matches:
                        matches[criterion] = []
                    matches[criterion].append({
                        "keyword": keyword,
                        "context": context
                    })
        
        return matches
    
    def _get_keyword_context(self, message: str, keyword: str, context_window: int = 10) -> str:
        """Extract context around a keyword match"""
        words = message.split()
        try:
            keyword_index = words.index(keyword)
            start = max(0, keyword_index - context_window)
            end = min(len(words), keyword_index + context_window + 1)
            return " ".join(words[start:end])
        except ValueError:
            return ""
    
    def update_bdi_scores(self, matches: Dict[str, List]) -> None:
        """Update BDI scores based on matched criteria"""
        for criterion, match_data in matches.items():
            if criterion not in self.depression_indicators["bdi_scores"]:
                self.depression_indicators["bdi_scores"][criterion] = []
            
            # Simple scoring based on keyword matches and context
            severity = min(len(match_data), 3)  # Cap at maximum BDI score of 3
            self.depression_indicators["bdi_scores"][criterion].append({
                "timestamp": datetime.now().isoformat(),
                "severity": severity,
                "matches": match_data
            })
    
    def analyze_depression_risk(self) -> Dict[str, Any]:
        """Analyze depression risk using BDI-II criteria"""
        if not self.depression_indicators["bdi_scores"]:
            return {
                "risk_level": "unknown",
                "confidence": 0.0,
                "criteria_matched": [],
                "recommendation": "Continue monitoring"
            }
        
        # Calculate recent severity scores
        recent_scores = {}
        for criterion, scores in self.depression_indicators["bdi_scores"].items():
            if scores:  # Get most recent score for each criterion
                recent_scores[criterion] = scores[-1]["severity"]
        
        # Calculate overall severity
        if recent_scores:
            avg_severity = sum(recent_scores.values()) / len(recent_scores)
            max_severity = max(recent_scores.values())
            
            # Determine risk level
            if max_severity >= 3 or avg_severity >= 2.5:
                risk_level = "high"
            elif max_severity >= 2 or avg_severity >= 1.5:
                risk_level = "moderate"
            elif max_severity >= 1 or avg_severity >= 0.5:
                risk_level = "low"
            else:
                risk_level = "minimal"
            
            # Calculate confidence based on number of criteria matched
            confidence = min(len(recent_scores) / len(BDI_CRITERIA), 1.0)
            
            return {
                "risk_level": risk_level,
                "confidence": confidence,
                "criteria_matched": list(recent_scores.keys()),
                "recommendation": self._get_recommendation(risk_level)
            }
        
        return {"risk_level": "unknown", "confidence": 0.0, "criteria_matched": []}
    
    def _get_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level"""
        recommendations = {
            "high": "Strongly recommend professional mental health evaluation",
            "moderate": "Consider speaking with a mental health professional",
            "low": "Monitor symptoms and practice self-care",
            "minimal": "Continue monitoring and maintaining mental well-being"
        }
        return recommendations.get(risk_level, "Continue monitoring")

def check_system_compatibility() -> Tuple[bool, str]:
    """Check system compatibility for GPU operations"""
    try:
        # Check PyTorch version and CUDA availability
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        
        # Try to get GPU info using nvidia-smi
        try:
            nvidia_smi = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader']
            ).decode('utf-8').strip().split('\n')[0].split(',')
            total_memory = int(nvidia_smi[0])
            used_memory = int(nvidia_smi[1])
            free_memory = total_memory - used_memory
        except (subprocess.SubprocessError, FileNotFoundError):
            total_memory = used_memory = free_memory = None

        # Build status message
        status = [
            f"PyTorch Version: {torch_version}",
            f"CUDA Available: {cuda_available}",
            f"CUDA Version: {cuda_version}",
            f"GPU Memory Total: {total_memory}MB" if total_memory else "GPU Memory: Unknown",
            f"GPU Memory Free: {free_memory}MB" if free_memory else ""
        ]
        
        # Check if system meets minimum requirements
        is_compatible = (
            cuda_available and 
            cuda_version is not None and 
            (free_memory is None or free_memory > 4000)  # Require at least 4GB free
        )
        
        return is_compatible, "\n".join(status)
        
    except Exception as e:
        return False, f"Error checking system compatibility: {str(e)}"

def create_rag_components(
    model_name: str = "facebook/blenderbot-400M-distill",
    model_cache_dir: str = None,
    device: str = None,
    vectorstore: Any = None
) -> RAGComponents:
    """Create RAG chain components."""
    try:
        logger.info("Initializing RAG components...")
        
        # Configure tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side='left',
            truncation_side='left',
            model_max_length=1024
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map='auto'
        )
        
        # Ensure model config is aligned with tokenizer
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        
        # Configure pipeline with detailed settings
        logger.info("Setting up pipeline...")
        pipe_config = {
            "task": "text2text-generation",  # For DialoGPT
            "model": model,
            "tokenizer": tokenizer,
            "max_new_tokens": 100,     # Keep responses concise
            "min_length": 20,        # Ensure complete sentences
            "do_sample": True,
            "temperature": 0.7,       # Balanced temperature
            "top_k": 50,             # Standard filtering
            "top_p": 0.9,            # Allow some variety
            "repetition_penalty": 1.2,  # Mild repetition penalty
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "return_full_text": False
        }
        
        # Create pipeline with error handling
        try:
            pipe = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=150,
                min_length=20,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            logger.info("Pipeline created successfully")
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            raise

        # Create LLM
        llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
        )
        
        # Create query transform chain
        query_transform_prompt = PromptTemplate(
            input_variables=["input"],
            template="Given this user message about their emotional state or mental health: {input}\n"
                     "Transform it into a search query that will find similar conversations about mental health and emotional well-being:"
        )
        query_transform_chain = LLMChain(
            llm=llm,
            prompt=query_transform_prompt,
            verbose=True
        )

        # Create conversation prompt
        conversation_prompt = PromptTemplate(
            input_variables=["context", "chat_history", "input"],
            template="""You are an empathetic mental health support chatbot having a conversation with a user. Be professional, supportive, and focused on helping them.

Rules:
1. Always respond with 2-4 complete, grammatically correct sentences
2. Maintain a caring, professional tone
3. If the user expresses distress, acknowledge their feelings and express concern
4. Use the context to provide relevant, empathetic responses

Relevant context from similar conversations:
{context}

Chat history:
{chat_history}

User: {input}
Assistant:"""
        )
        
        # Create conversation chain
        conversation_chain = LLMChain(
            llm=llm,
            prompt=conversation_prompt,
            verbose=True
        )

        # Create retriever with updated configuration
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # Create output parser
        output_parser = StrOutputParser()
        
        return RAGComponents(
            llm=llm,
            tokenizer=tokenizer,
            retriever=retriever,
            query_transform_chain=query_transform_chain,
            conversation_prompt=conversation_prompt,
            conversation_chain=conversation_chain,
            output_parser=output_parser
        )

    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
        raise RAGChainError(f"Failed to create RAG chain: {str(e)}")

def run_rag_chain(
    chain_components: Dict[str, Any],
    input_text: str,
    chat_history: Optional[List] = None,
    conversation_tracker: Optional[ConversationTracker] = None
) -> str:
    try:
        debug_logger.debug(f"Input text: {input_text}")
        
        # Format chat history properly
        formatted_history = []
        if chat_history:
            for msg in chat_history[-2:]:  # Keep only last 2 messages
                if isinstance(msg, dict):
                    role = str(msg.get('role', ''))
                    content = str(msg.get('content', ''))
                    formatted_history.append(f"{role}: {content}")
        
        formatted_history_str = "\n".join(formatted_history)

        try:
            # Transform query using predict
            search_query = chain_components["query_transform_chain"].predict(
                input=input_text
            )
            debug_logger.debug(f"Search query: {search_query}")

            # Get relevant documents
            try:
                docs = chain_components["retriever"].get_relevant_documents(search_query)
                # Combine content from top 3 documents
                context = "\n".join(doc.page_content[:200] for doc in docs[:3]) if docs else ""
            except Exception as e:
                debug_logger.error(f"Error retrieving documents: {e}")
                context = ""

            # Generate response using predict
            try:
                response = chain_components["conversation_chain"].predict(
                    context=context,
                    chat_history=formatted_history_str,
                    input=input_text  # Pass actual input text
                )
                
                if conversation_tracker:
                    matches = conversation_tracker.analyze_message(input_text)
                    if matches:
                        conversation_tracker.update_bdi_scores(matches)
                
                return response.strip()

            except Exception as e:
                debug_logger.error(f"Error in response generation: {e}")
                return "I understand you're sharing something important with me. While I process that, could you tell me more about how you're feeling?"

        except Exception as e:
            debug_logger.error(f"Error in chain execution: {e}")
            return "I'd like to hear more about what's on your mind."

    except Exception as e:
        debug_logger.error(f"Critical error in RAG chain: {e}")
        return "I'm here to listen. Would you like to tell me more?"

def format_docs(docs):
    """Format retrieved documents into a string, preserving conversation structure."""
    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        # Split context and conversation
        if "Context:" in content and "User:" in content:
            context, conversation = content.split("\n\n", 1)
            # Extract just the assistant's response
            if "Assistant:" in conversation:
                _, response = conversation.split("Assistant:", 1)
                formatted_docs.append(f"Similar example:\n{context}\nResponse:{response.strip()}")
    return "\n\n".join(formatted_docs)

def clear_gpu_memory():
    """Enhanced GPU memory management"""
    if torch.cuda.is_available():
        try:
            # Record memory before clearing
            before_memory = torch.cuda.memory_allocated()
            
            # Aggressive cleanup
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            
            # Force synchronization
            torch.cuda.synchronize()
            
            # Record memory after clearing
            after_memory = torch.cuda.memory_allocated()
            freed_memory = before_memory - after_memory
            
            logger.info(f"Cleared {freed_memory / 1024**2:.1f}MB of GPU memory")
            logger.info(f"Current GPU memory: {after_memory / 1024**2:.1f}MB")
            
            # Check for memory leaks
            if after_memory > (torch.cuda.get_device_properties(0).total_memory * 0.9):
                logger.warning("High GPU memory usage detected")
                
        except Exception as e:
            logger.error(f"Error in GPU memory cleanup: {e}")
            return False
        return True
    return False

def _extract_indicators(response: str) -> Dict[str, Any]:
    """Extract depression indicators from response with debugging"""
    debug_logger.debug(f"Extracting indicators from response length: {len(response)}")
    
    indicators = {
        "severity": "none",
        "action": "continue monitoring"
    }
    
    try:
        if "Depression indicators:" in response:
            parts = response.split("\n")
            for part in parts:
                debug_logger.debug(f"Processing part: {part[:50]}...")
                if "Severity:" in part:
                    severity = part.split(":")[1].strip().lower()
                    debug_logger.debug(f"Found severity: {severity}")
                    indicators["severity"] = severity
                if "Action:" in part:
                    action = part.split(":")[1].strip()
                    debug_logger.debug(f"Found action: {action}")
                    indicators["action"] = action
    except Exception as e:
        debug_logger.error(f"Error extracting indicators: {e}\n{traceback.format_exc()}")
    
    debug_logger.debug(f"Final indicators: {indicators}")
    return indicators

# Example usage in your main application:
def initialize_depression_detection():
    """Initialize the depression detection system"""
    tracker = ConversationTracker()
    # ... initialize other components ...
    return tracker

def handle_user_input(input_text: str, chain_components: Dict[str, Any], tracker: ConversationTracker):
    """Handle user input with debugging"""
    debug_logger.debug(f"Handling input: {input_text}")
    
    try:
        # Handle special commands
        if input_text.lower() == "assessment":
            debug_logger.debug("Processing assessment command")
            try:
                risk_assessment = tracker.analyze_depression_risk()
                debug_logger.debug(f"Risk assessment: {risk_assessment}")
                
                # Format assessment response
                matched_criteria = risk_assessment.get('criteria_matched', [])
                criteria_text = "\n".join([f"- {criterion}" for criterion in matched_criteria]) if matched_criteria else "None detected yet"
                
                return f"""Depression Assessment Results:
Risk Level: {risk_assessment.get('risk_level', 'unknown')}
Confidence: {risk_assessment.get('confidence', 0.0):.2f}
Areas of Concern:
{criteria_text}
Recommendation: {risk_assessment.get('recommendation', 'Continue monitoring')}

Remember, this is not a diagnosis. If you're concerned about your mental health, please speak with a qualified mental health professional."""
            except Exception as e:
                debug_logger.error(f"Error in assessment: {e}\n{traceback.format_exc()}")
                return "I apologize, but I couldn't generate an assessment at this time."

        # Regular message handling
        debug_logger.debug("Processing regular message")
        try:
            response = run_rag_chain(
                chain_components,
                input_text,  # Pass the actual input text
                chat_history=tracker.current_session[-2:] if tracker.current_session else None,
                conversation_tracker=tracker
            )
            
            # Save session periodically
            if len(tracker.current_session) % 5 == 0:
                debug_logger.debug("Saving session")
                tracker.save_session()
            
            return response
            
        except Exception as e:
            debug_logger.error(f"Error handling message: {e}\n{traceback.format_exc()}")
            return "I'm here to help. Could you tell me more?"
        
    except Exception as e:
        debug_logger.error(f"Critical error handling input: {e}\n{traceback.format_exc()}")
        return "I'm here to listen. Would you like to tell me more?"

def optimize_for_gpu():
    """Optimize settings based on GPU type"""
    if not torch.cuda.is_available():
        return "cpu", {}
    
    gpu_name = torch.cuda.get_device_name()
    logger.info(f"Detected GPU: {gpu_name}")
    
    if 'A100' in gpu_name:
        config = {
            'device': 'cuda',
            'max_memory': '35GB',
            'batch_size': 16,
            'max_new_tokens': 512,
            'model_parallel': True,
            'memory_fraction': 0.9
        }
        logger.info("Using A100 optimized settings")
    elif 'T4' in gpu_name:
        config = {
            'device': 'cuda',
            'max_memory': '4GB',
            'batch_size': 4,
            'max_new_tokens': 128,
            'model_parallel': False,
            'memory_fraction': 0.7
        }
        logger.info("Using T4 optimized settings")
    else:
        config = {
            'device': 'cuda',
            'max_memory': '4GB',
            'batch_size': 2,
            'max_new_tokens': 128,
            'model_parallel': False,
            'memory_fraction': 0.6
        }
        logger.info("Using default GPU settings")
    
    return config

def manage_gpu_memory():
    """Advanced GPU memory management for any GPU type"""
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name()
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Get memory info
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"Total Memory: {total_memory:.2f}GB")
            logger.info(f"Memory allocated: {memory_allocated/1024**3:.2f}GB")
            logger.info(f"Memory reserved: {memory_reserved/1024**3:.2f}GB")
            
            # Set threshold based on total memory
            memory_threshold = min(total_memory * 0.85, 30) * 1024**3  # 85% of total memory or 30GB
            
            # If memory usage is high, force cleanup
            if memory_allocated > memory_threshold:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
                logger.warning("Forced memory cleanup due to high usage")
                
            return True, f"Memory management complete for {gpu_name}"
            
        except Exception as e:
            logger.error(f"Error managing GPU memory: {e}")
            return False, str(e)