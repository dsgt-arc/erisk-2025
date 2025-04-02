import streamlit as st

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="Mental Health Support Chat",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys
import logging
from pathlib import Path
import uuid
import time

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from src.data_loader import load_and_process_data
from src.depression_detector import DepressionDetector
from src.rag_chain import create_rag_components, RAGComponents
from src.session_manager import UserSessionManager
from src.vector_store import setup_vector_store

# Disable file watcher to avoid PyTorch class issues
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_app():
    """Initialize the application components."""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Load data
        status_text.text("📚 Loading conversation data...")
        progress_bar.progress(10)
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "data", "MentalChat16K", "Synthetic_Data_10K.csv")
        logger.info(f"Loading data from: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
            
        data = load_and_process_data(data_path)
        progress_bar.progress(30)
        
        # Step 2: Setup vector store
        status_text.text("🔍 Setting up knowledge base...")
        vector_store = setup_vector_store(data)
        progress_bar.progress(50)
        
        # Step 3: Create RAG components
        status_text.text("🤖 Initializing AI model...")
        try:
            rag_components = create_rag_components(
                model_name="microsoft/DialoGPT-small",
                device="cpu",
                vectorstore=vector_store
            )
            if not isinstance(rag_components, RAGComponents):
                raise ValueError("Failed to create valid RAG components")
            progress_bar.progress(80)
        except Exception as e:
            logger.error(f"Error creating RAG components: {str(e)}")
            raise
        
        # Step 4: Initialize remaining components
        status_text.text("✨ Finalizing setup...")
        session_manager = UserSessionManager()
        detector = DepressionDetector(rag_components, session_manager)
        progress_bar.progress(100)
        status_text.empty()
        return detector, session_manager
        
    except Exception as e:
        logger.error(f"Error initializing app: {str(e)}")
        st.error("Failed to initialize the application. Please try again.")
        raise

def display_assessment():
    """Display the current depression assessment."""
    if "depression_detector" in st.session_state and "user_id" in st.session_state:
        try:
            assessment = st.session_state.depression_detector.get_assessment(st.session_state.user_id)
            
            # Setup severity colors
            severity_colors = {
                "minimal": "green",
                "mild": "yellow",
                "moderate": "orange",
                "severe": "red",
                "critical": "darkred"
            }
            severity = assessment.get('severity', 'Unknown')
            color = severity_colors.get(severity.lower(), "gray")
            
            # Format text content
            indicators = assessment.get('indicators', [])
            indicators_text = "\n".join(f"- {indicator}" for indicator in indicators) if indicators else "No significant indicators detected"
            
            recommendations = assessment.get('recommendations', [])
            recommendations_text = "\n".join(f"- {rec}" for rec in recommendations) if recommendations else "Continue the conversation"
            
            # Display single, consolidated assessment
            st.markdown(f"""
            ### Depression Assessment
            **Severity**: <span style='color: {color}'>{severity.title()}</span>
            
            **Indicators Detected**:
            {indicators_text}
            
            **Recommendations**:
            {recommendations_text}
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error displaying assessment: {str(e)}")
    else:
        st.info("Assessment not available yet.")

def display_chat():
    """Display the chat interface."""
    st.title("Mental Health Support Chat 🤗")
    st.caption("I'm here to listen and support you. Feel free to share how you're feeling.")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
        
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        with st.container():
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"<div style='color: #0066cc'><b>Bot:</b> {content}</div>", unsafe_allow_html=True)

    # Chat input
    if "depression_detector" not in st.session_state:
        st.error("Chat system is not initialized. Please refresh the page.")
        return
        
    # Input area
    user_input = st.text_input(
        "Type your message:",
        key="user_input",
        placeholder="Share how you're feeling..."
    )
    
    if st.button("Send") and user_input:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        try:
            # Add user message to chat first
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Get response from depression detector with error handling
            try:
                with st.spinner('Processing...'):
                    response = st.session_state.depression_detector.process_message(
                        message=user_input,
                        user_id=st.session_state.user_id
                    )
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                # Add a friendly error message to chat instead of showing error popup
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I understand you're sharing something important. While I process that, could you tell me more about how you're feeling?"
                })
                
        except Exception as e:
            logger.error(f"Critical error in chat interface: {str(e)}")
            st.error("An error occurred. Please refresh the page and try again.")
        
        # Clear input and refresh
        st.session_state.user_input = ""
        st.rerun()

def main():
    """Main entry point."""
    # Show title
    st.title("Mental Health Support Chat")
    st.write("I'm here to listen and support you. Feel free to share how you're feeling.")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    try:
        # Initialize components if not already done
        if 'depression_detector' not in st.session_state:
            with st.spinner('Initializing chat system...'):
                detector, session_manager = initialize_app()
                st.session_state.depression_detector = detector
                st.session_state.session_manager = session_manager
        
        # Create two columns
        col1, col2 = st.columns([2, 1])

        with col1:
            # Display chat interface
            display_chat()

        with col2:
            # Display current assessment
            display_assessment()
            
            # Helpful resources
            with st.expander("Helpful Resources"):
                st.markdown("""
                ### Crisis Resources 🆘
                - **National Crisis Line**: 988
                - **Crisis Text Line**: Text HOME to 741741
                
                ### Support Options 💚
                - Talk to a trusted friend or family member
                - Schedule an appointment with a counselor
                - Join a support group
                
                ### Self-Care Tips 🌱
                1. Practice deep breathing
                2. Get regular exercise
                3. Maintain a sleep schedule
                4. Stay connected with others
                5. Do activities you enjoy
                """)
            
            # Admin controls
            with st.expander("Admin Controls"):
                if "depression_detector" in st.session_state:
                    if st.button("Clear Conversation"):
                        st.session_state.messages = []
                        st.session_state.depression_detector.session_manager.clear_session(st.session_state.user_id)
                        st.session_state.user_id = str(uuid.uuid4())
                        st.success("Conversation cleared!")
                else:
                    st.error("Depression detector not initialized. Please check the logs.")

        # Sidebar with helpful information
        with st.sidebar:
            st.title("Options")
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
            
            with st.expander("About This Chat"):
                st.write("""
                This is a supportive space where you can talk about your feelings and thoughts.
                The chatbot uses AI to provide empathetic responses and can help identify signs
                of depression.
                
                Note: This is not a replacement for professional mental health care.
                If you're experiencing severe symptoms, please seek professional help.
                """)
                
            with st.expander("Crisis Resources", expanded=True):
                st.warning("""
                If you're having thoughts of suicide or experiencing a mental health crisis, please reach out:
                
                🆘 **National Crisis Line**: 988
                💬 **Crisis Text Line**: Text HOME to 741741
                📞 **National Suicide Prevention Lifeline**: 1-800-273-8255
                """)

        # Footer
        st.markdown("---")
        st.markdown("⚠️ **Disclaimer:** This is a demo application for educational purposes only. It is not a diagnostic tool and should not be used as a substitute for professional mental health advice.")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()