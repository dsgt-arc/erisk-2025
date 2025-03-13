import streamlit as st
import requests
import os
import sys
import pandas as pd
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data_loader import load_and_process_data
from src.vector_store import setup_vector_store
from src.rag_chain import create_rag_chain
from src.session_manager import UserSessionManager
from src.depression_detector import DepressionDetector

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = f"st_user_{int(time.time())}"

if "depression_detector" not in st.session_state:
    # Setup components
    data_path = os.environ.get("DATA_PATH", "data/MentalChat16K/Synthetic_Data_10K.csv")
    vector_db_path = os.environ.get("VECTOR_DB_PATH", "./chroma_db")

    try:
        # Load and process data if not already done
        if not os.path.exists(vector_db_path) or len(os.listdir(vector_db_path)) == 0:
            st.info("Setting up vector database for the first time. This might take a few minutes...")
            documents = load_and_process_data(data_path)
            vectorstore = setup_vector_store(documents, persist_directory=vector_db_path)
        else:
            # Just load the existing vector store
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
        
        st.session_state.depression_detector = depression_detector
    except Exception as e:
        st.error(f"Error setting up the depression detector: {str(e)}")
        st.session_state.depression_detector = None

# Set up the Streamlit UI
st.title("Depression Detection Chatbot")
st.subheader("A conversational agent to help identify signs of depression")

# UI tabs
tab1, tab2 = st.tabs(["Chat", "Assessment (Admin Only)"])

with tab1:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process the message if depression detector is initialized
        if st.session_state.depression_detector:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.depression_detector.process_message(
                        user_input,
                        st.session_state.user_id
                    )
                    st.write(result["response"])
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
        else:
            with st.chat_message("assistant"):
                st.error("Depression detector not initialized. Please check the logs.")

with tab2:
    # Admin view for assessment
    st.markdown("### Depression Assessment (Admin Only)")
    st.markdown("This view shows the current assessment of the user's depression indicators.")
    
    if st.session_state.depression_detector:
        assessment = st.session_state.depression_detector.get_assessment(st.session_state.user_id)
        
        # Display severity
        severity = assessment.get("severity", "none")
        if severity == "none":
            st.success("Severity: None detected")
        elif severity == "low":
            st.info("Severity: Low")
        elif severity == "moderate":
            st.warning("Severity: Moderate")
        elif severity == "high":
            st.error("Severity: High")
        
        # Display recommendation
        st.markdown(f"**Recommendation:** {assessment.get('recommendation', 'Continue monitoring')}")
        
        # Display indicators
        st.markdown("### Detected Indicators")
        indicators = assessment.get("indicators", {})
        if indicators:
            indicator_df = pd.DataFrame({
                "Indicator": list(indicators.keys()),
                "Count": list(indicators.values())
            })
            st.dataframe(indicator_df, use_container_width=True)
        else:
            st.info("No depression indicators detected yet.")
            
        # Clear conversation option
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.depression_detector.session_manager.clear_session(st.session_state.user_id)
            st.session_state.user_id = f"st_user_{int(time.time())}"
            st.success("Conversation cleared!")
    else:
        st.error("Depression detector not initialized. Please check the logs.")

# Footer
st.markdown("---")
st.markdown("⚠️ **Disclaimer:** This is a demo application for educational purposes only. It is not a diagnostic tool and should not be used as a substitute for professional mental health advice.")