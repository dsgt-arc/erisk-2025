import os
import argparse
from src.data_loader import load_and_process_data
from src.vector_store import setup_vector_store
from src.rag_chain import create_rag_chain
from src.session_manager import UserSessionManager
from src.depression_detector import DepressionDetector

def main():
    parser = argparse.ArgumentParser(description='Depression Detection Chatbot')
    parser.add_argument('--mode', type=str, default='cli', choices=['cli', 'api', 'ui'],
                      help='Run mode: cli for command line, api for FastAPI, ui for Streamlit')
    parser.add_argument('--data_path', type=str, default='data/MentalChat16K/Synthetic_Data_10K.csv',
                      help='Path to the MentalChat16K dataset')
    parser.add_argument('--vector_db_path', type=str, default='./chroma_db',
                      help='Path to store the vector database')
    
    args = parser.parse_args()
    
    # Load and process data
    documents = load_and_process_data(args.data_path)
    print(f"Loaded and processed {len(documents)} documents")
    
    # Setup vector store
    vectorstore = setup_vector_store(documents, persist_directory=args.vector_db_path)
    print(f"Vector store setup complete at {args.vector_db_path}")
    
    # Create RAG chain
    rag_chain = create_rag_chain(vectorstore)
    print("RAG chain created successfully")
    
    # Initialize session manager
    session_manager = UserSessionManager()
    
    # Initialize depression detector
    depression_detector = DepressionDetector(rag_chain, session_manager)
    
    # Run in selected mode
    if args.mode == 'cli':
        run_cli_mode(depression_detector)
    elif args.mode == 'api':
        run_api_mode(depression_detector, args.vector_db_path)
    elif args.mode == 'ui':
        run_ui_mode()
    
def run_cli_mode(depression_detector):
    from src.utils import clear_screen
    
    user_id = "cli_user"
    print("Depression Detection Chatbot CLI")
    print("Type 'exit' to quit, 'assessment' to see current assessment")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'assessment':
            assessment = depression_detector.get_assessment(user_id)
            print("\nCurrent Depression Assessment:")
            print(f"Severity: {assessment.get('severity', 'Unknown')}")
            print(f"Recommendation: {assessment.get('recommendation', 'Unknown')}")
            print("Indicators detected:", assessment.get('indicators', {}))
            continue
        
        # Process message
        result = depression_detector.process_message(user_input, user_id)
        print(f"\nBot: {result['response']}")
        
        if result.get('depression_assessment'):
            print("\n[Hidden Assessment Updated]")

def run_api_mode(depression_detector, vector_db_path):
    import uvicorn
    import os
    
    # Set environment variables for the API
    os.environ['VECTOR_DB_PATH'] = vector_db_path
    os.environ['DEPRESSION_DETECTOR'] = 'initialized'
    
    # Start the FastAPI server
    from api.app import app
    print("Starting FastAPI server...")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

def run_ui_mode():
    import subprocess
    
    # Start the streamlit app
    print("Starting Streamlit UI...")
    subprocess.run(["streamlit", "run", "ui/streamlit_app.py"])

if __name__ == "__main__":
    main()