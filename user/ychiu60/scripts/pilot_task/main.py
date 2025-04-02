import os
import sys
import signal
import argparse
import logging
from pathlib import Path
import yaml
from contextlib import contextmanager
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, Tuple

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import load_and_process_data
from src.depression_detector import DepressionDetector
from src.rag_chain import create_rag_components, RAGComponents
from src.session_manager import UserSessionManager
from src.vector_store import setup_vector_store
from src.exceptions import SetupError, ConfigError
from src.utils import clear_screen
from api.app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management class"""
    DEFAULT_CONFIG = {
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'llm_model': 'microsoft/DialoGPT-medium',
        'temperature': 0.7,
        'max_new_tokens': 128,  # Reduced from 256
        'do_sample': True,
        'data_path': 'data/MentalChat16K/Synthetic_Data_10K.csv',
        'vector_db_path': './chroma_db',
        'api_host': '0.0.0.0',
        'api_port': 8000,
        'required_env_vars': []
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path:
            self._load_config(config_path)
        
    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                if custom_config:
                    self.config.update(custom_config)
        except Exception as e:
            raise ConfigError(f"Error loading config file: {str(e)}")

    def get(self, key: str, default=None):
        return self.config.get(key, default)

@contextmanager
def graceful_shutdown(depression_detector):
    """Context manager for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal. Cleaning up...")
        if depression_detector and hasattr(depression_detector, 'cleanup'):
            depression_detector.cleanup()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        yield
    finally:
        if depression_detector and hasattr(depression_detector, 'cleanup'):
            depression_detector.cleanup()

def check_environment(config: Config):
    """Validate required environment variables"""
    required_vars = config.get('required_env_vars', [])
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

def validate_paths(config: Config):
    """Validate required paths exist"""
    data_path = Path(config.get('data_path'))
    if not data_path.exists():
        raise ConfigError(f"Data path does not exist: {data_path}")
    
    vector_db_path = Path(config.get('vector_db_path'))
    vector_db_path.mkdir(parents=True, exist_ok=True)

def setup_depression_detector(config: Config) -> DepressionDetector:
    """Setup the depression detector with error handling"""
    try:
        logger.info("Loading and processing data...")
        data_path = config.get('data_path', 'data/processed_posts.csv')
        data = load_and_process_data(data_path)
        logger.info(f"Loaded and processed {len(data)} documents")
        
        logger.info("Setting up vector store...")
        vector_store = setup_vector_store(data)
        logger.info("Vector store setup complete")
        
        logger.info(f"Creating RAG components with model {config.get('llm_model')}")
        rag_components = create_rag_components(
            model_name=config.get('llm_model'),
            device="cpu",
            vectorstore=vector_store
        )
        logger.info("RAG components created successfully")
        
        session_manager = UserSessionManager()
        detector = DepressionDetector(
            rag_components=rag_components,
            session_manager=session_manager
        )
        logger.info("Depression detector initialized successfully")
        return detector
    
    except Exception as e:
        logger.error(f"Error setting up depression detector: {str(e)}")
        raise SetupError(f"Error setting up depression detector: {str(e)}")

def initialize_app(config_path: str = "config.yaml") -> Tuple[DepressionDetector, UserSessionManager]:
    """
    Initialize the application components.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Tuple of (DepressionDetector, UserSessionManager)
    """
    try:
        # Load configuration
        config = Config(config_path)
        
        # Validate environment and paths
        check_environment(config)
        validate_paths(config)
        
        # Setup depression detector
        detector = setup_depression_detector(config)
        
        return detector, detector.session_manager
        
    except Exception as e:
        logger.error(f"Error initializing app: {str(e)}")
        raise

def run_cli_mode(depression_detector):
    """Run the CLI mode with improved error handling"""
    from src.utils import clear_screen
    
    user_id = "cli_user"
    logger.info("Starting CLI mode")
    print("\n=== Depression Detection Chatbot CLI ===")
    print("Commands: 'exit' to quit, 'assessment' for current assessment, 'clear' to clear screen")
    print("-" * 50)
    print("\nBot: Hello! I'm here to chat with you. How are you feeling today?")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'exit':
                print("\nBot: Goodbye! Take care.")
                break
            elif user_input.lower() == 'clear':
                clear_screen()
                print("=== Depression Detection Chatbot CLI ===")
                continue
            elif user_input.lower() == 'assessment':
                assessment = depression_detector.get_assessment(user_id)
                print("\nCurrent Depression Assessment:")
                print(f"Severity: {assessment.get('severity', 'Unknown')}")
                print(f"Recommendation: {assessment.get('recommendation', 'Unknown')}")
                print("Indicators detected:", assessment.get('indicators', {}))
                continue
            
            # Process message
            response = depression_detector.process_message(user_input, user_id)
            print(f"\nBot: {response}")
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            print("\nBot: I apologize, but I encountered an error. Please try again.")

def run_api_mode(depression_detector: DepressionDetector, config: Config) -> None:
    """Run the API mode"""
    try:
        import uvicorn
        
        # Store the detector in app state
        app.state.depression_detector = depression_detector
        
        # Run the FastAPI app
        logger.info(f"Starting API server on {config.get('api_host')}:{config.get('api_port')}")
        uvicorn.run(
            app,
            host=config.get('api_host'),
            port=config.get('api_port')
        )
        
    except Exception as e:
        logger.error(f"Error running API mode: {str(e)}")
        raise

def run_ui_mode() -> None:
    """Run the Streamlit UI mode"""
    try:
        import streamlit as st
        import subprocess
        import webbrowser
        
        # Start Streamlit server in a subprocess
        logger.info("Starting Streamlit UI")
        process = subprocess.Popen(
            ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=localhost"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Open browser automatically
        webbrowser.open("http://localhost:8501")
        
        # Wait for the process
        process.communicate()
        
    except Exception as e:
        logger.error(f"Error running UI mode: {str(e)}")
        raise

def main():
    """Main entry point with improved error handling and configuration"""
    parser = argparse.ArgumentParser(description='Depression Detection Chatbot')
    parser.add_argument('--mode', type=str, default='cli',
                      choices=['cli', 'api', 'ui'],
                      help='Run mode: cli for command line, api for FastAPI, ui for Streamlit')
    parser.add_argument('--config', type=str,
                      help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        detector, session_manager = initialize_app(args.config)
        
        # Run with graceful shutdown handling
        with graceful_shutdown(detector):
            if args.mode == 'cli':
                run_cli_mode(detector)
            elif args.mode == 'api':
                config = Config(args.config)
                run_api_mode(detector, config)
            elif args.mode == 'ui':
                run_ui_mode()
                
    except ConfigError as e:
        logger.error(f"Configuration error: {str(e)}")
        sys.exit(1)
    except SetupError as e:
        logger.error(f"Setup error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()