import logging
import os
from src.depression_detector import DepressionDetector
from src.session_manager import UserSessionManager
from src.rag_chain import create_rag_components
from src.vector_store import setup_vector_store, load_and_process_data
from src.erisk_task import ERiskTask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_components():
    """Initialize all required components"""
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "data", "MentalChat16K", "Synthetic_Data_10K.csv")
    data = load_and_process_data(data_path)
    
    # Setup vector store
    vector_store = setup_vector_store(data)
    
    # Create RAG components
    rag_components = create_rag_components(
        model_name="microsoft/DialoGPT-small",
        device="cpu",
        vectorstore=vector_store
    )
    
    # Initialize session manager and detector
    session_manager = UserSessionManager()
    detector = DepressionDetector(rag_components, session_manager)
    
    return detector

def run_automated_test(run_id: str):
    """Run automated test for eRisk task"""
    logger.info(f"Starting automated test run {run_id}")
    
    # Initialize components
    detector = initialize_components()
    
    # Create eRisk task handler
    task = ERiskTask(run_id=run_id, is_manual=False)
    
    # Test conversation prompts for each persona
    test_prompts = [
        "Hi, how are you doing today?",
        "What have you been up to lately?",
        "How has your week been?",
        "Do you have any hobbies or interests?",
        "How do you usually spend your evenings?",
        "Have you been sleeping well?",
        "What do you like to do for fun?",
        "How's your energy level been?",
        "What's been on your mind recently?",
        "How do you feel about the future?"
    ]
    
    for persona in task.PERSONAS:
        logger.info(f"Testing persona: {persona}")
        
        # Create unique user ID for this persona
        user_id = f"{run_id}_{persona.lower()}"
        
        # Run through test prompts
        for prompt in test_prompts:
            try:
                # Get response from detector
                response = detector.process_message(prompt, user_id)
                
                # Add messages to conversation log
                task.add_message(persona, "user", prompt)
                task.add_message(persona, persona, response)
                
            except Exception as e:
                logger.error(f"Error processing message for {persona}: {str(e)}")
                continue
        
        try:
            # Get final assessment
            assessment = detector.get_assessment(user_id)
            
            # Convert assessment to BDI-II format
            bdi_score = assessment.get("bdi_score", 0)
            key_symptoms = []
            
            # Map indicators to BDI-II symptoms (up to 4)
            symptom_map = {
                "sadness": "Sadness",
                "pessimism": "Pessimism",
                "past_failure": "Past Failure",
                "loss_of_pleasure": "Loss of Pleasure",
                "guilty_feelings": "Guilty Feelings",
                "self_dislike": "Self-Dislike",
                "suicidal_thoughts": "Suicidal Thoughts",
                "crying": "Crying",
                "agitation": "Agitation",
                "loss_of_interest": "Loss of Interest",
                "indecisiveness": "Indecisiveness",
                "changes_in_sleep": "Changes in Sleep",
                "changes_in_appetite": "Changes in Appetite"
            }
            
            # Get top 4 indicators by severity
            indicators = assessment.get("indicators", {})
            sorted_indicators = sorted(
                indicators.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:4]
            
            for indicator, _ in sorted_indicators:
                if indicator in symptom_map:
                    key_symptoms.append(symptom_map[indicator])
            
            # Set assessment for this persona
            task.set_assessment(persona, bdi_score, key_symptoms)
            
        except Exception as e:
            logger.error(f"Error getting assessment for {persona}: {str(e)}")
            continue
    
    # Save all results
    task.save_results()
    logger.info(f"Completed test run {run_id}")

if __name__ == "__main__":
    # Run automated test
    run_automated_test("run1")
