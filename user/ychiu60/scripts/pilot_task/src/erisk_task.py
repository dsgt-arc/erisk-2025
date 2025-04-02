import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class LLMPersona:
    name: str
    conversation: List[Dict[str, str]] = None
    bdi_score: Optional[int] = None
    key_symptoms: List[str] = None

    def __post_init__(self):
        self.conversation = self.conversation or []
        self.key_symptoms = self.key_symptoms or []

class ERiskTask:
    """Handler for eRisk 2025 pilot task"""
    
    PERSONAS = [
        "Ethan", "Maya", "Elena", "James", "Marco",
        "Noah", "Linda", "Gabriel", "Maria", "Alex",
        "Priya", "Laura"
    ]
    
    BDI_SYMPTOMS = [
        "Sadness", "Pessimism", "Past Failure", "Loss of Pleasure",
        "Guilty Feelings", "Punishment Feelings", "Self-Dislike",
        "Self-Criticalness", "Suicidal Thoughts", "Crying",
        "Agitation", "Loss of Interest", "Indecisiveness",
        "Worthlessness", "Loss of Energy", "Changes in Sleep",
        "Irritability", "Changes in Appetite", "Concentration Problems",
        "Fatigue", "Loss of Interest in Sex"
    ]

    def __init__(self, run_id: str, is_manual: bool = False):
        """Initialize eRisk task handler
        
        Args:
            run_id: Identifier for this run
            is_manual: Whether this is a manual run
        """
        self.run_id = run_id
        self.is_manual = is_manual
        self.personas = {name: LLMPersona(name=name) for name in self.PERSONAS}
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def add_message(self, persona_name: str, role: str, message: str):
        """Add a message to a persona's conversation
        
        Args:
            persona_name: Name of the LLM persona
            role: Either 'user' or the persona's name
            message: The message content
        """
        if persona_name not in self.personas:
            raise ValueError(f"Unknown persona: {persona_name}")
            
        self.personas[persona_name].conversation.append({
            "role": role,
            "message": message
        })

    def set_assessment(self, persona_name: str, bdi_score: int, key_symptoms: List[str]):
        """Set the depression assessment for a persona
        
        Args:
            persona_name: Name of the LLM persona
            bdi_score: BDI-II score (0-63)
            key_symptoms: List of up to 4 symptoms from BDI_SYMPTOMS
        """
        if persona_name not in self.personas:
            raise ValueError(f"Unknown persona: {persona_name}")
            
        if not 0 <= bdi_score <= 63:
            raise ValueError(f"Invalid BDI score: {bdi_score}")
            
        if len(key_symptoms) > 4:
            raise ValueError("Maximum 4 key symptoms allowed")
            
        for symptom in key_symptoms:
            if symptom not in self.BDI_SYMPTOMS:
                raise ValueError(f"Invalid symptom: {symptom}")
        
        persona = self.personas[persona_name]
        persona.bdi_score = bdi_score
        persona.key_symptoms = key_symptoms

    def save_results(self):
        """Save interaction logs and classification results"""
        prefix = "manual-run_" if self.is_manual else ""
        
        # Save interactions
        interactions_file = self.output_dir / f"{prefix}interactions_{self.run_id}.json"
        interactions = []
        for persona in self.personas.values():
            if persona.conversation:  # Only include personas with conversations
                interactions.append({
                    "LLM": persona.name,
                    "conversation": persona.conversation
                })
                
        with open(interactions_file, 'w') as f:
            json.dump(interactions, f, indent=4)
            
        # Save results
        results_file = self.output_dir / f"{prefix}results_{self.run_id}.json"
        results = []
        for persona in self.personas.values():
            if persona.bdi_score is not None:  # Only include assessed personas
                results.append({
                    "LLM": persona.name,
                    "bdi-score": persona.bdi_score,
                    "key-symptoms": persona.key_symptoms
                })
                
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Saved results to {self.output_dir}")
