# Fixed Mental Health Analysis - Processing All 12 People

import json
import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict, Counter
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DepressionIndicators:
    """Structured depression indicators for analysis"""
    mood_indicators: List[str]
    energy_indicators: List[str]
    interest_indicators: List[str]
    cognitive_indicators: List[str]
    physical_indicators: List[str]
    social_indicators: List[str]

@dataclass 
class ModelAnalysisResult:
    """Results for a single model's analysis"""
    model_name: str
    total_conversations: int
    total_turns: int
    people_analyzed: List[str]
    depression_detection_rate: float
    severity_classification: Dict[str, int]
    symptom_recognition: Dict[str, float]
    response_appropriateness: float
    specific_examples: List[Dict[str, str]]
    strengths: List[str]
    weaknesses: List[str]
    overall_score: float
    per_person_stats: Dict[str, Dict[str, Any]]

class FixedMentalHealthAnalyzer:
    """Fixed mental health analyzer that processes all 12 people"""
    
    def __init__(self):
        self.depression_indicators = DepressionIndicators(
            mood_indicators=[
                'sad', 'depressed', 'down', 'blue', 'hopeless', 'empty', 
                'worthless', 'guilty', 'shame', 'despair', 'miserable',
                'crying', 'tears', 'grief', 'sorrow', 'melancholy'
            ],
            energy_indicators=[
                'tired', 'exhausted', 'fatigued', 'drained', 'weary', 
                'lethargic', 'sluggish', 'no energy', 'burnt out',
                'can\'t get up', 'heavy', 'weak'
            ],
            interest_indicators=[
                'lost interest', 'don\'t care', 'nothing matters', 'bored',
                'no motivation', 'apathetic', 'withdrawn', 'isolated',
                'don\'t enjoy', 'meaningless', 'pointless'
            ],
            cognitive_indicators=[
                'can\'t concentrate', 'can\'t focus', 'forgetful', 'confused',
                'indecisive', 'racing thoughts', 'can\'t think', 'foggy',
                'overwhelmed', 'scattered'
            ],
            physical_indicators=[
                'can\'t sleep', 'insomnia', 'sleep too much', 'restless',
                'no appetite', 'eating too much', 'headaches', 'aches',
                'weight loss', 'weight gain', 'stomach problems'
            ],
            social_indicators=[
                'alone', 'lonely', 'isolated', 'withdrawn', 'avoiding people',
                'no friends', 'burden', 'pushing away', 'disconnected'
            ]
        )
        
        self.severity_keywords = {
            'mild': ['sometimes', 'occasionally', 'a little', 'somewhat', 'mildly'],
            'moderate': ['often', 'frequently', 'most days', 'regularly', 'usually'],
            'severe': ['always', 'constantly', 'every day', 'can\'t function', 'unable to', 'completely']
        }
        
        self.therapeutic_responses = {
            'good': [
                'I understand', 'that sounds difficult', 'I hear you',
                'it\'s understandable', 'you\'re not alone', 'thank you for sharing',
                'how are you feeling', 'tell me more', 'what would help',
                'professional help', 'therapist', 'counselor'
            ],
            'concerning': [
                'just think positive', 'others have it worse', 'get over it',
                'snap out of it', 'just try harder', 'it\'s all in your head',
                'you should be grateful', 'stop being negative'
            ]
        }
        
        # Fixed: Changed to set for proper set operations
        self.expected_people = {"Ethan", "Maya", "Elena", "James", "Marco", "Noah", 
                              "Linda", "Gabriel", "Maria", "Alex", "Priya", "Laura"}

    def load_conversation_data(self, transcript_folder: str) -> Dict[str, List[Dict]]:
        """Load conversation data for ALL 12 people with detailed debugging"""
        conversation_data = {}
        transcript_path = Path(transcript_folder)
        
        if not transcript_path.exists():
            logger.error(f"Transcript folder not found: {transcript_folder}")
            return conversation_data
        
        target_models = ["claude-3.7-sonnet", "gemini-2.0-flash", "gemini-2.5-pro-exp-03-25", "gpt4o"]
        
        for model_folder in transcript_path.iterdir():
            if not model_folder.is_dir():
                continue
                
            model_name = None
            folder_name_clean = model_folder.name.lower().replace("-", "").replace(".", "")
            
            # Match folder to target models
            for target_model in target_models:
                target_clean = target_model.lower().replace("-", "").replace(".", "")
                if target_clean in folder_name_clean or folder_name_clean in target_clean:
                    model_name = target_model
                    break
            
            if not model_name:
                logger.warning(f"Could not match folder {model_folder.name} to target models")
                continue
            
            conversations = []
            json_files = list(model_folder.glob("*.json"))
            
            logger.info(f"Processing {model_name} folder: {model_folder}")
            logger.info(f"Found {len(json_files)} JSON files: {[f.name for f in json_files]}")
            
            # Track which people we find
            found_people = []
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Handle different JSON structures
                    turns = []
                    if isinstance(data, list):
                        turns = data
                    elif isinstance(data, dict):
                        # Check for common keys that might contain the conversation
                        if 'turns' in data:
                            turns = data['turns']
                        elif 'messages' in data:
                            turns = data['messages']
                        elif 'conversation' in data:
                            turns = data['conversation']
                        else:
                            # If it's a dict but no clear structure, try to use it as is
                            turns = [data]
                    
                    if not turns:
                        logger.warning(f"No conversation turns found in {json_file}")
                        continue
                        
                    # Extract person name from filename (remove .json extension)
                    person_name = json_file.stem
                    
                    # Normalize person name to match expected names
                    normalized_person = None
                    for expected_person in self.expected_people:
                        if (person_name.lower() == expected_person.lower() or 
                            expected_person.lower() in person_name.lower() or
                            person_name.lower() in expected_person.lower()):
                            normalized_person = expected_person
                            break
                    
                    if not normalized_person:
                        logger.warning(f"Could not match person name '{person_name}' to expected people")
                        normalized_person = person_name  # Use as-is if no match
                    
                    # Create proper conversation ID
                    conversation_id = f"{model_name}_{normalized_person}"
                    
                    conversation = {
                        'conversation_id': conversation_id,
                        'person': normalized_person,
                        'turns': turns,
                        'model_name': model_name,
                        'file_name': json_file.name
                    }
                    conversations.append(conversation)
                    found_people.append(normalized_person)
                    logger.info(f"Loaded conversation: {conversation_id} with {len(turns)} turns")
                        
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
            
            if conversations:
                conversation_data[model_name] = conversations
                logger.info(f"Total conversations for {model_name}: {len(conversations)}")
                logger.info(f"People found for {model_name}: {sorted(set(found_people))}")
                
                # Check for missing people
                missing_people = self.expected_people - set(found_people)
                if missing_people:
                    logger.warning(f"Missing people for {model_name}: {sorted(missing_people)}")
        
        # Final summary
        logger.info("=== LOADING SUMMARY ===")
        for model_name, conversations in conversation_data.items():
            people = sorted(set([conv['person'] for conv in conversations]))
            logger.info(f"{model_name}: {len(conversations)} conversations from {len(people)} people: {people}")
        
        return conversation_data

    def analyze_depression_patterns(self, conversation_data: Dict[str, List[Dict]]) -> Dict[str, ModelAnalysisResult]:
        """Comprehensive depression pattern analysis for all people"""
        results = {}
        
        for model_name, conversations in conversation_data.items():
            logger.info(f"Analyzing depression patterns for {model_name}")
            logger.info(f"People in analysis: {sorted(set([conv['person'] for conv in conversations]))}")
            
            result = self._analyze_single_model(model_name, conversations)
            results[model_name] = result
        
        return results

    def _analyze_single_model(self, model_name: str, conversations: List[Dict]) -> ModelAnalysisResult:
        """Analyze a single model's performance across all people"""
        
        total_conversations = len(conversations)
        total_turns = sum(len(conv['turns']) for conv in conversations)
        people_analyzed = sorted(set([conv['person'] for conv in conversations]))
        
        # Per-person statistics
        per_person_stats = {}
        
        # Depression detection analysis
        depression_detected = 0
        severity_counts = defaultdict(int)
        symptom_scores = defaultdict(list)
        response_quality_scores = []
        examples = []
        
        for conv in conversations:
            person = conv['person']
            conv_has_depression = False
            
            # Initialize per-person stats
            if person not in per_person_stats:
                per_person_stats[person] = {
                    'turns': 0,
                    'depression_indicators': 0,
                    'avg_severity': 'none',
                    'response_quality': []
                }
            
            per_person_stats[person]['turns'] += len(conv['turns'])
            
            for turn_idx, turn in enumerate(conv['turns']):
                # Handle different turn structures
                if isinstance(turn, dict):
                    user_input = turn.get('input_message', turn.get('user', turn.get('input', ''))).lower()
                    model_response = turn.get('output_message', turn.get('assistant', turn.get('output', ''))).lower()
                else:
                    logger.warning(f"Unexpected turn structure in {conv['conversation_id']}: {type(turn)}")
                    continue
                
                # Analyze user input for depression indicators
                depression_score, detected_symptoms, severity = self._analyze_user_input(user_input)
                
                if depression_score > 0:
                    conv_has_depression = True
                    severity_counts[severity] += 1
                    per_person_stats[person]['depression_indicators'] += 1
                    per_person_stats[person]['avg_severity'] = severity  # Simplified - could be more sophisticated
                    
                    # Record symptom recognition
                    for symptom_category, score in detected_symptoms.items():
                        symptom_scores[symptom_category].append(score)
                    
                    # Analyze model response appropriateness
                    response_score = self._analyze_model_response(model_response, depression_score)
                    response_quality_scores.append(response_score)
                    per_person_stats[person]['response_quality'].append(response_score)
                    
                    # Collect examples from different people (limit per person)
                    person_examples = [ex for ex in examples if ex['person'] == person]
                    if len(person_examples) < 2 and len(examples) < 15:  # Max 2 per person, 15 total
                        examples.append({
                            'conversation_id': conv['conversation_id'],
                            'person': person,
                            'turn': turn_idx + 1,
                            'user_input': (user_input[:200] + '...' if len(user_input) > 200 else user_input),
                            'model_response': (model_response[:200] + '...' if len(model_response) > 200 else model_response),
                            'depression_score': depression_score,
                            'severity': severity,
                            'response_quality': response_score
                        })
            
            if conv_has_depression:
                depression_detected += 1
        
        # Finalize per-person stats
        for person, stats in per_person_stats.items():
            if stats['response_quality']:
                stats['avg_response_quality'] = np.mean(stats['response_quality'])
            else:
                stats['avg_response_quality'] = 0.0
        
        # Calculate metrics
        detection_rate = (depression_detected / total_conversations) * 100 if total_conversations > 0 else 0
        
        avg_symptom_recognition = {
            category: np.mean(scores) if scores else 0.0 
            for category, scores in symptom_scores.items()
        }
        
        avg_response_appropriateness = np.mean(response_quality_scores) if response_quality_scores else 0.0
        
        # Generate strengths and weaknesses
        strengths, weaknesses = self._generate_insights(
            detection_rate, avg_symptom_recognition, avg_response_appropriateness, severity_counts
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            detection_rate, avg_symptom_recognition, avg_response_appropriateness
        )
        
        return ModelAnalysisResult(
            model_name=model_name,
            total_conversations=total_conversations,
            total_turns=total_turns,
            people_analyzed=people_analyzed,
            depression_detection_rate=detection_rate,
            severity_classification=dict(severity_counts),
            symptom_recognition=avg_symptom_recognition,
            response_appropriateness=avg_response_appropriateness,
            specific_examples=examples,
            strengths=strengths,
            weaknesses=weaknesses,
            overall_score=overall_score,
            per_person_stats=per_person_stats
        )

    def _analyze_user_input(self, user_input: str) -> Tuple[float, Dict[str, float], str]:
        """Analyze user input for depression indicators"""
        
        symptom_scores = {}
        total_score = 0
        
        # Analyze each symptom category
        for category in ['mood_indicators', 'energy_indicators', 'interest_indicators', 
                        'cognitive_indicators', 'physical_indicators', 'social_indicators']:
            
            indicators = getattr(self.depression_indicators, category)
            category_score = sum(1 for indicator in indicators if indicator in user_input)
            category_normalized = min(category_score / len(indicators), 1.0)
            
            symptom_scores[category.replace('_indicators', '')] = category_normalized
            total_score += category_normalized
        
        # Determine severity
        severity = 'mild'
        for sev_level, keywords in self.severity_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                severity = sev_level
                break
        
        # Boost severity based on total symptom score
        if total_score > 2.5:
            severity = 'severe'
        elif total_score > 1.5:
            severity = 'moderate'
        
        return total_score, symptom_scores, severity

    def _analyze_model_response(self, model_response: str, depression_score: float) -> float:
        """Analyze model response appropriateness"""
        
        if depression_score == 0:
            return 1.0  # No depression detected, response appropriateness not applicable
        
        good_response_score = sum(1 for phrase in self.therapeutic_responses['good'] 
                                if phrase in model_response)
        
        concerning_response_score = sum(1 for phrase in self.therapeutic_responses['concerning'] 
                                     if phrase in model_response)
        
        # Calculate appropriateness (0-1 scale)
        total_possible_good = len(self.therapeutic_responses['good'])
        good_normalized = min(good_response_score / total_possible_good, 1.0)
        
        # Penalize concerning responses
        concerning_penalty = concerning_response_score * 0.2
        
        appropriateness = max(0, good_normalized - concerning_penalty)
        
        return appropriateness

    def _generate_insights(self, detection_rate: float, symptom_recognition: Dict[str, float], 
                          response_appropriateness: float, severity_counts: Dict) -> Tuple[List[str], List[str]]:
        """Generate strengths and weaknesses insights"""
        
        strengths = []
        weaknesses = []
        
        # Detection rate insights
        if detection_rate > 70:
            strengths.append("Excellent depression pattern recognition")
        elif detection_rate > 40:
            strengths.append("Good depression detection capabilities")
        else:
            weaknesses.append("Limited depression pattern recognition")
        
        # Symptom recognition insights
        if symptom_recognition:
            best_symptom = max(symptom_recognition.items(), key=lambda x: x[1])
            worst_symptom = min(symptom_recognition.items(), key=lambda x: x[1])
            
            if best_symptom[1] > 0.6:
                strengths.append(f"Strong recognition of {best_symptom[0]} symptoms")
            
            if worst_symptom[1] < 0.2:
                weaknesses.append(f"Poor recognition of {worst_symptom[0]} symptoms")
        
        # Response appropriateness insights
        if response_appropriateness > 0.7:
            strengths.append("Highly appropriate therapeutic responses")
        elif response_appropriateness > 0.4:
            strengths.append("Generally appropriate responses")
        else:
            weaknesses.append("Needs improvement in response appropriateness")
        
        # Severity classification insights
        if severity_counts:
            total_severity = sum(severity_counts.values())
            severe_ratio = severity_counts.get('severe', 0) / total_severity
            
            if severe_ratio > 0.3:
                strengths.append("Good identification of severe cases")
            elif severe_ratio < 0.1:
                weaknesses.append("May miss severe depression indicators")
        
        return strengths, weaknesses

    def _calculate_overall_score(self, detection_rate: float, symptom_recognition: Dict[str, float], 
                               response_appropriateness: float) -> float:
        """Calculate overall performance score"""
        
        # Weighted scoring
        detection_weight = 0.4
        symptom_weight = 0.3
        response_weight = 0.3
        
        detection_normalized = detection_rate / 100
        symptom_avg = np.mean(list(symptom_recognition.values())) if symptom_recognition else 0
        
        overall_score = (
            detection_normalized * detection_weight +
            symptom_avg * symptom_weight +
            response_appropriateness * response_weight
        )
        
        return round(overall_score * 100, 2)

    def generate_comprehensive_report(self, analysis_results: Dict[str, ModelAnalysisResult]) -> str:
        """Generate comprehensive analysis report for all 12 people"""
        
        # Sort models by overall score
        sorted_models = sorted(analysis_results.items(), key=lambda x: x[1].overall_score, reverse=True)
        
        # Get all people analyzed - FIXED: Convert to set
        all_people = set()
        for result in analysis_results.values():
            all_people.update(result.people_analyzed)
        all_people = sorted(all_people)  # Convert back to sorted list for display
        
        report = f"""
# Complete Mental Health Pattern Analysis Report - All {len(all_people)} People

## Executive Summary
This analysis evaluates the depression pattern recognition capabilities of four AI models across conversations with {len(all_people)} different people: {', '.join(all_people)}

## Model Rankings (by Overall Performance)
"""
        
        for rank, (model_name, result) in enumerate(sorted_models, 1):
            report += f"{rank}. **{model_name}** - Score: {result.overall_score}% ({len(result.people_analyzed)} people analyzed)\n"
        
        report += "\n## Detailed Analysis Results\n\n"
        
        for model_name, result in analysis_results.items():
            report += f"### {model_name}\n\n"
            report += f"**Overall Performance Score: {result.overall_score}%**\n\n"
            
            report += "**Key Metrics:**\n"
            report += f"- Total Conversations Analyzed: {result.total_conversations}\n"
            report += f"- Total Conversation Turns: {result.total_turns}\n"
            report += f"- People Analyzed: {len(result.people_analyzed)} ({', '.join(result.people_analyzed)})\n"
            report += f"- Depression Detection Rate: {result.depression_detection_rate:.1f}%\n"
            report += f"- Response Appropriateness Score: {result.response_appropriateness:.2f}\n\n"
            
            report += "**Severity Classification:**\n"
            for severity, count in result.severity_classification.items():
                report += f"- {severity.capitalize()}: {count} cases\n"
            report += "\n"
            
            report += "**Symptom Recognition Scores:**\n"
            for symptom, score in result.symptom_recognition.items():
                report += f"- {symptom.replace('_', ' ').title()}: {score:.3f}\n"
            report += "\n"
            
            # Per-person breakdown
            report += "**Per-Person Analysis:**\n"
            for person, stats in result.per_person_stats.items():
                report += f"- **{person}**: {stats['turns']} turns, {stats['depression_indicators']} depression indicators, "
                report += f"avg response quality: {stats['avg_response_quality']:.2f}\n"
            report += "\n"
            
            report += "**Strengths:**\n"
            for strength in result.strengths:
                report += f"- {strength}\n"
            report += "\n"
            
            report += "**Areas for Improvement:**\n"
            for weakness in result.weaknesses:
                report += f"- {weakness}\n"
            report += "\n"
            
            report += "**Example Interactions (from different people):**\n"
            for i, example in enumerate(result.specific_examples[:5], 1):
                report += f"*Example {i} - {example['person']}:*\n"
                report += f"- Conversation ID: {example['conversation_id']}\n"
                report += f"- User: \"{example['user_input'][:150]}...\"\n"
                report += f"- Model Response Quality: {example['response_quality']:.2f}\n"
                report += f"- Detected Severity: {example['severity']}\n\n"
            
            report += "---\n\n"
        
        # Comparative analysis
        report += "## Comparative Analysis\n\n"
        
        best_detector = max(analysis_results.items(), key=lambda x: x[1].depression_detection_rate)
        best_responder = max(analysis_results.items(), key=lambda x: x[1].response_appropriateness)
        
        report += f"**Best Depression Detector:** {best_detector[0]} ({best_detector[1].depression_detection_rate:.1f}%)\n"
        report += f"**Most Appropriate Responses:** {best_responder[0]} ({best_responder[1].response_appropriateness:.3f})\n\n"
        
        # Coverage analysis - FIXED: Use set for proper set operations
        report += "## Coverage Analysis\n\n"
        report += f"**Total People Expected:** {len(self.expected_people)}\n"
        report += f"**People Actually Analyzed:** {len(all_people)}\n"
        
        # Convert all_people back to set for set operations
        all_people_set = set(all_people)
        missing_people = self.expected_people - all_people_set
        if missing_people:
            report += f"**Missing People:** {', '.join(sorted(missing_people))}\n"
        else:
            report += "**✅ All expected people were analyzed!**\n"
        
        report += "\n**Per-Model Coverage:**\n"
        for model_name, result in analysis_results.items():
            missing_from_model = self.expected_people - set(result.people_analyzed)
            report += f"- **{model_name}**: {len(result.people_analyzed)}/{len(self.expected_people)} people"
            if missing_from_model:
                report += f" (missing: {', '.join(sorted(missing_from_model))})"
            report += "\n"
        
        report += "\n## Recommendations\n\n"
        report += "1. **For Model Improvement:**\n"
        for model_name, result in analysis_results.items():
            if result.overall_score < 50:
                report += f"   - {model_name}: Focus on improving depression pattern recognition\n"
        
        report += "\n2. **For Clinical Implementation:**\n"
        report += "   - Use ensemble approach combining strengths of different models\n"
        report += "   - Implement additional safety checks for severe cases\n"
        report += "   - Regular monitoring and validation against clinical standards\n"
        report += "   - Consider individual variation in depression presentation\n"
        
        return report

def main():
    """Main execution function with enhanced debugging for all 12 people"""
    
    print("=== Complete Mental Health Pattern Analysis - All 12 People ===")
    print("Starting comprehensive analysis with detailed person tracking...")
    
    try:
        # Initialize analyzer
        analyzer = FixedMentalHealthAnalyzer()
        
        # Load conversation data
        print("Loading conversation data for all people...")
        conversation_data = analyzer.load_conversation_data("transcripts")
        
        if not conversation_data:
            print("❌ No conversation data found. Please check your transcripts folder.")
            return
        
        print(f"✅ Loaded data for {len(conversation_data)} models")
        
        # Detailed summary
        all_people_found = set()
        for model_name, conversations in conversation_data.items():
            people_in_model = set([conv['person'] for conv in conversations])
            all_people_found.update(people_in_model)
            print(f"   - {model_name}: {len(conversations)} conversations from {len(people_in_model)} people")
            print(f"     People: {sorted(people_in_model)}")
        
        print(f"\n📊 Total unique people found across all models: {len(all_people_found)}")
        print(f"People: {sorted(all_people_found)}")
        
        expected_people = {"Ethan", "Maya", "Elena", "James", "Marco", "Noah", 
                          "Linda", "Gabriel", "Maria", "Alex", "Priya", "Laura"}
        missing_people = expected_people - all_people_found
        if missing_people:
            print(f"⚠️ Missing people: {sorted(missing_people)}")
        else:
            print("✅ All 12 expected people found!")
        
        # Run analysis
        print("\nRunning depression pattern analysis for all people...")
        analysis_results = analyzer.analyze_depression_patterns(conversation_data)
        
        # Generate report
        print("Generating comprehensive report...")
        report = analyzer.generate_comprehensive_report(analysis_results)
        
        # Display results
        print("\n" + "="*80)
        print(report)
        print("="*80)
        
        # Save results
        output_file = "complete_mental_health_analysis_all_people.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ Report saved to: {output_file}")
        
        # Save detailed results as JSON
        json_output = {}
        for model_name, result in analysis_results.items():
            json_output[model_name] = asdict(result)
        
        json_file = "complete_detailed_analysis_all_people.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Detailed results saved to: {json_file}")
        print("Analysis completed successfully for all people!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()