"""
Follow-up Question Generator
Generates mixed-domain follow-up questions based on symptom severity
"""

from typing import List, Dict, Tuple
import random


class FollowUpGenerator:
    """Generates intelligent follow-up questions balanced across all symptom domains"""
    
    # Comprehensive question bank organized by domain and severity
    QUESTIONS = {
        "social_communication": {
            "high": [
                "Can you describe a recent situation where your child had difficulty starting or maintaining a conversation?",
                "How does your child typically respond when peers try to talk to them?",
                "Does your child understand sarcasm, jokes, or figures of speech?",
                "How does your child share their interests or excitement with others?",
                "Does your child use gestures (pointing, waving) when communicating?"
            ],
            "moderate": [
                "How often does your child initiate conversations with family members?",
                "Does your child ask questions to learn more about topics they're interested in?",
                "How does your child react when someone doesn't understand what they're saying?"
            ],
            "low": [
                "Does your child enjoy talking about their favorite topics?",
                "How does your child communicate when they need help?"
            ]
        },
        "sensory_processing": {
            "high": [
                "What specific sounds, textures, or sensations seem to overwhelm your child?",
                "Does your child have strong preferences about clothing tags, seams, or fabrics?",
                "How does your child react to unexpected loud noises (vacuum, siren, etc.)?",
                "Are there certain foods your child refuses based on texture rather than taste?",
                "Does your child seek out or avoid certain physical sensations (spinning, tight hugs, etc.)?"
            ],
            "moderate": [
                "How does your child handle crowded or noisy environments like malls or parties?",
                "Does your child notice smells or sounds that others might not?",
                "Are there activities your child avoids due to sensory discomfort?"
            ],
            "low": [
                "Does your child have any specific sensory preferences?",
                "How does your child react to trying new foods?"
            ]
        },
        "emotional_recognition": {
            "high": [
                "Can your child identify basic emotions (happy, sad, angry) in others by their facial expressions?",
                "Does your child seem to understand when someone is upset or hurt without being told?",
                "How does your child respond when they see someone crying or in distress?",
                "Can your child explain how they're feeling when you ask them?",
                "Does your child recognize when their behavior has upset someone?"
            ],
            "moderate": [
                "Does your child pick up on subtle social cues like tone of voice or body language?",
                "How well does your child understand other people's perspectives in stories or real situations?",
                "Does your child show empathy when someone is hurt or sad?"
            ],
            "low": [
                "How does your child express their own emotions?",
                "Does your child comfort others when they're upset?"
            ]
        },
        "social_interaction": {
            "high": [
                "Does your child prefer to play alone or with others?",
                "How does your child engage during playtime - do they prefer parallel play or interactive games?",
                "Does your child show interest in what other children are doing or playing?",
                "How does your child respond when invited to join a group activity?",
                "Does your child initiate play with peers or wait to be invited?"
            ],
            "moderate": [
                "How many close friendships does your child have?",
                "Does your child understand concepts like sharing and taking turns?",
                "How does your child handle conflicts or disagreements with peers?"
            ],
            "low": [
                "Does your child enjoy group activities?",
                "How does your child interact with siblings or cousins?"
            ]
        }
    }
    
    def __init__(self):
        self.asked_questions = {}  # Track questions per session
    
    def generate_mixed_followups(self, session_id: str, symptom_severity: Dict, 
                                 questions_per_area: Dict, num_suggestions: int = 4) -> List[Dict]:
        """
        Generate mixed follow-up questions across all domains
        Prioritizes high-severity areas but ensures variety
        
        Args:
            session_id: Session identifier
            symptom_severity: Dict mapping symptom areas to severity levels
            questions_per_area: Dict tracking how many questions asked per area
            num_suggestions: Number of questions to generate
            
        Returns:
            List of dicts with question, symptom_area, severity, priority
        """
        if session_id not in self.asked_questions:
            self.asked_questions[session_id] = set()
        
        # Create priority list: (area, severity_level, priority_score)
        priority_areas = []
        for area, severity in symptom_severity.items():
            if severity in ["high", "moderate"]:
                asked_count = questions_per_area.get(area, 0)
                target_count = 4 if severity == "high" else 2
                
                if asked_count < target_count:
                    # Higher severity and fewer questions = higher priority
                    severity_weight = 10 if severity == "high" else 5
                    balance_weight = (target_count - asked_count) * 2
                    priority_score = severity_weight + balance_weight
                    
                    priority_areas.append((area, severity, priority_score, asked_count, target_count))
        
        # Sort by priority (highest first)
        priority_areas.sort(key=lambda x: x[2], reverse=True)
        
        # Generate questions with domain mixing
        suggested_followups = []
        used_areas = set()
        
        for area, severity, priority, asked, target in priority_areas:
            if len(suggested_followups) >= num_suggestions:
                break
            
            # Get available questions for this area
            available_questions = self.QUESTIONS.get(area, {}).get(severity, [])
            
            # Filter out already-asked questions
            available_questions = [
                q for q in available_questions 
                if f"{area}:{q}" not in self.asked_questions[session_id]
            ]
            
            if available_questions:
                # For high priority area, include 2 questions; otherwise 1
                num_from_area = 2 if priority >= 15 and len(suggested_followups) < num_suggestions - 1 else 1
                
                selected = random.sample(available_questions, min(num_from_area, len(available_questions)))
                
                for question in selected:
                    suggested_followups.append({
                        "question": question,
                        "symptom_area": area,
                        "severity": severity,
                        "priority": priority
                    })
                    
                    # Mark as asked
                    self.asked_questions[session_id].add(f"{area}:{question}")
                
                used_areas.add(area)
        
        # If we don't have enough questions, add from low-severity areas
        if len(suggested_followups) < num_suggestions:
            for area, severity in symptom_severity.items():
                if len(suggested_followups) >= num_suggestions:
                    break
                
                if area not in used_areas and severity == "low":
                    available_questions = self.QUESTIONS.get(area, {}).get(severity, [])
                    available_questions = [
                        q for q in available_questions 
                        if f"{area}:{q}" not in self.asked_questions[session_id]
                    ]
                    
                    if available_questions:
                        question = random.choice(available_questions)
                        suggested_followups.append({
                            "question": question,
                            "symptom_area": area,
                            "severity": severity,
                            "priority": 1
                        })
                        self.asked_questions[session_id].add(f"{area}:{question}")
        
        return suggested_followups[:num_suggestions]
    
    def clear_session(self, session_id: str):
        """Clear asked questions for a session"""
        if session_id in self.asked_questions:
            del self.asked_questions[session_id]
