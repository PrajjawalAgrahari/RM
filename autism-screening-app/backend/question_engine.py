"""
Follow-up Question Engine
Generates intelligent follow-up questions based on symptom severity
"""

from typing import List, Dict, Tuple, Optional
import random


class FollowUpQuestionEngine:
    """Generates intelligent follow-up questions based on symptom severity"""
    
    FOLLOW_UP_QUESTIONS = {
        "social_communication": {
            "high": [
                "Can you describe a recent situation where your child had difficulty maintaining a conversation with peers?",
                "How does your child typically respond when someone tries to start a conversation with them?",
                "Does your child understand when someone is joking or using sarcasm?",
                "How does your child express their needs or feelings to you?",
                "Does your child make eye contact during conversations?"
            ],
            "moderate": [
                "In group settings, how often does your child participate in conversations?",
                "Can your child take turns in a conversation, or do they tend to dominate or withdraw?",
                "How does your child react when meeting new people?",
                "Does your child ask questions when they don't understand something?"
            ]
        },
        "sensory_processing": {
            "high": [
                "What specific sounds or sensations seem to bother your child the most?",
                "Does your child have strong preferences for certain textures in clothing or food?",
                "How does your child react to loud or unexpected noises?",
                "Are there situations where sensory input causes visible distress?",
                "Does your child cover their ears or eyes in certain situations?"
            ],
            "moderate": [
                "Does your child notice or react to things that others might not?",
                "How does your child handle busy or crowded environments?",
                "Are there certain smells or tastes your child is particularly sensitive to?"
            ]
        },
        "emotional_recognition": {
            "high": [
                "Can your child identify basic emotions (happy, sad, angry) in others by looking at their faces?",
                "Does your child seem to understand when someone is upset or hurt?",
                "How does your child respond to others' emotions - do they offer comfort or seem unaware?",
                "Can your child explain how they're feeling when asked?",
                "Does your child understand facial expressions and body language?"
            ],
            "moderate": [
                "Does your child pick up on social cues like body language or tone of voice?",
                "How well does your child understand others' perspectives in stories or situations?",
                "Can your child predict how someone might feel in a given situation?"
            ]
        },
        "social_interaction": {
            "high": [
                "Does your child prefer to play alone, or do they seek out other children?",
                "How does your child engage in play - do they prefer solitary activities or cooperative games?",
                "Does your child show interest in what other children are doing or playing?",
                "How does your child respond when invited to join a group activity?",
                "Does your child initiate play with others, or wait to be approached?"
            ],
            "moderate": [
                "How many close friendships does your child have?",
                "Does your child understand the concept of sharing and taking turns?",
                "How does your child handle conflicts or disagreements with peers?"
            ]
        }
    }
    
    def get_follow_up_questions(self, symptom_severity: Dict, 
                               already_asked: List[str] = None) -> List[Tuple[str, str, str]]:
        """
        Generate follow-up questions based on symptom severity
        
        Args:
            symptom_severity: Dict of symptom areas and their severity levels
            already_asked: List of questions already asked
        
        Returns: 
            List of (question, symptom_area, severity_level) tuples
        """
        if already_asked is None:
            already_asked = []
        
        questions = []
        
        # Prioritize high severity symptoms
        sorted_symptoms = sorted(
            symptom_severity.items(),
            key=lambda x: {"high": 3, "moderate": 2, "low": 1}.get(x[1], 0),
            reverse=True
        )
        
        for symptom_area, severity_level in sorted_symptoms:
            if severity_level in ["high", "moderate"]:
                available_questions = self.FOLLOW_UP_QUESTIONS.get(symptom_area, {}).get(severity_level, [])
                
                # Filter out already asked questions
                new_questions = [q for q in available_questions if q not in already_asked]
                
                if new_questions:
                    # Pick 1-2 questions from this category
                    num_questions = 2 if severity_level == "high" else 1
                    selected = random.sample(new_questions, min(num_questions, len(new_questions)))
                    
                    for q in selected:
                        questions.append((q, symptom_area, severity_level))
        
        return questions[:3]  # Return top 3 questions
    
    def get_next_question_for_area(self, symptom_area: str, severity_level: str, 
                                   already_asked: List[str]) -> Optional[str]:
        """Get next question for specific symptom area"""
        available = self.FOLLOW_UP_QUESTIONS.get(symptom_area, {}).get(severity_level, [])
        remaining = [q for q in available if q not in already_asked]
        
        if remaining:
            return random.choice(remaining)
        return None
